import torch
import torch.nn as nn
import time
import torch.nn.functional as F

# Import necessary functions from the quantize.py file
def get_group_qparams(w, n_bit=4, groupsize=128):
    # needed for GPTQ with padding
    if groupsize > w.shape[-1]:
        groupsize = w.shape[-1]
    assert groupsize > 1
    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    max_val = to_quant.amax(dim=1, keepdim=True)
    min_val = to_quant.amin(dim=1, keepdim=True)
    max_int = 2**n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-6) / max_int
    zeros = min_val + scales * (2 ** (n_bit - 1))
    return scales.to(torch.bfloat16).reshape(w.shape[0], -1), zeros.to(
        torch.bfloat16
    ).reshape(w.shape[0], -1)

def pack_scales_and_zeros(scales, zeros):
    assert scales.shape == zeros.shape
    assert scales.dtype == torch.bfloat16
    assert zeros.dtype == torch.bfloat16
    return (
        torch.cat(
            [
                scales.reshape(scales.size(0), scales.size(1), 1),
                zeros.reshape(zeros.size(0), zeros.size(1), 1),
            ],
            2,
        )
        .transpose(0, 1)
        .contiguous()
    )

def group_quantize_tensor(w, n_bit=4, groupsize=128):
    scales, zeros = get_group_qparams(w, n_bit, groupsize)
    w_int32 = group_quantize_tensor_from_qparams(w, scales, zeros, n_bit, groupsize)
    scales_and_zeros = pack_scales_and_zeros(scales, zeros)
    return w_int32, scales_and_zeros

def group_quantize_tensor_from_qparams(w, scales, zeros, n_bit=4, groupsize=128):
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    w_int32 = (
        to_quant.sub(min_val)
        .div(scales)
        .round()
        .clamp_(min_int, max_int)
        .to(torch.int32)
        .reshape_as(w)
    )

    return w_int32

def prepare_int4_weight_and_scales_and_zeros(weight_bf16, groupsize, inner_k_tiles):
    weight_int32, scales_and_zeros = group_quantize_tensor(
        weight_bf16, n_bit=4, groupsize=groupsize
    )
    weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(weight_int32, inner_k_tiles)
    return weight_int4pack, scales_and_zeros

def linear_forward_int4(x, weight_int4pack, scales_and_zeros, out_features, groupsize):
    origin_x_size = x.size()
    x = x.reshape(-1, origin_x_size[-1])
    c = torch.ops.aten._weight_int4pack_mm(x, weight_int4pack, groupsize, scales_and_zeros)
    new_shape = origin_x_size[:-1] + (out_features,)
    c = c.reshape(new_shape)
    return c

class WeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
            self, in_features: int, out_features: int,
            bias=False, device=None, dtype=None, groupsize: int = 128, inner_k_tiles: int = 8
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert in_features % (inner_k_tiles * 16) == 0, "require in_features % (innerKTiles * 16) == 0"
        self.register_buffer(
            "weight",
            torch.empty((out_features // 8, in_features // (inner_k_tiles * 16), 32, inner_k_tiles // 2), dtype=torch.int32)
        )
        self.register_buffer(
            "scales_and_zeros",
            torch.empty((in_features // groupsize, out_features, 2), dtype=torch.bfloat16)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.to(torch.bfloat16)
        return linear_forward_int4(
            input,
            self.weight, self.scales_and_zeros, self.out_features, self.groupsize
        )

# Define dimensions that satisfy the requirements for INT4 quantization
# in_features must be divisible by inner_k_tiles * 16
# out_features must be divisible by 8
in_features = 1024  # Must be divisible by inner_k_tiles * 16
out_features = 2048  # Must be divisible by 8
groupsize = 128
inner_k_tiles = 8

# Create models
fp16_model = nn.Sequential(
    nn.Linear(in_features, out_features, bias=False)
)

# Create INT4 model
int4_model = nn.Sequential(
    WeightOnlyInt4Linear(in_features, out_features, bias=False, 
                         groupsize=groupsize, inner_k_tiles=inner_k_tiles)
)

# Quantize the weights and set up the INT4 model
with torch.no_grad():
    # Convert FP16 weights to INT4
    fp16_weight = fp16_model[0].weight.data.to(torch.bfloat16)
    weight_int4pack, scales_and_zeros = prepare_int4_weight_and_scales_and_zeros(
        fp16_weight, groupsize, inner_k_tiles
    )
    
    # Set the quantized weights in the INT4 model
    int4_model[0].weight.copy_(weight_int4pack)
    int4_model[0].scales_and_zeros.copy_(scales_and_zeros)

# Move models to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fp16_model = fp16_model.to(device)
int4_model = int4_model.to(device)

# Create random input tensor
batch_size = 1024
input_tensor = torch.randn(batch_size, in_features, device=device)
input_tensor_bf16 = input_tensor.to(torch.bfloat16)

# Speed test function
def speed_test(model, input_tensor, name, num_iterations=100):
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    # Actual timing
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_iterations):
        _ = model(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    print(f"{name} model: {avg_time:.6f} seconds per iteration")
    return avg_time

# Run speed tests
with torch.no_grad():  # Disable gradient calculation for inference
    print(f"Running benchmark with batch_size={batch_size}, in_features={in_features}, out_features={out_features}")
    print(f"INT4 parameters: groupsize={groupsize}, inner_k_tiles={inner_k_tiles}")
    
    fp16_time = speed_test(fp16_model, input_tensor_bf16, "FP16")
    int4_time = speed_test(int4_model, input_tensor, "INT4")
    
    # Calculate speedup
    speedup = fp16_time / int4_time
    print(f"INT4 is {speedup:.2f}x faster than FP16")
    
    # Calculate memory savings
    fp16_memory = fp16_model[0].weight.nelement() * fp16_model[0].weight.element_size()
    int4_memory = (int4_model[0].weight.nelement() * int4_model[0].weight.element_size() + 
                  int4_model[0].scales_and_zeros.nelement() * int4_model[0].scales_and_zeros.element_size())
    
    memory_reduction = fp16_memory / int4_memory
    print(f"Memory reduction: {memory_reduction:.2f}x ({fp16_memory/1024/1024:.2f} MB vs {int4_memory/1024/1024:.2f} MB)")
    
    # Check accuracy
    with torch.no_grad():
        fp16_output = fp16_model(input_tensor_bf16)
        int4_output = int4_model(input_tensor)
        
        # Calculate error metrics
        abs_error = torch.abs(fp16_output - int4_output)
        rel_error = abs_error / (torch.abs(fp16_output) + 1e-7)
        
        print(f"Mean absolute error: {abs_error.mean().item():.6f}")
        print(f"Max absolute error: {abs_error.max().item():.6f}")
        print(f"Mean relative error: {rel_error.mean().item():.6f}") 
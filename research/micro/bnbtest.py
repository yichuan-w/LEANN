import torch
import torch.nn as nn
import time

# import bitsandbytes as bnb
from bitsandbytes.nn import Linear8bitLt

# set default to half
import torch
torch.set_default_dtype(torch.float16)

M = 2048
N = 2048

bsz =  2048
import torch_int
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear, W8A8B8O8LinearReLU

fp16_model = nn.Sequential(
    nn.Linear(M, N),
    # nn.Linear(2048, 2048)
)

int8_model = nn.Sequential(
    Linear8bitLt(M, N, has_fp16_weights=False),
    # Linear8bitLt(2048, 2048, has_fp16_weights=False)
)

int8_model.load_state_dict(fp16_model.state_dict())
int8_model = int8_model.to(0) # Quantization happens here
fp16_model = fp16_model.to(0) # Move fp16 model to GPU as well

# Create random input tensor
input_tensor = torch.randn(bsz, M, device=0)  # Batch of 1000 vectors

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
    fp16_time = speed_test(fp16_model, input_tensor, "FP16")
    int8_time = speed_test(int8_model, input_tensor, "INT8")
    
    # Calculate speedup
    speedup = fp16_time / int8_time
    print(f"INT8 is {speedup:.2f}x faster than FP16")
import torch
import nvmath.bindings.cublas
import ctypes

# 创建 CUBLAS 句柄
handle = nvmath.bindings.cublas.create()

# 准备数据 - 使用 uint8 类型，并确保内存连续
m, n, k = 64, 32, 48
a = (torch.rand(m, k, device="cuda") * 255).to(torch.uint8).contiguous()
b = (torch.rand(k, n, device="cuda") * 255).to(torch.uint8).contiguous()
c = torch.zeros(m, n, device="cuda", dtype=torch.uint8).contiguous()

# 确保张量在 CUDA 上
assert a.is_cuda and b.is_cuda and c.is_cuda
# 确保张量是连续的
assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous()

# 获取指针
a_ptr = a.data_ptr()
b_ptr = b.data_ptr()
c_ptr = c.data_ptr()

# 设置参数
transa = 0  # CUBLAS_OP_N (不转置)
transb = 0  # CUBLAS_OP_N (不转置)
transc = 0  # CUBLAS_OP_N (不转置)

# 设置偏置值
a_bias = 0
b_bias = 0
c_bias = 0

# 设置正确的 leading dimensions
lda = k  # A 的 leading dimension
ldb = n  # B 的 leading dimension
ldc = n  # C 的 leading dimension

c_mult = 1
c_shift = 0

# 打印调试信息
print(f"a shape: {a.shape}, a_ptr: {a_ptr}")
print(f"b shape: {b.shape}, b_ptr: {b_ptr}")
print(f"c shape: {c.shape}, c_ptr: {c_ptr}")

try:
    # 调用 uint8gemm_bias
    nvmath.bindings.cublas.uint8gemm_bias(
        handle,
        transa, transb, transc,
        m, n, k,
        a_ptr, a_bias, lda,
        b_ptr, b_bias, ldb,
        c_ptr, c_bias, ldc,
        c_mult, c_shift
    )
except Exception as e:
    print(f"Error: {e}")
    # 尝试使用 ctypes 转换指针
    a_ptr_c = ctypes.c_void_p(a_ptr).value
    b_ptr_c = ctypes.c_void_p(b_ptr).value
    c_ptr_c = ctypes.c_void_p(c_ptr).value
    
    print(f"Using ctypes: a_ptr: {a_ptr_c}, b_ptr: {b_ptr_c}, c_ptr: {c_ptr_c}")
    
    # 再次尝试调用
    nvmath.bindings.cublas.uint8gemm_bias(
        handle,
        transa, transb, transc,
        m, n, k,
        a_ptr_c, a_bias, lda,
        b_ptr_c, b_bias, ldb,
        c_ptr_c, c_bias, ldc,
        c_mult, c_shift
    )

# 销毁 CUBLAS 句柄
nvmath.bindings.cublas.destroy(handle)

# 打印结果
print("Result:")
print(c)
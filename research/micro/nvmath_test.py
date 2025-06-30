# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates basic matrix multiplication of FP8 tensors.

In narrow-precision operations, quantization scales must be provided for each tensor. These
scales are used to dequantize input operands and quantize the result. Without proper
scaling, the results of FP8 operations will likely exceed the type's range.

FP8 is only supported with cuBLAS 12.8 or newer and on devices with compute
capability 8.9 or higher.
"""

import torch

import nvmath

# Prepare sample input data. Note that N, M and K must be divisible by 16 for FP8.
# cuBLAS requires B to be column-major, so we first create a row-major tensor and then
# transpose it.
m, n, k = 64, 32, 48
a = (torch.rand(m, k, device="cuda") * 10).type(torch.float8_e4m3fn)
b = (torch.rand(n, k, device="cuda") * 10).type(torch.float8_e4m3fn).T

# Prepare quantization scales. The scales must allow the result to fit within the dynamic
# range of the data type used. Scales can be provided either as a dictionary or as a
# MatmulQuantizationScales object. Note that scales are only allowed for FP8 operands.
scales = {"a": 1, "b": 1, "d": 0.1}

# Perform the multiplication. The result of the multiplication will be:
# (scales.a * A) @ (scales.b * B) * scales.d
result = nvmath.linalg.advanced.matmul(a, b, quantization_scales=scales)

# Check how scaling helped to fit into the dynamic range of float8_e4m3fn type.
result_without_scaling = nvmath.linalg.advanced.matmul(a, b, quantization_scales={"a": 1, "b": 1, "d": 1})
print("Without scaling, most of the elements were clamped to the maximum value of float8_e4m3fn type (448):")
print(result_without_scaling)
print(f"\nWith D scale set to {scales['d']}, they were scaled down to fit into the dynamic range of float8_e4m3fn:")
print(result)
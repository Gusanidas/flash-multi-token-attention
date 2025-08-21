#!/usr/bin/env python3
"""
Simplified test for forward convolution attention inference function
"""
import torch
import numpy as np
import sys
import os
import time

# Add parent directory to path for imports
#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
t0 = time.time()

from flash_multi_token_attention.conv_attn.flash_conv_attention_inference import (
    fwd_conv_attn_inference,
)
from flash_multi_token_attention.conv_attn.conv_attention_pytorch import ConvAttention


def print_stats(tensor, title):
    """Print statistics for debugging"""
    print(f"{title} - mean: {tensor.mean().item():.6f}")
    print(f"{title} - abs_mean: {tensor.abs().mean().item():.6f}")
    print(f"{title} - std: {tensor.std().item():.6f}")
    print(f"{title} - abs_max: {tensor.abs().max().item():.6f}")
    print()


# Hardcoded test parameters
batch_size = 1
n_heads = 1
query_len = 16  # Less than 16
seq_len = 240
d_head = 32
kernel_size_q = 5
kernel_size_k = 5
k_splits = 4
causal = False
dtype = torch.bfloat16
# Scale factor
softmax_scale = 1.0 / np.sqrt(d_head)
#softmax_scale = 1.0

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Q, K, V, and W tensors
Q = torch.randn(batch_size, n_heads, query_len, d_head, device=device, dtype=dtype)
K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
V = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
conv_kernel = (
    torch.randn(n_heads, kernel_size_q, kernel_size_k, device=device, dtype=dtype)
    * 0.02
)

print(
    f"Testing with: batch={batch_size}, heads={n_heads}, query_len={query_len}, seq_len={seq_len}, d_head={d_head}"
)
print(
    f"Conv kernel: {kernel_size_q}x{kernel_size_k}, k_splits={k_splits}, causal={causal}"
)
print()

print_stats(Q, "Q")
print_stats(K, "K")
print_stats(V, "V")
print_stats(conv_kernel, "Conv kernel")


# PyTorch forward pass - get the last value
conv_attn = ConvAttention(n_heads, kernel_size_q, kernel_size_k, init_random=True)
conv_attn.conv_weight.data = conv_kernel
conv_attn = conv_attn.to(device)

with torch.no_grad():
    pytorch_output, pytorch_M, pytorch_L = conv_attn(
        Q, K, V, causal=causal, softmax_scale=softmax_scale
    )
    # Get only the last query token output
    pytorch_output = pytorch_output[:, :, -1:, :]  # [batch, heads, 1, d_head]
    pytorch_M = pytorch_M[:, :, -1:]  # [batch, heads, 1]
    pytorch_L = pytorch_L[:, :, -1:]  # [batch, heads, 1]

print("PyTorch results:")
print(f"PyTorch output shape: {pytorch_output.shape}")
print_stats(pytorch_output, "PyTorch output")
print_stats(pytorch_M, "PyTorch M")
print_stats(pytorch_L, "PyTorch L")
print(f"Last M values: {pytorch_M[:, :, -1:].flatten()}")
print(f"Last L values: {pytorch_L[:, :, -1:].flatten()}")

# Print actual output values if seq_len < 33
if seq_len < 33 or True:
    print("\n--- PyTorch Output Values (seq_len < 33) ---")
    print(f"PyTorch output:\n{pytorch_output}")
    print(f"PyTorch M:\n{pytorch_M}")
    print(f"PyTorch L:\n{pytorch_L}")

print()

# CUDA forward pass
pad_q = kernel_size_q - 1
pad_k = (kernel_size_k - 1) // 2

cuda_output, cuda_M, cuda_L = fwd_conv_attn_inference(
    Q, K, V, conv_kernel, k_splits, softmax_scale, pad_q, pad_k, causal
)

print("CUDA results:")
print(f"CUDA output shape: {cuda_output.shape}")
print_stats(cuda_output, "CUDA output")
print_stats(cuda_M, "CUDA M")
print_stats(cuda_L, "CUDA L")

# Print actual output values if seq_len < 33
if seq_len < 33 or True:
    print("\n--- CUDA Output Values (seq_len < 33) ---")
    print(f"CUDA output:\n{cuda_output}")
    print(f"CUDA M:\n{cuda_M}")
    print(f"CUDA L:\n{cuda_L}")

# Extract only the last output (1, d_head) for comparison
print("\nExtracting last output for comparison:")
if cuda_output.dim() == 4:  # [batch, heads, seq, d_head]
    cuda_last_output = cuda_output[:, :, -1:, :]
elif cuda_output.dim() == 3:  # [batch, heads, d_head]
    cuda_last_output = cuda_output.unsqueeze(2)  # Add seq dimension
else:
    cuda_last_output = cuda_output

print(f"PyTorch last output shape: {pytorch_output.shape}")
print(f"CUDA last output shape: {cuda_last_output.shape}")

# Compare results
print("Differences:")
output_diff = torch.abs(pytorch_output.float() - cuda_last_output.float())
M_diff = torch.abs(pytorch_M.float() - cuda_M.float())
L_diff = torch.abs(pytorch_L.float() - cuda_L.float())

print_stats(output_diff, "Output diff")
print_stats(M_diff, "M diff")
print_stats(L_diff, "L diff")

# Check if results are close
tolerance = 1e-2
output_close = torch.allclose(
    pytorch_output.float(), cuda_last_output.float(), rtol=tolerance, atol=tolerance
)
M_close = torch.allclose(
    pytorch_M.float(), cuda_M.float(), rtol=tolerance, atol=tolerance
)
L_close = torch.allclose(
    pytorch_L.float(), cuda_L.float(), rtol=tolerance, atol=tolerance
)

print(f"Output matches: {output_close}")
print(f"M matches: {M_close}")
print(f"L matches: {L_close}")

if output_close and M_close and L_close:
    print("\n✓ Test passed!")
else:
    print("\n✗ Test failed!")

print(f"Time taken: {time.time() - t0} seconds")
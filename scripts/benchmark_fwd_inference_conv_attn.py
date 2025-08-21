#!/usr/bin/env python3
"""
Benchmark for forward convolution attention inference function
Compares PyTorch vs CUDA implementation performance
"""
import torch
import torch.nn.functional as F
import numpy as np
import time

from flash_multi_token_attention.conv_attn.flash_conv_attention_inference import (
    fwd_conv_attn_inference,
)
from flash_multi_token_attention.conv_attn.conv_attention_pytorch import ConvAttention


def benchmark_implementation(impl_name, run_func, warmup_iterations=5, benchmark_iterations=20):
    """Benchmark a function with warmup and multiple iterations"""
    print(f"\nBenchmarking {impl_name}...")
    
    # Warmup
    for _ in range(warmup_iterations):
        run_func()
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(benchmark_iterations):
        start_time = time.time()
        run_func()
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"{impl_name} - Mean: {mean_time*1000:.3f}ms, Std: {std_time*1000:.3f}ms")
    return mean_time


# Hardcoded test parameters
batch_size = 2
n_heads = 2
query_len = 16
seq_len = 1024
d_head = 64
kernel_size_q = 11
kernel_size_k = 11
k_splits = 32
causal = False
dtype = torch.bfloat16
softmax_scale = 1.0 / np.sqrt(d_head)

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Benchmarking with: batch={batch_size}, heads={n_heads}, query_len={query_len}, seq_len={seq_len}, d_head={d_head}")
print(f"Conv kernel: {kernel_size_q}x{kernel_size_k}, k_splits={k_splits}, causal={causal}")

# Create tensors
Q = torch.randn(batch_size, n_heads, query_len, d_head, device=device, dtype=dtype)
K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
V = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
conv_kernel = torch.randn(n_heads, kernel_size_q, kernel_size_k, device=device, dtype=dtype) * 0.02

# Setup PyTorch implementation
conv_attn = ConvAttention(n_heads, kernel_size_q, kernel_size_k, init_random=True)
conv_attn.conv_weight.data = conv_kernel
conv_attn = conv_attn.to(device)

def pytorch_forward():
    with torch.no_grad():
        output, M, L = conv_attn(Q, K, V, causal=causal, softmax_scale=softmax_scale)
        return output[:, :, -1:, :], M[:, :, -1:], L[:, :, -1:]

# Setup CUDA implementation
pad_q = kernel_size_q - 1
pad_k = (kernel_size_k - 1) // 2

def cuda_forward():
    output, M, L = fwd_conv_attn_inference(
        Q, K, V, conv_kernel, k_splits, softmax_scale, pad_q, pad_k, causal
    )
    return output, M, L

# Setup PyTorch Flash Attention (reference implementation)
def pytorch_flash_attention():
    with torch.no_grad():
        # Use only the last query token for inference (like the other implementations)
        q_last = Q[:, :, -1:, :]  # Shape: [batch, n_heads, 1, d_head]
        k = K  # Shape: [batch, n_heads, seq_len, d_head]
        v = V  # Shape: [batch, n_heads, seq_len, d_head]
        
        
        # Use PyTorch's scaled_dot_product_attention (Flash Attention)
        output = F.scaled_dot_product_attention(
            q_last, k, v, 
            scale=softmax_scale,
        )
        
        # Return dummy M and L values for compatibility (not computed by this implementation)
        batch, n_heads, query_len, d_head = output.shape
        M = torch.zeros(batch, n_heads, query_len, device=device, dtype=dtype)
        L = torch.zeros(batch, n_heads, query_len, device=device, dtype=dtype)
        
        return output, M, L

# Run benchmarks
pytorch_time = benchmark_implementation("PyTorch ConvAttn", pytorch_forward)
cuda_time = benchmark_implementation("CUDA ConvAttn", cuda_forward)
flash_time = benchmark_implementation("PyTorch Flash Attention (Reference)", pytorch_flash_attention)

# Calculate speedups
speedup_cuda = pytorch_time / cuda_time
speedup_flash = pytorch_time / flash_time
print(f"\nSpeedups vs PyTorch ConvAttn:")
print(f"CUDA ConvAttn: {speedup_cuda:.2f}x faster")
print(f"PyTorch Flash Attention: {speedup_flash:.2f}x faster")
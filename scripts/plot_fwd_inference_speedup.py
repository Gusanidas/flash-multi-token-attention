#!/usr/bin/env python3
"""
Plot forward inference speedup for different sequence lengths with kernel size 7.
Compares CUDA inference implementation vs PyTorch reference.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_multi_token_attention.conv_attn.flash_conv_attention_inference import (
    fwd_conv_attn_inference,
)
from flash_multi_token_attention.conv_attn.conv_attention_pytorch import ConvAttention


def benchmark_implementation(run_func, warmup_iterations=5, benchmark_iterations=20):
    """Benchmark a function with warmup and multiple iterations"""
    # Warmup
    for _ in range(warmup_iterations):
        run_func()
    
    torch.cuda.synchronize()
    
    # Benchmark using CUDA events for better accuracy
    times = []
    for _ in range(benchmark_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        run_func()
        end_event.record()
        torch.cuda.synchronize()
        
        times.append(start_event.elapsed_time(end_event) / 1000.0)  # Convert to seconds
    
    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return mean_time, std_time


def run_inference_benchmark_for_plotting():
    """Run forward inference benchmarks for different sequence lengths"""
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run benchmarks.")
        return None

    device = torch.device("cuda")
    seq_lengths = [512, 1024, 2048, 4096]
    
    # Fixed parameters
    batch_size = 2
    n_heads = 2
    query_len = 16
    d_head = 64
    kernel_size = 7  # Kernel size 7 as requested
    k_splits = 32
    causal = False
    dtype = torch.bfloat16
    softmax_scale = 1.0 / np.sqrt(d_head)

    results = []

    for seq_len in seq_lengths:
        print(f"Benchmarking sequence length: {seq_len}")
        
        # Create tensors
        torch.manual_seed(42)  # For reproducibility
        Q = torch.randn(batch_size, n_heads, query_len, d_head, device=device, dtype=dtype)
        K = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
        V = torch.randn(batch_size, n_heads, seq_len, d_head, device=device, dtype=dtype)
        conv_kernel = torch.randn(n_heads, kernel_size, kernel_size, device=device, dtype=dtype) * 0.02

        # Setup PyTorch implementation
        conv_attn = ConvAttention(n_heads, kernel_size, kernel_size, init_random=True)
        conv_attn.conv_weight.data = conv_kernel
        conv_attn = conv_attn.to(device)

        def pytorch_forward():
            with torch.no_grad():
                output, M, L = conv_attn(Q, K, V, causal=causal, softmax_scale=softmax_scale)
                return output[:, :, -1:, :], M[:, :, -1:], L[:, :, -1:]

        # Setup CUDA implementation
        pad_q = kernel_size - 1
        pad_k = (kernel_size - 1) // 2

        def cuda_forward():
            output, M, L = fwd_conv_attn_inference(
                Q, K, V, conv_kernel, k_splits, softmax_scale, pad_q, pad_k, causal
            )
            return output, M, L

        # Setup PyTorch Flash Attention (reference)
        def pytorch_flash_attention():
            with torch.no_grad():
                q_last = Q[:, :, -1:, :]  # Last query token for inference
                k = K
                v = V
                
                output = F.scaled_dot_product_attention(
                    q_last, k, v, 
                    scale=softmax_scale,
                )
                
                # Return dummy M and L values for compatibility
                batch, n_heads, query_len, d_head = output.shape
                M = torch.zeros(batch, n_heads, query_len, device=device, dtype=dtype)
                L = torch.zeros(batch, n_heads, query_len, device=device, dtype=dtype)
                
                return output, M, L

        # Benchmark all implementations
        try:
            pytorch_time, pytorch_std = benchmark_implementation(pytorch_forward)
            cuda_time, cuda_std = benchmark_implementation(cuda_forward)
            flash_time, flash_std = benchmark_implementation(pytorch_flash_attention)

            # Calculate speedups
            cuda_speedup = pytorch_time / cuda_time
            flash_speedup = pytorch_time / flash_time
            cuda_vs_flash_speedup = flash_time / cuda_time

            results.append({
                "seq_len": seq_len,
                "pytorch_time": pytorch_time,
                "cuda_time": cuda_time,
                "flash_time": flash_time,
                "pytorch_std": pytorch_std,
                "cuda_std": cuda_std,
                "flash_std": flash_std,
                "cuda_speedup": cuda_speedup,
                "flash_speedup": flash_speedup,
                "cuda_vs_flash_speedup": cuda_vs_flash_speedup,
            })

            print(f"  PyTorch ConvAttn: {pytorch_time*1000:.2f}ms ± {pytorch_std*1000:.2f}ms")
            print(f"  CUDA ConvAttn: {cuda_time*1000:.2f}ms ± {cuda_std*1000:.2f}ms")
            print(f"  Flash Attention: {flash_time*1000:.2f}ms ± {flash_std*1000:.2f}ms")
            print(f"  CUDA Speedup vs PyTorch: {cuda_speedup:.2f}x")
            print(f"  Flash Speedup vs PyTorch: {flash_speedup:.2f}x")
            print(f"  CUDA vs Flash: {cuda_vs_flash_speedup:.2f}x")

        except Exception as e:
            print(f"Benchmark failed for sequence length {seq_len}: {e}")
            continue

    return results


def plot_inference_results(results):
    """Plot the forward inference benchmark results"""
    if not results:
        print("No results to plot")
        return

    seq_lens = [r["seq_len"] for r in results]
    pytorch_times = [r["pytorch_time"] * 1000 for r in results]  # Convert to ms
    cuda_times = [r["cuda_time"] * 1000 for r in results]
    flash_times = [r["flash_time"] * 1000 for r in results]

    # Plot 1: Absolute times (only plot kept)
    plt.figure(figsize=(12, 6))
    plt.plot(seq_lens, pytorch_times, 'b-o', label='PyTorch ConvAttn', linewidth=2, markersize=8)
    plt.plot(seq_lens, cuda_times, 'r-o', label='CUDA ConvAttn', linewidth=2, markersize=8)
    plt.plot(seq_lens, flash_times, 'g-o', label='PyTorch Flash Attn', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Forward Inference Times (Kernel Size 7)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')

    plt.tight_layout()
    plt.savefig('fwd_inference_speedup.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Plot saved as 'fwd_inference_speedup.png'")


def main():
    """Main function"""
    print("Running forward inference benchmark for speedup analysis...")
    print("Kernel size: 7")
    print("Sequence lengths: 512, 1024, 2048, 4096")
    print("Comparing: PyTorch ConvAttn vs CUDA ConvAttn vs PyTorch Flash Attention")
    print()

    results = run_inference_benchmark_for_plotting()
    if results:
        plot_inference_results(results)
        
        # Print summary table
        print("\nSummary:")
        print("-" * 80)
        print(f"{'Seq Len':<8} {'PyTorch(ms)':<12} {'CUDA(ms)':<10} {'Flash(ms)':<10} {'CUDA vs PT':<10} {'Flash vs PT':<11} {'CUDA vs Flash':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['seq_len']:<8} {r['pytorch_time']*1000:<12.2f} {r['cuda_time']*1000:<10.2f} {r['flash_time']*1000:<10.2f} "
                  f"{r['cuda_speedup']:<10.2f} {r['flash_speedup']:<11.2f} {r['cuda_vs_flash_speedup']:<12.2f}")


if __name__ == "__main__":
    main()
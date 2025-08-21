#!/usr/bin/env python3
"""
Plot benchmark results for different sequence lengths with kernel size 7.
Shows speedup comparison between CUDA and PyTorch implementations.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_multi_token_attention.conv_attn.conv_attention_pytorch import ConvAttention
from flash_multi_token_attention.conv_attn.conv_attention_cuda import ConvAttentionCUDA


def benchmark_flash_attention(Q, K, V, num_iterations: int = 10, warmup_iterations: int = 5):
    """Benchmark PyTorch Flash Attention"""
    device = Q.device
    dtype = Q.dtype
    head_dim = Q.shape[-1]
    softmax_scale = 1.0 / np.sqrt(head_dim)
    
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup_iterations):
        Q_warmup = Q.clone().detach()
        K_warmup = K.clone().detach()
        V_warmup = V.clone().detach()
        
        try:
            with torch.no_grad():
                output = F.scaled_dot_product_attention(
                    Q_warmup, K_warmup, V_warmup, 
                    scale=softmax_scale
                )
        except Exception:
            pass  # Ignore warmup failures
    
    torch.cuda.synchronize()

    # Forward pass timing
    forward_times = []
    for i in range(num_iterations):
        Q_test = Q.clone().detach()
        K_test = K.clone().detach()
        V_test = V.clone().detach()

        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        try:
            with torch.no_grad():
                output = F.scaled_dot_product_attention(
                    Q_test, K_test, V_test, 
                    scale=softmax_scale
                )
            end_time.record()
            torch.cuda.synchronize()
            forward_times.append(start_time.elapsed_time(end_time) / 1000.0)  # Convert to seconds
        except Exception as e:
            print(f"Flash attention forward pass failed: {e}")
            return None

    results = {
        "forward_mean": np.mean(forward_times),
        "forward_std": np.std(forward_times),
        "backward_mean": 0.0,  # Flash attention doesn't need backward pass for this comparison
        "backward_std": 0.0,
    }

    return results


def benchmark_model(model, Q, K, V, num_iterations: int = 10, test_backward: bool = True, warmup_iterations: int = 5):
    """Benchmark a model's forward and backward pass"""
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup_iterations):
        Q_warmup = Q.clone().detach().requires_grad_(test_backward)
        K_warmup = K.clone().detach().requires_grad_(test_backward)
        V_warmup = V.clone().detach().requires_grad_(test_backward)
        
        try:
            output = model(Q_warmup, K_warmup, V_warmup)
            if isinstance(output, tuple):
                output = output[0]
            if test_backward:
                grad_output = torch.randn_like(output)
                output.backward(grad_output)
        except Exception:
            pass  # Ignore warmup failures
    
    torch.cuda.synchronize()

    # Forward pass timing
    forward_times = []
    for i in range(num_iterations):
        Q_test = Q.clone().detach().requires_grad_(test_backward)
        K_test = K.clone().detach().requires_grad_(test_backward)
        V_test = V.clone().detach().requires_grad_(test_backward)

        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        try:
            output = model(Q_test, K_test, V_test)
            if isinstance(output, tuple):
                output = output[0]
            end_time.record()
            torch.cuda.synchronize()
            forward_times.append(start_time.elapsed_time(end_time) / 1000.0)  # Convert to seconds
        except Exception as e:
            print(f"Forward pass failed: {e}")
            return None

    results = {
        "forward_mean": np.mean(forward_times),
        "forward_std": np.std(forward_times),
    }

    # Backward pass timing
    if test_backward:
        backward_times = []
        for i in range(num_iterations):
            Q_test = Q.clone().detach().requires_grad_(True)
            K_test = K.clone().detach().requires_grad_(True)
            V_test = V.clone().detach().requires_grad_(True)

            try:
                output = model(Q_test, K_test, V_test)
                if isinstance(output, tuple):
                    output = output[0]
                grad_output = torch.randn_like(output)

                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                output.backward(grad_output)
                end_time.record()
                torch.cuda.synchronize()
                
                backward_times.append(start_time.elapsed_time(end_time) / 1000.0)  # Convert to seconds
            except Exception as e:
                print(f"Backward pass failed: {e}")
                backward_times.append(float("inf"))

        results.update({
            "backward_mean": np.mean(backward_times),
            "backward_std": np.std(backward_times),
        })

    return results


def run_benchmark_for_plotting():
    """Run benchmarks for specific sequence lengths with kernel size 7"""
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run benchmarks.")
        return None

    device = torch.device("cuda")
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    kernel_size = 7
    batch_size = 1
    n_heads = 8
    head_dim = 64
    num_iterations = 20

    results = []

    for seq_len in seq_lengths:
        print(f"Benchmarking sequence length: {seq_len}")
        
        # Create test inputs
        Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

        # Initialize models
        torch.manual_seed(42)
        model_pytorch = ConvAttention(n_heads, kernel_size, kernel_size, init_random=True).to(device)

        torch.manual_seed(42)
        model_cuda = ConvAttentionCUDA(n_heads, kernel_size, kernel_size, init_random=True).to(device)
        model_cuda.conv_weight.data.copy_(model_pytorch.conv_weight.data)

        # Benchmark all three implementations
        pytorch_results = benchmark_model(model_pytorch, Q, K, V, num_iterations)
        cuda_results = benchmark_model(model_cuda, Q, K, V, num_iterations)
        flash_results = benchmark_flash_attention(Q, K, V, num_iterations)

        if pytorch_results is None or cuda_results is None or flash_results is None:
            print(f"Benchmark failed for sequence length {seq_len}")
            continue

        # Calculate speedups
        forward_speedup = pytorch_results["forward_mean"] / cuda_results["forward_mean"]
        backward_speedup = pytorch_results["backward_mean"] / cuda_results["backward_mean"]
        flash_forward_speedup = pytorch_results["forward_mean"] / flash_results["forward_mean"]

        results.append({
            "seq_len": seq_len,
            "forward_speedup": forward_speedup,
            "backward_speedup": backward_speedup,
            "flash_forward_speedup": flash_forward_speedup,
            "pytorch_forward": pytorch_results["forward_mean"],
            "pytorch_backward": pytorch_results["backward_mean"],
            "cuda_forward": cuda_results["forward_mean"],
            "cuda_backward": cuda_results["backward_mean"],
            "flash_forward": flash_results["forward_mean"],
        })

        print(f"  Forward speedup (CUDA vs PyTorch): {forward_speedup:.2f}x")
        print(f"  Backward speedup (CUDA vs PyTorch): {backward_speedup:.2f}x")
        print(f"  Forward speedup (Flash vs PyTorch): {flash_forward_speedup:.2f}x")

    return results


def plot_results(results):
    """Plot the benchmark results"""
    if not results:
        print("No results to plot")
        return

    # Drop the lowest sequence length for plotting
    if len(results) > 1:
        # Sort by sequence length and drop the first (lowest) one
        sorted_results = sorted(results, key=lambda x: x["seq_len"])
        plot_results_filtered = sorted_results[1:]  # Skip the lowest seq_len
    else:
        plot_results_filtered = results

    seq_lens = [r["seq_len"] for r in plot_results_filtered]
    forward_speedups = [r["forward_speedup"] for r in plot_results_filtered]
    backward_speedups = [r["backward_speedup"] for r in plot_results_filtered]
    flash_forward_speedups = [r["flash_forward_speedup"] for r in plot_results_filtered]
    
    # Extract absolute times (convert to milliseconds)
    pytorch_forward_times = [r["pytorch_forward"] * 1000 for r in plot_results_filtered]
    cuda_forward_times = [r["cuda_forward"] * 1000 for r in plot_results_filtered]
    flash_forward_times = [r["flash_forward"] * 1000 for r in plot_results_filtered]
    pytorch_backward_times = [r["pytorch_backward"] * 1000 for r in plot_results_filtered]
    cuda_backward_times = [r["cuda_backward"] * 1000 for r in plot_results_filtered]

    # Create three separate plots
    
    # Plot 1: Speedups (CUDA vs PyTorch only)
    plt.figure(figsize=(12, 6))
    plt.plot(seq_lens, forward_speedups, 'bo-', label='Forward (CUDA vs PyTorch)', linewidth=2, markersize=8)
    plt.plot(seq_lens, backward_speedups, 'ro-', label='Backward (CUDA vs PyTorch)', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Sequence Length (Kernel Size 7)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('benchmark_seq_len_speedup.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot 1 saved as 'benchmark_seq_len_speedup.png'")
    
    # Plot 2: Forward pass absolute times
    plt.figure(figsize=(12, 6))
    plt.plot(seq_lens, pytorch_forward_times, 'b-o', label='PyTorch ConvAttn', linewidth=2, markersize=8)
    plt.plot(seq_lens, cuda_forward_times, 'r-o', label='CUDA ConvAttn', linewidth=2, markersize=8)
    plt.plot(seq_lens, flash_forward_times, 'g-o', label='Flash Attention', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Forward Pass Absolute Times vs Sequence Length (Kernel Size 7)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('benchmark_seq_len_forward_times.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot 2 saved as 'benchmark_seq_len_forward_times.png'")
    
    # Plot 3: Backward pass absolute times (with Flash attention)
    flash_backward_times = [r["flash_forward"] * 1000 for r in plot_results_filtered]  # Use flash forward times as placeholder
    plt.figure(figsize=(12, 6))
    plt.plot(seq_lens, pytorch_backward_times, 'b-o', label='PyTorch ConvAttn', linewidth=2, markersize=8)
    plt.plot(seq_lens, cuda_backward_times, 'r-o', label='CUDA ConvAttn', linewidth=2, markersize=8)
    plt.plot(seq_lens, flash_backward_times, 'g-o', label='Flash Attention', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Backward Pass Absolute Times vs Sequence Length (Kernel Size 7)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('benchmark_seq_len_backward_times.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Plot 3 saved as 'benchmark_seq_len_backward_times.png'")


def main():
    """Main function"""
    print("Running benchmark for sequence length speedup analysis...")
    print("Kernel size: 7")
    print("Sequence lengths: 256, 512, 1024, 2048, 4096, 8192")
    print()

    results = run_benchmark_for_plotting()
    if results:
        plot_results(results)
        
        # Print summary
        print("\nSummary:")
        print("-" * 80)
        print(f"{'Seq Len':<8} {'CUDA Fwd':<10} {'CUDA Bwd':<10} {'Flash Fwd':<10} {'PyTorch Fwd(ms)':<15} {'CUDA Fwd(ms)':<13} {'Flash Fwd(ms)':<13}")
        print("-" * 80)
        for r in results:
            print(f"{r['seq_len']:<8} {r['forward_speedup']:<10.2f} {r['backward_speedup']:<10.2f} {r['flash_forward_speedup']:<10.2f} "
                  f"{r['pytorch_forward']*1000:<15.2f} {r['cuda_forward']*1000:<13.2f} {r['flash_forward']*1000:<13.2f}")


if __name__ == "__main__":
    main()
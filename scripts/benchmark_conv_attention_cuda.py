#!/usr/bin/env python3
"""
Benchmark script to compare performance between CUDA and PyTorch implementations.
Tests both forward and backward passes across different sequence lengths.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flash_multi_token_attention.conv_attn.conv_attention_pytorch import ConvAttention
from flash_multi_token_attention.conv_attn.conv_attention_cuda import ConvAttentionCUDA


def warmup_gpu():
    """Warm up GPU to get consistent timing results"""
    print("Warming up GPU...")
    dummy = torch.randn(1000, 1000, device="cuda")
    for _ in range(10):
        torch.matmul(dummy, dummy)
    torch.cuda.synchronize()
    print("GPU warmup complete")


def benchmark_model(
    model, Q, K, V, num_iterations: int = 10, test_backward: bool = True
):
    """
    Benchmark a model's forward and backward pass

    Args:
        model: The model to benchmark
        Q, K, V: Input tensors
        num_iterations: Number of iterations to run
        test_backward: Whether to test backward pass

    Returns:
        Dictionary with timing results
    """
    torch.cuda.synchronize()

    # Forward pass timing
    forward_times = []
    for i in range(num_iterations):
        # Create fresh inputs with gradients for backward pass
        Q_test = Q.clone().detach().requires_grad_(test_backward)
        K_test = K.clone().detach().requires_grad_(test_backward)
        V_test = V.clone().detach().requires_grad_(test_backward)

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        try:
            output = model(Q_test, K_test, V_test)
            if isinstance(output, tuple):
                output = output[0]
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            forward_times.append(end_time - start_time)
        except Exception as e:
            print(f"Forward pass failed: {e}")
            return None

    results = {
        "forward_mean": np.mean(forward_times),
        "forward_std": np.std(forward_times),
        "forward_min": np.min(forward_times),
    }

    # Backward pass timing
    if test_backward:
        backward_times = []
        for i in range(num_iterations):
            # Fresh inputs with gradients
            Q_test = Q.clone().detach().requires_grad_(True)
            K_test = K.clone().detach().requires_grad_(True)
            V_test = V.clone().detach().requires_grad_(True)

            # Forward pass
            try:
                output = model(Q_test, K_test, V_test)
                if isinstance(output, tuple):
                    output = output[0]
                grad_output = torch.randn_like(output)

                torch.cuda.synchronize()
                start_time = time.perf_counter()

                output.backward(grad_output)
                torch.cuda.synchronize()
                end_time = time.perf_counter()

                backward_times.append(end_time - start_time)
            except Exception as e:
                print(f"Backward pass failed: {e}")
                backward_times.append(float("inf"))

        results.update(
            {
                "backward_mean": np.mean(backward_times),
                "backward_std": np.std(backward_times),
                "backward_min": np.min(backward_times),
                "total_mean": np.mean(
                    [f + b for f, b in zip(forward_times, backward_times)]
                ),
            }
        )

    return results


def run_benchmark_suite():
    """Run comprehensive benchmarks across different configurations"""
    print("=" * 80)
    print("CONV ATTENTION CUDA vs PYTORCH BENCHMARK")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("CUDA not available. Cannot run benchmarks.")
        return

    print(f"Using device: {torch.cuda.get_device_name()}")
    warmup_gpu()
    print()

    # Test configurations
    configs = [
        {
            "batch_size": 1,
            "n_heads": 8,
            "seq_len": 256,
            "head_dim": 64,
            "kernel_size": 3,
        },
        {
            "batch_size": 1,
            "n_heads": 8,
            "seq_len": 512,
            "head_dim": 64,
            "kernel_size": 3,
        },
        {
            "batch_size": 1,
            "n_heads": 8,
            "seq_len": 1024,
            "head_dim": 64,
            "kernel_size": 3,
        },
        {
            "batch_size": 1,
            "n_heads": 8,
            "seq_len": 2048,
            "head_dim": 64,
            "kernel_size": 7,
        },
        {
            "batch_size": 1,
            "n_heads": 8,
            "seq_len": 2048,
            "head_dim": 64,
            "kernel_size": 11,
        },
        {
            "batch_size": 1,
            "n_heads": 8,
            "seq_len": 4096,
            "head_dim": 64,
            "kernel_size": 11,
        },
        {
            "batch_size": 1,
            "n_heads": 8,
            "seq_len": 8192,
            "head_dim": 64,
            "kernel_size": 11,
        },
    ]

    device = torch.device("cuda")
    num_iterations = 20

    results = []

    for config in configs:
        print(f"Testing configuration: {config}")
        print("-" * 60)

        batch_size = config["batch_size"]
        n_heads = config["n_heads"]
        seq_len = config["seq_len"]
        head_dim = config["head_dim"]
        kernel_size = config["kernel_size"]

        # Create test inputs
        Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

        # Initialize models with same parameters
        torch.manual_seed(42)
        model_pytorch = ConvAttention(
            n_heads, kernel_size, kernel_size, init_random=True
        ).to(device)

        torch.manual_seed(42)
        model_cuda = ConvAttentionCUDA(
            n_heads, kernel_size, kernel_size, init_random=True
        ).to(device)
        model_cuda.conv_weight.data.copy_(model_pytorch.conv_weight.data)

        # Benchmark PyTorch implementation
        print("Benchmarking PyTorch implementation...")
        pytorch_results = benchmark_model(model_pytorch, Q, K, V, num_iterations)

        if pytorch_results is None:
            print("PyTorch benchmark failed, skipping this configuration")
            continue

        # Benchmark CUDA implementation
        print("Benchmarking CUDA implementation...")
        cuda_results = benchmark_model(model_cuda, Q, K, V, num_iterations)

        if cuda_results is None:
            print("CUDA benchmark failed, skipping this configuration")
            continue

        # Calculate speedups
        forward_speedup = (
            pytorch_results["forward_mean"] / cuda_results["forward_mean"]
            if cuda_results["forward_mean"] > 0
            else 0
        )
        backward_speedup = (
            pytorch_results["backward_mean"] / cuda_results["backward_mean"]
            if cuda_results["backward_mean"] > 0
            else 0
        )
        total_speedup = (
            pytorch_results["total_mean"] / cuda_results["total_mean"]
            if cuda_results["total_mean"] > 0
            else 0
        )

        # Print results
        print(
            f"PyTorch - Forward: {pytorch_results['forward_mean']*1000:.2f}ms ± {pytorch_results['forward_std']*1000:.2f}ms"
        )
        print(
            f"CUDA    - Forward: {cuda_results['forward_mean']*1000:.2f}ms ± {cuda_results['forward_std']*1000:.2f}ms"
        )
        print(f"Forward Speedup: {forward_speedup:.2f}x")
        print()

        print(
            f"PyTorch - Backward: {pytorch_results['backward_mean']*1000:.2f}ms ± {pytorch_results['backward_std']*1000:.2f}ms"
        )
        print(
            f"CUDA    - Backward: {cuda_results['backward_mean']*1000:.2f}ms ± {cuda_results['backward_std']*1000:.2f}ms"
        )
        print(f"Backward Speedup: {backward_speedup:.2f}x")
        print()

        print(f"PyTorch - Total: {pytorch_results['total_mean']*1000:.2f}ms")
        print(f"CUDA    - Total: {cuda_results['total_mean']*1000:.2f}ms")
        print(f"Total Speedup: {total_speedup:.2f}x")
        print()

        # Store results
        result_entry = {
            "config": config,
            "pytorch": pytorch_results,
            "cuda": cuda_results,
            "speedup": {
                "forward": forward_speedup,
                "backward": backward_speedup,
                "total": total_speedup,
            },
        }
        results.append(result_entry)

        print("=" * 60)
        print()

    # Summary
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    if not results:
        print("No successful benchmark results")
        return

    # Calculate average speedups
    forward_speedups = [
        r["speedup"]["forward"] for r in results if r["speedup"]["forward"] > 0
    ]
    backward_speedups = [
        r["speedup"]["backward"] for r in results if r["speedup"]["backward"] > 0
    ]
    total_speedups = [
        r["speedup"]["total"] for r in results if r["speedup"]["total"] > 0
    ]

    if forward_speedups:
        print(
            f"Average Forward Speedup: {np.mean(forward_speedups):.2f}x (min: {np.min(forward_speedups):.2f}x, max: {np.max(forward_speedups):.2f}x)"
        )

    if backward_speedups:
        print(
            f"Average Backward Speedup: {np.mean(backward_speedups):.2f}x (min: {np.min(backward_speedups):.2f}x, max: {np.max(backward_speedups):.2f}x)"
        )

    if total_speedups:
        print(
            f"Average Total Speedup: {np.mean(total_speedups):.2f}x (min: {np.min(total_speedups):.2f}x, max: {np.max(total_speedups):.2f}x)"
        )

    print()

    # Print detailed table
    print("DETAILED RESULTS")
    print("-" * 80)
    print(f"{'Config':<25} {'Forward':<15} {'Backward':<15} {'Total':<15}")
    print("-" * 80)

    for result in results:
        config = result["config"]
        speedup = result["speedup"]
        config_str = (
            f"B{config['batch_size']}_H{config['n_heads']}_S{config['seq_len']}"
        )

        print(
            f"{config_str:<25} {speedup['forward']:<15.2f} {speedup['backward']:<15.2f} {speedup['total']:<15.2f}"
        )

    return results


def main():
    """Main benchmark function"""
    run_benchmark_suite()


if __name__ == "__main__":
    main()

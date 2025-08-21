import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flash_multi_token_attention.conv_attn.conv_attention_cuda import ConvAttentionCUDA
from flash_multi_token_attention.conv_attn.conv_attention_pytorch import ConvAttention


def compare_tensors(a, b, name=""):
    """
    Compare two tensors and print statistics
    
    Args:
        a: First tensor
        b: Second tensor  
        name: Optional name for the comparison
    """
    diff = a - b
    
    print(f"\n=== Tensor Comparison: {name} ===")
    print(f"Tensor A - mean_abs: {a.abs().mean():.6f}, max_abs: {a.abs().max():.6f}, std: {a.std():.6f}")
    print(f"Tensor B - mean_abs: {b.abs().mean():.6f}, max_abs: {b.abs().max():.6f}, std: {b.std():.6f}")
    print(f"Diff (A-B) - mean_abs: {diff.abs().mean():.6f}, max_abs: {diff.abs().max():.6f}, std: {diff.std():.6f}")
    
    # Compute ratio |a-b|/|a| (avoiding division by zero)
    c = a.abs().mean()*0.1
    ratio = diff.abs() / (a.abs() + c)
    print(f"Ratio |A-B|/|A| - mean: {ratio.mean():.6f}, max: {ratio.max():.6f}")


def test_conv_attn():
    """Test forward and backward pass of conv_attn implementations"""
    
    # ===== HARDCODED VARIABLES =====
    batch_size = 2
    num_heads = 4
    seq_len = 16
    d_head = 64
    kernel_size_q = 3
    kernel_size_k = 3
    causal = True
    dtype = torch.bfloat16
    device = "cuda"
    
    print(f"Testing with:")
    print(f"  batch_size: {batch_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_head: {d_head}")
    print(f"  kernel_size_q: {kernel_size_q}")
    print(f"  kernel_size_k: {kernel_size_k}")
    print(f"  causal: {causal}")
    print(f"  dtype: {dtype}")
    print(f"  device: {device}")
    
    # ===== INITIALIZE RANDOM INPUTS =====
    torch.manual_seed(42)
    
    Q = torch.randn(batch_size, num_heads, seq_len, d_head, dtype=dtype, device=device, requires_grad=True)
    K = torch.randn(batch_size, num_heads, seq_len, d_head, dtype=dtype, device=device, requires_grad=True)
    V = torch.randn(batch_size, num_heads, seq_len, d_head, dtype=dtype, device=device, requires_grad=True)
    
    # Clone inputs for pytorch implementation
    Q_torch = Q.clone().detach().requires_grad_(True)
    K_torch = K.clone().detach().requires_grad_(True)
    V_torch = V.clone().detach().requires_grad_(True)
    
    # ===== INITIALIZE MODELS =====
    # CUDA implementation
    cuda_model = ConvAttentionCUDA(
        n_heads=num_heads,
        kernel_size_q=kernel_size_q,
        kernel_size_k=kernel_size_k
    ).to(device).to(dtype)
    
    # PyTorch implementation  
    pytorch_model = ConvAttention(
        n_heads=num_heads,
        kernel_size_q=kernel_size_q,
        kernel_size_k=kernel_size_k
    ).to(device).to(dtype)
    
    # Sync the convolution kernels to make comparison fair
    with torch.no_grad():
        pytorch_model.conv_weight.copy_(cuda_model.conv_weight)
    
    # ===== FORWARD PASS =====
    print("\n" + "="*50)
    print("FORWARD PASS COMPARISON")
    print("="*50)
    
    # CUDA forward
    cuda_output = cuda_model(Q, K, V)
    print(f"CUDA output type: {cuda_output.dtype}")
    
    # PyTorch forward
    pytorch_output, M, L = pytorch_model(Q_torch, K_torch, V_torch, causal=causal)
    print(f"PyTorch output type: {pytorch_output.dtype}")
    
    # Compare forward results
    compare_tensors(cuda_output, pytorch_output, "Forward Output")
    
    # ===== BACKWARD PASS =====
    print("\n" + "="*50)
    print("BACKWARD PASS COMPARISON")
    print("="*50)
    
    # Create gradient for backward pass
    grad_output = torch.randn_like(cuda_output)
    grad_output_torch = grad_output.clone()
    
    # CUDA backward
    cuda_output.backward(grad_output, retain_graph=True)
    cuda_dQ = Q.grad.clone()
    cuda_dK = K.grad.clone() if K.grad is not None else torch.zeros_like(K)
    cuda_dV = V.grad.clone() if V.grad is not None else torch.zeros_like(V)
    cuda_dconv = cuda_model.conv_weight.grad.clone() if cuda_model.conv_weight.grad is not None else torch.zeros_like(cuda_model.conv_weight)
    
    # PyTorch backward
    pytorch_output.backward(grad_output_torch, retain_graph=True)
    pytorch_dQ = Q_torch.grad.clone()
    pytorch_dK = K_torch.grad.clone() if K_torch.grad is not None else torch.zeros_like(K_torch)
    pytorch_dV = V_torch.grad.clone() if V_torch.grad is not None else torch.zeros_like(V_torch)
    pytorch_dconv = pytorch_model.conv_weight.grad.clone() if pytorch_model.conv_weight.grad is not None else torch.zeros_like(pytorch_model.conv_weight)
    
    # Compare backward results
    compare_tensors(cuda_dQ, pytorch_dQ, "dQ")
    compare_tensors(cuda_dK, pytorch_dK, "dK") 
    compare_tensors(cuda_dV, pytorch_dV, "dV")
    compare_tensors(cuda_dconv, pytorch_dconv, "dConv_kernel")
    
    print("\n" + "="*50)
    print("TEST COMPLETED")
    print("="*50)


if __name__ == "__main__":
    test_conv_attn()
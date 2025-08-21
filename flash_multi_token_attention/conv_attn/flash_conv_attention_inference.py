import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import math

# Get the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# JIT compile the CUDA extension for inference
flash_conv_attention_inference_cuda = load(
    name="flash_conv_attention_inference_cuda",
    sources=[
        os.path.join(current_dir, "fwd_inference_bindings.cpp"),
        os.path.join(current_dir, "kernels", "fwd_inference_wrapper.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        #"-gencode=arch=compute_80,code=sm_80",  # For A100
        #"-gencode=arch=compute_86,code=sm_86",  # For RTX 30xx
        "-gencode=arch=compute_89,code=sm_89",  # For RTX 40xx
        # "-gencode=arch=compute_90,code=sm_90",  # For H100
    ],
    verbose=True,
)

# Export the compiled function
forward_inference = flash_conv_attention_inference_cuda.forward_inference


def fwd_conv_attn_inference(
    Q, K, V, conv_kernel, k_splits, scale, pad_q, pad_k, causal
):
    """
    Forward pass function for inference that calls the CUDA kernel with K-dimension splitting

    Args:
        Q: Query tensor [batch, heads, q_len, d_head] - query tokens (q_len <= 16)
        K: Key tensor [batch, heads, seq_len, d_head]
        V: Value tensor [batch, heads, seq_len, d_head]
        conv_kernel: Convolution kernel [heads, kx, ky]
        k_splits: Number of splits for K dimension
        scale: Attention scale factor
        pad_q: Query padding for convolution
        pad_k: Key padding for convolution
        causal: Whether to apply causal masking

    Returns:
        O: Combined output tensor [batch, heads, 1, d_head] - final attention output
        M: Global max values [batch, heads, 1]
        L: Global sum values [batch, heads, 1]
    """
    import torch

    # Pad Q to 16 tokens if needed
    batch_size, num_heads, q_seq_len, d_head = Q.shape
    if q_seq_len < 16:
        # Create zero-padded tensor
        Q_padded = torch.zeros((batch_size, num_heads, 16, d_head), 
                              dtype=Q.dtype, device=Q.device)
        # Copy original Q to the beginning
        Q_padded[:, :, -q_seq_len:, :] = Q
        Q = Q_padded
    elif q_seq_len > 16:
        raise ValueError(f"Q sequence length must be <= 16 for inference, got {q_seq_len}")

    # Calculate number of K blocks needed for sequential processing
    _, _, seq_len_k, _ = K.shape
    block_size_k = 32
    reduced_k_block_size = block_size_k - 2 * pad_k  # Account for padding
    num_k_blocks = (seq_len_k + reduced_k_block_size - 1) // reduced_k_block_size
    
    # For inference, we use k_splits=1 and process sequentially within the kernel
    num_k_blocks = (num_k_blocks + k_splits-1) // k_splits

    # Call CUDA kernel which handles sequential tiling internally
    O_splits, M_splits, L_splits = forward_inference(
        Q, K, V, conv_kernel, k_splits, scale, pad_q, pad_k, causal, num_k_blocks
    )
    #print("--------------------------------")
    #print(f"In pytorch, O_splits shape: {O_splits.shape}, max: {O_splits.max()}, mean abs: {O_splits.abs().mean()}")
    #print("--------------------------------")
    #print(f"M_splits: {M_splits}")

    # The combination of splits is now performed in the CUDA kernel
    # Just return the results from the first split (index 0) which contains the combined results
    O_global = O_splits[:, :, 0:1, :]  # [batch, heads, 1, d_head] - first split contains combined result
    M_global = M_splits[:, :, 0:1]     # [batch, heads, 1] - first split contains combined M
    L_global = L_splits[:, :, 0:1]     # [batch, heads, 1] - first split contains combined L

    return O_global, M_global, L_global


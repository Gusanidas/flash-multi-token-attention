import os
import torch
from torch.utils.cpp_extension import load

# Get the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# JIT compile the CUDA extension
flash_conv_attention_cuda = load(
    name="flash_conv_attention_cuda",
    sources=[
        os.path.join(current_dir, "bindings.cpp"),
        os.path.join(current_dir, "kernels", "fwd_wrapper.cu"),
        os.path.join(current_dir, "kernels", "dq_bwd_wrapper.cu"),
        os.path.join(current_dir, "kernels", "dkdv_wrapper.cu"),
    ],
    extra_cflags=["-O3"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        #"-gencode=arch=compute_80,code=sm_80",  # For A100
        #"-gencode=arch=compute_86,code=sm_86",  # For RTX 30xx
        "-gencode=arch=compute_89,code=sm_89",  # For RTX 40xx
    ],
    verbose=True
)

# Export the compiled functions
forward = flash_conv_attention_cuda.forward
dq_backward = flash_conv_attention_cuda.dq_backward
dkdv_backward = flash_conv_attention_cuda.dkdv_backward

# Import and export the Python classes
from .flash_conv_attention import FlashConvAttention, FlashConvAttentionFunction, fwd_conv_attn, dq_bwd_conv_attn, dkdv_bwd_conv_attn

__all__ = ["FlashConvAttention", "FlashConvAttentionFunction", "forward", "dq_backward", "dkdv_backward", "fwd_conv_attn", "dq_bwd_conv_attn", "dkdv_bwd_conv_attn"]
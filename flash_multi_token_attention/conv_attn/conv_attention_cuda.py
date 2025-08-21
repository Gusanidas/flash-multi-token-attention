import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import forward, dq_backward, dkdv_backward


class ConvAttentionCUDA(nn.Module):
    """
    CUDA-accelerated conv attention using the optimized CUDA kernels.
    Mirrors the interface of ConvAttention from conv_attention_pytorch.py but uses
    CUDA kernels for forward and backward passes.
    """

    def __init__(
        self,
        n_heads: int,
        kernel_size_q: int = 3,
        kernel_size_k: int = 3,
        init_random: bool = True,
        init_first: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.kernel_size_q = kernel_size_q
        self.kernel_size_k = kernel_size_k
        self.init_random = init_random
        self.init_first = init_first

        # Convolution weight
        self.conv_weight = torch.nn.parameter.Parameter(
            torch.empty(n_heads, kernel_size_q, kernel_size_k)
        )

        self._init_parameters()

    def _init_parameters(self):
        """Initialize with identity-like kernel (center weight = 1, others = 0) or random weights"""
        with torch.no_grad():
            if self.init_random:
                # Initialize with random weights
                torch.nn.init.normal_(self.conv_weight, mean=0.0, std=0.2)
            elif self.init_first:
                # Initialize with first kernel
                self.conv_weight.zero_()
                self.conv_weight[:, 0, 0] = 1.0
            else:
                # Initialize with identity-like kernel
                self.conv_weight.zero_()
                center_q = self.kernel_size_q - 1  # Last row
                center_k = self.kernel_size_k // 2  # Center column
                self.conv_weight[:, center_q, center_k] = 1.0

    def forward(self, Q, K, V, causal=True, softmax_scale=None):
        """
        Forward pass with CUDA kernels
        Args:
            Q: [batch_size, n_heads, seq_len, head_dim]
            K: [batch_size, n_heads, seq_len, head_dim]
            V: [batch_size, n_heads, seq_len, head_dim]
            causal: whether to apply causal masking
            softmax_scale: scaling factor for softmax
        Returns:
            output: [batch_size, n_heads, seq_len, head_dim]
            M: max values [batch_size, n_heads, seq_len]
            L: sum of softmax [batch_size, n_heads, seq_len]
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape

        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)

        # Convert to bfloat16 for CUDA kernels
        Q_bf16 = Q.to(torch.bfloat16)
        K_bf16 = K.to(torch.bfloat16)
        V_bf16 = V.to(torch.bfloat16)
        conv_weight_bf16 = self.conv_weight.to(torch.bfloat16)

        # Calculate padding for convolution
        pad_k = (self.kernel_size_k - 1) // 2
        pad_q = self.kernel_size_q - 1

        # Use the CUDA forward kernel through the FlashConvAttentionFunction
        return FlashConvAttentionCUDAFunction.apply(
            Q_bf16,
            K_bf16,
            V_bf16,
            conv_weight_bf16,
            softmax_scale,
            pad_q,
            pad_k,
            causal,
        )


class FlashConvAttentionCUDAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, conv_kernel, scale, pad_q, pad_k, causal):
        """
        Forward pass using CUDA kernel
        """
        batch_size, num_heads, seq_len, d_head = Q.shape

        # Ensure tensors are in bfloat16 format
        Q = Q.to(torch.bfloat16)
        K = K.to(torch.bfloat16)
        V = V.to(torch.bfloat16)
        conv_kernel = conv_kernel.to(torch.bfloat16)

        # Call the CUDA forward kernel - returns tuple (O, M, L)
        O_output, M, L = forward(Q, K, V, conv_kernel, scale, pad_q, pad_k, causal)

        # Save tensors for backward pass
        ctx.save_for_backward(
            Q, K, V, conv_kernel, M.to(torch.bfloat16), L.to(torch.bfloat16), O_output
        )
        ctx.scale = scale
        ctx.pad_q = pad_q
        ctx.pad_k = pad_k
        ctx.causal = causal

        return O_output

    @staticmethod
    def backward(ctx, dO):
        """
        Backward pass using CUDA kernels
        """
        Q, K, V, conv_kernel, M, L, O = ctx.saved_tensors
        L = M + torch.log(L)
        scale = ctx.scale
        pad_q = ctx.pad_q
        pad_k = ctx.pad_k
        causal = ctx.causal

        # Ensure dO is in bfloat16 format
        dO = dO.to(torch.bfloat16)

        # Calculate D = sum(dO * O, dim=-1)
        # In a proper implementation, this should be computed efficiently
        # For now, we'll use a placeholder approach
        batch_size, num_heads, seq_len, d_head = Q.shape

        D = (dO.float() * O).sum(dim=-1).to(torch.bfloat16)

        # Call the CUDA backward kernels
        dQ, dconv_kernel = dq_backward(
            Q, K, V, dO, D, L, conv_kernel, scale, pad_q, pad_k, causal
        )
        dK, dV = dkdv_backward(
            Q, K, V, dO, D, L, conv_kernel, scale, pad_q, pad_k, causal
        )

        # Convert back to original dtype
        dQ = dQ.to(Q.dtype) if dQ is not None else None
        dK = dK.to(K.dtype) if dK is not None else None
        dV = dV.to(V.dtype) if dV is not None else None
        dconv_kernel = (
            dconv_kernel.to(conv_kernel.dtype) if dconv_kernel is not None else None
        )

        # Return gradients for all inputs (None for non-differentiable parameters)
        return dQ, dK, dV, dconv_kernel, None, None, None, None

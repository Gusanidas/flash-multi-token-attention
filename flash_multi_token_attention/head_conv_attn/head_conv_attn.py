import torch
import torch.nn as nn
from torch.autograd import Function
import math

from .fwd_kernel import head_conv_attention_forward
from .bwd_dq_dw import head_conv_attention_bwd_dq_dw
from .bwd_dk_dv import head_conv_attention_bwd_dkdv


class HeadConvAttentionFunction(Function):
    @staticmethod
    def forward(ctx, q, k, v, w, causal=False, scale=None):
        """
        Head Convolution Attention forward pass using Triton kernels.
        
        Args:
            q: Query tensor [batch, heads, seq_len, d_model]
            k: Key tensor [batch, heads, seq_len, d_model]
            v: Value tensor [batch, heads, seq_len, d_model]
            w: Linear transformation weights [heads_out, heads_in]
            causal: Whether to apply causal masking
            scale: Attention scale factor (default: 1/sqrt(d_model))
            
        Returns:
            output: Attention output [batch, heads, seq_len, d_model]
        """
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
            
        # Call the forward kernel
        output, l, m = head_conv_attention_forward(q, k, v, w, scale=scale, causal=causal)
        
        # Save tensors for backward pass
        ctx.save_for_backward(q, k, v, w, output, l, m)
        ctx.causal = causal
        ctx.scale = scale
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Head Convolution Attention backward pass using Triton kernels.
        """
        q, k, v, w, output, l, m = ctx.saved_tensors
        causal = ctx.causal
        scale = ctx.scale
        
        # Initialize gradients
        grad_q = grad_k = grad_v = grad_w = None
        
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:  # Need dQ or dW
            grad_q, grad_w = head_conv_attention_bwd_dq_dw(
                q, k, v, w, output, grad_output, l, m, causal=causal, scale=scale
            )
            
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:  # Need dK or dV
            grad_k, grad_v = head_conv_attention_bwd_dkdv(
                q, k, v, w, output, grad_output, l, m, causal=causal, scale=scale
            )
            
        return grad_q, grad_k, grad_v, grad_w, None, None


class HeadConvAttention(nn.Module):
    """
    Head Convolution Attention module that applies a linear transformation
    across the head dimension before softmax.
    """
    
    def __init__(self, n_heads, dtype=torch.float16):
        super().__init__()
        self.n_heads = n_heads
        self.dtype = dtype
        
        # Linear transformation weights across head dimension
        self.head_conv_weights = nn.Parameter(
            torch.randn(n_heads, n_heads, dtype=dtype) / math.sqrt(n_heads)
        )
        
    def forward(self, q, k, v, causal=False, scale=None):
        """
        Forward pass of Head Convolution Attention.
        
        Args:
            q: Query tensor [batch, heads, seq_len, d_model]
            k: Key tensor [batch, heads, seq_len, d_model]
            v: Value tensor [batch, heads, seq_len, d_model]
            causal: Whether to apply causal masking
            scale: Attention scale factor
            
        Returns:
            output: Attention output [batch, heads, seq_len, d_model]
        """
        return HeadConvAttentionFunction.apply(
            q, k, v, self.head_conv_weights, causal, scale
        )


def head_conv_attention(q, k, v, w, causal=False, scale=None):
    """
    Functional interface for Head Convolution Attention.
    
    Args:
        q: Query tensor [batch, heads, seq_len, d_model]
        k: Key tensor [batch, heads, seq_len, d_model]
        v: Value tensor [batch, heads, seq_len, d_model]
        w: Linear transformation weights [heads_out, heads_in]
        causal: Whether to apply causal masking
        scale: Attention scale factor
        
    Returns:
        output: Attention output [batch, heads, seq_len, d_model]
    """
    return HeadConvAttentionFunction.apply(q, k, v, w, causal, scale)
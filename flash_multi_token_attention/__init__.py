"""
Flash Multi-Token Attention

This package implements various Flash Attention algorithm variations with efficient CUDA kernels.
"""

from .conv_attn import FlashConvAttention, FlashConvAttentionFunction

__version__ = "0.1.0"
__all__ = ["FlashConvAttention", "FlashConvAttentionFunction"]
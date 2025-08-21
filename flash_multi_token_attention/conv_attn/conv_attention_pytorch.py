import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def print_matrix_in_quadrants(name: str, matrix: torch.Tensor, scale: float = 1.0) -> None:
    """Print a matrix in four quadrants for quick inspection.

    If the input tensor has more than 2 dimensions (e.g. [batch, heads, rows, cols]),
    the function prints the first batch and first head. No output is printed if
    either dimension > 32.
    """
    print(f"In print_matrix_in_quadrants, conv_attention_pytorch.py, name: {name}")
    # Max, abs mean
    print(f"Max: {matrix.max()}, Abs mean: {matrix.abs().mean()}")
    with torch.no_grad():
        if matrix.dim() < 2:
            return

        if matrix.dim() >= 4:
            matrix_2d = matrix[0, 0]
        elif matrix.dim() == 2:
            matrix_2d = matrix
        else:
            print(f"Returning, dim = {matrix.dim()}")
            return

        rows = int(matrix_2d.shape[-2])
        cols = int(matrix_2d.shape[-1])
        
        if rows > 32 or cols > 32:
            print(f"Returning, shape = {matrix_2d.shape} (dimensions too large)")
            return

        matrix_cpu = matrix_2d.detach().to("cpu").float()
        mid_row = (rows + 1) // 2
        mid_col = (cols + 1) // 2

        top = slice(0, mid_row)
        bottom = slice(mid_row, rows)
        left = slice(0, mid_col)
        right = slice(mid_col, cols)

        print(f"=== {name} (first batch/head), shape=({rows}, {cols}) ===")
        
        # Helper function to print a matrix section with indices
        def print_matrix_section(section_name: str, matrix_section: torch.Tensor, row_offset: int, col_offset: int):
            print(f"{section_name}:")
            section_rows, section_cols = matrix_section.shape
            
            # Print column headers
            col_header = "        "
            for c in range(section_cols):
                col_header += f"{c + col_offset:7d}"
            print(col_header)
            
            # Print each row with row index
            for r in range(section_rows):
                row_str = f"Row {r + row_offset:2d}: "
                for c in range(section_cols):
                    row_str += f"{(matrix_section[r, c].item() * scale):7.2f}"
                print(row_str)
            print()
        
        print_matrix_section("Top-Left", matrix_cpu[top, left], 0, 0)
        print_matrix_section("Top-Right", matrix_cpu[top, right], 0, mid_col)
        print_matrix_section("Bottom-Left", matrix_cpu[bottom, left], mid_row, 0)
        print_matrix_section("Bottom-Right", matrix_cpu[bottom, right], mid_row, mid_col)


class ConvAttention(nn.Module):
    """
    Simplified conv attention that only applies convolution before softmax.
    Similar to mta_attention_pytorch.py but much simpler.
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
        Forward pass with convolution before softmax
        Args:
            Q: [batch_size, n_heads, seq_len, head_dim]
            K: [batch_size, n_heads, seq_len, head_dim]
            V: [batch_size, n_heads, seq_len, head_dim]
            causal: whether to apply causal masking
            softmax_scale: scaling factor for softmax
        Returns:
            output: [batch_size, n_heads, seq_len, head_dim]
        """
        batch_size, n_heads, seq_len, head_dim = Q.shape

        if softmax_scale is None:
            softmax_scale = 1.0 / (head_dim**0.5)

        # Compute QK scores
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, seq, seq]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device), diagonal=1)
        mask = mask.bool()
        if causal:
            scores = scores.masked_fill(mask, 0)

        scores = scores * softmax_scale


        # Apply convolution to scores
        conv_scores = self._apply_convolution(scores)

        # Apply causal mask if needed
        if causal:
            conv_scores = conv_scores.masked_fill(mask, float("-inf"))

        # Apply softmax
        conv_scores_scaled = conv_scores # Scale before

        M = conv_scores_scaled.max(dim=-1, keepdim=False).values

        exp_scores = torch.exp(conv_scores_scaled - M.unsqueeze(-1))

        L = exp_scores.sum(dim=-1, keepdim=False)
        attn_weights = F.softmax(conv_scores_scaled, dim=-1)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)

        return output, M, L

    def _apply_convolution(self, scores):
        """Apply convolution to attention scores"""
        batch_size, n_heads, seq_len, _ = scores.shape

        # Pad scores for convolution
        # For "both" padding strategy like in MTA
        pad_k = (self.kernel_size_k - 1) // 2
        pad_q = self.kernel_size_q - 1

        scores_padded = F.pad(scores, (pad_k, pad_k, pad_q, 0), value=0.0)

        # Apply convolution with grouped convolution
        conv_scores = F.conv2d(
            scores_padded,
            self.conv_weight.unsqueeze(1),  # Add channel dim
            padding=0,
            groups=n_heads,
        )

        return conv_scores

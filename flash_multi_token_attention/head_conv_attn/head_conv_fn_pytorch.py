import torch
import torch.nn.functional as F
import math


def pytorch_reference_head_conv_attention(xq, xk, xv, w, scale=1.0, causal=False):
    # Reference implementation adapted from the user provided PyTorch code
    # xq, xk, xv: [B, H, N, D]
    # w: [H_out, H_in] (Weights of the nn.Linear layer)
    batch_size, n_heads, seq_len, head_dim = xq.shape

    # 1. Compute QK.T
    scores = torch.matmul(xq, xk.transpose(2, 3)) * scale  # [B, H, N, N]

    # 2. Apply linear transformation over head dimension
    # This matches: scores = self.wpsm(scores.transpose(1, -1)).transpose(1, -1)

    # Transpose to [B, N, N, H] (using permute to handle the generalized transpose(1, -1))
    scores_t = scores.transpose(1, -1)
    # Apply linear layer: y = xW^T. F.linear expects weight (out, in).
    scores_t = F.linear(scores_t, w)
    # Transpose back to [B, H, N, N]
    scores = scores_t.transpose(1, -1)

    # 3. Apply causal mask
    if causal:
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=xq.device, dtype=xq.dtype
        )
        mask = torch.triu(mask, diagonal=1)
        # Mask is broadcasted over Batch and Heads dimensions.
        scores = scores + mask[None, None, :, :]

    M = scores.max(dim=-1, keepdim=False).values
    L = torch.exp(scores - M[:, :, :, None]).sum(dim=-1)
    # print(f"Shape of M: {M.shape}, Shape of L: {L.shape}")
    # print(f"Abs max of M: {M.abs().max().item()}, Abs max of L: {L.abs().max().item()}")
    # print("0-0-0")

    # 4. Softmax
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)

    # 5. Output calculation
    output = torch.matmul(scores, xv)  # [B, H, N, D]

    return output, L, M

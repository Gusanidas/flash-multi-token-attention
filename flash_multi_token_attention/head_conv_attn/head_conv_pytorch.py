import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadConvAttention(nn.Module):
    def __init__(self, n_heads: int):
        super().__init__()

        self.n_heads = n_heads

        self.wpsm = nn.Linear(
            n_heads,
            n_heads,
            bias=False,
        )

    def forward(self, xq, xk, xv, causal=False):
        mask = self._update_mask(mask=False, bsz=xq.size(0), xq=xq, xk_size=xk.size(-2))

        head_dim = xq.size(-1)
        scores = torch.matmul(xq, xk.transpose(2, 3)) * torch.rsqrt(
            torch.tensor(head_dim, requires_grad=False, device="cuda")
        )

        # Apply linear transformation over head dimension
        scores = self.wpsm(scores.transpose(1, -1)).transpose(1, -1)

        if causal:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2)

        return output

    def _update_mask(
        self, mask: torch.tensor, bsz: int, xq: torch.tensor, xk_size: int
    ):
        if not isinstance(mask, torch.Tensor):
            # causal mask
            mask = torch.full((xq.size(-2), xk_size), float("-inf"), device=xq.device)
            mask = torch.triu(mask, diagonal=1).type_as(xq)
            mask = mask.repeat(bsz, self.n_heads, 1, 1)
        else:
            # generation task, mask is provided and reflects concatenated docs
            if mask.dtype == torch.bool:
                mask = torch.where(mask, 0.0, float("-inf")).to(xq.dtype)
            if mask.dtype != xq.dtype:
                mask = mask.type(xq.dtype)
            if len(mask.shape) == 2:
                assert mask.size(0) == bsz
                mask = mask.repeat(1, self.n_heads, xq.size(-2), 1)
                mask_i, mask_j = torch.triu_indices(xq.size(-2), xk_size, offset=1)
                mask[:, :, mask_i, mask_j] = float("-inf")
            else:
                if mask.size(0) == 1 and mask.size(1) == 1:
                    # in prefilling mask is defined for 1 head
                    mask = mask.repeat(bsz, self.n_heads, 1, 1)
        return mask

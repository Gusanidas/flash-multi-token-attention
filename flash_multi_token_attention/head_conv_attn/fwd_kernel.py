import torch
import triton
import triton.language as tl
import math


@triton.jit
def head_conv_attn_fwd_kernel(
    Q,
    K,
    V,
    W,  # Linear transformation weights [n_heads_out, n_heads]
    O,  # Output tensor
    L,
    M,  # Logsumexp accumulators
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  # Q strides
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  # K strides
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,  # V strides
    stride_wh,
    stride_wi,  # W strides
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,  # O strides
    stride_lz,
    stride_lh,
    stride_lm,  # L strides
    stride_mz,
    stride_mh,
    stride_mm,  # M strides
    Z,
    H,
    SCALE,
    N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    N_HEADS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_D: tl.constexpr = 32,  # Chunk size for d_head dimension
):
    """
    Head Convolution Attention forward kernel with chunked QK and V computation.

    Implements attention with linear transformation across head dimension:
    1. For each N block, load Q and K in chunks of BLOCK_D along d_head
    2. Compute QK^T by accumulating across d_head chunks
    3. Apply linear transformation W @ scores_reshaped
    4. For each V chunk, compute partial output and write directly to global memory
    """
    # Get program IDs
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)  # Each thread block handles one batch element, all heads

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, N_HEADS)

    # Initialize logsumexp trackers for all heads
    m_hi = tl.zeros([N_HEADS, BLOCK_M], dtype=tl.float32) - 200.0
    l_hi = tl.zeros([N_HEADS, BLOCK_M], dtype=tl.float32) + 0.1

    # Load linear transformation weights W: [n_heads_out, n_heads]
    w_ptrs = W + offs_h[:, None] * stride_wh + offs_h[None, :] * stride_wi
    w = tl.load(w_ptrs, mask=(offs_h[:, None] < N_HEADS) & (offs_h[None, :] < N_HEADS))

    # Initialize output tensor to zero (we'll accumulate across N blocks)
    for start_d_out in range(0, BLOCK_DMODEL, BLOCK_D):
        start_d_out = tl.multiple_of(start_d_out, BLOCK_D)
        offs_d_out_curr = start_d_out + offs_d

        o_ptrs = (
            O
            + off_z * stride_oz
            + offs_h[:, None, None] * stride_oh
            + offs_m[None, :, None] * stride_om
            + offs_d_out_curr[None, None, :] * stride_ok
        )
        tl.store(
            o_ptrs,
            tl.zeros([N_HEADS, BLOCK_M, BLOCK_D], dtype=tl.float32),
            mask=(offs_h[:, None, None] < N_HEADS)
            & (offs_m[None, :, None] < N_CTX)
            & (offs_d_out_curr[None, None, :] < BLOCK_DMODEL),
        )

    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n

        # Causal mask
        if IS_CAUSAL:
            mask = offs_m[:, None] >= offs_n_curr[None, :]
        else:
            mask = True

        # Initialize attention scores for this block: [n_heads, block_m, block_n]
        s = tl.zeros([N_HEADS, BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Chunked computation over d_head dimension for Q@K^T
        for start_d in range(0, BLOCK_DMODEL, BLOCK_D):
            start_d = tl.multiple_of(start_d, BLOCK_D)
            offs_d_curr = start_d + offs_d

            # Load Q chunk for all heads: [n_heads, block_m, block_d]
            q_ptrs = (
                Q
                + off_z * stride_qz
                + offs_h[:, None, None] * stride_qh
                + offs_m[None, :, None] * stride_qm
                + offs_d_curr[None, None, :] * stride_qk
            )
            q_chunk = tl.load(
                q_ptrs,
                mask=(offs_h[:, None, None] < N_HEADS)
                & (offs_m[None, :, None] < N_CTX)
                & (offs_d_curr[None, None, :] < BLOCK_DMODEL),
                other=0.0,
            )

            # Load K chunk for all heads: [n_heads, block_n, block_d]
            k_ptrs = (
                K
                + off_z * stride_kz
                + offs_h[:, None, None] * stride_kh
                + offs_n_curr[None, :, None] * stride_kn
                + offs_d_curr[None, None, :] * stride_kk
            )
            k_chunk = tl.load(
                k_ptrs,
                mask=(offs_h[:, None, None] < N_HEADS)
                & (offs_n_curr[None, :, None] < N_CTX)
                & (offs_d_curr[None, None, :] < BLOCK_DMODEL),
                other=0.0,
            )

            # Accumulate QK^T for this chunk: [n_heads, block_m, block_n]
            # We need to transpose k_chunk for the matmul: k_chunk -> [n_heads, block_d, block_n]
            k_chunk_t = tl.trans(k_chunk, (0, 2, 1))
            s += tl.dot(q_chunk, k_chunk_t)

        # Apply scale to final scores
        s *= SCALE

        # Reshape scores to [block_m*block_n, n_heads] for linear transformation
        # Note: This is conceptual - in practice we need to handle this differently in Triton
        # We'll apply the transformation head-wise instead
        s_reshaped = tl.reshape(s, (N_HEADS, BLOCK_M * BLOCK_N))
        s_transformed = tl.dot(w, s_reshaped.to(w.dtype))
        s = tl.reshape(s_transformed, (N_HEADS, BLOCK_M, BLOCK_N))

        if IS_CAUSAL:
            s = tl.where(mask[None, :, :], s, -1000.0)

        # Standard attention computation for normalization
        m_hij = tl.max(s, axis=-1)
        m_hi_new = tl.maximum(m_hij, m_hi)
        s = s - m_hi_new[:, :, None]
        p = tl.exp(s)
        alpha = tl.exp(m_hi - m_hi_new)
        l_hi_new = alpha * l_hi + tl.sum(p, axis=-1)

        # Update normalization factors
        l_hi = l_hi_new
        m_hi = m_hi_new

        # Loop over V chunks and write partial results directly to global memory
        for start_d_out in range(0, BLOCK_DMODEL, BLOCK_D):
            start_d_out = tl.multiple_of(start_d_out, BLOCK_D)
            offs_d_out_curr = start_d_out + offs_d

            # Load V chunk for all heads: [n_heads, block_n, block_d]
            v_ptrs = (
                V
                + off_z * stride_vz
                + offs_h[:, None, None] * stride_vh
                + offs_n_curr[None, :, None] * stride_vn
                + offs_d_out_curr[None, None, :] * stride_vk
            )
            v_chunk = tl.load(
                v_ptrs,
                mask=(offs_h[:, None, None] < N_HEADS)
                & (offs_n_curr[None, :, None] < N_CTX)
                & (offs_d_out_curr[None, None, :] < BLOCK_DMODEL),
                other=0.0,
            )

            # Compute partial output: P @ V_chunk -> [n_heads, block_m, block_d]
            partial_out = tl.dot(p.to(v_chunk.dtype), v_chunk)

            # Load existing output chunk and add to it (accumulate across N blocks)
            o_ptrs = (
                O
                + off_z * stride_oz
                + offs_h[:, None, None] * stride_oh
                + offs_m[None, :, None] * stride_om
                + offs_d_out_curr[None, None, :] * stride_ok
            )

            existing_out = tl.load(
                o_ptrs,
                mask=(offs_h[:, None, None] < N_HEADS)
                & (offs_m[None, :, None] < N_CTX)
                & (offs_d_out_curr[None, None, :] < BLOCK_DMODEL),
                other=0.0,
            )

            # Apply alpha scaling for online softmax and accumulate
            scaled_existing = existing_out * alpha[:, :, None]
            updated_out = scaled_existing + partial_out

            # Write back to global memory
            tl.store(
                o_ptrs,
                updated_out,
                mask=(offs_h[:, None, None] < N_HEADS)
                & (offs_m[None, :, None] < N_CTX)
                & (offs_d_out_curr[None, None, :] < BLOCK_DMODEL),
            )

    # Final normalization: divide by accumulated logsumexp
    for start_d_out in range(0, BLOCK_DMODEL, BLOCK_D):
        start_d_out = tl.multiple_of(start_d_out, BLOCK_D)
        offs_d_out_curr = start_d_out + offs_d

        o_ptrs = (
            O
            + off_z * stride_oz
            + offs_h[:, None, None] * stride_oh
            + offs_m[None, :, None] * stride_om
            + offs_d_out_curr[None, None, :] * stride_ok
        )

        out_chunk = tl.load(
            o_ptrs,
            mask=(offs_h[:, None, None] < N_HEADS)
            & (offs_m[None, :, None] < N_CTX)
            & (offs_d_out_curr[None, None, :] < BLOCK_DMODEL),
            other=0.0,
        )

        # Normalize by logsumexp
        normalized_out = out_chunk / l_hi[:, :, None]

        tl.store(
            o_ptrs,
            normalized_out,
            mask=(offs_h[:, None, None] < N_HEADS)
            & (offs_m[None, :, None] < N_CTX)
            & (offs_d_out_curr[None, None, :] < BLOCK_DMODEL),
        )

    # Store logsumexp values for all heads
    l_ptrs = (
        L
        + off_z * stride_lz
        + offs_h[:, None] * stride_lh
        + offs_m[None, :] * stride_lm
    )
    m_ptrs = (
        M
        + off_z * stride_mz
        + offs_h[:, None] * stride_mh
        + offs_m[None, :] * stride_mm
    )
    tl.store(l_ptrs, l_hi, mask=(offs_h[:, None] < N_HEADS) & (offs_m[None, :] < N_CTX))
    tl.store(m_ptrs, m_hi, mask=(offs_h[:, None] < N_HEADS) & (offs_m[None, :] < N_CTX))


def head_conv_attention_forward(q, k, v, w, scale=1.0, causal=False):
    """
    Head Convolution Attention forward pass.

    Args:
        q: Query tensor [batch, heads, seq_len, d_model]
        k: Key tensor [batch, heads, seq_len, d_model]
        v: Value tensor [batch, heads, seq_len, d_model]
        w: Linear transformation weights [heads_out, heads_in]
        causal: Whether to apply causal masking

    Returns:
        output: Attention output [batch, heads, seq_len, d_model]
        l: Logsumexp values (for backward pass)
        m: Max values (for backward pass)
    """
    # Get dimensions
    batch, heads, seq_len, d_model = q.shape
    assert heads in [16, 32], f"n_heads must be 16 or 32, got {heads}"

    # Allocate output tensors
    o = torch.empty_like(q)
    l = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)
    m = torch.empty((batch, heads, seq_len), device=q.device, dtype=torch.float32)

    # Set block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_DMODEL = d_model

    # Ensure d_model is within Triton's constraints
    assert d_model <= 128, "d_model must be <= 128"

    # Grid dimensions - one program per batch element, each handles all heads
    grid = (triton.cdiv(seq_len, BLOCK_M), batch)

    # Launch kernel
    head_conv_attn_fwd_kernel[grid](
        q,
        k,
        v,
        w,
        o,
        l,
        m,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        w.stride(0),
        w.stride(1),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        l.stride(0),
        l.stride(1),
        l.stride(2),
        m.stride(0),
        m.stride(1),
        m.stride(2),
        batch,
        heads,
        scale,
        seq_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        N_HEADS=heads,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )

    return o, l, m


def print_stats(a, name):
    print(f"Stats for {name}:")
    print(f"Max: {a.max().item()}")
    print(f"Min: {a.min().item()}")
    print(f"Abs Mean: {a.abs().mean().item()}")
    print(f"Std: {a.std().item()}")
    print()


# Example usage
if __name__ == "__main__":
    from head_conv_fn_pytorch import pytorch_reference_head_conv_attention
    from head_conv_pytorch import HeadConvAttention

    torch.manual_seed(34)

    # Test the implementation
    batch_size = 2
    n_heads = 16
    seq_len = 201
    d_model = 32
    dtype = torch.bfloat16
    causal = False
    device = "cuda"

    # Create random inputs
    q = torch.randn(batch_size, n_heads, seq_len, d_model, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, d_model, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, d_model, device=device, dtype=dtype)
    w = torch.randn(n_heads, n_heads, device=device, dtype=dtype) / 2

    # Run head conv attention
    output, l, m = head_conv_attention_forward(q, k, v, w, causal=causal)
    print_stats(output, "output")
    print_stats(l, "l")
    print_stats(m, "m")
    # print(f"L={l}")
    # print(f"M={m}")

    print(f"Output shape: {output.shape}")
    print(f"Memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

    # Verify against PyTorch reference implementation

    # head_conv_attn = HeadConvAttention(n_heads=n_heads)
    # expected = head_conv_attn(q, k, v, mask=None, chunk_start_ids=None, causal=causal)
    O_e, L_e, M_e = pytorch_reference_head_conv_attention(q, k, v, w, causal=causal)
    print()
    print_stats(O_e, "O_e")
    print_stats(M_e, "M_e")
    print_stats(L_e, "L_e")

    abs_diff_o = (output - O_e).abs()
    abs_diff_l = (l - L_e).abs()
    abs_diff_m = (m - M_e).abs()
    print()
    print_stats(abs_diff_o, "abs_diff_o")
    print_stats(abs_diff_l, "abs_diff_l")
    print_stats(abs_diff_m, "abs_diff_m")

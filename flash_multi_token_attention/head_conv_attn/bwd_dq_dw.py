# bwd_dq_dw.py
import torch
import triton
import triton.language as tl
import math


@triton.jit
def head_conv_attn_bwd_dq_dw_kernel(
    Q,
    K,
    V,
    W,
    O,
    DO,
    L,
    M,
    DQ,
    DW,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_wh,
    stride_wi,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_dok,
    stride_lz,
    stride_lh,
    stride_lm,
    stride_mz,
    stride_mh,
    stride_mm,
    stride_dqz,
    stride_dqh,
    stride_dqm,
    stride_dqk,
    stride_dwh,
    stride_dwi,
    Z,
    H,
    N_CTX,
    SCALE,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    N_HEADS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_D: tl.constexpr = 16,
):
    # Program IDs
    start_m = tl.program_id(0)
    off_z = tl.program_id(1)

    # Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, N_HEADS)

    # Masks
    mask_m = offs_m < N_CTX
    mask_h = offs_h < H
    mask_m_h = mask_h[:, None] & mask_m[None, :]
    mask_w = mask_h[:, None] & mask_h[None, :]

    # Load L, M - these are needed for all chunks
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
    l = tl.load(l_ptrs, mask=mask_m_h, other=0.0)
    m_val = tl.load(m_ptrs, mask=mask_m_h, other=0.0)

    # Load W and W^T - these are needed for all chunks
    w_ptrs = W + offs_h[:, None] * stride_wh + offs_h[None, :] * stride_wi
    w = tl.load(w_ptrs, mask=mask_w)
    wt_ptrs = W + offs_h[None, :] * stride_wh + offs_h[:, None] * stride_wi
    wt = tl.load(wt_ptrs, mask=mask_w)

    # Initialize dW accumulator (accumulates across all chunks)
    acc_dw = tl.zeros([N_HEADS, N_HEADS], dtype=tl.float32)

    # Initialize dQ output to zero (we'll accumulate across N blocks)
    for start_d_out in range(0, BLOCK_DMODEL, BLOCK_D):
        start_d_out = tl.multiple_of(start_d_out, BLOCK_D)
        offs_d_out_curr = start_d_out + offs_d
        mask_m_h_d = (
            mask_h[:, None, None]
            & mask_m[None, :, None]
            & (offs_d_out_curr[None, None, :] < BLOCK_DMODEL)
        )

        dq_ptrs = (
            DQ
            + off_z * stride_dqz
            + offs_h[:, None, None] * stride_dqh
            + offs_m[None, :, None] * stride_dqm
            + offs_d_out_curr[None, None, :] * stride_dqk
        )
        tl.store(
            dq_ptrs,
            tl.zeros([N_HEADS, BLOCK_M, BLOCK_D], dtype=tl.float32),
            mask=mask_m_h_d,
        )

    # Determine loop range (Causal optimization)
    loop_end = N_CTX
    if IS_CAUSAL:
        loop_end = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)

    # Loop over N blocks (K, V)
    for start_n in range(0, loop_end, BLOCK_N):
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < N_CTX

        # Causal mask
        if IS_CAUSAL:
            mask_causal = offs_m[:, None] >= offs_n_curr[None, :]
        else:
            mask_causal = True

        # Initialize attention scores for this block: [n_heads, block_m, block_n]
        s_raw_full = tl.zeros([N_HEADS, BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Chunked computation over d_head dimension for Q@K^T
        for start_d in range(0, BLOCK_DMODEL, BLOCK_D):
            start_d = tl.multiple_of(start_d, BLOCK_D)
            offs_d_curr = start_d + offs_d
            mask_d = offs_d_curr < BLOCK_DMODEL

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
                mask=(
                    mask_h[:, None, None]
                    & mask_m[None, :, None]
                    & mask_d[None, None, :]
                ),
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
                mask=(
                    mask_h[:, None, None]
                    & mask_n[None, :, None]
                    & mask_d[None, None, :]
                ),
                other=0.0,
            )

            # Accumulate QK^T for this chunk: [n_heads, block_m, block_n]
            k_chunk_t = tl.trans(k_chunk, (0, 2, 1))
            s_raw_full += tl.dot(q_chunk, k_chunk_t)

        # Apply scale to final scores
        s_raw_full *= SCALE

        # Apply linear transformation: S = W @ S_raw
        s_raw_reshaped = tl.reshape(s_raw_full, [N_HEADS, BLOCK_M * BLOCK_N])
        s_reshaped = tl.dot(w, s_raw_reshaped.to(w.dtype))
        s = tl.reshape(s_reshaped, [N_HEADS, BLOCK_M, BLOCK_N])

        # Apply causal mask
        if IS_CAUSAL:
            s = tl.where(mask_causal[None, :, :], s, float("-inf"))

        # Compute P = softmax(S) (FP32)
        p = tl.exp(s.to(tl.float32) - m_val[:, :, None]) / l[:, :, None]

        # Process dO and O in chunks to compute dP and D
        dp_full = tl.zeros([N_HEADS, BLOCK_M, BLOCK_N], dtype=tl.float32)
        D = tl.zeros([N_HEADS, BLOCK_M], dtype=tl.float32)

        for start_d in range(0, BLOCK_DMODEL, BLOCK_D):
            start_d = tl.multiple_of(start_d, BLOCK_D)
            offs_d_curr = start_d + offs_d
            mask_d = offs_d_curr < BLOCK_DMODEL

            # Load dO chunk: [n_heads, block_m, block_d]
            do_ptrs = (
                DO
                + off_z * stride_doz
                + offs_h[:, None, None] * stride_doh
                + offs_m[None, :, None] * stride_dom
                + offs_d_curr[None, None, :] * stride_dok
            )
            do_chunk = tl.load(
                do_ptrs,
                mask=(
                    mask_h[:, None, None]
                    & mask_m[None, :, None]
                    & mask_d[None, None, :]
                ),
                other=0.0,
            )

            # Load O chunk: [n_heads, block_m, block_d]
            o_ptrs = (
                O
                + off_z * stride_oz
                + offs_h[:, None, None] * stride_oh
                + offs_m[None, :, None] * stride_om
                + offs_d_curr[None, None, :] * stride_ok
            )
            o_chunk = tl.load(
                o_ptrs,
                mask=(
                    mask_h[:, None, None]
                    & mask_m[None, :, None]
                    & mask_d[None, None, :]
                ),
                other=0.0,
            )

            # Load V chunk: [n_heads, block_n, block_d]
            v_ptrs = (
                V
                + off_z * stride_vz
                + offs_h[:, None, None] * stride_vh
                + offs_n_curr[None, :, None] * stride_vn
                + offs_d_curr[None, None, :] * stride_vk
            )
            v_chunk = tl.load(
                v_ptrs,
                mask=(
                    mask_h[:, None, None]
                    & mask_n[None, :, None]
                    & mask_d[None, None, :]
                ),
                other=0.0,
            )

            # Accumulate D = rowsum(dO * O)
            D += tl.sum((do_chunk * o_chunk).to(tl.float32), axis=-1)

            # Compute dP chunk: dO @ V^T
            v_chunk_t = tl.trans(v_chunk, (0, 2, 1))
            dp_full += tl.dot(do_chunk, v_chunk_t)

        # Compute dS = P * (dP - D) with numerical stability
        dp_minus_d = dp_full.to(tl.float32) - D[:, :, None]
        # Clamp values to prevent overflow
        dp_minus_d = tl.maximum(tl.minimum(dp_minus_d, 100.0), -100.0)
        ds = p * dp_minus_d

        # Apply mask to dS for safety
        mask_mn = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            mask_final = mask_causal & mask_mn
        else:
            mask_final = mask_mn
        ds = tl.where(mask_final[None, :, :], ds, 0.0)

        # dW contribution: dW += dS @ S_raw^T
        ds_reshaped = tl.reshape(ds, [N_HEADS, BLOCK_M * BLOCK_N])
        s_raw_reshaped_t = tl.trans(
            s_raw_reshaped, (1, 0)
        )  # Transpose to get [BLOCK_M*BLOCK_N, N_HEADS]
        dw_contrib = tl.dot(ds_reshaped, s_raw_reshaped_t.to(ds_reshaped.dtype))
        # Clamp dW contribution to prevent overflow
        dw_contrib = tl.maximum(tl.minimum(dw_contrib, 1000.0), -1000.0)
        acc_dw += dw_contrib

        # Compute dS_raw = W^T @ dS
        ds_raw_reshaped = tl.dot(wt, ds_reshaped.to(wt.dtype))
        ds_raw = tl.reshape(ds_raw_reshaped, [N_HEADS, BLOCK_M, BLOCK_N])

        # Process dQ in chunks: dQ += dS_raw @ K * scale
        for start_d_out in range(0, BLOCK_DMODEL, BLOCK_D):
            start_d_out = tl.multiple_of(start_d_out, BLOCK_D)
            offs_d_out_curr = start_d_out + offs_d
            mask_d_out = offs_d_out_curr < BLOCK_DMODEL

            # Load K chunk: [n_heads, block_n, block_d]
            k_ptrs = (
                K
                + off_z * stride_kz
                + offs_h[:, None, None] * stride_kh
                + offs_n_curr[None, :, None] * stride_kn
                + offs_d_out_curr[None, None, :] * stride_kk
            )
            k_chunk = tl.load(
                k_ptrs,
                mask=(
                    mask_h[:, None, None]
                    & mask_n[None, :, None]
                    & mask_d_out[None, None, :]
                ),
                other=0.0,
            )

            # Compute partial dQ: dS_raw @ K_chunk * scale
            partial_dq = tl.dot(ds_raw.to(k_chunk.dtype), k_chunk) * SCALE

            # Load existing dQ chunk and accumulate
            dq_ptrs = (
                DQ
                + off_z * stride_dqz
                + offs_h[:, None, None] * stride_dqh
                + offs_m[None, :, None] * stride_dqm
                + offs_d_out_curr[None, None, :] * stride_dqk
            )

            existing_dq = tl.load(
                dq_ptrs,
                mask=(
                    mask_h[:, None, None]
                    & mask_m[None, :, None]
                    & mask_d_out[None, None, :]
                ),
                other=0.0,
            )

            updated_dq = existing_dq + partial_dq

            # Store back to global memory
            tl.store(
                dq_ptrs,
                updated_dq,
                mask=(
                    mask_h[:, None, None]
                    & mask_m[None, :, None]
                    & mask_d_out[None, None, :]
                ),
            )

    # Store dW using atomic add with proper masking
    dw_ptrs = DW + offs_h[:, None] * stride_dwh + offs_h[None, :] * stride_dwi
    # Additional clamp for final dW values to prevent extreme accumulation
    acc_dw = tl.maximum(tl.minimum(acc_dw, 10000.0), -10000.0)
    tl.atomic_add(dw_ptrs, acc_dw, mask=mask_w)


def head_conv_attention_bwd_dq_dw(q, k, v, w, o, do, l, m, causal=False, scale=None):
    batch, heads, seq_len, d_model = q.shape
    assert d_model <= 128, "d_model must be <= 128"

    if scale is None:
        scale = 1.0 / math.sqrt(d_model)

    # Allocate outputs (FP32 accumulation buffers)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dw = torch.zeros_like(w, dtype=torch.float32)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    w = w.contiguous()
    o = o.contiguous()
    do = do.contiguous()
    l = l.contiguous()

    dq = dq.contiguous()
    dw = dw.contiguous()

    # Block sizes (can be tuned)
    BLOCK_M = 16
    BLOCK_N = 16

    grid = (triton.cdiv(seq_len, BLOCK_M), batch)

    head_conv_attn_bwd_dq_dw_kernel[grid](
        q,
        k,
        v,
        w,
        o,
        do,
        l,
        m,
        dq,
        dw,
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
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        l.stride(0),
        l.stride(1),
        l.stride(2),
        m.stride(0),
        m.stride(1),
        m.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        dw.stride(0),
        dw.stride(1),
        batch,
        heads,
        seq_len,
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=d_model,
        N_HEADS=heads,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=2,
    )

    # Return gradients cast to the input dtype
    return dq.to(q.dtype), dw.to(w.dtype)

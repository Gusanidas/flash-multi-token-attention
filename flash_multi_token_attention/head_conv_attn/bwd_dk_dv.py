# bwd_dkdv.py
import torch
import triton
import triton.language as tl
import math


@triton.jit
def head_conv_attn_bwd_dkdv_kernel(
    Q,
    K,
    V,
    W,
    O,
    DO,
    L,
    M,
    DK,
    DV,
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
    stride_dkz,
    stride_dkh,
    stride_dkn,
    stride_dkk,
    stride_dvz,
    stride_dvh,
    stride_dvn,
    stride_dvk,
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
    start_n = tl.program_id(0)
    off_z = tl.program_id(1)

    # Offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_h = tl.arange(0, N_HEADS)

    # Masks
    mask_n = offs_n < N_CTX
    mask_h = offs_h < H
    mask_w = mask_h[:, None] & mask_h[None, :]

    # Load W and W^T - these are needed for all chunks
    w_ptrs = W + offs_h[:, None] * stride_wh + offs_h[None, :] * stride_wi
    w = tl.load(w_ptrs, mask=mask_w)
    wt_ptrs = W + offs_h[None, :] * stride_wh + offs_h[:, None] * stride_wi
    wt = tl.load(wt_ptrs, mask=mask_w)

    # Initialize dK and dV outputs to zero 
    for start_d_out in range(0, BLOCK_DMODEL, BLOCK_D):
        start_d_out = tl.multiple_of(start_d_out, BLOCK_D)
        offs_d_out_curr = start_d_out + offs_d
        mask_h_n_d = (
            mask_h[:, None, None]
            & mask_n[None, :, None]
            & (offs_d_out_curr[None, None, :] < BLOCK_DMODEL)
        )

        dk_ptrs = (
            DK
            + off_z * stride_dkz
            + offs_h[:, None, None] * stride_dkh
            + offs_n[None, :, None] * stride_dkn
            + offs_d_out_curr[None, None, :] * stride_dkk
        )
        dv_ptrs = (
            DV
            + off_z * stride_dvz
            + offs_h[:, None, None] * stride_dvh
            + offs_n[None, :, None] * stride_dvn
            + offs_d_out_curr[None, None, :] * stride_dvk
        )

        tl.store(
            dk_ptrs,
            tl.zeros([N_HEADS, BLOCK_N, BLOCK_D], dtype=tl.float32),
            mask=mask_h_n_d,
        )
        tl.store(
            dv_ptrs,
            tl.zeros([N_HEADS, BLOCK_N, BLOCK_D], dtype=tl.float32),
            mask=mask_h_n_d,
        )

    # Loop over M blocks (Q)
    start_m_begin = 0
    if IS_CAUSAL:
        start_m_begin = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M

    for start_m in range(start_m_begin, N_CTX, BLOCK_M):
        offs_m_curr = start_m + offs_m
        mask_m = offs_m_curr < N_CTX

        # Causal mask
        if IS_CAUSAL:
            mask_causal = offs_m_curr[:, None] >= offs_n[None, :]
        else:
            mask_causal = True

        # Load L, M for this M block [H, M]
        mask_lm = mask_h[:, None] & mask_m[None, :]
        l_ptrs = (
            L
            + off_z * stride_lz
            + offs_h[:, None] * stride_lh
            + offs_m_curr[None, :] * stride_lm
        )
        m_ptrs = (
            M
            + off_z * stride_mz
            + offs_h[:, None] * stride_mh
            + offs_m_curr[None, :] * stride_mm
        )
        l = tl.load(l_ptrs, mask=mask_lm, other=0.0)
        m_val = tl.load(m_ptrs, mask=mask_lm, other=0.0)

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
                + offs_m_curr[None, :, None] * stride_qm
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
                + offs_n[None, :, None] * stride_kn
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

        # Mask P explicitly for padding/causality
        mask_mn = mask_m[:, None] & mask_n[None, :]
        if IS_CAUSAL:
            mask_p = mask_causal & mask_mn
        else:
            mask_p = mask_mn
        p = tl.where(mask_p[None, :, :], p, 0.0)

        # Process dO, O, and V in chunks to compute dV, dP, and D
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
                + offs_m_curr[None, :, None] * stride_dom
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
                + offs_m_curr[None, :, None] * stride_om
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
                + offs_n[None, :, None] * stride_vn
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

            # Compute dV contribution: dO^T @ P and accumulate to global memory
            do_chunk_t = tl.trans(do_chunk, (0, 2, 1))
            partial_dv_t = tl.dot(do_chunk_t, p.to(do_chunk_t.dtype))
            # Transpose to match expected output layout [N_HEADS, BLOCK_N, BLOCK_D]
            partial_dv = tl.trans(partial_dv_t, (0, 2, 1))

            # Load existing dV chunk and accumulate
            dv_ptrs = (
                DV
                + off_z * stride_dvz
                + offs_h[:, None, None] * stride_dvh
                + offs_n[None, :, None] * stride_dvn
                + offs_d_curr[None, None, :] * stride_dvk
            )

            existing_dv = tl.load(
                dv_ptrs,
                mask=(
                    mask_h[:, None, None]
                    & mask_n[None, :, None]
                    & mask_d[None, None, :]
                ),
                other=0.0,
            )

            updated_dv = existing_dv + partial_dv

            # Store back to global memory
            tl.store(
                dv_ptrs,
                updated_dv,
                mask=(
                    mask_h[:, None, None]
                    & mask_n[None, :, None]
                    & mask_d[None, None, :]
                ),
            )

        # Compute dS = P * (dP - D)
        ds = p * (dp_full.to(tl.float32) - D[:, :, None])

        # Compute dS_raw = W^T @ dS
        ds_reshaped = tl.reshape(ds, [N_HEADS, BLOCK_M * BLOCK_N])
        ds_raw_reshaped = tl.dot(wt, ds_reshaped.to(wt.dtype))
        ds_raw = tl.reshape(ds_raw_reshaped, [N_HEADS, BLOCK_M, BLOCK_N])

        # Process dK in chunks: dK += Q^T @ dS_raw * scale
        for start_d_out in range(0, BLOCK_DMODEL, BLOCK_D):
            start_d_out = tl.multiple_of(start_d_out, BLOCK_D)
            offs_d_out_curr = start_d_out + offs_d
            mask_d_out = offs_d_out_curr < BLOCK_DMODEL

            # Load Q chunk: [n_heads, block_m, block_d]
            q_ptrs = (
                Q
                + off_z * stride_qz
                + offs_h[:, None, None] * stride_qh
                + offs_m_curr[None, :, None] * stride_qm
                + offs_d_out_curr[None, None, :] * stride_qk
            )
            q_chunk = tl.load(
                q_ptrs,
                mask=(
                    mask_h[:, None, None]
                    & mask_m[None, :, None]
                    & mask_d_out[None, None, :]
                ),
                other=0.0,
            )

            # Compute partial dK: Q^T @ dS_raw * scale
            q_chunk_t = tl.trans(q_chunk, (0, 2, 1))
            partial_dk_t = tl.dot(q_chunk_t, ds_raw.to(q_chunk_t.dtype)) * SCALE
            # Transpose to match expected output layout [N_HEADS, BLOCK_N, BLOCK_D]
            partial_dk = tl.trans(partial_dk_t, (0, 2, 1))

            # Load existing dK chunk and accumulate
            dk_ptrs = (
                DK
                + off_z * stride_dkz
                + offs_h[:, None, None] * stride_dkh
                + offs_n[None, :, None] * stride_dkn
                + offs_d_out_curr[None, None, :] * stride_dkk
            )

            existing_dk = tl.load(
                dk_ptrs,
                mask=(
                    mask_h[:, None, None]
                    & mask_n[None, :, None]
                    & mask_d_out[None, None, :]
                ),
                other=0.0,
            )

            updated_dk = existing_dk + partial_dk

            # Store back to global memory
            tl.store(
                dk_ptrs,
                updated_dk,
                mask=(
                    mask_h[:, None, None]
                    & mask_n[None, :, None]
                    & mask_d_out[None, None, :]
                ),
            )


def head_conv_attention_bwd_dkdv(q, k, v, w, o, do, l, m, causal=False, scale=None):
    batch, heads, seq_len, d_model = q.shape
    assert d_model <= 128, "d_model must be <= 128"

    if scale is None:
        scale = 1.0 / math.sqrt(d_model)

    # Allocate outputs (FP32 accumulation buffers)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)

    # Block sizes (can be tuned)
    BLOCK_M = 16
    BLOCK_N = 16

    grid = (triton.cdiv(seq_len, BLOCK_N), batch)

    head_conv_attn_bwd_dkdv_kernel[grid](
        q,
        k,
        v,
        w,
        o,
        do,
        l,
        m,
        dk,
        dv,
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
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dk.stride(3),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        dv.stride(3),
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
    return dk.to(k.dtype), dv.to(v.dtype)

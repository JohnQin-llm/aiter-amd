# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon (gfx950) main kernel of the token-balanced MLA decode (bf16, 576-wide latent cache, page 1).

Same schedule and output contract as the Triton kernel in
aiter/ops/triton/_triton_kernels/attention/mla_decode_balanced.py: program p of the fixed grid takes
the units (UNIT tokens of one sequence) [p * T // P, (p + 1) * T // P) of the sequence-major unit
list, computed on the device from kv_indptr. Whole sequences are written as bf16; pieces of split
sequences write fp32 (m, l, acc) partials for the real heads and a merge task, combined by
_mla_decode_balanced_merge_kernel.

The tile loop streams the latent rows straight into LDS (async global->LDS copies, NUM_BUFFERS
buffers): tile i + 1 is in flight while tile i runs QK (512 NoPE + 64 RoPE, two MFMA chains) and PV
(512) on the MFMA units. The slot ids of tile i + 2 are read during tile i, so the copy of a tile is
issued without waiting on an index load.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _pick(vec, offs, idx):
    # vec[idx] for a register vector (idx is a runtime scalar)
    return gl.sum(gl.where(offs == idx, vec, 0), axis=0)


@gluon.jit
def _slots(idx_base, tok0, tok_hi, rng):
    # Clamped to the last token of the range: the tail of a tile re-reads a row the range
    # attends anyway (an L2 hit) and its score is masked, so the copy needs no mask and the
    # LDS tile never holds stale or uninitialized rows (0 * NaN would reach the PV MFMA).
    t = gl.minimum(tok0 + rng, tok_hi - 1)
    return gl.load(idx_base + t)


@gluon.jit
def _issue(kv_ptr, stride_kv, slot, slot_r, kv_buf, r_buf, col, col_r, DV: gl.constexpr):
    """One commit group = one tile (NoPE/V rows and RoPE rows)."""
    row = kv_ptr + slot.to(gl.int64) * stride_kv
    gl.amd.cdna4.async_copy.global_load_to_shared(
        kv_buf, row[:, None] + col[None, :], cache_modifier=".cg"
    )
    row_r = kv_ptr + slot_r.to(gl.int64) * stride_kv + DV
    gl.amd.cdna4.async_copy.global_load_to_shared(
        r_buf, row_r[:, None] + col_r[None, :], cache_modifier=".cg"
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _mla_decode_balanced_gluon_kernel(
    out_ptr,  # [num_seqs, 16, DV]
    q_ptr,  # [num_seqs, 16, DV + DR]
    kv_ptr,  # [num_slots, DV + DR]
    kv_indptr_ptr,  # [num_seqs + 1] int32
    kv_indices_ptr,  # [*] int32 token slots
    part_m_ptr,  # [16, num_progs, 2] fp32
    part_l_ptr,  # [16, num_progs, 2] fp32
    part_o_ptr,  # [16, num_progs, 2, DV] fp32
    task_ptr,  # [num_seqs, 4] int32
    num_seqs,
    num_progs,
    scale,
    stride_q_t,
    stride_q_h,
    stride_o_t,
    stride_o_h,
    stride_kv,
    DV: gl.constexpr,
    DR: gl.constexpr,
    UNIT: gl.constexpr,  # schedule granularity (tokens)
    TILE: gl.constexpr,  # tokens per LDS tile
    BLOCK_B: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    KV_PAD: gl.constexpr,
):
    BLOCK_M: gl.constexpr = 16  # 16 query heads, one MFMA row block
    sched_l: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])
    mma_l: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 16], transposed=True, warps_per_cta=[1, NUM_WARPS]
    )
    q_dl: gl.constexpr = gl.DotOperandLayout(0, mma_l, 8)
    k_dl: gl.constexpr = gl.DotOperandLayout(1, mma_l, 8)
    q_bl: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, NUM_WARPS], [1, 0])
    # async copies: 16 B per lane, a 1 KB NoPE row per warp instruction / 8 RoPE rows
    cp_l: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [NUM_WARPS, 1], [1, 0])
    cpr_l: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NUM_WARPS, 1], [1, 0])
    kv_sh: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[DV, KV_PAD]], [TILE, DV], [1, 0]
    )
    r_sh: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[8 * DR, KV_PAD]], [TILE, DR], [1, 0]
    )

    pid = gl.program_id(0)
    # ---- on-device schedule ----
    offs_b = gl.arange(0, BLOCK_B, layout=sched_l)
    bmask = offs_b < num_seqs
    b_lo = gl.load(kv_indptr_ptr + offs_b, mask=bmask, other=0)
    b_hi = gl.load(kv_indptr_ptr + offs_b + 1, mask=bmask, other=0)
    lens = gl.maximum(b_hi - b_lo, 0)
    units = (lens + UNIT - 1) // UNIT
    cum = gl.associative_scan(units, 0, _add)
    total = gl.sum(units, axis=0).to(gl.int64)
    nprog = gl.minimum(total, num_progs)
    if pid < nprog:
        u0 = (pid * total) // nprog
        u1 = ((pid + 1) * total) // nprog

        offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, q_bl))
        offs_dq = gl.arange(0, DV, layout=gl.SliceLayout(0, q_bl))
        offs_rq = gl.arange(0, DR, layout=gl.SliceLayout(0, q_bl))
        col = gl.arange(0, DV, layout=gl.SliceLayout(0, cp_l))
        col_r = gl.arange(0, DR, layout=gl.SliceLayout(0, cpr_l))
        rng = gl.arange(0, TILE, layout=gl.SliceLayout(1, cp_l))
        rng_r = gl.arange(0, TILE, layout=gl.SliceLayout(1, cpr_l))
        rng_s = gl.arange(0, TILE, layout=gl.SliceLayout(0, mma_l))
        om = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, mma_l))
        od = gl.arange(0, DV, layout=gl.SliceLayout(0, mma_l))
        qk_scale = scale * 1.4426950408889634

        kv_smem = gl.allocate_shared_memory(gl.bfloat16, [NUM_BUFFERS, TILE, DV], kv_sh)
        r_smem = gl.allocate_shared_memory(gl.bfloat16, [NUM_BUFFERS, TILE, DR], r_sh)

        u = u0
        while u < u1:
            seq = gl.sum((cum <= u).to(gl.int32), axis=0)
            c_end = _pick(cum, offs_b, seq).to(gl.int64)
            n_units = _pick(units, offs_b, seq)
            seq_len = _pick(lens, offs_b, seq)
            kv_base = _pick(b_lo, offs_b, seq).to(gl.int64)
            c_start = c_end - n_units
            t_lo = (u - c_start).to(gl.int32)
            t_hi = (gl.minimum(u1, c_end) - c_start).to(gl.int32)
            tok_lo = t_lo * UNIT
            tok_hi = gl.minimum(t_hi * UNIT, seq_len)
            n_iter = (tok_hi - tok_lo + TILE - 1) // TILE
            idx_base = kv_indices_ptr + kv_base

            m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, layout=gl.SliceLayout(1, mma_l))
            l_i = gl.zeros([BLOCK_M], gl.float32, layout=gl.SliceLayout(1, mma_l))
            acc = gl.zeros([BLOCK_M, DV], gl.float32, layout=mma_l)

            # prologue: tile 0 in flight, slots of tile 1 requested
            s0 = _slots(idx_base, tok_lo, tok_hi, rng)
            s0r = _slots(idx_base, tok_lo, tok_hi, rng_r)
            _issue(kv_ptr, stride_kv, s0, s0r, kv_smem.index(0), r_smem.index(0), col, col_r, DV)
            # Q after the first copy, so its latency overlaps the copy's
            q_base = q_ptr + seq.to(gl.int64) * stride_q_t
            q_row = offs_m * stride_q_h
            q_n = gl.load(q_base + q_row[:, None] + offs_dq[None, :])
            q_r = gl.load(q_base + q_row[:, None] + DV + offs_rq[None, :])
            s1 = _slots(idx_base, tok_lo + TILE, tok_hi, rng)
            s1r = _slots(idx_base, tok_lo + TILE, tok_hi, rng_r)
            q_n = gl.convert_layout(q_n, q_dl)
            q_r = gl.convert_layout(q_r, q_dl)
            for i in range(n_iter):
                buf = i % NUM_BUFFERS
                # tile i (and the slot ids of tile i + 1) have landed; every warp is done with
                # the buffer the next copy overwrites
                gl.amd.cdna4.async_copy.wait_group(0)
                gl.barrier()
                if i + 1 < n_iter:
                    nb = (i + 1) % NUM_BUFFERS
                    _issue(kv_ptr, stride_kv, s1, s1r, kv_smem.index(nb), r_smem.index(nb),
                           col, col_r, DV)
                    t2 = tok_lo + (i + 2) * TILE
                    s1 = _slots(idx_base, t2, tok_hi, rng)
                    s1r = _slots(idx_base, t2, tok_hi, rng_r)
                kv_t = kv_smem.index(buf)
                k = gl.amd.cdna4.async_copy.load_shared_relaxed(kv_t.permute([1, 0]), k_dl)
                k_r = gl.amd.cdna4.async_copy.load_shared_relaxed(r_smem.index(buf).permute([1, 0]), k_dl)
                s = gl.zeros([BLOCK_M, TILE], gl.float32, layout=mma_l)
                s = gl.amd.cdna4.mfma(q_n, k, s)
                s = gl.amd.cdna4.mfma(q_r, k_r, s)
                valid = (tok_lo + i * TILE + rng_s) < tok_hi
                s = gl.where(valid[None, :], s, float("-inf"))
                m_new = gl.maximum(m_i, gl.max(s, axis=1) * qk_scale)
                m_new = gl.where(m_new > float("-inf"), m_new, 0.0)
                p = gl.exp2(s * qk_scale - m_new[:, None])
                alpha = gl.exp2(m_i - m_new)
                l_i = l_i * alpha + gl.sum(p, axis=1)
                v = gl.amd.cdna4.async_copy.load_shared_relaxed(kv_t, k_dl)
                p_dot = gl.convert_layout(p.to(gl.bfloat16), q_dl)
                acc = acc * alpha[:, None]
                acc = gl.amd.cdna4.mfma(p_dot, v, acc)
                m_i = m_new

            if (t_lo == 0) & (t_hi == n_units):
                o = acc / gl.where(l_i == 0.0, 1.0, l_i)[:, None]
                o_base = out_ptr + seq.to(gl.int64) * stride_o_t
                gl.store(o_base + (om * stride_o_h)[:, None] + od[None, :], o.to(gl.bfloat16))
                gl.store(task_ptr + seq * 4, -1)
            else:
                slot_p = (u0 < c_start).to(gl.int64)
                pidx = om.to(gl.int64) * (num_progs * 2) + pid * 2 + slot_p
                gl.store(part_m_ptr + pidx, m_i)
                gl.store(part_l_ptr + pidx, l_i)
                gl.store(part_o_ptr + pidx[:, None] * DV + od[None, :], acc)
                if t_lo == 0:
                    gl.store(task_ptr + seq * 4, pid)
                    gl.store(task_ptr + seq * 4 + 1, ((c_end * nprog - 1) // total).to(gl.int32))
                    gl.store(task_ptr + seq * 4 + 2, slot_p.to(gl.int32))
            u = c_start + t_hi

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Token-balanced persistent MLA decode for a 576-wide latent cache (512 NoPE + 64 RoPE), page size 1.

The latent row is both K (all 576 columns) and V (the first 512 columns), so every KV row is read
from memory once and used for QK and PV. The KV of the batch is cut into tiles of TILE tokens per
sequence; program p of the fixed persistent grid takes tiles [p * T // P, (p + 1) * T // P) of the
sequence-major tile list, where T is the total tile count and P the number of programs in use.
Both are computed inside every program from kv_indptr on the device, so the split needs no
metadata kernel and no host sync, and a captured HIP graph stays correct and balanced for any
lengths.

All query rows of a sequence (QLEN tokens x NUM_HEADS heads) form one MFMA tile. A sequence covered
by one program is written to the output in its final dtype. Otherwise every program that covers
part of it writes an fp32 partial (max, sum, acc) for the real heads only, and
_mla_decode_balanced_merge_kernel combines them by log-sum-exp.
"""

import triton
import triton.language as tl


@triton.jit
def _pick(vec, offs, idx):
    # vec[idx] for a register vector (idx is a runtime scalar)
    return tl.sum(tl.where(offs == idx, vec, 0), axis=0)


@triton.jit
def _schedule(kv_indptr_ptr, num_seqs, num_progs, TILE: tl.constexpr, BLOCK_B: tl.constexpr):
    """Per-sequence KV start and length, tile counts, inclusive tile prefix sums, total tiles and
    programs in use."""
    offs_b = tl.arange(0, BLOCK_B)
    bmask = offs_b < num_seqs
    lo = tl.load(kv_indptr_ptr + offs_b, mask=bmask, other=0)
    hi = tl.load(kv_indptr_ptr + offs_b + 1, mask=bmask, other=0)
    lens = tl.maximum(hi - lo, 0)
    tiles = (lens + TILE - 1) // TILE
    cum = tl.cumsum(tiles, axis=0)
    total = tl.sum(tiles, axis=0).to(tl.int64)
    nprog = tl.minimum(num_progs, total)
    return offs_b, lo, lens, tiles, cum, total, nprog


@triton.jit
def _mla_decode_balanced_kernel(
    out_ptr,  # [num_seqs * QLEN, NUM_HEADS, DV]
    q_ptr,  # [num_seqs * QLEN, NUM_HEADS, DV + DR]
    kv_ptr,  # [num_slots, DV + DR] (page size 1)
    kv_indptr_ptr,  # [num_seqs + 1] int32
    kv_indices_ptr,  # [*] int32 token slots
    part_m_ptr,  # [BLOCK_M, num_progs, 2] fp32
    part_l_ptr,  # [BLOCK_M, num_progs, 2] fp32
    part_o_ptr,  # [BLOCK_M, num_progs, 2, DV] fp32
    task_ptr,  # [num_seqs, 4] int32 merge task per sequence (see the merge kernel)
    num_seqs,
    num_progs,
    scale,
    stride_q_t,
    stride_q_h,
    stride_o_t,
    stride_o_h,
    stride_kv,
    NUM_HEADS: tl.constexpr,
    QLEN: tl.constexpr,
    DV: tl.constexpr,  # 512 (NoPE, also V)
    DR: tl.constexpr,  # 64 (RoPE)
    TILE: tl.constexpr,
    BLOCK_M: tl.constexpr,  # >= QLEN * NUM_HEADS, >= 16
    BLOCK_B: tl.constexpr,  # >= num_seqs, power of 2
):
    pid = tl.program_id(0)
    offs_b, kv_lo, lens, tiles, cum, total, nprog = _schedule(
        kv_indptr_ptr, num_seqs, num_progs, TILE, BLOCK_B
    )
    if pid >= nprog:
        return
    u0 = (pid * total) // nprog
    u1 = ((pid + 1) * total) // nprog

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, DV)
    offs_r = tl.arange(0, DR)
    offs_t = tl.arange(0, TILE)
    row_mask = offs_m < QLEN * NUM_HEADS
    q_tok = offs_m // NUM_HEADS  # query token of the row within its sequence
    q_head = offs_m % NUM_HEADS
    q_row_off = q_tok.to(tl.int64) * stride_q_t + q_head * stride_q_h
    o_row_off = q_tok.to(tl.int64) * stride_o_t + q_head * stride_o_h
    qk_scale = scale * 1.4426950408889634  # exp2 domain

    u = u0
    while u < u1:
        seq = tl.sum((cum <= u).to(tl.int32), axis=0)
        c_end = _pick(cum, offs_b, seq).to(tl.int64)
        n_tiles = _pick(tiles, offs_b, seq)
        seq_len = _pick(lens, offs_b, seq)
        kv_base = _pick(kv_lo, offs_b, seq).to(tl.int64)
        c_start = c_end - n_tiles
        t_lo = (u - c_start).to(tl.int32)
        t_hi = (tl.minimum(u1, c_end) - c_start).to(tl.int32)

        q_base = q_ptr + seq.to(tl.int64) * QLEN * stride_q_t
        q_n = tl.load(q_base + q_row_off[:, None] + offs_d[None, :], mask=row_mask[:, None], other=0.0)
        q_r = tl.load(q_base + q_row_off[:, None] + DV + offs_r[None, :], mask=row_mask[:, None], other=0.0)
        # bottom-right causal limit per row: query token i sees kv_len - QLEN + i + 1 tokens
        row_lim = seq_len - QLEN + 1 + q_tok

        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, DV], dtype=tl.float32)
        # software pipeline: the rows of tile t + 1 and the slots of tile t + 2 are in flight while
        # tile t is computed (each tile is a dependent slot -> row gather; loads beyond this
        # program's range are masked off, so there is no extra memory traffic)
        tok_end = t_hi * TILE
        tok = t_lo * TILE + offs_t
        slot = tl.load(kv_indices_ptr + kv_base + tok, mask=tok < seq_len, other=0).to(tl.int64)
        row = kv_ptr + slot * stride_kv
        nmask = (tok < seq_len)[:, None]
        # every latent row is streamed once: non-temporal loads
        kv_n_nx = tl.load(row[:, None] + offs_d[None, :], mask=nmask, other=0.0, cache_modifier=".cg")
        kv_r_nx = tl.load(row[:, None] + DV + offs_r[None, :], mask=nmask, other=0.0, cache_modifier=".cg")
        tok2 = tok + TILE
        slot_nx = tl.load(kv_indices_ptr + kv_base + tok2, mask=(tok2 < seq_len) & (tok2 < tok_end),
                          other=0).to(tl.int64)
        for t in range(t_lo, t_hi):
            tok = t * TILE + offs_t
            t_mask = tok < seq_len
            kv_n = kv_n_nx  # [TILE, DV]: K NoPE and V
            kv_r = kv_r_nx  # [TILE, DR]: K RoPE
            # issue tile t + 1 (rows) and t + 2 (slots)
            tok1 = tok + TILE
            m1 = ((tok1 < seq_len) & (tok1 < tok_end))[:, None]
            row1 = kv_ptr + slot_nx * stride_kv
            kv_n_nx = tl.load(row1[:, None] + offs_d[None, :], mask=m1, other=0.0, cache_modifier=".cg")
            kv_r_nx = tl.load(row1[:, None] + DV + offs_r[None, :], mask=m1, other=0.0, cache_modifier=".cg")
            tok2 = tok + 2 * TILE
            slot_nx = tl.load(kv_indices_ptr + kv_base + tok2, mask=(tok2 < seq_len) & (tok2 < tok_end),
                              other=0).to(tl.int64)

            s = tl.dot(q_n, tl.trans(kv_n))
            s = tl.dot(q_r, tl.trans(kv_r), acc=s)
            s = s * qk_scale
            if QLEN > 1:
                s = tl.where(tok[None, :] < row_lim[:, None], s, float("-inf"))
            else:
                s = tl.where(t_mask[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            p = tl.math.exp2(s - m_safe[:, None])
            alpha = tl.math.exp2(m_i - m_safe)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(kv_n.dtype), kv_n, acc=acc)
            m_i = m_new

        if (t_lo == 0) & (t_hi == n_tiles):
            # the whole sequence is in this program: final output
            o = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
            o_base = out_ptr + seq.to(tl.int64) * QLEN * stride_o_t
            tl.store(o_base + o_row_off[:, None] + offs_d[None, :], o.to(out_ptr.dtype.element_ty),
                     mask=row_mask[:, None])
            tl.store(task_ptr + seq * 4, -1)  # nothing to merge
        else:
            # slot 1 when the sequence is not the first one of this program's range
            slot_p = (u0 < c_start).to(tl.int64)
            # row-major partials [row, program * 2 + slot]: the pieces of one sequence are
            # neighbours, so the merge of a row reads one contiguous range
            pidx = offs_m.to(tl.int64) * (num_progs * 2) + pid * 2 + slot_p
            tl.store(part_m_ptr + pidx, m_i)
            tl.store(part_l_ptr + pidx, l_i)
            tl.store(part_o_ptr + pidx[:, None] * DV + offs_d[None, :], acc, mask=row_mask[:, None])
            if t_lo == 0:
                # first piece of a split sequence: record its merge task
                # (programs covering it: pid .. the owner of its last tile)
                tl.store(task_ptr + seq * 4, pid)
                tl.store(task_ptr + seq * 4 + 1, ((c_end * nprog - 1) // total).to(tl.int32))
                tl.store(task_ptr + seq * 4 + 2, slot_p.to(tl.int32))
        u = c_start + t_hi


@triton.jit
def _mla_decode_balanced_merge_kernel(
    out_ptr,
    task_ptr,  # [num_seqs, 4] int32, written by the main kernel
    kv_indptr_ptr,
    part_m_ptr,
    part_l_ptr,
    part_o_ptr,
    num_progs,
    stride_o_t,
    stride_o_h,
    num_seqs,
    total_max,  # > 0: hybrid dispatch threshold on the step's KV tokens
    red_indptr_ptr,  # hybrid, asm regime: the inputs of mla_reduce_v1 (reduce_indptr [n_tiles + 1],
    red_final_ptr,  # reduce_final_map [n_tiles, 2], reduce_partial_map, partial lse [P, H]
    red_pmap_ptr,  # (natural log) and partial output [P, H, DV], both fp32)
    red_lse_ptr,
    red_out_ptr,
    n_tiles,
    NUM_HEADS: tl.constexpr,
    QLEN: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_P: tl.constexpr,  # partials merged per step
    BLOCK_D: tl.constexpr,  # output columns per program
    HYBRID: tl.constexpr,
):
    """Merge the partials of split sequences by log-sum-exp.

    Fixed grid (sequence, query row, block of BLOCK_D columns). For every sequence with KV the main
    kernel wrote task[seq]: -1 when one program covered it (output already final), else the first
    and last program covering it and the partial slot of the first piece. Sequences without KV
    (e.g. graph padding) are skipped."""
    seq = tl.program_id(0)
    if HYBRID:
        # Above total_max KV tokens the balanced main kernel did nothing and the asm main kernel did
        # the work: this launch then does the job of mla_reduce_v1 on the asm partials (one reduce
        # launch per layer in both regimes). Tile t merges the partials
        # reduce_partial_map[reduce_indptr[t]:reduce_indptr[t + 1]] into output row
        # reduce_final_map[t][0] with weights exp(lse_k - lse), as mla_reduce_v1 does.
        # the reduce-tile scalars are read together with the regime test (one round trip)
        t = seq
        tv = t < n_tiles
        r_last = tl.load(red_indptr_ptr + n_tiles)
        r_s = tl.load(red_indptr_ptr + t, mask=tv, other=0)
        r_e = tl.load(red_indptr_ptr + t + 1, mask=tv, other=0)
        q0 = tl.load(red_final_ptr + 2 * t, mask=tv, other=0).to(tl.int64)
        tok_total = tl.load(kv_indptr_ptr + num_seqs) - tl.load(kv_indptr_ptr)
        small = tok_total <= total_max
        if not small:
            if tv:
                if (r_s != r_last) & (r_e > r_s):
                    h = tl.program_id(1)
                    offs_rp = tl.arange(0, BLOCK_P)
                    offs_rd = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
                    rm = tl.max(tl.full([BLOCK_P], float("-inf"), dtype=tl.float32), axis=0)
                    rl = tl.sum(tl.zeros([BLOCK_P], dtype=tl.float32), axis=0)
                    ro = tl.zeros([BLOCK_D], dtype=tl.float32)
                    for p0 in range(r_s, r_e, BLOCK_P):
                        pp = p0 + offs_rp
                        pmk = pp < r_e
                        prow = tl.load(red_pmap_ptr + pp, mask=pmk, other=0).to(tl.int64)
                        lse = tl.load(red_lse_ptr + prow * NUM_HEADS + h, mask=pmk, other=float("-inf"))
                        ov = tl.load(red_out_ptr + (prow * NUM_HEADS + h)[:, None] * DV + offs_rd[None, :],
                                     mask=pmk[:, None], other=0.0)
                        mn = tl.maximum(rm, tl.max(lse, axis=0))
                        ms = tl.where(mn == float("-inf"), 0.0, mn)
                        a = tl.exp(rm - ms)
                        w = tl.exp(lse - ms)
                        rl = rl * a + tl.sum(w, axis=0)
                        ro = ro * a + tl.sum(ov * w[:, None], axis=0)
                        rm = mn
                    res = ro / tl.where(rl == 0.0, 1.0, rl)
                    tl.store(out_ptr + q0 * stride_o_t + h * stride_o_h + offs_rd,
                             res.to(out_ptr.dtype.element_ty))
            return
        if seq >= num_seqs:
            return
    kv_lo = tl.load(kv_indptr_ptr + seq)
    kv_hi = tl.load(kv_indptr_ptr + seq + 1)
    p_first = tl.load(task_ptr + seq * 4)
    if (kv_hi <= kv_lo) | (p_first < 0):
        return
    p_last = tl.load(task_ptr + seq * 4 + 1)
    first_slot = tl.load(task_ptr + seq * 4 + 2)
    row = tl.program_id(1)

    # BLOCK_P partials per step, so even the longest sequence of a skewed batch (split over many
    # programs) is merged in a few wide steps
    offs_p = tl.arange(0, BLOCK_P)
    offs_d = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    m_t = tl.max(tl.full([BLOCK_P], float("-inf"), dtype=tl.float32), axis=0)
    l_t = tl.sum(tl.zeros([BLOCK_P], dtype=tl.float32), axis=0)
    o_t = tl.zeros([BLOCK_D], dtype=tl.float32)
    for p0 in range(p_first, p_last + 1, BLOCK_P):
        pp = p0 + offs_p
        pmask = pp <= p_last
        sl = tl.where(pp == p_first, first_slot, 0)
        idx = row.to(tl.int64) * (num_progs * 2) + pp * 2 + sl
        mp = tl.load(part_m_ptr + idx, mask=pmask, other=float("-inf"))
        lp = tl.load(part_l_ptr + idx, mask=pmask, other=0.0)
        op = tl.load(part_o_ptr + idx[:, None] * DV + offs_d[None, :], mask=pmask[:, None], other=0.0)
        m_n = tl.maximum(m_t, tl.max(mp, axis=0))
        m_s = tl.where(m_n == float("-inf"), 0.0, m_n)
        a = tl.math.exp2(m_t - m_s)
        w = tl.math.exp2(mp - m_s)
        l_t = l_t * a + tl.sum(lp * w, axis=0)
        o_t = o_t * a + tl.sum(op * w[:, None], axis=0)
        m_t = m_n
    o = o_t / tl.where(l_t == 0.0, 1.0, l_t)
    q_tok = row // NUM_HEADS
    q_head = row % NUM_HEADS
    dst = out_ptr + (seq.to(tl.int64) * QLEN + q_tok) * stride_o_t + q_head * stride_o_h + offs_d
    tl.store(dst, o.to(out_ptr.dtype.element_ty))

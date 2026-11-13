# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Token-balanced persistent GQA paged decode (one query token per sequence).

The KV work of a batch is cut into tiles of TILE tokens per sequence. Every column of the grid
(one column per group of HEADS kv heads) walks the same global tile list (sequence-major), and
program ``p`` of a column takes tiles ``[p * T // P, (p + 1) * T // P)``, where ``T`` is the total
tile count and ``P`` the number of programs in use. ``T`` and the per-sequence tile prefix sums are
computed inside every program from ``seq_lens`` (device), so the split is decided on the GPU on
every launch: a captured HIP graph stays correct and balanced for any lengths, and no metadata
kernel or host sync is needed.

A program walks the sequences its range intersects. A sequence covered by one program is written
to the output directly; otherwise every program covering it writes a partial (max, sum, acc) and
``_balanced_gqa_decode_merge_kernel`` combines them by log-sum-exp.
"""

import triton
import triton.language as tl


@triton.jit
def _pick(vec, offs, idx):
    # vec[idx] for a register vector (idx is a runtime scalar)
    return tl.sum(tl.where(offs == idx, vec, 0), axis=0)


@triton.jit
def _schedule(seq_lens_ptr, num_seqs, num_progs, TILE: tl.constexpr, BLOCK_B: tl.constexpr,
              MIN_TILES: tl.constexpr):
    """Per-sequence tile counts, inclusive prefix sums, total tiles and programs in use."""
    offs_b = tl.arange(0, BLOCK_B)
    lens = tl.load(seq_lens_ptr + offs_b, mask=offs_b < num_seqs, other=0)
    tiles = (lens + TILE - 1) // TILE
    cum = tl.cumsum(tiles, axis=0)
    total = tl.sum(tiles, axis=0).to(tl.int64)
    nprog = tl.minimum(num_progs, (total + MIN_TILES - 1) // MIN_TILES)
    return offs_b, lens, tiles, cum, total, nprog


@triton.jit
def _balanced_gqa_decode_kernel(
    out_ptr,  # [num_seqs, num_q_heads, HEAD_D]
    q_ptr,  # [num_seqs, num_q_heads, HEAD_D]
    k_cache_ptr,  # [num_pages, PAGE_SIZE, num_kv_heads, HEAD_D]
    v_cache_ptr,  # [num_pages, PAGE_SIZE, num_kv_heads, HEAD_D]
    seq_lens_ptr,  # [num_seqs] int32
    table_ptr,  # block table [num_seqs, max_pages] or flat page indices (RAGGED_TABLE)
    table_indptr_ptr,  # [num_seqs + 1] (RAGGED_TABLE only)
    part_m_ptr,  # [max_progs, 2, NUM_GROUPS, BLOCK_M] fp32
    part_l_ptr,  # [max_progs, 2, NUM_GROUPS, BLOCK_M] fp32
    part_o_ptr,  # [max_progs, 2, NUM_GROUPS, BLOCK_M, HEAD_D] fp32
    num_seqs,
    num_progs,  # programs per kv head group (grid axis 0)
    scale,
    stride_q_b,
    stride_q_h,
    stride_o_b,
    stride_o_h,
    stride_k_p,
    stride_k_s,
    stride_k_h,
    stride_v_p,
    stride_v_s,
    stride_v_h,
    stride_t_b,
    NUM_GROUPS: tl.constexpr,  # num_kv_heads // HEADS
    GQA: tl.constexpr,  # query heads per kv head
    HEADS: tl.constexpr,  # kv heads per program (adjacent in the cache row: stride_k_h == HEAD_D)
    HEAD_D: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_M: tl.constexpr,  # >= HEADS * GQA, >= 16 (MFMA rows)
    BLOCK_B: tl.constexpr,  # >= num_seqs, power of 2
    MIN_TILES: tl.constexpr,  # minimum tiles per program
    RAGGED_TABLE: tl.constexpr,
):
    pid = tl.program_id(0)
    grp = tl.program_id(1)

    # ---- on-device schedule: equal tile share per program ----
    offs_b, lens, tiles, cum, total, nprog = _schedule(
        seq_lens_ptr, num_seqs, num_progs, TILE, BLOCK_B, MIN_TILES
    )
    if pid >= nprog:
        return
    u0 = (pid * total) // nprog
    u1 = ((pid + 1) * total) // nprog

    # rows: the HEADS * GQA query heads of the group; columns: HEADS * HEAD_D (HEADS kv heads
    # side by side). Q is block-diagonal (row r only has the columns of its own kv head), so one
    # QK dot over the HEADS heads' keys gives each row the scores of its own head; the PV result
    # is read back on the same block diagonal.
    offs_m = tl.arange(0, BLOCK_M)
    offs_c = tl.arange(0, HEADS * HEAD_D)
    offs_t = tl.arange(0, TILE)
    row_mask = offs_m < HEADS * GQA
    head = grp * (HEADS * GQA) + offs_m
    col_d = offs_c % HEAD_D
    blk_mask = row_mask[:, None] & ((offs_m // GQA)[:, None] == (offs_c // HEAD_D)[None, :])
    kv_head_off_k = grp * HEADS * stride_k_h
    kv_head_off_v = grp * HEADS * stride_v_h
    qk_scale = scale * 1.4426950408889634  # exp2 domain

    u = u0
    while u < u1:
        seq = tl.sum((cum <= u).to(tl.int32), axis=0)
        c_end = _pick(cum, offs_b, seq).to(tl.int64)
        n_tiles = _pick(tiles, offs_b, seq)
        seq_len = _pick(lens, offs_b, seq)
        c_start = c_end - n_tiles
        t_lo = (u - c_start).to(tl.int32)
        t_hi = (tl.minimum(u1, c_end) - c_start).to(tl.int32)

        if RAGGED_TABLE:
            t_base = tl.load(table_indptr_ptr + seq).to(tl.int64)
        else:
            t_base = seq.to(tl.int64) * stride_t_b

        q = tl.load(
            q_ptr + seq.to(tl.int64) * stride_q_b + head[:, None] * stride_q_h + col_d[None, :],
            mask=blk_mask,
            other=0.0,
        )
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEADS * HEAD_D], dtype=tl.float32)

        for t in range(t_lo, t_hi):
            tok = t * TILE + offs_t
            t_mask = tok < seq_len
            page = tl.load(table_ptr + t_base + tok // PAGE_SIZE, mask=t_mask, other=0).to(tl.int64)
            slot = tok % PAGE_SIZE
            k_off = page * stride_k_p + slot * stride_k_s + kv_head_off_k
            v_off = page * stride_v_p + slot * stride_v_s + kv_head_off_v
            # KV is streamed once: ".cg" (non-temporal) loads
            k = tl.load(k_cache_ptr + k_off[None, :] + offs_c[:, None], mask=t_mask[None, :], other=0.0,
                        cache_modifier=".cg")  # [HEADS * D, TILE]
            v = tl.load(v_cache_ptr + v_off[:, None] + offs_c[None, :], mask=t_mask[:, None], other=0.0,
                        cache_modifier=".cg")  # [TILE, HEADS * D]
            s = tl.dot(q, k) * qk_scale
            s = tl.where(t_mask[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            p = tl.math.exp2(s - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc=acc)
            m_i = m_new

        if (t_lo == 0) & (t_hi == n_tiles):
            # the whole sequence is in this program
            o = acc / l_i[:, None]
            tl.store(
                out_ptr + seq.to(tl.int64) * stride_o_b + head[:, None] * stride_o_h + col_d[None, :],
                o.to(out_ptr.dtype.element_ty),
                mask=blk_mask,
            )
        else:
            # partial slot 1 when the sequence is not the first one of this program's range
            slot_p = (u0 < c_start).to(tl.int64)
            pidx = (pid.to(tl.int64) * 2 + slot_p) * NUM_GROUPS + grp
            tl.store(part_m_ptr + pidx * BLOCK_M + offs_m, m_i)
            tl.store(part_l_ptr + pidx * BLOCK_M + offs_m, l_i)
            tl.store(
                part_o_ptr + (pidx * BLOCK_M + offs_m)[:, None] * HEAD_D + col_d[None, :],
                acc,
                mask=blk_mask,
            )
        u = c_start + t_hi


@triton.jit
def _balanced_gqa_decode_merge_kernel(
    out_ptr,
    seq_lens_ptr,
    part_m_ptr,
    part_l_ptr,
    part_o_ptr,
    num_seqs,
    num_progs,
    stride_o_b,
    stride_o_h,
    NUM_GROUPS: tl.constexpr,
    ROWS: tl.constexpr,  # query heads per group (HEADS * GQA)
    HEAD_D: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_B: tl.constexpr,
    MIN_TILES: tl.constexpr,
    BLOCK_P: tl.constexpr,  # partials merged per step
):
    """Merge the partials of the split sequences by log-sum-exp.

    A sequence is split only where a program boundary falls inside it, so the grid is one
    program per (program boundary j = 1 .. P-1, query head), independent of the batch size and
    lengths. Program j recomputes the schedule, finds the sequence holding boundary j and merges
    it only if j is the first boundary inside that sequence; every other program (boundary at a
    sequence start, later boundary of the same sequence, unused program) exits at once.
    Sequences covered by one program were written by the main kernel and are not touched."""
    j = tl.program_id(0) + 1
    head = tl.program_id(1)
    grp = head // ROWS
    row = head % ROWS

    offs_b, lens, tiles, cum, total, nprog = _schedule(
        seq_lens_ptr, num_seqs, num_progs, TILE, BLOCK_B, MIN_TILES
    )
    if j >= nprog:
        return
    u = (j * total) // nprog  # first tile of program j
    seq = tl.sum((cum <= u).to(tl.int32), axis=0)
    c_end = _pick(cum, offs_b, seq).to(tl.int64)
    c_start = c_end - _pick(tiles, offs_b, seq)
    if u == c_start:
        return  # boundary at a sequence start: nothing split here
    # programs covering tiles [c_start, c_end): tile u belongs to program ((u + 1) * P - 1) // T
    p_first = ((c_start + 1) * nprog - 1) // total
    if j != p_first + 1:
        return  # merged by the first boundary inside this sequence
    p_last = (c_end * nprog - 1) // total
    first_slot = ((p_first * total) // nprog < c_start).to(tl.int64)

    offs_p = tl.arange(0, BLOCK_P)
    offs_d = tl.arange(0, HEAD_D)
    m_t = tl.max(tl.full([BLOCK_P], float("-inf"), dtype=tl.float32), axis=0)
    l_t = tl.sum(tl.zeros([BLOCK_P], dtype=tl.float32), axis=0)
    o_t = tl.zeros([HEAD_D], dtype=tl.float32)
    for p0 in range(p_first, p_last + 1, BLOCK_P):
        pp = p0 + offs_p
        pmask = pp <= p_last
        sl = tl.where(pp == p_first, first_slot, 0)
        idx = ((pp * 2 + sl) * NUM_GROUPS + grp) * BLOCK_M + row
        mp = tl.load(part_m_ptr + idx, mask=pmask, other=float("-inf"))
        lp = tl.load(part_l_ptr + idx, mask=pmask, other=0.0)
        op = tl.load(part_o_ptr + idx[:, None] * HEAD_D + offs_d[None, :], mask=pmask[:, None], other=0.0)
        m_n = tl.maximum(m_t, tl.max(mp, axis=0))
        a = tl.math.exp2(m_t - m_n)
        w = tl.math.exp2(mp - m_n)
        l_t = l_t * a + tl.sum(lp * w, axis=0)
        o_t = o_t * a + tl.sum(op * w[:, None], axis=0)
        m_t = m_n
    o = o_t / l_t
    tl.store(out_ptr + seq.to(tl.int64) * stride_o_b + head * stride_o_h + offs_d, o.to(out_ptr.dtype.element_ty))

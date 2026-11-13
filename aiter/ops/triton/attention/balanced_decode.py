# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Token-balanced, graph-friendly GQA paged decode.

``balanced_gqa_decode`` computes one decode step (one query token per sequence) of grouped-query
attention over a paged NHD KV cache. Unlike a per-sequence split, the work is divided by tokens:
a fixed persistent grid of programs takes equal shares of the batch's KV tiles, and the share is
computed on the GPU from ``seq_lens`` on every launch (no host sync, no metadata kernel). A captured
HIP graph therefore stays correct and balanced when the sequence lengths change between replays.
Sequences split across programs are combined by a small log-sum-exp merge kernel.

Supported: bf16/fp16 Q and KV, power-of-2 head size (128 tested), GQA ratios up to 16, page size
1 or 16 (any power of 2), a vLLM-style block table or an SGLang/FlashInfer-style
(kv_indptr, kv_indices) page list. Not supported: q_len > 1, sliding window, sinks, alibi,
softcap, fp8 KV, LSE output.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.balanced_decode import (
    _balanced_gqa_decode_kernel,
    _balanced_gqa_decode_merge_kernel,
)


def _num_cus(device) -> int:
    try:
        from aiter.ops.triton.utils.device_info import get_num_sms

        return int(get_num_sms())
    except Exception:  # noqa: BLE001
        return torch.cuda.get_device_properties(device).multi_processor_count


def _block_m(rows: int) -> int:
    return max(16, triton.next_power_of_2(rows))


@dataclass
class BalancedDecodeWorkspace:
    """Partial-result buffers, allocated once (e.g. at engine start, before graph capture) and
    reused for every step. Sized by the program count, independent of the sequence lengths."""

    num_kv_heads: int
    heads_per_program: int
    num_progs: int  # programs per kv head group
    part_m: torch.Tensor
    part_l: torch.Tensor
    part_o: torch.Tensor


def balanced_gqa_decode_workspace(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    device="cuda",
    programs_per_cu: float = 2,
    heads_per_program: int = 1,
) -> BalancedDecodeWorkspace:
    gqa = num_q_heads // num_kv_heads
    assert num_kv_heads % heads_per_program == 0
    groups = num_kv_heads // heads_per_program
    block_m = _block_m(heads_per_program * gqa)
    num_progs = max(1, int(_num_cus(device) * programs_per_cu) // groups)
    n = num_progs * 2 * groups * block_m
    return BalancedDecodeWorkspace(
        num_kv_heads=num_kv_heads,
        heads_per_program=heads_per_program,
        num_progs=num_progs,
        part_m=torch.empty(n, dtype=torch.float32, device=device),
        part_l=torch.empty(n, dtype=torch.float32, device=device),
        part_o=torch.empty(n * head_size, dtype=torch.float32, device=device),
    )


def balanced_gqa_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[BalancedDecodeWorkspace] = None,
    tile: int = 64,
    min_tiles: int = 1,
    num_warps: int = 4,
    num_stages: int = 3,
    merge_block_p: int = 0,
    merge_num_warps: int = 1,
) -> torch.Tensor:
    """
    q:          [num_seqs, num_q_heads, head_size], one decode token per sequence.
    k_cache:    [num_pages, page_size, num_kv_heads, head_size] (NHD), same for v_cache.
    seq_lens:   [num_seqs] int32 on the device: KV length of every sequence (0 = no work and
                its output is left untouched, e.g. graph padding slots).
    block_table [num_seqs, max_pages] int32 (vLLM), or kv_indptr [num_seqs + 1] and kv_indices
                (SGLang / FlashInfer page list): page j of sequence b is kv_indices[kv_indptr[b] + j].
    workspace:  from balanced_gqa_decode_workspace(); allocate once and reuse (required for graph
                capture). Allocated per call when None.
    Returns out [num_seqs, num_q_heads, head_size].
    """
    assert q.dim() == 3 and k_cache.dim() == 4 and v_cache.shape == k_cache.shape
    num_seqs, num_q_heads, head_size = q.shape
    _, page_size, num_kv_heads, kd = k_cache.shape
    assert kd == head_size and num_q_heads % num_kv_heads == 0
    assert head_size == triton.next_power_of_2(head_size), "head_size must be a power of 2"
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
    assert seq_lens.dtype == torch.int32 and seq_lens.numel() >= num_seqs
    gqa = num_q_heads // num_kv_heads
    if workspace is None:
        workspace = balanced_gqa_decode_workspace(num_q_heads, num_kv_heads, head_size, q.device)
    heads = workspace.heads_per_program
    assert workspace.num_kv_heads == num_kv_heads
    if heads > 1:
        assert k_cache.stride(2) == head_size and v_cache.stride(2) == head_size
    rows = heads * gqa
    assert rows <= 16, "more than 16 query heads per program is not supported"
    groups = num_kv_heads // heads
    block_m = _block_m(rows)
    assert workspace.part_m.numel() >= workspace.num_progs * 2 * groups * block_m
    ragged = block_table is None
    if ragged:
        assert kv_indptr is not None and kv_indices is not None
        table, indptr, stride_t = kv_indices, kv_indptr, 0
    else:
        table, indptr, stride_t = block_table, block_table, block_table.stride(0)
    if softmax_scale is None:
        softmax_scale = head_size**-0.5
    if out is None:
        out = torch.empty_like(q)

    block_b = max(16, triton.next_power_of_2(num_seqs))
    _balanced_gqa_decode_kernel[(workspace.num_progs, groups)](
        out,
        q,
        k_cache,
        v_cache,
        seq_lens,
        table,
        indptr,
        workspace.part_m,
        workspace.part_l,
        workspace.part_o,
        num_seqs,
        workspace.num_progs,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        out.stride(0),
        out.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        stride_t,
        NUM_GROUPS=groups,
        GQA=gqa,
        HEADS=heads,
        HEAD_D=head_size,
        PAGE_SIZE=page_size,
        TILE=tile,
        BLOCK_M=block_m,
        BLOCK_B=block_b,
        MIN_TILES=min_tiles,
        RAGGED_TABLE=ragged,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    # partials per split sequence: ~ programs per sequence + 2 for even lengths, more for the long
    # sequences of a skewed batch; a wide step is cheap (masked loads), extra steps cost latency
    if not merge_block_p:
        per_seq = triton.cdiv(workspace.num_progs, num_seqs) + 2
        merge_block_p = min(64, max(8, triton.next_power_of_2(per_seq)))
    # one program per (program boundary, query head): fixed grid, captured once in a graph
    _balanced_gqa_decode_merge_kernel[(max(1, workspace.num_progs - 1), num_q_heads)](
        out,
        seq_lens,
        workspace.part_m,
        workspace.part_l,
        workspace.part_o,
        num_seqs,
        workspace.num_progs,
        out.stride(0),
        out.stride(1),
        NUM_GROUPS=groups,
        ROWS=rows,
        HEAD_D=head_size,
        TILE=tile,
        BLOCK_M=block_m,
        BLOCK_B=block_b,
        MIN_TILES=min_tiles,
        BLOCK_P=merge_block_p,
        num_warps=merge_num_warps,
    )
    return out


# ---------------------------------------------------------------------------------------------
# Dispatch between balanced_gqa_decode and unified_attention on graph-safe host quantities.
# ---------------------------------------------------------------------------------------------

BALANCED_MIN_SEQS = 16
BALANCED_MAX_SEQS = 128
BALANCED_MIN_MAX_SEQLEN = 2048
BALANCED_MAX_WORK = 1 << 20  # num_seqs * max_seqlen_k


def use_balanced_decode(num_seqs: int, num_kv_heads: int, page_size: int, max_seqlen_k: int) -> bool:
    """Pick balanced_gqa_decode over unified_attention from host-known values only (no
    device->host sync, so the choice is fixed per captured graph and valid for any lengths).

    Rule (MI355X, bf16, head size 128, GQA 4 and 8, page 16 and 1):
      * 16 <= num_seqs <= 128: with fewer sequences unified_attention's per-sequence split
        already fills the GPU and the balanced op's second (merge) launch dominates; with more,
        unified_attention has enough programs to hide skew.
      * max_seqlen_k >= 2048: below that every sequence is at most a few tiles, the batch is
        latency-bound and unified_attention's single launch wins even when lengths differ.
      * num_seqs * max_seqlen_k <= 2**20: above that bound unified_attention streams near the
        bandwidth ceiling and the balanced kernel's loop is up to ~6 % slower.
    Inside that range the balanced op is within a few percent of unified_attention on uniform
    lengths and up to ~2x faster when lengths are skewed. num_kv_heads and page_size are part
    of the signature for future tuning; the measured GQA ratios (4, 8) and page sizes (1, 16)
    share the rule.
    """
    return (
        BALANCED_MIN_SEQS <= num_seqs <= BALANCED_MAX_SEQS
        and max_seqlen_k >= BALANCED_MIN_MAX_SEQLEN
        and num_seqs * max_seqlen_k <= BALANCED_MAX_WORK
    )


def gqa_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[BalancedDecodeWorkspace] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode step (q_len 1) through balanced_gqa_decode or unified_attention, chosen by
    use_balanced_decode() on host values. max_seqlen_k is the host bound the caller already
    passes to unified_attention (it may be an upper bound, e.g. under graph capture).
    cu_seqlens_q ([num_seqs + 1], arange) is the caller's query_start_loc; created here when
    None (one extra small kernel)."""
    num_seqs, num_q_heads, head_size = q.shape
    num_kv_heads, page_size = k_cache.shape[2], k_cache.shape[1]
    if softmax_scale is None:
        softmax_scale = head_size**-0.5
    if out is None:
        out = torch.empty_like(q)
    if use_balanced_decode(num_seqs, num_kv_heads, page_size, max_seqlen_k):
        return balanced_gqa_decode(
            q, k_cache, v_cache, seq_lens, block_table=block_table,
            softmax_scale=softmax_scale, out=out, workspace=workspace,
        )
    from aiter.ops.triton.attention.unified_attention import unified_attention

    if cu_seqlens_q is None:
        cu_seqlens_q = torch.arange(num_seqs + 1, device=q.device, dtype=torch.int32)
    unified_attention(
        q, k_cache, v_cache, out, cu_seqlens_q, 1, seq_lens, max_seqlen_k, softmax_scale, True,
        (-1, -1), block_table, 0, None, None, None,
    )
    return out

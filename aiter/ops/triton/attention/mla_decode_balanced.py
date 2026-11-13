# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Token-balanced, graph-safe MLA decode for small head counts (bf16, 576-wide latent cache, page 1).

``mla_decode_balanced`` computes one decode step of absorbed MLA: q [B * q_len, H, 576] against a
latent cache whose row is K (576 columns) and V (first 512 columns). The work is split by tokens on
the device from ``kv_indptr`` on every launch (fixed grid, no metadata kernel, no host sync), so a
captured HIP graph stays balanced when the lengths change. Heads are not padded: H query heads x
q_len tokens form one tile, and fp32 partials are written only for sequences split across programs.

Two main kernels share the schedule and the merge: a Gluon one for gfx950 with 16 heads and q_len 1
(async global->LDS copies, about 5 TB/s on long streams), and a portable Triton one (any head count
that fits one MFMA tile, q_len >= 1).

``mla_decode_fwd`` routes here (see ``use_mla_decode_balanced``) only when
``AITER_MLA_DECODE_BALANCED=1``. On MI355X with 1-64 sequences it is 1.3-2x faster than the
persistent asm path up to ~1M KV tokens and within ~1 % at 64 x 32k. With 2M+ tokens per step
(e.g. 32-64 x 128k) the asm kernel streams faster (~5.8 against ~4.9 TB/s, 13-17 % slower), and
the dispatch cannot see the KV lengths (frameworks pass padded bounds and whole-pool index buffers
under graph capture), so it stays opt-in.
"""

import os

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.mla_decode_balanced import (
    _mla_decode_balanced_kernel,
    _mla_decode_balanced_merge_kernel,
)

_DV = 512
_DR = 64


def _num_cus(device) -> int:
    try:
        from aiter.ops.triton.utils.device_info import get_num_sms

        return int(get_num_sms())
    except Exception:  # noqa: BLE001
        return torch.cuda.get_device_properties(device).multi_processor_count


def mla_decode_balanced(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    max_seqlen_q: int,
    sm_scale: float,
    programs_per_cu: float = 0,
    tile: int = 16,
    num_warps: int = 4,
    num_stages: int = 1,
    backend: str | None = None,
    unit: int = 64,
    g_tile: int = 64,
    g_warps: int = 4,
    g_buffers: int = 2,
    g_waves: int = 1,
    merge_block_p: int = 32,
    merge_block_d: int = 256,
    merge_num_warps: int = 4,
) -> torch.Tensor:
    """
    q:          [num_seqs * q_len, H, 576] bf16 (512 NoPE + 64 RoPE), q_len = max_seqlen_q for every
                sequence (uniform decode / MTP).
    kv_buffer:  latent cache viewed as page size 1, [num_slots, 1, 1, 576] (any shape whose last dim
                is 576 and whose rows are contiguous).
    o:          [num_seqs * q_len, H, 512], written in place.
    kv_indptr:  [num_seqs + 1] int32, kv_indices: token slots. A sequence with no KV is skipped (its
                output is left untouched, e.g. graph padding slots).
    Query token i of a sequence sees the first kv_len - q_len + i + 1 tokens (bottom-right causal).
    """
    qlen = int(max_seqlen_q)
    total_q, nhead, dqk = q.shape
    assert dqk == _DV + _DR and o.shape[-1] == _DV and total_q % qlen == 0
    num_seqs = total_q // qlen
    assert kv_indptr.numel() >= num_seqs + 1
    kv = kv_buffer.reshape(-1, dqk)
    assert kv.stride(-1) == 1 and q.stride(-1) == 1 and o.stride(-1) == 1
    if backend is None:
        from aiter.jit.utils.chip_info import get_gfx

        gluon_ok = get_gfx() == "gfx950" and nhead == 16 and qlen == 1
        backend = "gluon" if gluon_ok else "triton"
    if backend == "triton":
        unit = tile
    rows = qlen * nhead
    block_m = max(16, triton.next_power_of_2(rows))
    if not programs_per_cu:
        # the Gluon kernel holds 2 x 64-token LDS tiles (150 KB): one program per CU
        programs_per_cu = 1 if backend == "gluon" else 2
    num_progs = max(1, int(_num_cus(q.device) * programs_per_cu))
    n = num_progs * 2 * block_m
    part_m = torch.empty(n, dtype=torch.float32, device=q.device)
    part_l = torch.empty(n, dtype=torch.float32, device=q.device)
    part_o = torch.empty(n * _DV, dtype=torch.float32, device=q.device)
    task = torch.empty(num_seqs * 4, dtype=torch.int32, device=q.device)
    block_b = max(16, triton.next_power_of_2(num_seqs))

    if backend == "gluon":
        assert nhead == 16 and qlen == 1, "the Gluon kernel covers 16 heads, q_len 1"
        from aiter.ops.triton._gluon_kernels.gfx950.attention.mla_decode_balanced import (
            _mla_decode_balanced_gluon_kernel,
        )

        _mla_decode_balanced_gluon_kernel[(num_progs,)](
            o, q, kv, kv_indptr, kv_indices, part_m, part_l, part_o, task,
            num_seqs, num_progs, float(sm_scale),
            q.stride(0), q.stride(1), o.stride(0), o.stride(1), kv.stride(0),
            DV=_DV, DR=_DR, UNIT=unit, TILE=g_tile, BLOCK_B=block_b, NUM_WARPS=g_warps,
            NUM_BUFFERS=g_buffers, KV_PAD=8, num_warps=g_warps, waves_per_eu=g_waves,
        )
    else:
        _mla_decode_balanced_kernel[(num_progs,)](
            o, q, kv, kv_indptr, kv_indices, part_m, part_l, part_o, task,
            num_seqs, num_progs, float(sm_scale),
            q.stride(0), q.stride(1), o.stride(0), o.stride(1), kv.stride(0),
            NUM_HEADS=nhead, QLEN=qlen, DV=_DV, DR=_DR, TILE=unit, BLOCK_M=block_m, BLOCK_B=block_b,
            num_warps=num_warps, num_stages=num_stages,
        )
    # one program per (sequence, query row, column block): fixed grid for a given batch size
    _mla_decode_balanced_merge_kernel[(num_seqs, rows, _DV // merge_block_d)](
        o, task, kv_indptr, part_m, part_l, part_o, num_progs, o.stride(0), o.stride(1),
        NUM_HEADS=nhead, QLEN=qlen, DV=_DV,
        BLOCK_P=merge_block_p, BLOCK_D=merge_block_d, num_warps=merge_num_warps,
    )
    return o


# ---------------------------------------------------------------------------------------------
# Dispatch from mla_decode_fwd, on host-known values only (no device->host sync, graph-safe).
# ---------------------------------------------------------------------------------------------

BALANCED_MIN_SEQS = 1
BALANCED_MAX_SEQS = 64  # above, the asm kernel is faster (256 x 8k: 1.07x, 256 x 32k: 1.20x)


def use_mla_decode_balanced(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen_q: int,
    page_size: int,
    nhead_kv: int,
) -> bool:
    """True when ``AITER_MLA_DECODE_BALANCED=1`` and the call is a bf16 gfx950 decode step this kernel
    covers: 16 query heads, q_len 1 (the Triton kernel also handles q_len 2 / MTP, but is slower than
    the asm path there), 576-wide latent cache with page size 1, and
    BALANCED_MIN_SEQS..BALANCED_MAX_SEQS sequences. The caller checks the remaining mla_decode_fwd options
    (no LSE / logits output, no logit cap, no scales, no context parallelism, token-major layout)."""
    if os.environ.get("AITER_MLA_DECODE_BALANCED", "0") != "1":
        return False
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        return False
    if q.dtype != torch.bfloat16 or kv_buffer.dtype != torch.bfloat16 or o.dtype != torch.bfloat16:
        return False
    if q.dim() != 3 or q.shape[1] != 16 or q.shape[2] != _DV + _DR or o.shape[-1] != _DV:
        return False
    if page_size != 1 or nhead_kv != 1 or kv_buffer.shape[-1] != _DV + _DR:
        return False
    if isinstance(max_seqlen_q, bool) or max_seqlen_q != 1:
        return False
    num_seqs = qo_indptr.shape[0] - 1
    if q.shape[0] != num_seqs * max_seqlen_q or not BALANCED_MIN_SEQS <= num_seqs <= BALANCED_MAX_SEQS:
        return False
    return q.stride(-1) == 1 and o.stride(-1) == 1 and kv_buffer.is_contiguous()

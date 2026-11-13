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

``mla_decode_fwd`` routes gfx950 bf16 decode with 16 heads, q_len 1, page size 1 and 1-64 sequences
here by default (``AITER_MLA_DECODE_BALANCED=0`` turns it off). With the persistent metadata the
frameworks pass, it runs as a device-side hybrid: the balanced main kernel works when the step has
at most ``BALANCED_MAX_TOKENS`` KV tokens and the persistent asm main kernel otherwise (it gets an
empty work list in the first case), and one reduce launch serves both. The choice is made on the
GPU from kv_indptr on every launch, so it needs no host sync and holds under HIP-graph replay.
Without persistent metadata the balanced path alone runs only with ``AITER_MLA_DECODE_BALANCED=1``.
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
    hybrid: dict | None = None,
    total_max: int = 0,
    dual: bool = False,
):
    """
    q:          [num_seqs * q_len, H, 576] bf16 (512 NoPE + 64 RoPE), q_len = max_seqlen_q for every
                sequence (uniform decode / MTP).
    kv_buffer:  latent cache viewed as page size 1, [num_slots, 1, 1, 576] (any shape whose last dim
                is 576 and whose rows are contiguous).
    o:          [num_seqs * q_len, H, 512], written in place.
    kv_indptr:  [num_seqs + 1] int32, kv_indices: token slots. A sequence with no KV is skipped (its
                output is left untouched, e.g. graph padding slots).
    Query token i of a sequence sees the first kv_len - q_len + i + 1 tokens (bottom-right causal).
    hybrid:     the persistent asm metadata of the call (work_meta_data, work_indptr, work_info_set,
                reduce_indptr). Then the kernels work only when the step has at most total_max KV
                tokens (decided on the device). Returns the effective work_meta_data for the asm main
                kernel that follows (empty work at or below total_max, the caller's above it) and a
                function that, given the inputs of mla_reduce_v1 as a dict, launches the one reduce
                kernel of the layer (the balanced merge at or below total_max, the reduce of the
                asm partials above it). Without hybrid, returns o.
    dual:       (Gluon) also launch the long-step configuration (LONG_CONFIG): steps with at most
                total_max KV tokens run the configured kernel, longer steps the long-step kernel; the
                choice is made on the device, graph-safe. Returns o.
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
    num_progs_long = 0
    if dual:
        assert backend == "gluon" and hybrid is None and total_max > 0
        num_progs_long = max(1, int(_num_cus(q.device) * LONG_CONFIG["programs_per_cu"]))
    n = max(num_progs, num_progs_long) * 2 * block_m
    part_m = torch.empty(n, dtype=torch.float32, device=q.device)
    part_l = torch.empty(n, dtype=torch.float32, device=q.device)
    part_o = torch.empty(n * _DV, dtype=torch.float32, device=q.device)
    task = torch.empty(num_seqs * 4, dtype=torch.int32, device=q.device)
    block_b = max(16, triton.next_power_of_2(num_seqs))

    if hybrid is not None:
        assert backend == "gluon", "the hybrid dispatch runs the Gluon kernel"
        meta = hybrid["work_meta_data"]
        eff_meta = torch.empty_like(meta)
        zero_indptr = torch.empty_like(hybrid["work_indptr"])
        assert meta.element_size() == 8 and meta.is_contiguous()
        cap = num_progs * max(16, triton.next_power_of_2(num_seqs))
        assert max(meta.numel(), zero_indptr.numel()) <= cap
        hyb_args = (int(total_max), meta, eff_meta, meta.numel(), zero_indptr, zero_indptr.numel(),
                    zero_indptr.data_ptr())
    else:
        hyb_args = (0, task, task, 0, task, 0, 0)
    if backend == "gluon":
        assert nhead == 16 and qlen == 1, "the Gluon kernel covers 16 heads, q_len 1"
        from aiter.ops.triton._gluon_kernels.gfx950.attention.mla_decode_balanced import (
            _mla_decode_balanced_gluon_kernel,
        )

        def _main(nprogs, t_max, t_min, unit_, tile_, warps_, buffers_, hyb, flag):
            _mla_decode_balanced_gluon_kernel[(nprogs,)](
                o, q, kv, kv_indptr, kv_indices, part_m, part_l, part_o, task,
                num_seqs, nprogs, float(sm_scale),
                q.stride(0), q.stride(1), o.stride(0), o.stride(1), kv.stride(0),
                t_max, t_min, *hyb[1:],
                DV=_DV, DR=_DR, UNIT=unit_, TILE=tile_, BLOCK_B=block_b, NUM_WARPS=warps_,
                NUM_BUFFERS=buffers_, KV_PAD=8, HYBRID=flag, num_warps=warps_,
                waves_per_eu=g_waves,
            )

        if dual:
            # at most total_max tokens: this configuration; more: the long-step configuration
            _main(num_progs, int(total_max), 0, unit, g_tile, g_warps, g_buffers, hyb_args, False)
            lc = LONG_CONFIG
            _main(num_progs_long, 0, int(total_max), lc["unit"], lc["tile"], lc["warps"],
                  lc["buffers"], hyb_args, False)
        else:
            _main(num_progs, hyb_args[0], 0, unit, g_tile, g_warps, g_buffers, hyb_args,
                  hybrid is not None)
    else:
        _mla_decode_balanced_kernel[(num_progs,)](
            o, q, kv, kv_indptr, kv_indices, part_m, part_l, part_o, task,
            num_seqs, num_progs, float(sm_scale),
            q.stride(0), q.stride(1), o.stride(0), o.stride(1), kv.stride(0),
            NUM_HEADS=nhead, QLEN=qlen, DV=_DV, DR=_DR, TILE=unit, BLOCK_M=block_m, BLOCK_B=block_b,
            num_warps=num_warps, num_stages=num_stages,
        )
    def _merge(red=None):
        # one program per (sequence, query row, column block): fixed grid
        if red is None:
            ra, n_tiles = (task, task, task, part_l, part_o), 0
        else:
            ra = (red["reduce_indptr"], red["reduce_final_map"], red["reduce_partial_map"],
                  red["partial_lse"], red["partial_output"])
            n_tiles = ra[0].numel() - 1
            assert ra[3].dtype == torch.float32 and ra[4].dtype == torch.float32
            assert ra[4].shape[-1] == _DV and ra[4].shape[-2] == nhead and ra[4].is_contiguous()
        _mla_decode_balanced_merge_kernel[(num_seqs, rows, _DV // merge_block_d)](
            o, task, kv_indptr, part_m, part_l, part_o, num_progs, o.stride(0), o.stride(1),
            num_seqs, int(total_max) if dual else hyb_args[0], *ra, n_tiles, num_progs_long,
            NUM_HEADS=nhead, QLEN=qlen, DV=_DV,
            BLOCK_P=merge_block_p, BLOCK_D=merge_block_d, HYBRID=hybrid is not None, DUAL=dual,
            num_warps=merge_num_warps,
        )

    if hybrid is not None:
        # the caller runs the asm main kernel with eff_meta, then this single reduce launch
        return eff_meta, _merge
    _merge()
    return o


# ---------------------------------------------------------------------------------------------
# Dispatch from mla_decode_fwd, on host-known values only (no device->host sync, graph-safe).
# ---------------------------------------------------------------------------------------------

# Hybrid crossover: at most this many KV tokens per step run the balanced kernels, more run the
# persistent asm path (decided on the device per launch; see mla_decode_fwd).
BALANCED_MAX_TOKENS = 1_572_864  # MI355X: the balanced path wins up to ~1.5-2M tokens per step

# Long-step Gluon configuration: 1-warp programs with 16-token tiles, 4 per CU. Each wave streams its
# own tiles independently (no cross-warp softmax exchange or barrier), 6.4 TB/s at 64 x 128k.
LONG_CONFIG = dict(programs_per_cu=4, unit=64, tile=16, warps=1, buffers=2)
# Below this many KV tokens per step the 4-warp configuration is faster (fewer split pieces to merge)
DUAL_SWITCH_TOKENS = int(os.environ.get("AITER_MLA_BALANCED_SWITCH_TOKENS", 163_840))


def long_steps_on_asm() -> bool:
    """AITER_MLA_BALANCED_LONG=asm: run long steps on the persistent asm kernel (device hybrid)."""
    return os.environ.get("AITER_MLA_BALANCED_LONG", "") == "asm"


def balanced_forced() -> bool:
    """AITER_MLA_DECODE_BALANCED=1: also use the balanced path without persistent metadata."""
    return os.environ.get("AITER_MLA_DECODE_BALANCED", "") == "1"
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
    """True unless ``AITER_MLA_DECODE_BALANCED=0``, when the call is a bf16 gfx950 decode step this
    kernel covers: 16 query heads, q_len 1 (the Triton kernel also handles q_len 2 / MTP, but is slower than
    the asm path there), 576-wide latent cache with page size 1, and
    BALANCED_MIN_SEQS..BALANCED_MAX_SEQS sequences. The caller checks the remaining mla_decode_fwd options
    (no LSE / logits output, no logit cap, no scales, no context parallelism, token-major layout)."""
    if os.environ.get("AITER_MLA_DECODE_BALANCED", "") == "0":
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

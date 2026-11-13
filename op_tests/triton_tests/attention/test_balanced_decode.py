# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch

from aiter.ops.triton.attention.balanced_decode import (
    balanced_gqa_decode,
    balanced_gqa_decode_workspace,
)

HEAD = 128


def _make(lens, num_q_heads, num_kv_heads, page_size, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    b = len(lens)
    pages = [(n + page_size - 1) // page_size for n in lens]
    max_pages = max(max(pages), 1)
    total = b * max_pages
    k_cache = torch.randn(total, page_size, num_kv_heads, HEAD, device="cuda", dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    block_table = torch.randperm(total, device="cuda").to(torch.int32).view(b, max_pages)
    q = torch.randn(b, num_q_heads, HEAD, device="cuda", dtype=dtype)
    seq_lens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    return q, k_cache, v_cache, block_table, seq_lens, pages


def _ref(q, k_cache, v_cache, block_table, lens, scale):
    b, hq, d = q.shape
    _, ps, hk, _ = k_cache.shape
    g = hq // hk
    out = torch.empty(b, hq, d, device=q.device, dtype=torch.float32)
    for i, n in enumerate(lens):
        pg = block_table[i, : (n + ps - 1) // ps].long()
        k = k_cache[pg].reshape(-1, hk, d)[:n].float()
        v = v_cache[pg].reshape(-1, hk, d)[:n].float()
        s = torch.einsum("hgd,thd->hgt", q[i].float().view(hk, g, d), k) * scale
        out[i] = torch.einsum("hgt,thd->hgd", torch.softmax(s, -1), v).reshape(hq, d)
    return out


LENS = {
    "single_token": [1],
    "one_seq": [1000],
    "uniform": [1024] * 16,
    "odd": [5, 300, 4097, 63, 64, 129, 1, 2000],
    "one_long": [16384] + [256] * 31,
    "bimodal": [64, 4096] * 16,
}


@pytest.mark.parametrize("lens_name", list(LENS))
@pytest.mark.parametrize("num_kv_heads", [4, 8])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("ragged", [False, True])
def test_balanced_gqa_decode(lens_name, num_kv_heads, page_size, ragged):
    lens = LENS[lens_name]
    num_q_heads = 32
    q, kc, vc, bt, seq_lens, pages = _make(lens, num_q_heads, num_kv_heads, page_size)
    scale = 1.0 / math.sqrt(HEAD)
    ws = balanced_gqa_decode_workspace(num_q_heads, num_kv_heads, HEAD, "cuda")
    if ragged:
        indptr = torch.zeros(len(lens) + 1, device="cuda", dtype=torch.int32)
        indptr[1:] = torch.tensor(pages, device="cuda").cumsum(0)
        indices = torch.cat([bt[i, :n] for i, n in enumerate(pages)])
        out = balanced_gqa_decode(q, kc, vc, seq_lens, kv_indptr=indptr, kv_indices=indices,
                                  softmax_scale=scale, workspace=ws)
    else:
        out = balanced_gqa_decode(q, kc, vc, seq_lens, block_table=bt, softmax_scale=scale, workspace=ws)
    ref = _ref(q, kc, vc, bt, lens, scale)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


def test_balanced_gqa_decode_graph_replay():
    """Capture once, replay with other (skewed) lengths in the same seq_lens tensor."""
    b, num_q_heads, num_kv_heads, ps, max_len = 32, 32, 8, 16, 8192
    q, kc, vc, bt, seq_lens, _ = _make([max_len] * b, num_q_heads, num_kv_heads, ps)
    scale = 1.0 / math.sqrt(HEAD)
    ws = balanced_gqa_decode_workspace(num_q_heads, num_kv_heads, HEAD, "cuda")
    out = torch.empty_like(q)

    def run():
        balanced_gqa_decode(q, kc, vc, seq_lens, block_table=bt, softmax_scale=scale, out=out, workspace=ws)

    seq_lens.fill_(512)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    for lens in ([512] * b, [max_len] + [100] * (b - 1), [1, 3000] * (b // 2)):
        seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), _ref(q, kc, vc, bt, lens, scale), atol=2e-2, rtol=2e-2)

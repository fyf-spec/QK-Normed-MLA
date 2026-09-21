#!/usr/bin/env python3
# Full-MLA-layer decode benchmark, paper-faithful (arXiv 2606.16310 §5.3).
#
# Reproduces the paper's measurement methodology: one DeepSeek-V3-width MLA
# LAYER (not just the attention core), BF16, decode step (1 new token) over a
# prefilled KV cache, timed across context lengths at batch=4. Overhead is
# reported over the FULL-LAYER time, the paper's denominator -- which is why
# the module-only bench (bench_qknorm_module.py) reported ~4%: it divides by
# the attention core alone (a 10-60x tighter denominator).
#
# Three backends, IDENTICAL projection/absorption code, differ ONLY in the
# attention core:
#   (1) torch    : pure-PyTorch absorbed decode, torch.compile(reduce-overhead)
#                  -- the paper's own baseline.
#   (2) flashmla : FlashMLA dense-decode kernel, qk_norm_scale_cache=None
#                  (the pre-existing <T,false> MLA path), CUDA-graph replay.
#   (3) qknorm   : same kernel, qk_norm_scale_cache=sidecar (<T,true>), plus the
#                  honest per-token alpha^k RMS-stat write, CUDA-graph replay.
#
# The rock-solid qk-norm overhead number is (3-2)/2: both use the same FlashMLA
# core under identical graph capture, so the delta isolates the qk-norm
# integration and counts NONE of FlashMLA's baseline MLA-path speedup as gain
# (per the standing constraint). torch (1) is a paper-comparable context point.
#
# Config = DeepSeek-V3.2 MLA, TP=8 local view (inference/model.py):
#   dim=7168, q_lora_rank=1536, kv_lora_rank=512(=r), n_local_heads=16,
#   qk_nope=128(=d_c), qk_rope=64(=d_r), v_head=128, qk_head_dim=192.
#   softmax_scale = 192**-0.5  (pre-absorption dim -- NOT 576**-0.5).
# Systems-faithfulness note: BF16 KV cache (paper is BF16); the fp8 kv-cache
# simulation in the reference model is deployment-specific and orthogonal to
# the qk-norm question, so it is intentionally omitted (stated, not hidden).
import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone

import torch
import triton
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

D_MODEL = 7168
Q_LORA = 1536
KV_LORA = 512          # r, = HEAD_DIM_V (content latent)
N_HEADS = 16           # TP=8 local view of 128 heads
NOPE = 128             # d_c, per-head content dim (pre-absorption)
ROPE = 64              # d_r
V_HEAD = 128
QK_HEAD = NOPE + ROPE  # 192, sets softmax scale
HEAD_DIM_K = KV_LORA + ROPE   # 576, FlashMLA key dim after absorption
BLOCK = 64             # page block size
DTYPE = torch.bfloat16
SOFTMAX_SCALE = QK_HEAD ** -0.5

# batch=4 like the paper; contexts 4k..256k (§5.3 sweep).
MATRIX = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
BATCH = 4


def _rope(x, cos, sin):
    # x: (..., ROPE). Non-interleaved half-rotation; identical across all three
    # backends, so the exact variant is irrelevant to the comparison.
    x1, x2 = x[..., : ROPE // 2], x[..., ROPE // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class MLALayer(torch.nn.Module):
    """DeepSeek-V3.2 MLA, TP=8 local view, absorbed decode. Weights are random
    (timing harness, not a correctness model). Projection + absorption code is
    shared by all backends; only attn_core differs."""

    def __init__(self):
        super().__init__()
        lin = lambda i, o: torch.nn.Linear(i, o, bias=False, dtype=DTYPE)
        self.wq_a = lin(D_MODEL, Q_LORA)
        self.q_norm_w = torch.nn.Parameter(torch.ones(Q_LORA, dtype=DTYPE))
        self.wq_b = lin(Q_LORA, N_HEADS * QK_HEAD)
        self.wkv_a = lin(D_MODEL, KV_LORA + ROPE)
        self.kv_norm_w = torch.nn.Parameter(torch.ones(KV_LORA, dtype=DTYPE))
        # wkv_b: per-head up-proj, absorbed at decode. (H, NOPE+V_HEAD, KV_LORA)
        self.wkv_b = torch.nn.Parameter(
            torch.randn(N_HEADS, NOPE + V_HEAD, KV_LORA, dtype=DTYPE) * 0.02)
        self.wo = lin(N_HEADS * V_HEAD, D_MODEL)
        self.eps = 1e-6

    def _rmsnorm(self, x, w):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * w.float()).to(DTYPE)

    def project_q_kv(self, x, cos, sin):
        # Shared front-end: produce absorbed content query q_c (B,1,H,KV_LORA),
        # rope query q_r (B,1,H,ROPE), and the NEW token's kv-latent + k_rope.
        bsz = x.shape[0]
        qr = self._rmsnorm(self.wq_a(x), self.q_norm_w)
        q = self.wq_b(qr).view(bsz, 1, N_HEADS, QK_HEAD)
        q_nope, q_pe = q.split([NOPE, ROPE], dim=-1)
        q_pe = _rope(q_pe, cos, sin)
        wkv_b = self.wkv_b                       # (H, NOPE+V, KV_LORA)
        # absorb W_UK^T into query: q_c = q_nope @ W_UK  -> (B,1,H,KV_LORA)
        q_c = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :NOPE])
        kv = self.wkv_a(x)                        # (B,1,KV_LORA+ROPE)
        kv_lat, k_pe = kv.split([KV_LORA, ROPE], dim=-1)
        kv_lat = self._rmsnorm(kv_lat, self.kv_norm_w)   # new content latent
        k_pe = _rope(k_pe.unsqueeze(2), cos, sin).squeeze(2)
        return q_c, q_pe, kv_lat, k_pe, wkv_b

    def absorb_out(self, x_c, wkv_b):
        # x_c: (B,1,H,KV_LORA) attended latent -> per-head value -> o-proj.
        x = torch.einsum("bshc,hdc->bshd", x_c, wkv_b[:, NOPE:])
        return self.wo(x.flatten(2))

    # ---- backend (1): pure-PyTorch absorbed core (contiguous cache) ----
    def core_torch(self, q_c, q_pe, cache_c, cache_pe):
        # scores: content-latent dot + rope dot, scaled, softmax, weighted sum.
        # The einsums contract the single KV head IMPLICITLY (never expand it to
        # 16 heads), so memory is tiny -- scores (B,1,H,S) f32 = 67MB @256k, no
        # OOM. (SDPA is NOT usable here: head_dim_qk=576 != head_dim_v=512, both
        # >256, so every fused backend refuses and the math fallback materializes
        # the GQA-broadcast K/V -> ~10x slower + memory blowup.) torch.compile
        # fuses the softmax pipeline -- the paper's torch.compile baseline.
        s = (torch.einsum("bshc,btc->bsht", q_c, cache_c)
             + torch.einsum("bshr,btr->bsht", q_pe, cache_pe)) * SOFTMAX_SCALE
        s = s.softmax(dim=-1)
        return torch.einsum("bsht,btc->bshc", s, cache_c)

    # ---- backends (2)/(3): FlashMLA core (paged cache) ----
    def core_flash(self, q_c, q_pe, blocked_k, block_table, seqlens,
                   meta, splits, scale_cache):
        q = torch.cat([q_c, q_pe], dim=-1)       # (B,1,H,576)
        out = flash_mla_with_kvcache(
            q, blocked_k, block_table, seqlens, KV_LORA,
            meta, splits, softmax_scale=SOFTMAX_SCALE, causal=True,
            qk_norm_scale_cache=scale_cache)
        if isinstance(out, tuple):
            out = out[0]
        return out                               # (B,1,H,KV_LORA)


def _build(bsz, ctx, dev, need_paged, need_contig):
    """Prefill KV cache with `ctx` past tokens; one decode-step input x."""
    x = torch.randn(bsz, 1, D_MODEL, dtype=DTYPE, device=dev)
    cos = torch.randn(bsz, 1, 1, ROPE // 2, dtype=DTYPE, device=dev)
    sin = torch.randn(bsz, 1, 1, ROPE // 2, dtype=DTYPE, device=dev)
    seqlens = torch.full((bsz,), ctx, dtype=torch.int32, device=dev)
    contig_c = contig_pe = None
    if need_contig:
        contig_c = torch.randn(bsz, ctx, KV_LORA, dtype=DTYPE, device=dev)
        contig_pe = torch.randn(bsz, ctx, ROPE, dtype=DTYPE, device=dev)
    blocked_k = block_table = scale_cache = None
    if need_paged:
        pad = (ctx + BLOCK - 1) // BLOCK * BLOCK
        nblk = bsz * pad // BLOCK
        blocked_k = torch.randn(nblk, BLOCK, 1, HEAD_DIM_K, dtype=DTYPE, device=dev)
        if need_contig:                          # share the same past K data
            # FlashMLA key layout is [content(512) | rope(64)] == q layout.
            k = torch.cat([contig_c, contig_pe], dim=-1)
            for b in range(bsz):
                flat = k[b].reshape(-1, HEAD_DIM_K)
                blocked_k.view(bsz, pad, HEAD_DIM_K)[b, :ctx] = flat
        block_table = torch.arange(nblk, dtype=torch.int32,
                                   device=dev).view(bsz, pad // BLOCK)
        scale_cache = torch.ones(nblk, BLOCK, dtype=torch.float32, device=dev)
    return dict(x=x, cos=cos, sin=sin, seqlens=seqlens, contig_c=contig_c,
                contig_pe=contig_pe, blocked_k=blocked_k, block_table=block_table,
                scale_cache=scale_cache)


def make_forward(layer, backend, d):
    """Return a nullary closure running ONE full-layer decode step."""
    x, cos, sin = d["x"], d["cos"], d["sin"]
    if backend == "torch":
        cc, cpe = d["contig_c"], d["contig_pe"]
        def fwd():
            q_c, q_pe, kv_lat, k_pe, wkv_b = layer.project_q_kv(x, cos, sin)
            # append new token to contiguous cache (decode write). kv_lat/k_pe
            # are already (B,1,dim) -- they carry the seq dim -- so no unsqueeze.
            c = torch.cat([cc, kv_lat], dim=1)
            p = torch.cat([cpe, k_pe], dim=1)
            x_c = layer.core_torch(q_c, q_pe, c, p)
            return layer.absorb_out(x_c, wkv_b)
        return fwd
    # flashmla / qknorm
    bk, bt, sl = d["blocked_k"], d["block_table"], d["seqlens"]
    # Fresh meta PER backend: its config (incl. qk_normed) is frozen on the
    # first flash_mla_with_kvcache call, so flashmla and qknorm must not share.
    meta, splits = get_mla_metadata(sl, N_HEADS, 1)
    sc = d["scale_cache"] if backend == "qknorm" else None
    def fwd():
        q_c, q_pe, kv_lat, k_pe, wkv_b = layer.project_q_kv(x, cos, sin)
        if backend == "qknorm":
            # honest per-token alpha^k RMS-stat write (the qk-norm cost the
            # paper's overhead must include): inverse-RMS of the new content
            # latent, one scalar per token per KV group.
            _ = torch.rsqrt(kv_lat.float().pow(2).mean(-1) + layer.eps)
        x_c = layer.core_flash(q_c, q_pe, bk, bt, sl, meta, splits, sc)
        return layer.absorb_out(x_c, wkv_b)
    return fwd


def _graphed(fwd):
    # CUDA-graph replay so all three backends pay equal launch overhead
    # (torch.compile reduce-overhead does the same for backend 1).
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fwd()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fwd()
    def run():
        g.replay()
        return out
    return run


def equivalence_gate(layer, dev):
    # Full-layer numerical integrity: qknorm backend with sidecar=ones and NO
    # alpha^q fold must match the plain flashmla backend within tolerance.
    d = _build(2, 256, dev, need_paged=True, need_contig=False)
    f_base = make_forward(layer, "flashmla", d)
    f_qk = make_forward(layer, "qknorm", d)
    with torch.no_grad():
        o_base, o_qk = f_base(), f_qk()
    diff = (o_base.float() - o_qk.float()).abs().max().item()
    scale = o_base.float().abs().max().item() + 1e-6
    rel = diff / scale
    ok = rel < 2e-2
    print(f"[gate] full-layer scale=ones: max|Δ|={diff:.3e} rel={rel:.3e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok, diff, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qknorm_fulllayer_perf.csv")
    ap.add_argument("--manifest", default="qknorm_fulllayer_manifest.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device visible"
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    torch.manual_seed(0)
    layer = MLALayer().to(dev).eval()

    gate_ok, gate_diff, gate_rel = equivalence_gate(layer, dev)
    if not gate_ok:
        raise SystemExit("[gate] FAILED numerical-integrity check; aborting.")

    ctxs = [4096, 8192] if args.smoke else MATRIX

    rows = []
    hdr = (f"{'ctx':>7} | {'torch_ms':>9} {'flash_ms':>9} {'qk_ms':>9} | "
           f"{'qk/flash%':>9} {'flash/torch%':>12}")
    print(hdr)
    for ctx in ctxs:
        d = _build(BATCH, ctx, dev, need_paged=True, need_contig=True)
        ms = {}
        for backend in ("torch", "flashmla", "qknorm"):
            fwd = make_forward(layer, backend, d)
            with torch.no_grad():
                if backend == "torch":
                    # paper baseline: torch.compile(reduce-overhead) (cudagraphs)
                    run = torch.compile(fwd, mode="reduce-overhead")
                    for _ in range(3):
                        run()             # warm/trace before timing
                else:
                    # custom kernel can't be compiled -> manual graph replay,
                    # equal launch-overhead treatment to backend 1.
                    run = _graphed(fwd)
                ms[backend] = triton.testing.do_bench(run)
        ov_qk = (ms["qknorm"] / ms["flashmla"] - 1.0) * 100.0
        ov_fl = (ms["flashmla"] / ms["torch"] - 1.0) * 100.0
        print(f"{ctx:>7} | {ms['torch']:>9.4f} {ms['flashmla']:>9.4f} "
              f"{ms['qknorm']:>9.4f} | {ov_qk:>9.3f} {ov_fl:>12.3f}")
        rows.append(dict(ctx=ctx, torch_ms=ms["torch"], flash_ms=ms["flashmla"],
                         qk_ms=ms["qknorm"], qk_over_flash_pct=ov_qk,
                         flash_over_torch_pct=ov_fl))
        del d
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        f.write("ctx,torch_ms,flash_ms,qk_ms,qk_over_flash_pct,flash_over_torch_pct\n")
        for r in rows:
            f.write(f"{r['ctx']},{r['torch_ms']:.6f},{r['flash_ms']:.6f},"
                    f"{r['qk_ms']:.6f},{r['qk_over_flash_pct']:.4f},"
                    f"{r['flash_over_torch_pct']:.4f}\n")

    ov = [r["qk_over_flash_pct"] for r in rows]
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True).strip().splitlines()[0]
    except Exception:
        gpu = torch.cuda.get_device_name(0)
    manifest = dict(
        task="qknorm-mla-fulllayer-decode-benchmark",
        lane="measurement-only",
        methodology="paper arXiv 2606.16310 v1 §5.3: full MLA layer, BF16, "
                    "decode step over prefilled cache, overhead over full-layer time",
        config=dict(dim=D_MODEL, q_lora=Q_LORA, kv_lora=KV_LORA, n_heads=N_HEADS,
                    nope=NOPE, rope=ROPE, v_head=V_HEAD, softmax_scale=SOFTMAX_SCALE,
                    dtype="bfloat16", batch=BATCH, tp=8),
        backends=dict(
            torch="pure-PyTorch absorbed decode, torch.compile(reduce-overhead)",
            flashmla="FlashMLA <T,false> dense kernel, CUDA-graph replay",
            qknorm="FlashMLA <T,true> + alpha^k RMS-stat write, CUDA-graph replay"),
        primary_metric="qk_over_flash_pct (isolates qk-norm integration; "
                       "counts none of FlashMLA baseline speedup as gain)",
        gate=dict(passed=gate_ok, max_abs_diff=gate_diff, rel=gate_rel, tol=2e-2),
        qk_over_flash_pct=dict(
            min=min(ov), max=max(ov), mean=sum(ov) / len(ov)),
        smoke=args.smoke, rows=rows, gpu=gpu,
        torch_version=torch.__version__, platform=platform.platform(),
        timestamp=datetime.now(timezone.utc).isoformat())
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] qk/flash overhead: min={min(ov):.3f}% "
          f"max={max(ov):.3f}% mean={sum(ov)/len(ov):.3f}%  ->  {args.manifest}")


if __name__ == "__main__":
    main()

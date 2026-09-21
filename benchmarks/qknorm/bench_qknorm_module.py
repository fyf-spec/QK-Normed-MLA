#!/usr/bin/env python3
"""Module-level MLA decode benchmark: original dense MLA vs QK-norm path.

Measurement-only. Toggles ONLY `qk_norm_scale_cache` on the exact same
FlashMLA dense-decode kernel and shapes:
  - None            -> original dense MLA (FlashMLA baseline path)
  - float32 tensor  -> QK-normed MLA path (reads per-token/head scale sidecar)

Reports ms / TFLOPS / GB-s in the SAME units as FlashMLA's own
benchmark/bench_flash_mla.py (identical FLOPS & bytes formulas), plus the
qk-norm overhead percentage per shape.

Run on an H200 node inside the megatron-h200-cu128-py312 venv:
  python bench_qknorm_module.py --out qknorm_module_perf.csv
"""
import argparse
import json
import math
import platform
import subprocess
from datetime import datetime, timezone

import torch
import triton
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

# QK-Normed MLA config for Qwen3.5-9B (matches the qk-normed test + checkpoint):
#   h_q = 24 query heads, h_kv = 1 (latent), d_content = 512, d_rope = 64
#   full head dim d = 512 + 64 = 576, value head dim dv = 512, page block = 64
H_Q, H_KV = 24, 1
D_CONTENT, D_ROPE = 512, 64
D = D_CONTENT + D_ROPE          # 576
DV = D_CONTENT                  # 512
BLOCK_SIZE = 64
S_Q = 1                         # single-token decode step
DTYPE = torch.bfloat16
CAUSAL = True

# Decode matrix from the formal run manifest:
#   4k:b1,4,16,32,64,128 ; 16k:b1,4,8,16,32 ; 32k:b1,4,8,16 ; 64k:b1,2,4,8
MATRIX = {
    4096:  [1, 4, 16, 32, 64, 128],
    16384: [1, 4, 8, 16, 32],
    32768: [1, 4, 8, 16],
    65536: [1, 2, 4, 8],
}


def _build_inputs(b, seqlen):
    max_seqlen_pad = triton.cdiv(seqlen, 256) * 256
    n_blocks = b * max_seqlen_pad // BLOCK_SIZE
    q = torch.randn(b, S_Q, H_Q, D)
    block_table = torch.arange(n_blocks, dtype=torch.int32).view(b, max_seqlen_pad // BLOCK_SIZE)
    blocked_k = torch.randn(n_blocks, BLOCK_SIZE, H_KV, D)
    cache_seqlens = torch.full((b,), seqlen, dtype=torch.int32)
    # alpha^k sidecar shaped (num_blocks, block_size), float32 -> qk-norm path.
    # Paper 2606.16310: per-token per-KV-group key scalar; MLA G=1 so one scalar
    # per token shared across all q-heads (T x 1, not T x 24). Token-contiguous
    # so the kernel coalesces adjacent token scales into one float2 load. The
    # per-head alpha^q is a projection-side fold, not part of the KV-cache cost.
    scale_cache = torch.ones(n_blocks, BLOCK_SIZE, dtype=torch.float32)
    return q, block_table, blocked_k, cache_seqlens, scale_cache


def _make(q, block_table, blocked_k, cache_seqlens, scale):
    # A fresh sched_meta is required per path: get_mla_metadata() returns an
    # empty meta whose config (including qk_normed) is frozen on the FIRST
    # flash_mla_with_kvcache call. Sharing one meta across the baseline
    # (scale=None) and qk-norm (scale=tensor) paths trips the
    # sched_meta.config.qk_normed assertion.
    meta, splits = get_mla_metadata(cache_seqlens, S_Q * H_Q // H_KV, H_KV)

    def run():
        return flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, DV,
            meta, splits, causal=CAUSAL, qk_norm_scale_cache=scale,
        )
    return run


def _bench_one(q, block_table, blocked_k, cache_seqlens, scale_cache):
    base_ms = triton.testing.do_bench(_make(q, block_table, blocked_k, cache_seqlens, None))
    qk_ms = triton.testing.do_bench(_make(q, block_table, blocked_k, cache_seqlens, scale_cache))
    return base_ms, qk_ms


def _perf(seqlen, b, ms):
    total = seqlen * b
    flops = S_Q * total * H_Q * (D + DV) * 2
    nbytes = (total * H_KV * D + b * S_Q * H_Q * D + b * S_Q * H_Q * DV) * (torch.finfo(DTYPE).bits // 8)
    return flops / 1e9 / ms, nbytes / 1e6 / ms  # TFLOPS, GB/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qknorm_module_perf.csv")
    ap.add_argument("--manifest", default="qknorm_module_manifest.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device visible"
    dev = torch.device("cuda:0")
    torch.set_default_dtype(DTYPE)
    torch.set_default_device(dev)
    torch.cuda.set_device(dev)
    torch.manual_seed(0)

    rows = []
    print(f"{'ctx':>6} {'b':>4} | {'base_ms':>9} {'qk_ms':>9} {'ovhd%':>7} | "
          f"{'base_GB/s':>10} {'qk_GB/s':>10} | {'base_TF':>8} {'qk_TF':>8}")
    for seqlen, batches in MATRIX.items():
        for b in batches:
            q, bt, bk, cs, sc = _build_inputs(b, seqlen)
            base_ms, qk_ms = _bench_one(q, bt, bk, cs, sc)
            base_tf, base_bw = _perf(seqlen, b, base_ms)
            qk_tf, qk_bw = _perf(seqlen, b, qk_ms)
            ovhd = (qk_ms / base_ms - 1.0) * 100.0
            print(f"{seqlen:>6} {b:>4} | {base_ms:>9.4f} {qk_ms:>9.4f} {ovhd:>7.2f} | "
                  f"{base_bw:>10.0f} {qk_bw:>10.0f} | {base_tf:>8.0f} {qk_tf:>8.0f}")
            rows.append(dict(ctx=seqlen, batch=b, base_ms=base_ms, qk_ms=qk_ms,
                             overhead_pct=ovhd, base_gbs=base_bw, qk_gbs=qk_bw,
                             base_tflops=base_tf, qk_tflops=qk_tf))

    with open(args.out, "w") as f:
        f.write("ctx,batch,base_ms,qk_ms,overhead_pct,base_gbs,qk_gbs,base_tflops,qk_tflops\n")
        for r in rows:
            f.write(f"{r['ctx']},{r['batch']},{r['base_ms']:.6f},{r['qk_ms']:.6f},"
                    f"{r['overhead_pct']:.4f},{r['base_gbs']:.2f},{r['qk_gbs']:.2f},"
                    f"{r['base_tflops']:.2f},{r['qk_tflops']:.2f}\n")

    ov = [r["overhead_pct"] for r in rows]
    geo_base_bw = math.exp(sum(math.log(r["base_gbs"]) for r in rows) / len(rows))
    geo_qk_bw = math.exp(sum(math.log(r["qk_gbs"]) for r in rows) / len(rows))
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True).strip().splitlines()[0]
    except Exception:
        gpu = torch.cuda.get_device_name(0)
    manifest = dict(
        task="qknorm-mla-module-decode-benchmark",
        lane="measurement-only",
        created=datetime.now(timezone.utc).isoformat(),
        gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda,
        host=platform.node(),
        flashmla_head="9241ae3ef9bac614dd25e45e507e089f888280e0",
        config=dict(h_q=H_Q, h_kv=H_KV, d_content=D_CONTENT, d_rope=D_ROPE,
                    dv=DV, block_size=BLOCK_SIZE, dtype="bf16", causal=CAUSAL),
        toggle="qk_norm_scale_cache: None=dense-MLA-baseline, ones=qk-norm-path",
        n_shapes=len(rows),
        overhead_pct_min=min(ov), overhead_pct_max=max(ov),
        overhead_pct_mean=sum(ov) / len(ov),
        geomean_base_gbs=geo_base_bw, geomean_qk_gbs=geo_qk_bw,
        rows=rows,
    )
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nqk-norm overhead over dense MLA baseline: "
          f"min={min(ov):.2f}%  mean={sum(ov)/len(ov):.2f}%  max={max(ov):.2f}%")
    print(f"geomean bandwidth: baseline={geo_base_bw:.0f} GB/s  qk-norm={geo_qk_bw:.0f} GB/s")
    print(f"wrote {args.out} and {args.manifest}")


if __name__ == "__main__":
    main()

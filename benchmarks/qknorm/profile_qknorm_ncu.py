#!/usr/bin/env python3
"""ncu profiling harness for the FlashMLA dense-decode qk-norm kernel.

Measurement-only. Profiles ONE code path per process (baseline `<T,false>` or
qk-norm `<T,true>` instance of flash_fwd_splitkv_mla_kernel) on ONE shape, so
Nsight Compute can attribute SpeedOfLight / Occupancy / LaunchStats /
MemoryWorkloadAnalysis cleanly to a single kernel launch.

Numerical integrity is verified BEFORE the profiled region (ncu replays the
kernel many times; we must not profile a numerically broken kernel):
  * all outputs finite,
  * qk-norm(scale=ones) == baseline(scale=None) within FlashMLA test tolerance
    (inverse-RMS=1 makes the sidecar an identity, so outputs must match),
  * qk-norm(random scale in [0.5,1.5]) finite.

The profiled launch is wrapped in cudaProfilerStart/Stop (torch.cuda.profiler)
so ncu is run with `--profile-from-start off`. Warmup runs stay outside it.

Usage (inside the H200 venv, under ncu):
  ncu --profile-from-start off --set ... -o report -f \
      python profile_qknorm_ncu.py --ctx 16384 --batch 8 --path qk
"""
import argparse

import torch
import triton
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

H_Q_DEFAULT, H_KV = 16, 1   # 16 = DeepSeek-V3 MLA local heads at TP=8 (paper cfg)
D_CONTENT, D_ROPE = 512, 64
D = D_CONTENT + D_ROPE          # 576
DV = D_CONTENT                  # 512
BLOCK_SIZE = 64
S_Q = 1
DTYPE = torch.bfloat16
CAUSAL = True

# FlashMLA qk-normed correctness tolerances (tests/test_flash_mla_qk_normed_*).
OUT_RTOL, OUT_ATOL = 2.01 / 128, 8e-4
LSE_RTOL, LSE_ATOL = 8.01 / 65536, 1e-6


def _build_inputs(b, seqlen, h_q):
    max_seqlen_pad = triton.cdiv(seqlen, 256) * 256
    n_blocks = b * max_seqlen_pad // BLOCK_SIZE
    q = torch.randn(b, S_Q, h_q, D)
    block_table = torch.arange(n_blocks, dtype=torch.int32).view(b, max_seqlen_pad // BLOCK_SIZE)
    blocked_k = torch.randn(n_blocks, BLOCK_SIZE, H_KV, D)
    cache_seqlens = torch.full((b,), seqlen, dtype=torch.int32)
    # Paper 2606.16310 alpha^k: per-token per-KV-group scalar; MLA G=1 -> one
    # scalar per token shared across all q-heads, [block, token] token-contiguous.
    ones = torch.ones(n_blocks, BLOCK_SIZE, dtype=torch.float32)
    rand = torch.empty(n_blocks, BLOCK_SIZE, dtype=torch.float32).uniform_(0.5, 1.5)
    return q, block_table, blocked_k, cache_seqlens, ones, rand


def _make(q, block_table, blocked_k, cache_seqlens, scale, h_q):
    # Fresh sched_meta per path: config (incl. qk_normed) is frozen on the first
    # flash_mla_with_kvcache call, so baseline and qk-norm need separate metas.
    meta, splits = get_mla_metadata(cache_seqlens, S_Q * h_q // H_KV, H_KV)

    def run():
        return flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, DV,
            meta, splits, causal=CAUSAL, qk_norm_scale_cache=scale,
        )
    return run


def _check_integrity(q, bt, bk, cs, ones, rand, h_q):
    """Verify finiteness + qk-norm(ones)==baseline before profiling."""
    base_out, base_lse = _make(q, bt, bk, cs, None, h_q)()
    ones_out, ones_lse = _make(q, bt, bk, cs, ones, h_q)()
    rand_out, rand_lse = _make(q, bt, bk, cs, rand, h_q)()
    torch.cuda.synchronize()
    for name, t in [("base_out", base_out), ("base_lse", base_lse),
                    ("ones_out", ones_out), ("rand_out", rand_out),
                    ("rand_lse", rand_lse)]:
        assert torch.isfinite(t).all(), f"non-finite in {name}"
    # scale=ones is an identity -> qk-norm path must equal baseline.
    torch.testing.assert_close(ones_out, base_out, rtol=OUT_RTOL, atol=OUT_ATOL)
    torch.testing.assert_close(ones_lse, base_lse, rtol=LSE_RTOL, atol=LSE_ATOL)
    print("INTEGRITY_OK finite=all qk_norm(ones)==baseline within tol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--path", choices=["base", "qk"], required=True)
    ap.add_argument("--heads", type=int, default=H_Q_DEFAULT,
                    help="local q-heads (default 16 = DeepSeek-V3 MLA @ TP=8)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--skip-integrity", action="store_true",
                    help="skip correctness check (only for isolating profile cost)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA device visible"
    dev = torch.device("cuda:0")
    torch.set_default_dtype(DTYPE)
    torch.set_default_device(dev)
    torch.cuda.set_device(dev)
    torch.manual_seed(0)

    q, bt, bk, cs, ones, rand = _build_inputs(args.batch, args.ctx, args.heads)
    print(f"shape ctx={args.ctx} batch={args.batch} heads={args.heads} "
          f"path={args.path} gpu={torch.cuda.get_device_name(0)}")

    if not args.skip_integrity:
        _check_integrity(q, bt, bk, cs, ones, rand, args.heads)

    scale = None if args.path == "base" else ones
    run = _make(q, bt, bk, cs, scale, args.heads)

    # Warmup OUTSIDE the profiled region.
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()

    # Profiled region: exactly one launch, captured via --profile-from-start off.
    torch.cuda.profiler.start()
    run()
    torch.cuda.profiler.stop()
    torch.cuda.synchronize()
    print("PROFILED_ONE_LAUNCH_DONE")


if __name__ == "__main__":
    main()

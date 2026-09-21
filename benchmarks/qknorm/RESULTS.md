# QK-Normed MLA on FlashMLA — Benchmark Results

Paper-faithful QK-Normalization ([arXiv 2606.16310](https://arxiv.org/abs/2606.16310))
integrated into FlashMLA's SM90 dense-decode kernel, and measured on **NVIDIA H200**.

All numbers are measurement-only (local manifests, no W&B). Baseline FlashMLA
commit `9241ae3`. GPU driver 595.71.05, torch 2.9.0+cu128, BF16.

## What "QK-norm integration" costs

QK-norm scales the **content** part of the QK^T score by a per-token per-KV-group
scalar `alpha^k` (RoPE dims left unscaled). The `<T,true>` kernel differs from the
baseline `<T,false>` kernel by **exactly** that `alpha^k` read + apply — so the delta
below is the isolated integration cost, counting **none** of FlashMLA's own speedup
as a gain.

Numerical integrity gate (both benches): with `alpha^k = ones`, qk-norm output is
**bit-exact** vs baseline (max_abs_diff = 0.0). Random scale stays finite.

## 1. Full MLA layer, three-way (paper §5.3 methodology)

DeepSeek-V3.2 MLA @ TP=8, batch=4, decode step over prefilled cache. Three backends,
identical projections, differ only in the attention core.
Data: [`results/fulllayer_manifest.json`](results/fulllayer_manifest.json)

| ctx | torch.compile (ms) | FlashMLA (ms) | qk-norm (ms) | qk-over-FlashMLA | FlashMLA-over-torch |
|------:|------:|------:|------:|------:|------:|
| 4096   | 0.1748 | 0.1420 | 0.1534 | +8.01% | −18.8% |
| 8192   | 0.3255 | 0.1489 | 0.1608 | +8.05% | −54.3% |
| 16384  | 0.5767 | 0.1623 | 0.1750 | +7.77% | −71.9% |
| 32768  | 1.0669 | 0.1785 | 0.1953 | +9.42% | −83.3% |
| 65536  | 2.0351 | 0.2144 | 0.2298 | +7.18% | −89.5% |
| 131072 | 3.9154 | 0.2827 | 0.3013 | +6.58% | −92.8% |
| 262144 | 13.1214 | 0.4168 | 0.4406 | +5.70% | −96.8% |

- **qk-norm over FlashMLA baseline:** min 5.70%, **mean 7.53%**, max 9.42%.
- **qk-norm over torch.compile baseline (paper's denominator):** 0.18–6.5%, **mean 2.19%, <2% for ctx ≥ 32k** — matches the paper's <2% claim.
- The same kernel looks like 7.5% or 2.2% depending purely on the baseline: the faster the MLA baseline, the larger a fixed µs cost reads as a percentage. FlashMLA itself is 18.8–96.8% faster than torch.compile.

## 2. Attention-core kernel, isolated (19 shapes)

`flash_mla_with_kvcache` only (tightest denominator).
Data: [`results/qknorm_module_manifest.json`](results/qknorm_module_manifest.json),
[`results/qknorm_module_perf.csv`](results/qknorm_module_perf.csv)

| version | overhead min | mean | max |
|---|--:|--:|--:|
| naive per-(token,head) T×24 fp32 sidecar | 1.01% | **16.34%** | 23.0% |
| **re-arch: per-KV-group rank-1 T×1** | −0.10% | **3.90%** | 6.61% |

The re-architecture removed a batch-dependent memory-bandwidth floor (the old
per-head sidecar read 24× redundant fp32), cutting mean overhead ~4×.

## 3. What did NOT help (recorded negatives)

- **`alpha^k` load-prefetch / dedup:** mean 3.90% → 4.00% (within noise). Register
  count identical → bottleneck is the GMMA drain ordering, not load latency.
- **RoPE-own-accumulator bubble fix:** register-bound. Giving the RoPE score tile its
  own accumulator in the steady-state pipeline pushes `<T,true>` to 255 regs +
  20 B spill (baseline 240, accepted 242). Spill in the decode-hot loop costs more
  than the bubble it removes. Reverted.
  Data: [`results/ncu_probe_manifest.json`](results/ncu_probe_manifest.json).

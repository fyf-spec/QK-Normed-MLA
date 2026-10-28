# XTuner QK-Normed MLA Decode Kernel 交付

## 范围

基于 FlashMLA `9241ae3`，实现 `e0a2c6d`，实验记录 `418c9c3`。支持 SM90 decode 的 rank-1 QK-Norm：每 token、每 KV-group 保存 FP32 `alpha_k`；单 KV-group sidecar 为 `[num_blocks,64]`。调用方将 `alpha_q` 折叠进 content query，RoPE 与 latent V 不缩放。

## 代码位置

- `flash_mla/flash_mla_interface.py`：Python 接口。
- `csrc/api/dense_decode.h`、`csrc/params.h`：校验和参数传递。
- `csrc/sm90/decode/dense/splitkv_mla.cuh`：8 个 content tile 累加后乘 `alpha_k`，再加 RoPE tile。
- `tests/test_flash_mla_qk_normed_dense_decoding.py`：正确性测试。
- `benchmarks/qknorm/`：H200/NCU 结果。

## 结果

H200 attention-core 中，naive `T×24` 平均开销 16.34%，rank-1 `T×1` 约 3.9%。RoPE 独立 accumulator 因 255 registers 和 spill 已回退。

验证：`python tests/test_flash_mla_qk_normed_dense_decoding.py`。

限制：Qwen 的 per-head `T×H` scale 必须保留 head 维，不能广播为 `T×1`。

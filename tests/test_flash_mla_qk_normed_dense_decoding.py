"""H200 correctness test for QK-Normed MLA, paper-faithful rank-1 form.

Paper arXiv 2606.16310 score-stage scaling:
    L^c_{h,t} = L^{c,raw}_{h,t} * alpha^q_{h} * alpha^k_{t,g(h)}
This is rank-1: a per-head QUERY scalar alpha^q times a per-token per-KV-group
KEY scalar alpha^k. MLA has G = num_heads_k = 1, so alpha^k is ONE scalar per
token, SHARED across all q-heads. The sidecar `qk_norm_scale_cache` carries only
alpha^k with shape (num_blocks, page_block_size) -- T x 1, not T x 24. The
per-head alpha^q is "absorbed into the query projection" (paper), i.e. folded by
the caller into the content dims of q before the kernel runs, so the kernel
applies only alpha^k. The RoPE dims and latent V are never scaled.

Two checks:
  (1) integrity gate  : alpha^k = ones AND alpha^q = ones  -> bit-exact vs the
                         baseline dense-MLA path (qk_norm_scale_cache=None).
  (2) equivalence test: random per-token alpha^k + random per-head alpha^q folded
                         into q_content -> matches an explicit reference that
                         applies the same rank-1 scaling.
"""

import math

import torch

import flash_mla


@torch.inference_mode()
def _decode(q, kv, block_table, lengths, content_dim, scale_cache):
    metadata, splits = flash_mla.get_mla_metadata()
    return flash_mla.flash_mla_with_kvcache(
        q, kv, block_table, lengths, content_dim, metadata, splits,
        causal=True, qk_norm_scale_cache=scale_cache,
    )


@torch.inference_mode()
def main() -> None:
    torch.set_default_device("cuda")
    torch.manual_seed(1234)
    batch, query_heads, content_dim, rope_dim, block_size = 2, 24, 512, 64, 64
    head_dim = content_dim + rope_dim
    lengths = torch.tensor([65, 191], dtype=torch.int32)
    max_blocks = 3
    block_table = torch.arange(batch * max_blocks, dtype=torch.int32).view(batch, max_blocks)
    kv = (torch.randn(batch * max_blocks, block_size, 1, head_dim) / 10).to(torch.bfloat16)
    q0 = (torch.randn(batch, 1, query_heads, head_dim) / 10).to(torch.bfloat16)

    # ---- (1) integrity gate: ones must be bit-exact with the baseline path ----
    out_base, lse_base = _decode(q0, kv, block_table, lengths, content_dim, None)
    ones_k = torch.ones(batch * max_blocks, block_size, dtype=torch.float32)
    out_ones, lse_ones = _decode(q0, kv, block_table, lengths, content_dim, ones_k)
    torch.testing.assert_close(out_ones, out_base, rtol=0, atol=0)
    torch.testing.assert_close(lse_ones, lse_base, rtol=0, atol=0)
    print("PASS integrity gate: alpha^k=ones + alpha^q=ones is bit-exact vs baseline")

    # ---- (2) equivalence test: random rank-1 alpha^q (per head) x alpha^k (per token) ----
    # alpha^k: one scalar per token, shared across all q-heads (T x 1 sidecar).
    alpha_k = torch.empty(batch * max_blocks, block_size, dtype=torch.float32).uniform_(0.5, 1.5)
    # alpha^q: one scalar per q-head, folded into q_content by the caller.
    alpha_q = torch.empty(query_heads, dtype=torch.float32).uniform_(0.5, 1.5)
    q = q0.clone().float()
    q[:, :, :, :content_dim] *= alpha_q[None, None, :, None]  # fold alpha^q into content dims only
    q = q.to(torch.bfloat16)

    out, lse = _decode(q, kv, block_table, lengths, content_dim, alpha_k)

    scale = 1.0 / math.sqrt(head_dim)
    out_ref = torch.empty_like(out)
    lse_ref = torch.empty_like(lse)
    for batch_idx, length in enumerate(lengths.cpu().tolist()):
        block_indices = block_table[batch_idx, : (length + block_size - 1) // block_size]
        dense_kv = kv[block_indices].view(-1, head_dim)[:length].float()
        # alpha^k per token, shared across heads -> broadcast over the head (row) axis.
        dense_scale = alpha_k[block_indices].reshape(-1)[:length]  # (length,)
        query = q[batch_idx, 0].float()  # alpha^q already folded into content dims
        logits = (
            (query[:, :content_dim] @ dense_kv[:, :content_dim].T) * dense_scale[None, :]
            + query[:, content_dim:] @ dense_kv[:, content_dim:].T
        ) * scale
        probability = torch.softmax(logits, dim=-1)
        out_ref[batch_idx, 0] = (probability @ dense_kv[:, :content_dim]).to(torch.bfloat16)
        lse_ref[batch_idx, :, 0] = torch.logsumexp(logits, dim=-1)

    torch.testing.assert_close(out, out_ref, rtol=2.01 / 128, atol=8e-4)
    torch.testing.assert_close(lse, lse_ref, rtol=8.01 / 65536, atol=1e-6)
    print("PASS equivalence: rank-1 alpha^q(per-head) x alpha^k(per-token, T x 1), h_q=24")


if __name__ == "__main__":
    main()

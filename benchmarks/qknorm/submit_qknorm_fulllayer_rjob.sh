#!/usr/bin/env bash
# Full-MLA-layer decode benchmark on H200 (paper arXiv 2606.16310 §5.3 method).
# Runs bench_qknorm_fulllayer.py: 3 backends (torch.compile / FlashMLA /
# qk-normed-FlashMLA) over a DeepSeek-V3-width MLA LAYER, overhead reported over
# full-layer time. NO kernel rebuild this run -- the FlashMLA .so is already
# built + gated PASS (prior build, persisted inplace on the shared mount). We
# verify the import, then bench. CUDA_HOME is still assembled because
# torch.compile(reduce-overhead) drives Triton/inductor codegen (needs ptxas +
# a complete toolchain) at runtime.
set -Eeuo pipefail

ROOT=/mnt/shared-storage-gpfs2/plm-gpfs/yffeng
ENV_PREFIX="$ROOT/megatron-h200-cu128-py312"
FLASHMLA_ROOT="$ROOT/FlashMLA"
WORK="$ROOT/mla-kernel-opt-20260908"

source "$ROOT/.bashrc" 2>/dev/null || true
proxy_off 2>/dev/null || true
export KUBEBRAIN_CLUSTER_ENTRY=http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451

STAMP="$(date +%m%d-%H%M%S)"
JOB_NAME="${JOB_NAME:-yffeng-qknorm-fulllayer-${STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/results/fulllayer-${STAMP}}"
mkdir -p "$OUTPUT_DIR"
SMOKE="${SMOKE:-1}"

rjob submit \
  --name "$JOB_NAME" \
  --metadata-name "$JOB_NAME" \
  --namespace ailab-llmarchitecture \
  --task-type normal \
  --charged-group llmarchitecture_gpu \
  --private-machine group \
  --priority "${PRIORITY:-5}" \
  --image registry.h.pjlab.org.cn/ailab/ml-base:22.04-pjlab-20260323 \
  --image-pull-policy IfNotPresent \
  --cpu 16 \
  --gpu 1 \
  --memory 131072 \
  --mount gpfs://gpfs2/plm-gpfs:/mnt/shared-storage-gpfs2/plm-gpfs \
  -- bash -c "
    set -eo pipefail
    export PATH='$ENV_PREFIX/bin':\$PATH
    export PYTHONPATH='$FLASHMLA_ROOT'
    NV='$ENV_PREFIX/lib/python3.12/site-packages/nvidia'
    set +e
    is_complete() { [ -e \"\$1/include/curand_kernel.h\" ] && [ -e \"\$1/include/cuda_runtime_api.h\" ] && [ -x \"\$1/bin/nvcc\" ]; }
    CUDA_HOME=\"\"
    for cand in /usr/local/cuda; do
      is_complete \"\$cand\" && { CUDA_HOME=\"\$cand\"; echo \"[cuda] using existing \$cand\"; break; }
    done
    if [ -z \"\$CUDA_HOME\" ]; then
      CUDA_HOME=\$(mktemp -d /tmp/cudahome.XXXXXX)
      mkdir -p \"\$CUDA_HOME/bin\" \"\$CUDA_HOME/include\" \"\$CUDA_HOME/lib64\"
      for t in nvcc nvcc.profile ptxas cudafe++ fatbinary nvlink bin2c cu++filt nvprune nvdisasm __nvcc_device_query; do
        [ -e '$ENV_PREFIX/bin'/\"\$t\" ] && ln -sf '$ENV_PREFIX/bin'/\"\$t\" \"\$CUDA_HOME/bin/\$t\"
      done
      ln -sfn '$ENV_PREFIX/nvvm' \"\$CUDA_HOME/nvvm\"
      cp -srf '$ENV_PREFIX/targets/x86_64-linux/include/.' \"\$CUDA_HOME/include/\" 2>/dev/null
      cp -srf '$ENV_PREFIX/targets/x86_64-linux/lib/.'     \"\$CUDA_HOME/lib64/\"   2>/dev/null
      for d in \"\$NV\"/*/include; do [ -d \"\$d\" ] && cp -srf \"\$d/.\" \"\$CUDA_HOME/include/\" 2>/dev/null; done
      for d in \"\$NV\"/*/lib;     do for f in \"\$d\"/*.so*; do [ -e \"\$f\" ] && cp -sf \"\$f\" \"\$CUDA_HOME/lib64/\" 2>/dev/null; done; done
      echo \"[cuda] assembled CUDA_HOME=\$CUDA_HOME\"
    fi
    export CUDA_HOME
    set -e
    export PATH=\"\$CUDA_HOME/bin\":\$PATH
    export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64\":'$ENV_PREFIX/lib':\${LD_LIBRARY_PATH:-}
    echo \"[host] \$(hostname)\"; nvidia-smi -L || true
    echo '=== VERIFY FlashMLA import (no rebuild) ==='
    ls -la '$FLASHMLA_ROOT'/flash_mla/cuda*.so
    '$ENV_PREFIX/bin/python' -c \"import flash_mla; from flash_mla import flash_mla_with_kvcache, get_mla_metadata; print('flash_mla OK')\"
    cd '$WORK'
    if [[ '$SMOKE' == '1' ]]; then
      echo '=== SMOKE (gate + 2 ctx points) ==='
      '$ENV_PREFIX/bin/python' bench_qknorm_fulllayer.py --smoke \
        --out '$OUTPUT_DIR/fulllayer_smoke_perf.csv' \
        --manifest '$OUTPUT_DIR/fulllayer_smoke_manifest.json' 2>&1 | tee '$OUTPUT_DIR/bench.txt'
    else
      echo '=== FORMAL (gate + full ctx sweep 4k..256k) ==='
      '$ENV_PREFIX/bin/python' bench_qknorm_fulllayer.py \
        --out '$OUTPUT_DIR/fulllayer_perf.csv' \
        --manifest '$OUTPUT_DIR/fulllayer_manifest.json' 2>&1 | tee '$OUTPUT_DIR/bench.txt'
    fi
    echo DONE_QKNORM_FULLLAYER
  " 2>&1 | tee "$OUTPUT_DIR/submit.log"

printf 'job_name=%s\n' "$JOB_NAME"
printf 'output_dir=%s\n' "$OUTPUT_DIR"
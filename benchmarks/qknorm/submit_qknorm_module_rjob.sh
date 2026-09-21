#!/usr/bin/env bash
# Submit an rjob IDLE task that benchmarks the MLA decode module:
#   original dense MLA (qk_norm_scale_cache=None) vs qk-norm path (float32 sidecar).
# Uses the concept_olmo repo's rjob config (namespace/group/image/mount) and the
# yffeng megatron-h200-cu128-py312 env + FlashMLA build.
set -Eeuo pipefail

ROOT=/mnt/shared-storage-gpfs2/plm-gpfs/yffeng
ENV_PREFIX="$ROOT/megatron-h200-cu128-py312"
FLASHMLA_ROOT="$ROOT/FlashMLA"
WORK="$ROOT/mla-kernel-opt-20260908"

# H-cluster control-plane routing (h-cluster-rjob-rlaunch skill): internal
# scheduler proxy, external proxy OFF for internal endpoints.
source "$ROOT/.bashrc" 2>/dev/null || true
proxy_off 2>/dev/null || true
export KUBEBRAIN_CLUSTER_ENTRY=http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451

STAMP="$(date +%m%d-%H%M%S)"
JOB_NAME="${JOB_NAME:-yffeng-qknorm-mla-module-bench-${STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/results/qknorm-module-${STAMP}}"
mkdir -p "$OUTPUT_DIR"

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
    export CUDA_HOME='$ENV_PREFIX'
    export LD_LIBRARY_PATH='$ENV_PREFIX/lib':\${LD_LIBRARY_PATH:-}
    cd '$WORK'
    echo \"[host] \$(hostname)\"; nvidia-smi -L || true
    '$ENV_PREFIX/bin/python' bench_qknorm_module.py \
      --out '$OUTPUT_DIR/qknorm_module_perf.csv' \
      --manifest '$OUTPUT_DIR/qknorm_module_manifest.json' \
      2>&1 | tee '$OUTPUT_DIR/job.log'
    echo DONE_QKNORM_MODULE_BENCH
  "

printf 'job_name=%s\n' "$JOB_NAME"
printf 'output_dir=%s\n' "$OUTPUT_DIR"

#!/usr/bin/env bash
# Rebuild FlashMLA on H200, then (1) run the qk-normed correctness test
# (integrity gate: alpha^k=ones + alpha^q=ones bit-exact vs baseline; plus the
# paper's rank-1 equivalence: per-token alpha^k [T x 1, shared across heads] x
# per-head alpha^q folded into q_content) and (2) run the decode overhead bench.
# Paper-faithful re-arch to per-KV-group scalar (arXiv 2606.16310), replacing the
# earlier T x 24 per-head sidecar. nvcc build-temp -> node-local /tmp (shared
# mount is ~full);
# only the final .so overwrites in place on the shared mount.
set -Eeuo pipefail

ROOT=/mnt/shared-storage-gpfs2/plm-gpfs/yffeng
ENV_PREFIX="$ROOT/megatron-h200-cu128-py312"
FLASHMLA_ROOT="$ROOT/FlashMLA"
WORK="$ROOT/mla-kernel-opt-20260908"

source "$ROOT/.bashrc" 2>/dev/null || true
proxy_off 2>/dev/null || true
export KUBEBRAIN_CLUSTER_ENTRY=http://wangyixiuan-cpu.linzhouhan.ailab-llmarchitecture.svc.pjlab.local:11451

STAMP="$(date +%m%d-%H%M%S)"
JOB_NAME="${JOB_NAME:-yffeng-qknorm-build-bench-${STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORK/results/build-${STAMP}}"
mkdir -p "$OUTPUT_DIR"
SMOKE="${SMOKE:-0}"

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
    # torch adds \$CUDA_HOME/include to the host compile -I, so CUDA_HOME must be
    # a COMPLETE toolkit root (must contain curand_kernel.h + cuda_runtime_api.h).
    # Prefer a pre-existing system toolkit; else assemble one from the pip cu128
    # env whose headers are split across targets/ and site-packages/nvidia/*.
    # NOTE: errexit is OFF for this whole block -- cp over dirs with no .so match
    # returns nonzero and would kill the job under set -e before any output.
    NV='$ENV_PREFIX/lib/python3.12/site-packages/nvidia'
    set +e
    is_complete() { [ -e \"\$1/include/curand_kernel.h\" ] && [ -e \"\$1/include/cuda_runtime_api.h\" ] && [ -x \"\$1/bin/nvcc\" ]; }
    CUDA_HOME=\"\"
    for cand in /usr/local/cuda \"\$CUDA_HOME\"; do
      [ -n \"\$cand\" ] && is_complete \"\$cand\" && { CUDA_HOME=\"\$cand\"; echo \"[cuda] using existing \$cand\"; break; }
    done
    if [ -z \"\$CUDA_HOME\" ]; then
      CUDA_HOME=\$(mktemp -d /tmp/cudahome.XXXXXX)
      mkdir -p \"\$CUDA_HOME/bin\" \"\$CUDA_HOME/include\" \"\$CUDA_HOME/lib64\"
      # nvcc reads nvcc.profile relative to its own dir and needs the WHOLE
      # toolchain (ptxas, cudafe++, fatbinary, nvlink) + nvvm/bin/cicc, else
      # \"sh: cicc: not found\" kills every TU. Symlink them all in.
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
    for h in cuda_runtime_api.h curand_kernel.h crt/host_config.h; do
      [ -e \"\$CUDA_HOME/include/\$h\" ] || { echo \"FATAL: CUDA_HOME=\$CUDA_HOME missing include/\$h\"; exit 3; }
    done
    for t in bin/ptxas nvvm/bin/cicc; do
      [ -e \"\$CUDA_HOME/\$t\" ] || { echo \"FATAL: CUDA_HOME=\$CUDA_HOME missing \$t (nvcc toolchain incomplete)\"; exit 3; }
    done
    set -e
    echo \"[cuda] CUDA_HOME=\$CUDA_HOME\"; \"\$CUDA_HOME/bin/nvcc\" --version | tail -1
    export PATH=\"\$CUDA_HOME/bin\":\$PATH
    export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib64\":'$ENV_PREFIX/lib':\${LD_LIBRARY_PATH:-}
    export MAX_JOBS=8 NVCC_THREADS=8   # heavy CUTLASS TUs; cap parallelism to avoid OOM
    export FLASH_MLA_DISABLE_SM100=1   # env has NVCC 12.8; we only target sm_90a (H200)
    echo \"[host] \$(hostname)\"; nvidia-smi -L || true
    git config --global --add safe.directory '$FLASHMLA_ROOT' 2>/dev/null || true
    cd '$FLASHMLA_ROOT'
    echo '=== REBUILD (build-temp -> /tmp, inplace .so) ==='
    BT=\$(mktemp -d /tmp/fmla_build.XXXXXX)
    set +e
    '$ENV_PREFIX/bin/python' setup.py build_ext --inplace --build-temp \"\$BT\" > '$OUTPUT_DIR/build.log' 2>&1
    BUILD_RC=\$?
    set -e
    echo \"[build rc=\$BUILD_RC] --- ninja errors ---\"
    grep -nE 'error:|FAILED:|Error [0-9]|ninja: build stopped|No space|Killed' '$OUTPUT_DIR/build.log' | head -30 || true
    echo '--- last 12 lines of build.log ---'; tail -12 '$OUTPUT_DIR/build.log'
    rm -rf \"\$BT\"
    [ \$BUILD_RC -eq 0 ] || { echo BUILD_FAILED_RC=\$BUILD_RC; exit \$BUILD_RC; }
    echo '=== SO built ==='; ls -la flash_mla/cuda*.so
    echo '=== CORRECTNESS (integrity gate: random per-head scale + ones==base) ==='
    '$ENV_PREFIX/bin/python' tests/test_flash_mla_qk_normed_dense_decoding.py 2>&1 | tail -20
    if [[ '$SMOKE' != '1' ]]; then
      echo '=== FULL DECODE OVERHEAD BENCH ==='
      cd '$WORK'
      '$ENV_PREFIX/bin/python' bench_qknorm_module.py 2>&1 | tee '$OUTPUT_DIR/bench.txt'
    fi
    echo DONE_QKNORM_BUILD_BENCH
  " 2>&1 | tee "$OUTPUT_DIR/submit.log"

printf 'job_name=%s\n' "$JOB_NAME"
printf 'output_dir=%s\n' "$OUTPUT_DIR"

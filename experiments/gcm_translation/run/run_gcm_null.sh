#!/bin/bash -l

#$ -N gcm_null
#$ -l h_rt=2:00:00
#$ -l gpus=1
#$ -l gpu_memory=80G
#$ -l gpu_type=A100
#$ -m ea
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing/experiments/gcm_translation/run/$JOB_NAME.$TASK_ID.out
#$ -V

# Phase 2 null sweep:
#   - 56 cross-language directions (matching Phase 1 grid) +
#   - 8 same-language directions (eng->eng, spa->spa, ...) for the
#     content-discrimination floor.
# Total: 64 tasks.
#
# Submit with: qsub -t 1-64 experiments/gcm_translation/run/run_gcm_null.sh
#
# NB: do NOT add a `-q <buy-in-queue>` line — combining `-q` with `-l gpu_memory`
# silently routes onto undersized cards (Phase 1 OOM lesson; see CLAUDE.md).
#
# We pin `gpu_type=A100` (in practice A100-80G after the gpu_memory=80G filter)
# because the conda env's PyTorch lacks sm_120 kernels, so RTX Pro 6000
# (Blackwell, gpu_compute_capability=12.0) hosts like scc-b04 produce
# "no kernel image is available for execution on the device". `gpu_compute_
# capability` is treated as a minimum by SGE, so it can't cap from above.

set -euo pipefail

module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
# expandable_segments reduces fragmentation; +1-2 GB effective headroom
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda activate probes

cd /projectnb/mcnet/jbrin/lang-probing

# Build the directions list:
#   - 56 cross-lang ordered pairs (8 langs all-vs-all, src != tgt)
#   - 8 same-lang pairs (src == tgt) appended at the end
LANGS=(English Spanish German French Turkish Arabic Hindi Hebrew)
DIRECTIONS=()
for src in "${LANGS[@]}"; do
    for tgt in "${LANGS[@]}"; do
        if [ "$src" != "$tgt" ]; then
            DIRECTIONS+=("${src}|${tgt}")
        fi
    done
done
for src in "${LANGS[@]}"; do
    DIRECTIONS+=("${src}|${src}")
done

# SGE_TASK_ID is 1-indexed
TASK_INDEX=$((SGE_TASK_ID - 1))
DIR="${DIRECTIONS[$TASK_INDEX]}"
SRC_LANG="${DIR%%|*}"
TGT_LANG="${DIR##*|}"

echo "[$JOB_NAME.$SGE_TASK_ID] host=$HOSTNAME  null direction=${SRC_LANG} -> ${TGT_LANG}"
nvidia-smi -L || true

python experiments/gcm_translation/run.py \
    --src_lang "$SRC_LANG" \
    --tgt_lang "$TGT_LANG" \
    --n_pairs 100 \
    --seed 42 \
    --components both \
    --null_control \
    --output_dir outputs/gcm_translation_null

echo "[$JOB_NAME.$SGE_TASK_ID] done"

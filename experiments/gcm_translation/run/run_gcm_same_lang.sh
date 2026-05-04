#!/bin/bash -l

#$ -N gcm_same_lang
#$ -l h_rt=2:00:00
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -l gpu_c=8.0
#$ -m ea
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing/experiments/gcm_translation/run/$JOB_NAME.$TASK_ID.out
#$ -V

# Phase 1 sanity-check addendum: 8 same-language directions (eng->eng, etc.)
# in REAL (anchored, gold-scored) mode. Combined with the null same-lang from
# Phase 2, this gives the second leg of the four-quadrant decomposition:
#   real_cross - real_same = translation-circuit beyond monolingual completion
#   real_same - null_same  = monolingual identity-completion circuit
#
# Submit with: qsub -t 1-8 experiments/gcm_translation/run/run_gcm_same_lang.sh
#
# NB: same-lang sequences are shorter than cross-lang non-Eng pairs (src tokens
# align tgt tokens), so 48G is plenty. No -q line per CLAUDE.md.

set -euo pipefail

if [ -z "${SGE_TASK_ID:-}" ]; then
    echo "ERROR: SGE_TASK_ID is unset (run via 'qsub -t 1-8')" >&2
    exit 1
fi

module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda activate probes

cd /projectnb/mcnet/jbrin/lang-probing

LANGS=(English Spanish German French Turkish Arabic Hindi Hebrew)
TASK_INDEX=$((SGE_TASK_ID - 1))
if [ "$TASK_INDEX" -lt 0 ] || [ "$TASK_INDEX" -ge "${#LANGS[@]}" ]; then
    echo "ERROR: TASK_INDEX $TASK_INDEX out of range [0, ${#LANGS[@]})" >&2
    exit 1
fi
LANG="${LANGS[$TASK_INDEX]}"

echo "[$JOB_NAME.$SGE_TASK_ID] host=$HOSTNAME  same-lang real direction=${LANG} -> ${LANG}"
nvidia-smi -L || true

python experiments/gcm_translation/run.py \
    --src_lang "$LANG" \
    --tgt_lang "$LANG" \
    --n_pairs 100 \
    --seed 42 \
    --components both \
    --output_dir outputs/gcm_translation

echo "[$JOB_NAME.$SGE_TASK_ID] done"

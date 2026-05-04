#!/bin/bash -l

#$ -N gcm_sweep
#$ -l h_rt=2:00:00
#$ -l gpus=1
#$ -l gpu_memory=64G
#$ -q a100,ece-pub,joshigroup-gpu-pub,chapmangroup-gpu-pub,labcigroup-gpu-pub,cds-gpu-pub,thinfilament-gpu-pub,h200
#$ -m ea
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing/experiments/gcm_translation/run/$JOB_NAME.$TASK_ID.out
#$ -V

# Submit with: qsub -t 1-56 experiments/gcm_translation/run/run_gcm_sweep.sh

set -euo pipefail

module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
# expandable_segments reduces fragmentation; +1-2 GB effective headroom on 44 GB GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda activate probes

cd /projectnb/mcnet/jbrin/lang-probing

# Build the directions list: 8 langs all-vs-all (excluding identity) = 56 ordered pairs.
LANGS=(English Spanish German French Turkish Arabic Hindi Hebrew)
DIRECTIONS=()
for src in "${LANGS[@]}"; do
    for tgt in "${LANGS[@]}"; do
        if [ "$src" != "$tgt" ]; then
            DIRECTIONS+=("${src}|${tgt}")
        fi
    done
done

# SGE_TASK_ID is 1-indexed
TASK_INDEX=$((SGE_TASK_ID - 1))
DIR="${DIRECTIONS[$TASK_INDEX]}"
SRC_LANG="${DIR%%|*}"
TGT_LANG="${DIR##*|}"

echo "[$JOB_NAME.$SGE_TASK_ID] host=$HOSTNAME  direction=${SRC_LANG} -> ${TGT_LANG}"
nvidia-smi -L || true

python experiments/gcm_translation/run.py \
    --src_lang "$SRC_LANG" \
    --tgt_lang "$TGT_LANG" \
    --n_pairs 100 \
    --seed 42 \
    --components both \
    --output_dir outputs/gcm_translation

echo "[$JOB_NAME.$SGE_TASK_ID] done"

#!/bin/bash -l

#$ -N gcm_null_smoke
#$ -l h_rt=1:00:00
#$ -l gpus=1
#$ -l gpu_memory=48G
#$ -l gpu_c=8.0
#$ -m ea
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing/experiments/gcm_translation/run/$JOB_NAME.out
#$ -V

set -euo pipefail

module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda activate probes

cd /projectnb/mcnet/jbrin/lang-probing

echo "[$JOB_NAME] host=$HOSTNAME"
nvidia-smi -L || true

# 5-pair null on eng->spa: cross-lang
python experiments/gcm_translation/run.py \
    --src_lang English --tgt_lang Spanish \
    --n_pairs 5 --seed 42 --components both \
    --null_control \
    --output_dir outputs/gcm_translation_null_smoke

# 5-pair null on eng->eng: same-lang, exercises src==tgt branch
python experiments/gcm_translation/run.py \
    --src_lang English --tgt_lang English \
    --n_pairs 5 --seed 42 --components both \
    --null_control \
    --output_dir outputs/gcm_translation_null_smoke

echo "[$JOB_NAME] done"

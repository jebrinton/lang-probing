#!/bin/bash -l

#$ -l h_rt=2:00:00
#$ -l gpus=1
#$ -l gpu_c=8.9
#$ -l gpu_memory=32G
#$ -m ea
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing/experiments/counterfactual_attribution/run/$JOB_NAME.out
#$ -V

# LANG_CODE and PAIRS_FILE passed via -v (qsub -v LANG_CODE=fra,PAIRS_FILE=data/multilingual_pairs/fra.json)

set -euo pipefail

module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
conda activate probes

cd /projectnb/mcnet/jbrin/lang-probing

echo "[$JOB_NAME] host=$HOSTNAME lang=$LANG_CODE pairs=$PAIRS_FILE"
nvidia-smi -L || true
nvidia-smi --query-gpu=memory.free --format=csv || true

python experiments/counterfactual_attribution/attribute_multilingual.py \
    --pairs_file "$PAIRS_FILE" \
    --lang_code "$LANG_CODE" \
    --output_dir outputs/counterfactual_attribution/attribution \
    --max_pairs_per_cell 300 \
    --top_k 50 \
    --holdout_frac 0.2 \
    --seed 42

echo "[$JOB_NAME] done"


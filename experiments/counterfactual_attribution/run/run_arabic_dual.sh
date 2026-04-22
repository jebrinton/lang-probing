#!/bin/bash -l

#$ -l h_rt=2:00:00
#$ -l gpus=1
#$ -l gpu_c=8.9
#$ -l gpu_memory=32G
#$ -m ea
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing/experiments/counterfactual_attribution/run/$JOB_NAME.out
#$ -V

set -euo pipefail
module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
conda activate probes
cd /projectnb/mcnet/jbrin/lang-probing

echo "[$JOB_NAME] host=$HOSTNAME"
python experiments/counterfactual_attribution/analyze_arabic_dual_english.py
echo "[$JOB_NAME] done"

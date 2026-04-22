#!/bin/bash -l

#$ -l h_rt=2:00:00
#$ -l gpus=1
#$ -l gpu_c=8.9
#$ -l gpu_memory=32G
#$ -m ea
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing-overnight/run/$JOB_NAME.out
#$ -V

# WHICH=arabic_dual | ablate_validate
set -euo pipefail
module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
conda activate probes
cd /projectnb/mcnet/jbrin/lang-probing-overnight

echo "[$JOB_NAME] host=$HOSTNAME which=$WHICH"
case "$WHICH" in
  arabic_dual)
    python scripts/analyze_arabic_dual_english.py
    ;;
  ablate_validate)
    python scripts/ablate_validate.py
    ;;
  *)
    echo "unknown WHICH"; exit 1 ;;
esac
echo "[$JOB_NAME] done"

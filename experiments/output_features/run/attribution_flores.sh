#!/bin/bash -l

# Request GPU
#$ -l gpus=1
#$ -l gpu_memory=32G
#$ -l gpu_c=8.9

#$ -l h_rt=3:00:00

#$ -m ea
#$ -N attribution_flores
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing/run/$JOB_NAME.out
#$ -V

module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
conda activate probes

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "HF_HOME: $HF_HOME"
echo "CONDA_ENV: $CONDA_DEFAULT_ENV"
echo "=========================================================="

cd /projectnb/mcnet/jbrin/lang-probing
python experiments/output_features/run.py

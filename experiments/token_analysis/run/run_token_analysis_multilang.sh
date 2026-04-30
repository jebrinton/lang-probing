#!/bin/bash -l

# Match `qrjb 2` resources: 1 GPU with compute capability >= 8.9, 2 hour walltime.
#$ -l gpus=1
#$ -l gpu_memory=32G
#$ -l gpu_c=8.9
#$ -l h_rt=2:00:00

#$ -m ea
#$ -N token_analysis_multilang
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

CONFIG="experiments/token_analysis/configs/multilang.yaml"
OUTPUT_DIR="outputs/token_analysis/multilang"
HTML_DIR="outputs/token_analysis/html/multilang"

echo "Config: $CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "HTML dir: $HTML_DIR"
echo "=========================================================="

# Stage 1: GPU — per-token SAE activations + logprob deltas -> JSON
python experiments/token_analysis/run.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"

echo "Stage 1 complete. JSON cache in $OUTPUT_DIR"
echo "=========================================================="

# Stage 2: CPU — render self-contained HTML dashboards
python experiments/token_analysis/visualize.py \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$HTML_DIR"

echo "Stage 2 complete. HTML dashboards in $HTML_DIR"
echo "=========================================================="
echo "End date : $(date)"

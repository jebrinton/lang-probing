#!/bin/bash -l

# Request GPU
#$ -l gpus=1
#$ -l gpu_memory=32G

# Specify the minimum GPU compute capability.
#$ -l gpu_c=8.9
# Note for future: we have some H200s with compute capability 9.0 and L40s with compute capability 8.9

#$ -l h_rt=3:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N token_analysis

# Combine output and error files into a single file
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing/run/$JOB_NAME.out

# Export all environment variables
#$ -V

module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
conda activate probes

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "HF_HOME: $HF_HOME"
echo "CONDA_ENV: $CONDA_DEFAULT_ENV"
echo "=========================================================="

cd /projectnb/mcnet/jbrin/lang-probing

CONFIG=${CONFIG:-experiments/token_analysis/configs/example.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/token_analysis}
HTML_DIR=${HTML_DIR:-outputs/token_analysis/html}

echo "Config: $CONFIG"
echo "Output dir: $OUTPUT_DIR"
echo "HTML dir: $HTML_DIR"
echo "=========================================================="

# Stage 1: GPU inference — collect per-token SAE activations and logprob deltas
python experiments/token_analysis/run.py \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"

echo "Stage 1 complete. JSON cache in $OUTPUT_DIR"
echo "=========================================================="

# Stage 2: HTML generation — no GPU needed
python experiments/token_analysis/visualize.py \
    --input_dir "$OUTPUT_DIR" \
    --output_dir "$HTML_DIR"

echo "Stage 2 complete. HTML dashboards in $HTML_DIR"
echo "=========================================================="
echo "End date : $(date)"

#!/bin/bash -l

# Request GPU
#$ -l gpus=1
#$ -l gpu_memory=32G

# Specify the minimum GPU compute capability.
#$ -l gpu_c=8.9

#$ -l h_rt=1:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N cf_attr

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

DATA_FILE=${DATA_FILE:-data/grammatical_pairs.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/counterfactual_attribution}

echo "Data file: $DATA_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================================="

# Stage 1: GPU inference — gradient-based SAE feature attribution
python experiments/counterfactual_attribution/run.py \
    --data_file "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --save_raw_tensors

echo "Stage 1 complete. Results in $OUTPUT_DIR"
echo "=========================================================="

# Stage 2: Analysis — no GPU needed
python experiments/counterfactual_attribution/analyze.py \
    --results_dir "$OUTPUT_DIR"

echo "Stage 2 complete."
echo "=========================================================="
echo "End date : $(date)"

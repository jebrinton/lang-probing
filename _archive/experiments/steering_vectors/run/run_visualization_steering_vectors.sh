#!/bin/bash -l

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_memory=24

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.6

#$ -l h_rt=06:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N nova_visualization_layers_4

# Combine output and error files into a single file
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing/run/$JOB_NAME.out

# Export all environment variables
#$ -V

module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
conda activate urop-env

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "HF_HOME: $HF_HOME"
echo "CONDA_ENV: $CONDA_DEFAULT_ENV"
echo "=========================================================="

cd /projectnb/mcnet/jbrin/lang-probing
# loop through all layer numbers from 0 to 31, every 4 layers
for layer in {0..31..4}; do
    python scripts/visualize_steering_vectors.py --concept_key Number --concept_value Plur --layer $layer --generate_heatmap
    python scripts/visualize_steering_vectors.py --concept_key Number --concept_value Sing --layer $layer --generate_heatmap
    python scripts/visualize_steering_vectors.py --concept_key Tense  --concept_value Past --layer $layer --generate_heatmap
    python scripts/visualize_steering_vectors.py --concept_key Tense  --concept_value Pres --layer $layer --generate_heatmap
done
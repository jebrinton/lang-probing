#!/bin/bash -l

# Request GPU
#$ -l gpus=1
#$ -l gpu_memory=24G

# Specify the minimum GPU compute capability. 
#$ -l gpu_c=8.9
# Note for future: we have some H200s with compute capability 9.0 and L40s with compute capability 8.9

#$ -l h_rt=12:00:00

# Send an email when the job finishes or if it is aborted (by default no email is sent).
#$ -m ea

# Give job a name
#$ -N word_probes_layer16

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
python experiments/probes/run.py --concepts Tense Number --languages English French German Spanish Turkish Arabic Hindi Hebrew Chinese Indonesian --layers 16 --max_samples 1024

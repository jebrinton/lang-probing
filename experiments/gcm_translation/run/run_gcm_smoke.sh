#!/bin/bash -l

#$ -N gcm_smoke
#$ -l h_rt=1:00:00
#$ -l gpus=1
#$ -l gpu_memory=24G
#$ -q a40,a100,l40s,ece-pub,neuro-autonomy-pub,iris-gpu-pub,ivcbuyin-pub,li-rbsp-gpu-pub,joshigroup-gpu-pub,chapmangroup-gpu-pub,labcigroup-gpu-pub,batcomputer-pub,csgpu-pub,cds-gpu-pub,thinfilament-gpu-pub
#$ -m ea
#$ -j y
#$ -o /projectnb/mcnet/jbrin/lang-probing/experiments/gcm_translation/run/$JOB_NAME.out
#$ -V

set -euo pipefail

module load miniconda
export HF_HOME="/projectnb/mcnet/jbrin/.cache/huggingface"
# expandable_segments reduces fragmentation; +1-2 GB effective headroom on 44 GB GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda activate probes

cd /projectnb/mcnet/jbrin/lang-probing

echo "[$JOB_NAME] host=$HOSTNAME"
nvidia-smi -L || true
nvidia-smi --query-gpu=memory.free --format=csv || true

# First: run the full GPU test suite (catches bugs faster than the smoke run)
echo "[$JOB_NAME] === pytest GPU tests ==="
python -m pytest tests/test_gcm_translation.py -v --tb=short || {
    echo "[$JOB_NAME] GPU tests FAILED — aborting smoke run"
    exit 1
}

# Then: smoke test eng -> spa, 5 pairs, both components
echo "[$JOB_NAME] === GCM smoke run: English -> Spanish, 5 pairs ==="
python experiments/gcm_translation/run.py \
    --src_lang English \
    --tgt_lang Spanish \
    --n_pairs 5 \
    --seed 42 \
    --components both \
    --output_dir outputs/gcm_translation_smoke

echo "[$JOB_NAME] done"

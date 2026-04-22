
# Run probe-targeted ablation experiment
python scripts/ablate.py --experiment mono_random --concept Tense --value Past --k 5 --max_samples 256 --batch_size 32 --use_probe --probe_layer 32 --probe_n 1024 --output_dir "outputs/ablation"

python scripts/ablate.py --experiment mono_input --concept Tense --value Past --k 5 --max_samples 256 --batch_size 32 --use_probe --probe_layer 32 --probe_n 1024 --output_dir "outputs/ablation"

python scripts/ablate.py --experiment mono_output --concept Tense --value Past --k 5 --max_samples 256 --batch_size 32 --use_probe --probe_layer 32 --probe_n 1024 --output_dir "outputs/ablation"

python scripts/visualize_ablate_bar.py --input_dir "/projectnb/mcnet/jbrin/lang-probing/outputs/ablation_results" --output_dir "/projectnb/mcnet/jbrin/lang-probing/img/ablation" --num_samples 128 --probe_layer 32
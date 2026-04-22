# `_archive/scripts/`

Scripts archived during the 2026-04-22 restructure. Why each one is here:

| Script | Reason |
|---|---|
| `zzz_scratchpad.py` | Throwaway (counts tokens per UD treebank). |
| `zzz_input_space.py` | Early exploratory diff-in-means; superseded by `sentence_input_features.py`. |
| `zzz_collect_steering_vectors.py` | Pickle-based SV collection; superseded by the parquet-based variant. |
| `run_ablation.py` | Dead — imports `lang_probing_src.ablation` which doesn't exist. Legacy pair with `src/lang_probing_src/zzz_ablation.py`. |
| `jannik_output_features.py` | Reference implementation only; `attribution_flores.py` is canonical for output features. |
| `nnsight_practice.py` | 16-line nnsight demo. |
| `input_features.py` | Word-level draft superseded by `sentence_input_features.py`. The word-level procedure will be re-implemented as a config variant under `experiments/input_features/`. |
| `collect_sae_activations.py` | Never produced output (`outputs/activations_flores/` was always empty); SAE is applied on-the-fly by consumers. |
| `generate_steering_vectors.py` | Older, self-contained SV generator; superseded by `collect_steering_vectors.py`. |
| `collect_steering_vectors.py` | LEGACY — steering-vector pipeline is archived; project uses SAE-latent input features now. |
| `pca_steering_vectors.py` | LEGACY. |
| `visualize_steering_vectors.py` | LEGACY. |
| `steer.py` | LEGACY qualitative steering demo. |
| `visualize_ablate.py` | v1 heatmap visualization; superseded by `visualize_ablate_bar.py` (v2/v3). |
| `visualize_perplexity_bleu.py` | v1 legacy; superseded by `visualize_perplexity_bleu_correlation.py`. |
| `visualize_perplexity_bleu_sorted.py` | v2; superseded. |
| `filter_probe_training_log.py` | Trivial utility (strips non-INFO lines from a log). |
| `find_features.py` | Older probe-weight-based feature discovery; superseded by sentence-level diff-in-means + attribution. |
| `train_probes.py` | Legacy sentence-level probe trainer; `word_probes.py` is canonical. If sentence-level is ever wanted back, add it as a YAML config under `experiments/probes/` rather than resurrecting this script. |

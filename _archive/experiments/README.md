# `_archive/experiments/`

Legacy experiment stacks retained for reference but not part of the active pipeline.

## Steering vectors (LEGACY — Oct/Nov 2025)

The full steering-vector / PCA / cosine-similarity family:

- **Collection:** `_archive/scripts/collect_steering_vectors.py` (parquet-based, modern), `generate_steering_vectors.py` (older, self-contained), `zzz_collect_steering_vectors.py` (earliest, pickle-based).
- **Analysis:** `_archive/scripts/pca_steering_vectors.py`.
- **Visualization:** `_archive/scripts/visualize_steering_vectors.py`.
- **Demo:** `_archive/scripts/steer.py` + `_archive/examples/*.txt`.
- **Outputs:** `_archive/outputs/steering_vectors/`, `_archive/outputs/steering/`, `_archive/outputs/visualization_steering_vectors/` (~180 per-layer cosine-similarity PNGs).
- **Figures:** `_archive/img/pca_steering_vectors/` (25 PNGs), `_archive/img/sv_cs_heatmaps/` (18 PNGs).

**Why archived:** The project transitioned to SAE-latent input features as the primary H2 evidence. Steering vectors are diff-in-means in residual space — conceptually the ancestor of `sentence_input_features.py`, which operates in SAE space instead. Left in `_archive` for historical comparison and potential reuse; not used by any active experiment as of the Wave 1 restructure.

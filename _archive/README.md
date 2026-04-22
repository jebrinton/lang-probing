# `_archive/`

Legacy, deprecated, reference-only, or superseded artifacts. Kept in-repo but out of the active code paths. If it lives here, no active experiment consumes it.

Structure mirrors the top-level repo:

- `scripts/` — legacy / dead / reference-only Python scripts.
- `src/` — deprecated library modules.
- `outputs/` — old output directories (superseded, empty, or never populated).
- `img/` — superseded figure directories.
- `experiments/` — full legacy experiment stacks (e.g., steering vectors).
- `docs/` — stale documentation.
- `examples/` — legacy text-file examples consumed only by the dead `run_ablation.py`.

Large legacy blobs (multi-GB tar.gz archives and the 19.7 GB `zzz_dep_activations/`) live **off-tree** at `/projectnb/mcnet/jbrin/archive/lang-probing/`.

See [LEDGER.md](../LEDGER.md) for what's currently active.

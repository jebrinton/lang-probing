# gcm_translation

Generative Causal Mediation (GCM, Sankaranarayanan et al. arXiv:2602.16080 §2.2)
applied to FLORES translation pairs. For each (orig, cf) pair of source
sentences, computes the indirect effect

    IE_hat = grad_z log[pi(r_cf | p_orig) / pi(r_orig | p_orig)] . (z_orig - z_cf)

at every (a) attention-head output and (b) SAE feature in layer 16, patched
at the **last source-token position only**.

## Inputs

- FLORES dev split per language (`gsarti/flores_101`).
- Llama-3.1-8B + L16 SAE (jbrinkma/sae-llama-3-8b-layer16).

## Outputs

`outputs/gcm_translation/<src>__<tgt>/`:
- `heads_ie.pt`        — `[N_pairs, n_layers, n_heads]` IE per pair
- `sae_ie.pt`          — `[N_pairs, SAE_DIM]` IE per pair
- `top_rankings.json`  — top-20 heads + top-50 SAE features by mean-abs IE
- `summary.json`       — metadata + sanity stats
- `per_pair_records.json` — per-pair clean/patched metric values + drift checks
- `acp_faithfulness.json` (after `validate_faithfulness.py`) — ACP-vs-ATP correlation

`experiments/gcm_translation/img/`:
- `<src>__<tgt>_heads_heatmap.png`           — mean-abs IE per (layer, head)
- `<src>__<tgt>_heads_signed_heatmap.png`    — signed mean IE
- `<src>__<tgt>_top_heads_bar.png`           — top-20 heads with error bars
- `<src>__<tgt>_top_sae_bar.png`             — top-50 SAE features
- `<src>__<tgt>_acp_vs_atp.png`              — faithfulness scatter
- `universal_heads_heatmap.png`              — cross-direction "translation heads"

Figures are stored inside this experiment folder (not the repo-level `img/`)
to keep the experiment self-contained and portable.

## How to run

Smoke test (eng->spa, 5 pairs, ~10 min):
```
qsub experiments/gcm_translation/run/run_gcm_smoke.sh
```

Full sweep (56 directions, ~3 h wall-clock as 4-task array):
```
qsub -t 1-56 experiments/gcm_translation/run/run_gcm_sweep.sh
```

## Companion docs

- [REPORT.md](REPORT.md) — scientific writeup: background, methodology, sanity-check results, findings, figures.
- [REDTEAM.md](REDTEAM.md) — internal red-team review (math, nnsight idioms, code correctness) and the bug-fix log.
- [tests/test_gcm_translation.py](../../tests/test_gcm_translation.py) — 15 unit tests (9 CPU + 6 GPU).

## Status

v0.1 — implementation complete, awaiting overnight sweep.
Last source-token patching only; greedy 2-shot prompt; eng/spa/deu/fra/tur/ara/hin/heb sweep.

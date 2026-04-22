# lang-probing

Mechanistic investigation of how multilingual LLMs translate. We probe for grammatical concepts in Llama-3.1-8B (compared with Aya-23-8B), look for shared multilingual features in the layer-16 SAE (`jbrinkma/sae-llama-3-8b-layer16`), and causally test their role with ablations and counterfactual attribution.

## Where to start

- **[LEDGER.md](LEDGER.md)** — curated per-experiment view. What each experiment measures, current status, findings, caveats, figure links.
- **[LAB_NOTEBOOK.md](LAB_NOTEBOOK.md)** — chronological log (append-only). What was done each day, what surprised us.
- **[TODO.md](TODO.md)** — blockers, silent-failure fixes, scientific anomalies to investigate, nice-to-haves.

## Layout

```
src/lang_probing_src/        # library — primitives for activations, probes, features, interventions, eval, viz, io
experiments/<name>/          # one folder per experiment; thin run scripts + YAML configs + run/ job scripts
outputs/<name>/              # each experiment writes its outputs here (latest only)
img/<name>/                  # each experiment writes its figures here (latest only)
data/                        # small committed data (grammatical_pairs.json + README)
tests/                       # mirrors src/
docs/                        # paper, slides (reference material only)
_archive/                    # legacy code / outputs / figures / docs
```

Large legacy blobs (multi-GB tar.gz backups and the 19.7 GB `zzz_dep_activations/`) live off-tree at `/projectnb/mcnet/jbrin/archive/lang-probing/`.

## Experiments

| Folder | Hypothesis | Status | What it does |
|---|---|---|---|
| [experiments/ablation/](experiments/ablation/) | H2, H4 | active (v3) | Zero-ablate top-K SAE features; measure Δp(reference). 7 configs (mono/multi × input/output/random). |
| [experiments/counterfactual_attribution/](experiments/counterfactual_attribution/) | H2 | prototype | Gradient attribution on English minimal pairs; multilingual extension planned. |
| [experiments/token_analysis/](experiments/token_analysis/) | H2 | active | Per-token SAE ablation with HTML visualization. |
| [experiments/perplexity_bleu_linear/](experiments/perplexity_bleu_linear/) | H1 | active | `BLEU ≈ f(P_src, P_tgt)` + rank-1 SVD (**88% faithful for Llama**). |
| [experiments/input_features/](experiments/input_features/) | H2 | active | Diff-in-means in SAE latent space per (language, concept, value). |
| [experiments/output_features/](experiments/output_features/) | H2 | active | Gradient attribution from probe logits during translation. |
| [experiments/input_output_overlap/](experiments/input_output_overlap/) | H2 | active | Jaccard + signal plots: input features ↔ output features. |
| [experiments/probes/](experiments/probes/) | INFRA | active | cuML word-level logistic regression probes. |
| [experiments/activations_collection/](experiments/activations_collection/) | INFRA | stable | UD residual-stream activation cache. |
| [experiments/monolingual_ft/](experiments/monolingual_ft/) | H3 | scaffold | Evaluation stub; actual training lives in Jannik's separate repo. |

## Hypotheses

- **H1** — BLEU predictable from monolingual src/tgt competence.
- **H2** — The "noisy channel" is multilingual. Input and output feature spaces overlap across languages.
- **H3** — Adding a language ≈ improving monolingual capability.
- **H4** — Translation uses the same monolingual circuits the model uses for language modeling.

See [LEDGER.md](LEDGER.md) for which experiments speak to which hypothesis.

## Running an experiment

Every experiment has a README with exact commands. Generally:

```bash
python experiments/<name>/run.py --config experiments/<name>/configs/<variant>.yaml
```

Most experiments need a GPU (Llama-3.1-8B + SAE). Per-experiment `run/*.sh` scripts are qsub-ready templates for BU SCC batch jobs.

## Acknowledgements

Much of the sentence-level probing code is derived from Jannik Brinkmann's [multilingual-features](https://github.com/jannik-brinkmann/multilingual-features).

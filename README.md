# lang-probing

Mechanistic investigation of how multilingual LLMs translate. We probe for grammatical concepts in Llama-3.1-8B (compared with Aya-23-8B), look for shared multilingual features in the layer-16 SAE (`jbrinkma/sae-llama-3-8b-layer16`), and causally test their role with ablations and counterfactual attribution.

## Where to start

- **[LEDGER.md](LEDGER.md)** — curated per-experiment view. What each experiment measures, current status, findings, caveats, figure links.
- **[LAB_NOTEBOOK.md](LAB_NOTEBOOK.md)** — chronological log (append-only). What was done each day, what surprised us.
- **[TODO.md](TODO.md)** — blockers, silent-failure fixes, scientific anomalies to investigate, nice-to-haves.

## Layout

```
src/lang_probing_src/        # library — primitives for activations, probes, features, interventions, eval, viz
experiments/<name>/          # one folder per experiment; thin run scripts + YAML configs + co-located run/ job scripts
outputs/<name>/              # each experiment writes its outputs here (latest only)
img/<name>/                  # each experiment writes its figures here (latest only)
data/                        # small committed data (grammatical_pairs.json)
tests/                       # mirrors src/
docs/                        # paper, slides, old inventory
_archive/                    # legacy code / outputs / figures (git-tracked, in-repo)
```

Large legacy blobs (multi-GB tar.gz backups and the 19.7 GB `zzz_dep_activations/`) live off-tree at `/projectnb/mcnet/jbrin/archive/lang-probing/`.

## Running an experiment

Every experiment has a YAML config and a one-liner. From the repo root:

```bash
python experiments/<name>/run.py --config experiments/<name>/configs/<variant>.yaml
```

For BU SCC batch jobs, each experiment has its own `run/` folder with qsub-ready scripts. See `experiments/<name>/README.md` for specifics.

## Hypotheses

- **H1** — BLEU predictable from monolingual src/tgt competence.
- **H2** — The "noisy channel" is multilingual. Input and output feature spaces overlap across languages.
- **H3** — Adding a language ≈ improving monolingual capability.
- **H4** — Translation uses the same monolingual circuits the model uses for language modeling.

See [LEDGER.md](LEDGER.md) for which experiments speak to which hypothesis.

## Acknowledgements

Much of the sentence-level probing code is derived from Jannik Brinkmann's [multilingual-features](https://github.com/jannik-brinkmann/multilingual-features).

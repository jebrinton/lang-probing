# `monolingual_ft` (scaffold)

**Hypotheses:** H3 — adding a language via monolingual FT ≈ via parallel-corpus FT.

**Status:** scaffold only. No code in this repo.

## Where the work actually lives

Finetuning runs and checkpoints live in Jannik Brinkmann's separate repo. This folder hosts the **evaluation** side of H3: once a checkpoint arrives, we evaluate it on FLORES BLEU + Multi-BLiMP PER + representation-similarity.

## Planned entry point

```bash
python experiments/monolingual_ft/evaluate.py \
    --checkpoint <path-to-finetuned-llama> \
    --baseline-checkpoint <path-to-baseline-llama> \
    --config experiments/monolingual_ft/configs/evaluate_template.yaml
```

`evaluate.py` will call into:
- `lang_probing_src.eval.bleu` (FLORES BLEU)
- `lang_probing_src.eval.per` (Multi-BLiMP PER)
- `lang_probing_src.activations.collect` (for representation-similarity comparison)

None of those library modules are built out yet (they're empty shells pending Wave 3b extraction).

## TODOs

- [ ] Get a checkpoint from Jannik.
- [ ] Fill in `evaluate.py` stub.
- [ ] Write `configs/evaluate_template.yaml`.
- [ ] Implement representation-similarity eval.

See [LEDGER.md](../../LEDGER.md#monolingual_ft-h3--scaffold-only).

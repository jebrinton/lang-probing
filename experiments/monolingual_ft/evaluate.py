"""Evaluate a finetuned checkpoint for H3 (scaffold).

This is intentionally a stub. Fill it in when Jannik's finetuning repo
produces a checkpoint.

Planned behavior:
    1. Load the checkpoint (path from --checkpoint).
    2. Run FLORES BLEU on all (src, tgt) in a language grid.
    3. Run Multi-BLiMP PER per language.
    4. Compute representation-similarity against a baseline checkpoint
       (--baseline-checkpoint).
    5. Write results to outputs/monolingual_ft/<run_id>/.

Call `lang_probing_src.eval.bleu`, `lang_probing_src.eval.per`,
`lang_probing_src.activations.collect` once those are built out (Wave 3b).
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to finetuned checkpoint")
    parser.add_argument("--baseline-checkpoint", required=True,
                        help="Path to baseline (pre-finetune) checkpoint")
    parser.add_argument("--config", required=True,
                        help="YAML config (see configs/evaluate_template.yaml)")
    args = parser.parse_args()

    raise NotImplementedError(
        "experiments/monolingual_ft/evaluate.py is a scaffold. "
        "See README.md for the plan."
    )


if __name__ == "__main__":
    main()

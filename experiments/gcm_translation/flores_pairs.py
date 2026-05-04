"""
FLORES sentence-pair sampling for GCM translation attribution.

Yields disjoint (orig, cf) pairs from the same FLORES split, with the
two-shot examples held out from the pool.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from datasets import load_dataset

from lang_probing_src.config import NAME_TO_LANG_CODE


# Reserved indices for the 2-shot prompt examples. Held out from the pair pool
# so we never sample a shot as orig/cf. Same indices for every language because
# FLORES rows are aligned by index.
SHOT_INDICES = (0, 1)


@dataclass
class TranslationPair:
    pair_id: str
    orig_idx: int
    cf_idx: int
    src_orig: str
    tgt_orig: str
    src_cf: str
    tgt_cf: str


@dataclass
class NullTriple:
    """For the null control: A is the prompt source, B and C are the two
    scored responses. Neither B nor C corresponds to the gold translation
    of A, so the gradient ∂M/∂z does not carry a "preference for the right
    answer" component — only content discrimination at L16 survives.

    For src_lang == tgt_lang (same-lang null), tgt_B = src_B and tgt_C = src_C
    (the "translation" task is identity).
    """
    pair_id: str
    a_idx: int
    b_idx: int
    c_idx: int
    src_a: str
    src_b: str
    tgt_b: str
    src_c: str
    tgt_c: str


def _load_split(lang_code: str, split: str = "dev") -> List[str]:
    ds = load_dataset("gsarti/flores_101", lang_code, split=split)
    return [row["sentence"] for row in ds]


def get_shots(src_lang: str, tgt_lang: str, split: str = "dev"):
    """Return [(src_text, tgt_text), ...] for the two reserved shot rows."""
    src_code = NAME_TO_LANG_CODE[src_lang]
    tgt_code = NAME_TO_LANG_CODE[tgt_lang]
    src_rows = _load_split(src_code, split=split)
    tgt_rows = _load_split(tgt_code, split=split)
    return [(src_rows[i], tgt_rows[i]) for i in SHOT_INDICES]


def sample_pairs(
    src_lang: str,
    tgt_lang: str,
    n_pairs: int,
    seed: int = 42,
    split: str = "dev",
) -> List[TranslationPair]:
    """
    Sample n_pairs disjoint (orig, cf) tuples from the FLORES split.

    Each FLORES row is used at most once across all pairs (consecutive
    shuffled indices form (orig, cf) couples). Shot indices are excluded.
    """
    src_code = NAME_TO_LANG_CODE[src_lang]
    tgt_code = NAME_TO_LANG_CODE[tgt_lang]

    src_rows = _load_split(src_code, split=split)
    tgt_rows = _load_split(tgt_code, split=split)

    if len(src_rows) != len(tgt_rows):
        raise ValueError(
            f"FLORES row count mismatch: {src_lang}={len(src_rows)} "
            f"vs {tgt_lang}={len(tgt_rows)}"
        )

    candidate_idxs = [i for i in range(len(src_rows)) if i not in SHOT_INDICES]
    rng = random.Random(seed)
    rng.shuffle(candidate_idxs)

    max_pairs = len(candidate_idxs) // 2
    n_pairs = min(n_pairs, max_pairs)

    pairs: List[TranslationPair] = []
    for k in range(n_pairs):
        orig_idx = candidate_idxs[2 * k]
        cf_idx = candidate_idxs[2 * k + 1]
        pairs.append(
            TranslationPair(
                pair_id=f"{src_code}__{tgt_code}__{orig_idx}_{cf_idx}",
                orig_idx=orig_idx,
                cf_idx=cf_idx,
                src_orig=src_rows[orig_idx],
                tgt_orig=tgt_rows[orig_idx],
                src_cf=src_rows[cf_idx],
                tgt_cf=tgt_rows[cf_idx],
            )
        )
    return pairs


def sample_null_triples(
    src_lang: str,
    tgt_lang: str,
    n_pairs: int,
    seed: int = 42,
    split: str = "dev",
) -> List[NullTriple]:
    """
    Sample n_pairs disjoint (A, B, C) triples for the null control.

    A = source sentence in the prompt (analogous to the orig in Phase 1)
    B = first scored response (NOT the gold translation of A)
    C = second scored response (also unrelated to A)

    Disjoint: each FLORES row appears at most once across all 3*n_pairs
    slots, mirroring Phase 1's no-replacement convention.

    For src_lang == tgt_lang, only one language file is loaded, and
    tgt_B := src_B, tgt_C := src_C (the same-lang "translation" is identity).
    """
    src_code = NAME_TO_LANG_CODE[src_lang]
    same_lang = (src_lang == tgt_lang)

    src_rows = _load_split(src_code, split=split)
    if same_lang:
        tgt_rows = src_rows
    else:
        tgt_code = NAME_TO_LANG_CODE[tgt_lang]
        tgt_rows = _load_split(tgt_code, split=split)
        if len(src_rows) != len(tgt_rows):
            raise ValueError(
                f"FLORES row count mismatch: {src_lang}={len(src_rows)} "
                f"vs {tgt_lang}={len(tgt_rows)}"
            )

    candidate_idxs = [i for i in range(len(src_rows)) if i not in SHOT_INDICES]
    rng = random.Random(seed)
    rng.shuffle(candidate_idxs)

    max_pairs = len(candidate_idxs) // 3
    n_pairs = min(n_pairs, max_pairs)

    triples: List[NullTriple] = []
    for k in range(n_pairs):
        a_idx = candidate_idxs[3 * k]
        b_idx = candidate_idxs[3 * k + 1]
        c_idx = candidate_idxs[3 * k + 2]
        triples.append(
            NullTriple(
                pair_id=f"{src_code}__{NAME_TO_LANG_CODE[tgt_lang]}__null__{a_idx}_{b_idx}_{c_idx}",
                a_idx=a_idx,
                b_idx=b_idx,
                c_idx=c_idx,
                src_a=src_rows[a_idx],
                src_b=src_rows[b_idx],
                tgt_b=tgt_rows[b_idx],
                src_c=src_rows[c_idx],
                tgt_c=tgt_rows[c_idx],
            )
        )
    return triples


def make_prompt(src_lang: str, tgt_lang: str, shots, src_text: str) -> str:
    """
    2-shot translation prompt.

    Ends with `"{tgt_lang}: "` (trailing space) so the response can be a bare
    target string in any script — including non-Latin / RTL languages where
    a leading-space-prefixed first token is unnatural.

    Single-newline separators between blocks (not double-newline).
    """
    parts = []
    for s_src, s_tgt in shots:
        parts.append(f"{src_lang}: {s_src}")
        parts.append(f"{tgt_lang}: {s_tgt}")
    parts.append(f"{src_lang}: {src_text}")
    parts.append(f"{tgt_lang}: ")
    return "\n".join(parts)

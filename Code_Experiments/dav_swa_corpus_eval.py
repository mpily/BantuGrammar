# -*- coding: utf-8 -*-
"""
dav_swa_corpus_eval.py

2026-10-28. Direct user request: the thinkKenya/kenyan-low-resource-
language-data HF dataset (Mbogho et al. 2025 / Lacuna Fund 2021 corpus --
the SAME release this project's own CLAUDE.md s4 had flagged as
"check release status") is now public with a `dav_swa` config: 21,329
train + 5,333 test real Swahili<->Kidaw'ida sentence pairs. Confirmed
DIRECTLY via the HF Hub API before any of this was written (gated repo,
required the user's own access approval -- not assumed from the dataset
name). This script (1) mines Swahili noun -> Kitaita noun candidates from
the TRAIN split only, reusing this project's existing Dice-cooccurrence
alignment method (multi_version_pipeline.build_cooccurrence/
best_alignment, built for the 3-version Bible corpus -- extended here to
a second corpus, not forked) and (2) evaluates three conditions on the
HELD-OUT TEST split using the metrics already defined for machine-vs-
reference comparison in this project (hf_space/api/stats.chrf,
normalize_kitaita, mean_ci/rate_ci -- the same functions /api/progress
uses for machine-vs-speaker-output comparison).

SCOPE NOTE, stated plainly: kitaita_eval_metrics.py's metrics 1
(agreement accuracy) and 2 (vocabulary choice) both require GOLD slot-
level annotation or multi-speaker-choice data for the SAME sentences --
this raw parallel corpus is sentence pairs only, with neither. Those two
metrics are NOT run here; only the sentence-level exact-match/chrF
metric (metric used by /api/progress) applies to this data, and that is
what is reported below. Not a silent substitution -- flagged here and
in docs/status.md.

Three conditions, operationalizing this project's own null hypothesis
rule ("relatedness helps must be beaten, not assumed"):
  A = CURRENT (production tables: VERIFIED_NOUN_TRANSLATIONS + existing
      Bible-corpus-derived noun_kitaita, via build_lookup_tables -- i.e.
      exactly what the live pipeline uses today).
  B = CURRENT + dav_swa-train-mined noun candidates filling GAPS ONLY
      (a lemma with no existing VERIFIED_NOUN_TRANSLATIONS/noun_kitaita
      entry gets the new mined candidate; nothing already present is
      overridden -- monotonic coverage increase, never a silent
      overwrite of an existing/verified entry).
  C = NULL floor: no noun dictionary at all (every noun lemma passes
      through unchanged) -- the thing "mining helps" has to beat.

Uses translate_sentence() (heuristic/rule-based path, no NLLB) since the
test-set Swahili text is REAL corpus text, not something that needs
generating -- consistent with the project's post-pivot primary mechanism
(probe-guided rule-based rewriting), not the probes/steering research
track.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))

from huggingface_hub import hf_hub_download
import pandas as pd

from multi_version_pipeline import build_cooccurrence, best_alignment, MIN_FREQ, MIN_DICE
from kitaita_sentence_generator import load_pages, build_lookup_tables
from kitaita_e2e_pipeline import translate_sentence
from kitaita_verified_corrections import VERIFIED_NOUN_TRANSLATIONS
from swahili_noun_lexicon_agreement_v2 import NOUNS as SWAHILI_LEXICON
from api.stats import chrf, mean_ci, rate_ci
from api.util import normalize_kitaita
from dav_swa_eval_common import tokens_match

DATASET_REPO = "thinkKenya/kenyan-low-resource-language-data"
CONFIG = "dav_swa"
RANDOM_SEED = 20261028
# translate_sentence() is pure-Python/CPU (no NLLB call) -- benchmarked at <1s for
# the full 5,333-row test split, so no subsampling is needed; the full split is used.
TEST_SAMPLE_N = 5333


def git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_split(split: str) -> list[dict]:
    path = Path(hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset",
                                 filename=f"{CONFIG}/{split}-00000-of-00001.parquet"))
    df = pd.read_parquet(path)
    rows = [{"id": int(r["id"]), "dav": r["translation"]["dav"], "swa": r["translation"]["swa"]}
            for _, r in df.iterrows()]
    return rows, path


_QUOTES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'"})


def tokenize(sentence: str) -> list[str]:
    """Lowercase, straighten quotes (w'/ng' are phonemic in Kidaw'ida -- kept,
    not stripped), split on whitespace, strip leading/trailing punctuation
    from each token only (internal apostrophes preserved)."""
    out = []
    for tok in (sentence or "").translate(_QUOTES).lower().split():
        tok = tok.strip(".,!?;:\"()[]")
        if tok:
            out.append(tok)
    return out


def mine_noun_candidates(train_rows: list[dict]) -> dict:
    """Dice-cooccurrence alignment, TRAIN split only, reusing multi_version_
    pipeline's generic build_cooccurrence/best_alignment unchanged. That
    function is direction-specific (indexes cooccur by its 'ta' argument and
    looks candidates up by 'ta' word), so to query "best Kitaita candidate
    for a KNOWN SWAHILI LEMMA" we feed Swahili tokens as the 'ta' side and
    Kitaita tokens as the 'sw' side -- the reverse of the function's own
    Bible-era naming (where Kitaita was always 'ta'). The function's
    returned dict keys are hardcoded "kitaita_word"/"swahili_word" tied to
    that ta/sw argument order, not to real language identity, so they are
    relabeled correctly below rather than trusted as-is."""
    verses = [{"ta_tokens": tokenize(r["swa"]), "sw_tokens": tokenize(r["dav"])} for r in train_rows]
    ta_counts, sw_counts, cooccur = build_cooccurrence(verses)

    candidates = {}
    for lemma, swa_class, gloss in SWAHILI_LEXICON:
        raw = best_alignment(lemma, ta_counts, sw_counts, cooccur)
        if raw is None:
            continue
        candidates[lemma] = {
            "swahili_word": lemma, "swahili_class": swa_class, "gloss": gloss,
            "kitaita_word": raw["swahili_word"], "dice": raw["score"],
            "swahili_count_in_train": raw["kitaita_count"],
            "kitaita_count_in_train": raw["swahili_count"],
            "cooccur_count": raw["cooccur_count"], "n_competitors": raw["n_competitors"],
        }
    return candidates


def validate_mining_against_verified(candidates: dict) -> dict:
    """Sanity/validity check of the mining method itself, BEFORE using it for
    anything -- this project's own rule (never trust a new signal without
    checking it against something already known). Compares each mined
    candidate against VERIFIED_NOUN_TRANSLATIONS (native-speaker-confirmed)
    where both exist for the same Swahili lemma."""
    overlap = [(lemma, c["kitaita_word"], VERIFIED_NOUN_TRANSLATIONS[lemma])
               for lemma, c in candidates.items() if lemma in VERIFIED_NOUN_TRANSLATIONS]
    matches = [1 for _, mined, verified in overlap if tokens_match(mined, verified)]
    return {
        "n_overlap": len(overlap),
        "exact_match": rate_ci(len(matches), len(overlap)) if overlap else None,
        "mismatches": [{"swahili": l, "mined": m, "verified": v} for l, m, v in overlap if not tokens_match(m, v)],
    }


def build_condition_tables(mined: dict):
    """Returns (tables_A_current, tables_B_plus_mined, tables_C_null).
    B only FILLS gaps in noun_kitaita (never overrides an existing entry --
    VERIFIED_NOUN_TRANSLATIONS is checked separately inside lookup_noun_
    kitaita() regardless of noun_kitaita, so a verified entry is untouched
    either way; this only affects lemmas with NO verified entry AND no
    existing statistical noun_kitaita entry)."""
    p1, p2, p3 = load_pages()
    noun_kitaita, root_kitaita, root_kitaita_fallback, noun_class_of, allomorphy, root_kitaita_derived = \
        build_lookup_tables(p1, p2, p3)
    tables_a = (noun_kitaita, root_kitaita, root_kitaita_fallback, noun_class_of, allomorphy, root_kitaita_derived)

    noun_kitaita_b = dict(noun_kitaita)
    added = []
    for lemma, c in mined.items():
        if lemma in VERIFIED_NOUN_TRANSLATIONS or lemma in noun_kitaita_b:
            continue
        # n_versions_attested/confidence: the existing Bible-derived noun_kitaita
        # entries always carry these (trace-string formatting at kitaita_sentence_
        # generator.translate() line ~457 reads them unconditionally when a noun is
        # used as the sentence SUBJECT) -- found via a real KeyError on 5/1200 test
        # sentences during this run, not anticipated in advance. dav_swa is a single
        # corpus (not multiple independently-translated versions), so the honest
        # equivalent is 1/1, flagged as not cross-corpus-corroborated.
        noun_kitaita_b[lemma] = {"kitaita_word": c["kitaita_word"], "best_dice": c["dice"],
                                  "n_versions_attested": 1,
                                  "confidence": "statistical (dav_swa train corpus, single-source, "
                                                 "not cross-corpus corroborated)",
                                  "source": "dav_swa_train_mined_2026-10-28"}
        added.append(lemma)
    tables_b = (noun_kitaita_b, root_kitaita, root_kitaita_fallback, noun_class_of, allomorphy, root_kitaita_derived)

    tables_c = ({}, root_kitaita, root_kitaita_fallback, noun_class_of, allomorphy, root_kitaita_derived)
    return tables_a, tables_b, tables_c, added


def run_condition(test_sample: list[dict], tables, no_dict: bool = False) -> dict:
    """no_dict=True (condition C) additionally disables VERIFIED_NOUN_TRANSLATIONS
    via monkeypatch of the module-level dict translate_sentence's call chain reads,
    restored immediately after -- the pipeline's noun lookup checks
    VERIFIED_NOUN_TRANSLATIONS before noun_kitaita (see lookup_noun_kitaita()),
    so an empty noun_kitaita alone is not a true null-dictionary condition."""
    import kitaita_e2e_pipeline as pipeline_mod
    saved = dict(VERIFIED_NOUN_TRANSLATIONS)
    if no_dict:
        VERIFIED_NOUN_TRANSLATIONS.clear()
    try:
        exact_k = exact_n = 0
        chrf_scores = []
        errors = 0
        for row in test_sample:
            try:
                out = translate_sentence(row["swa"], tables)
            except Exception:
                errors += 1
                continue
            hyp = out.get("kitaita_sentence") or ""
            ref = row["dav"]
            exact_n += 1
            exact_k += int(tokens_match(normalize_kitaita(hyp), normalize_kitaita(ref)))
            chrf_scores.append(chrf(normalize_kitaita(hyp), normalize_kitaita(ref)))
        return {"exact_match": rate_ci(exact_k, exact_n), "chrf": mean_ci(chrf_scores),
                "n_sentences": len(test_sample), "n_scored": exact_n, "n_pipeline_errors": errors}
    finally:
        VERIFIED_NOUN_TRANSLATIONS.clear()
        VERIFIED_NOUN_TRANSLATIONS.update(saved)


def main():
    train_rows, train_path = load_split("train")
    test_rows, test_path = load_split("test")

    mined = mine_noun_candidates(train_rows)
    mining_validity = validate_mining_against_verified(mined)

    tables_a, tables_b, tables_c, added_lemmas = build_condition_tables(mined)

    rng = random.Random(RANDOM_SEED)
    sample_idx = rng.sample(range(len(test_rows)), min(TEST_SAMPLE_N, len(test_rows)))
    test_sample = [test_rows[i] for i in sample_idx]

    result_a = run_condition(test_sample, tables_a)
    result_b = run_condition(test_sample, tables_b)
    result_c = run_condition(test_sample, tables_c, no_dict=True)

    # Localized check: A vs B restricted to sentences that actually contain one of
    # the lemmas B added -- the aggregate A-vs-B delta above is diluted by every
    # sentence that can't possibly be affected (the other ~300 lexicon words were
    # already covered either way). Still subject to this project's n>=8 rule.
    added_set = set(added_lemmas)
    affected = [row for row in test_sample if added_set & set(tokenize(row["swa"]))]
    result_a_affected = run_condition(affected, tables_a) if len(affected) >= 1 else None
    result_b_affected = run_condition(affected, tables_b) if len(affected) >= 1 else None

    report = {
        "config": {
            "dataset_repo": DATASET_REPO, "dataset_config": CONFIG,
            "train_n": len(train_rows), "test_n": len(test_rows),
            "test_sample_n": len(test_sample), "random_seed": RANDOM_SEED,
            "min_freq": MIN_FREQ, "min_dice": MIN_DICE,
            "git_sha": git_sha(),
            "train_parquet_sha256_16": file_sha256(train_path),
            "test_parquet_sha256_16": file_sha256(test_path),
        },
        "mining": {
            "swahili_lexicon_n": len(SWAHILI_LEXICON),
            "n_mined_candidates": len(mined),
            "n_added_to_dictionary_B": len(added_lemmas),
            "added_lemmas": sorted(added_lemmas),
            "validity_vs_verified_nouns": mining_validity,
        },
        "conditions": {
            "A_current_production": result_a,
            "B_current_plus_train_mined": result_b,
            "C_null_no_dictionary": result_c,
        },
        "localized_affected_subset": {
            "n_affected_sentences": len(affected),
            "note": ("sentences in the test sample containing >=1 of the "
                     f"{len(added_lemmas)} lemmas B added; A vs B here isolates whether "
                     "the added vocabulary helped where it could possibly matter"),
            "A_on_affected": result_a_affected,
            "B_on_affected": result_b_affected,
        },
    }
    out_path = HERE / "dav_swa_corpus_eval_results.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    cand_path = HERE / "dav_swa_mined_noun_candidates.json"
    cand_path.write_text(json.dumps(mined, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

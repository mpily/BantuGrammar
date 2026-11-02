"""
build_pipeline_lexicon.py

2026-11-02. Turns the round-1 grammar-mining output into a word-level table
the deterministic pipeline can use as its LAST-RESORT tier: a Swahili word the
pipeline would otherwise keep as Swahili (a passthrough token, or a secondary
noun with no Kitaita form) gets the mined Kidaw'ida form instead.

Input:  ~/languagemodels/mining/round1/<area>.json  (not in git: built from the
        gated dav_swa corpus and the Bible text; see docs/status.md 2026-10-31)
Output: hf_space/kitaita_mined_lexicon_round1.json  (derived word pairs only,
        no corpus sentences)

Per Swahili word the table keeps the first-listed form from the highest-priority
area that has it (nouns > verb_morphology > phrases_and_rephrasings >
roots_and_function_words), its alternatives, and a confidence:
  high / medium / low  -- the agent's confidence for the entry whose lemma is
                          this word
  derived              -- the word is only in that area's lexicon_for_check
                          (e.g. an inflected verb built from a mined root)

Left out on purpose:
  * the noun_classes_concord area, and possessive / demonstrative /
    associative words from the other areas: their Kidaw'ida form depends on
    the controlling noun's Kidaw'ida class, which a word-level fallback
    cannot see (about 46% of nouns change class between the two languages);
  * entries whose form equals the Swahili word (no change);
  * multi-word keys.

Nothing in the table is speaker-verified. The pipeline uses it only after
every verified and Bible-derived tier has failed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MINING = Path.home() / "languagemodels" / "mining" / "round1"
OUT = Path(__file__).resolve().parents[2] / "hf_space" / "kitaita_mined_lexicon_round1.json"
AREAS = ["nouns", "verb_morphology", "phrases_and_rephrasings", "roots_and_function_words"]
CLASS_DEPENDENT_POS = {"possessive", "possessive_stem", "demonstrative", "associative", "concord",
                       "relative", "quantifier"}
CONF_RANK = {"high": 3, "medium": 2, "low": 1}


def class_dependent_by_shape():
    """The pipeline's own detectors for Swahili words whose form depends on a
    noun's class (possessives like yangu/zake, demonstratives, associative
    particles, -ote forms), so the fallback never fills those by Swahili class."""
    import sys
    sys.path.insert(0, str(OUT.parent))
    import kitaita_e2e_pipeline as pl
    fixed = set(pl.SWAHILI_DEM_PROXIMAL) | set(pl.SWAHILI_DEM_REFERENTIAL) | set(pl.SWAHILI_DEM_DISTAL) \
        | set(pl.ASSOC_PARTICLES) | set(pl.OTE_TOKENS)
    return lambda w: w in fixed or pl.find_possessive_pronoun(w) is not None


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    table: dict[str, dict] = {}
    skipped_class_dependent = 0
    sources = {}
    shape_dependent = class_dependent_by_shape()
    for area in AREAS:
        path = MINING / f"{area}.json"
        sources[area] = sha16(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        conf_of, pos_of = {}, {}
        for e in data["entries"]:
            sw = (e.get("swahili") or "").strip().lower()
            if not sw or " " in sw:
                continue
            c = e.get("confidence")
            if c in CONF_RANK and CONF_RANK[c] > CONF_RANK.get(conf_of.get(sw), 0):
                conf_of[sw] = c
                pos_of[sw] = (e.get("pos") or "").lower()
        for sw, forms in (data.get("lexicon_for_check") or {}).items():
            sw = sw.strip().lower()
            if not sw or " " in sw or sw in table:
                continue
            if any(p in pos_of.get(sw, "") for p in CLASS_DEPENDENT_POS) or shape_dependent(sw):
                skipped_class_dependent += 1
                continue
            forms = [f for f in forms if f and " " not in f]
            if not forms or forms[0].replace("'", "") == sw.replace("'", ""):
                continue
            table[sw] = {"kitaita": forms[0], "alternatives": forms[1:],
                         "confidence": conf_of.get(sw, "derived"), "area": area}
    # Register evidence (davswa pairs of MINE only; the holdout and dav_swa test
    # are not in MINE): per listed form, in how many davswa pairs containing the
    # Swahili word that form appears on the Kidaw'ida side ("=swahili" = the
    # translator kept the Swahili word). Used by kitaita_mined_lexicon's optional
    # davswa-preferred form choice.
    import collections
    pairs_n, form_n = collections.Counter(), collections.defaultdict(collections.Counter)
    norm = lambda w: w.replace("'", "").lower()
    for line in open(MINING.parent / "mine.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["src"] != "davswa":
            continue
        dav = {norm(t) for t in r["dav_tok"]}
        for sw in set(r["swa_tok"]):
            e = table.get(sw)
            if e is None:
                continue
            pairs_n[sw] += 1
            for f in [e["kitaita"]] + e["alternatives"]:
                if norm(f) != norm(sw) and norm(f) in dav:
                    form_n[sw][f] += 1
            if norm(sw) in dav:
                form_n[sw]["=swahili"] += 1
    for sw, e in table.items():
        e["davswa"] = {"pairs": pairs_n[sw], "form_counts": dict(form_n[sw])}
    counts = {}
    for v in table.values():
        counts[v["confidence"]] = counts.get(v["confidence"], 0) + 1
    meta = {
        "built_by": "Code_Experiments/grammar_mining/build_pipeline_lexicon.py",
        "date": "2026-11-02",
        "source": "round-1 grammar mining (dav_swa train minus a 1,000-pair holdout + Bible NEN); "
                  "see docs/status.md 2026-10-31",
        "source_sha256_16": sources,
        "areas_in_priority_order": AREAS,
        "excluded": "noun_classes_concord area; class-dependent POS " + ", ".join(sorted(CLASS_DEPENDENT_POS))
                    + "; words the pipeline's own detectors classify as possessive / demonstrative / "
                      "associative / -ote",
        "skipped_class_dependent": skipped_class_dependent,
        "counts_by_confidence": counts,
        "status": "machine-mined, NOT speaker-verified",
    }
    OUT.write_text(json.dumps({"_meta": meta, "entries": table}, ensure_ascii=False, indent=0, sort_keys=True),
                   encoding="utf-8")
    print(OUT, len(table), counts, "skipped class-dependent:", skipped_class_dependent)


if __name__ == "__main__":
    main()

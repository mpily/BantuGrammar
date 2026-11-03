"""
build_subject_markers.py

2026-11-02. Surface subject marker by (class, tense marker), for the version-B
subject-agreement step (hf_space/rulebook_correct.py). User request: correct
the fine-tuned draft's verb agreement with the pipeline, not by reranking the
model's candidates ("Kuku alipokuwa anarudi ... " -> draft "Nguku ... orekoghe
ukiwuya ... ukakwana": person-class o-/u- carried over from Swahili a-, but
nguku is a speaker-verified i- noun).

Each class is the pipeline's own subject-marker value (VERIFIED_SUBJECT_MARKERS
in kitaita_verified_corrections.py, i_nasal -> i via TAM_SM_ALIAS). For every
noun of that class in the training text (MINE + HOLDOUT), the word directly
after it is split as marker + tense + rest, where the marker is the class value
with its vowel changed or a glide added (i -> e, ya; u -> o, wa) and tense is
one of TAMS, or zero (present, marker ending in -a: wa-, ya-). Counts only, no
sentences. Not every word after a noun is its verb: the step uses a marker only
where one surface dominates (see rulebook_correct.py).
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
from kitaita_verified_corrections import TAM_SM_ALIAS, VERIFIED_SUBJECT_MARKERS  # noqa: E402
from rulebook_flagger import norm  # noqa: E402

MINING = Path.home() / "languagemodels" / "mining"
OUT = ROOT / "hf_space" / "kitaita_subject_markers.json"
TAMS = ("cha", "re", "de", "ka", "ki", "we")  # tried longest first; "" = zero-marked present
GLIDES = {"u": "w", "i": "y"}


def surfaces(base: str) -> list[str]:
    """Candidate surface forms of a class's subject marker, longest first.
    Consonant classes: the consonant with any vowel (chi -> cha/che/chi).
    One-vowel classes have no consonant to anchor them, so only the attested
    alternations are allowed (u -> o, wa, we; i -> e, ya, ye): with every vowel,
    i-nouns pick up the person-class verb of a following clause."""
    if base in GLIDES:
        return sorted({base, {"u": "o", "i": "e"}[base]} | {GLIDES[base] + v for v in "ae"}, key=len, reverse=True)
    stem = base[:-1] if base[-1] in "aeiou" else base
    return sorted({base} | {stem + v for v in "aeiou"}, key=len, reverse=True)


def split(token: str, base: str):
    """(surface, tense) if token starts with a subject marker of this class."""
    cands = surfaces(base)
    for t in TAMS:
        for s in cands:
            if token.startswith(s + t) and len(token) > len(s + t) + 1:
                return s, t
    for s in cands:
        rest = token[len(s):]
        if s.endswith("a") and token.startswith(s) and len(rest) >= 3 and rest[0] not in "aeiou":
            return s, ""
    return None


if __name__ == "__main__":
    classes = collections.defaultdict(set)
    for noun, sm in VERIFIED_SUBJECT_MARKERS.items():
        classes[TAM_SM_ALIAS.get(sm, sm)].add(norm(noun))
    counts = {b: collections.defaultdict(collections.Counter) for b in classes}
    sources = {}
    for name in ("mine.jsonl", "holdout.jsonl"):
        path = MINING / name
        sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        for line in open(path, encoding="utf-8"):
            toks = [norm(t) for t in json.loads(line)["dav_tok"]]
            for a, b in zip(toks, toks[1:]):
                for base, nouns in classes.items():
                    if a in nouns:
                        hit = split(b, base)
                        if hit:
                            counts[base][hit[1]][hit[0]] += 1
    vc = ROOT / "hf_space" / "kitaita_verified_corrections.py"
    OUT.write_text(json.dumps({
        "_meta": {"built_by": "Code_Experiments/rulebook/build_subject_markers.py", "date": "2026-11-02",
                  "source_sha256_16": {**sources, "kitaita_verified_corrections.py":
                                       hashlib.sha256(vc.read_bytes()).hexdigest()[:16]},
                  "nouns_per_class": {b: len(n) for b, n in sorted(classes.items())},
                  "tams": list(TAMS) + [""],
                  "status": "corpus counts over speaker-verified class nouns; surfaces NOT speaker-verified"},
        "counts": {b: {t: dict(c.most_common()) for t, c in sorted(by_t.items())} for b, by_t in sorted(counts.items())},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    for b, by_t in sorted(counts.items()):
        print(b, {t or "pres": c.most_common(3) for t, c in sorted(by_t.items())})

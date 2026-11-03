"""
build_noun_classes_round1.py

2026-11-03. Round-1 noun lexemes (mining/round1/nouns.json, lexeme entries,
confidence high/medium) -> Swahili noun -> Kidaw'ida forms + class pair, for
the subject-class resolver (hf_space/subject_class_resolver.py).

Class strings are parsed into (singular class, plural class):
  "9/10"   -> (9, 10)   one form for both numbers
  "1 / 2"  -> (1, 2)
  "9 / 2"  -> (9, 2)    animate plural
  "9"      -> (9, None) singular only listed;  "6" -> (None, 6) plural / mass
Anything else (e.g. "5, 6, 7") is left out. Output: word pairs and classes
only, no sentences: hf_space/kitaita_noun_classes_round1.json. Not
speaker-verified.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
from rulebook_flagger import norm  # noqa: E402

SRC = Path.home() / "languagemodels" / "mining" / "round1" / "nouns.json"
OUT = ROOT / "hf_space" / "kitaita_noun_classes_round1.json"
SG = {1, 3, 5, 7, 9, 11, 14, 15}
PL = {2, 4, 6, 8, 10}


def parse(cls: str):
    cls = (cls or "").strip()
    if not re.fullmatch(r"\d+(/\d+)?( / \d+(/\d+)?)?", cls):
        return None
    parts = cls.split(" / ")
    first = [int(x) for x in parts[0].split("/")]
    sg = first[0] if first[0] in SG else None
    pl = first[1] if len(first) > 1 else (first[0] if first[0] in PL else None)
    if len(parts) > 1:
        second = [int(x) for x in parts[1].split("/")]
        pl = second[-1] if second[-1] in PL else pl
    if sg is None and pl is None:
        return None
    return sg, pl


if __name__ == "__main__":
    entries = json.loads(SRC.read_text(encoding="utf-8"))["entries"]
    out, skipped = {}, 0
    for e in entries:
        if e["kind"] != "lexeme" or e.get("confidence") not in ("high", "medium") or not e.get("swahili"):
            continue
        p = parse(e.get("kitaita_class"))
        if not p:
            skipped += 1
            continue
        out[norm(e["swahili"])] = {"kitaita": [norm(k) for k in e["kitaita"] if k], "sg": p[0], "pl": p[1],
                                   "class": e["kitaita_class"], "confidence": e["confidence"]}
    OUT.write_text(json.dumps({"_meta": {"built_by": "Code_Experiments/rulebook/build_noun_classes_round1.py",
                                         "date": "2026-11-03",
                                         "source_sha256_16": hashlib.sha256(SRC.read_bytes()).hexdigest()[:16],
                                         "status": "round-1 agent mining, NOT speaker-verified"},
                               "nouns": out}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(len(out), "nouns,", skipped, "skipped (class string not parsed)")

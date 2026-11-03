"""
build_root_alignment.py

2026-11-03. Swahili verb root -> Kidaw'ida verb root(s), for the version-B tense
and extension steps (hf_space/rulebook_correct.py), which find the draft verb
that translates a Swahili verb by its root instead of by word position.

Sources, in priority order (first one that has the root wins; the others are
kept as alternatives):
  verified   VERIFIED_ROOTS (speaker)
  round1     kitaita_verb_roots.json (round-1 mining)
  infinitive mined lexicon pairs ku-X-a -> ku-Y-a (confidence high/derived)
  pipeline   the pipeline's root dictionary (root_kitaita, Dice >= MIN_DICE)
Output: hf_space/kitaita_root_alignment.json (word pairs only). Not
speaker-verified except the first tier.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
import os  # noqa: E402

os.chdir(ROOT / "hf_space")
from kitaita_verified_corrections import VERIFIED_ROOTS  # noqa: E402
from kitaita_sentence_generator import build_lookup_tables, load_pages  # noqa: E402
from rulebook_flagger import norm  # noqa: E402

MIN_DICE = 0.25
OUT = ROOT / "hf_space" / "kitaita_root_alignment.json"

if __name__ == "__main__":
    table = {}

    def add(sw, kit, source, score=None):
        sw, kit = norm(sw).strip("-"), norm(kit).strip("-")
        if not re.fullmatch(r"[a-z]{1,}", sw or "") or not re.fullmatch(r"[a-z]{1,}", kit or ""):
            return
        e = table.setdefault(sw, {"kitaita": [], "source": source, "score": score})
        if kit not in e["kitaita"]:
            e["kitaita"].append(kit)

    for sw, kit in VERIFIED_ROOTS.items():
        if isinstance(kit, str):
            add(sw, kit, "verified")
    r1 = json.loads(Path("kitaita_verb_roots.json").read_text(encoding="utf-8"))["roots"]
    for sw, v in r1.items():
        for kit in v["kitaita_roots"]:
            add(sw, kit, "round1")
    mined = json.loads(Path("kitaita_mined_lexicon_round1.json").read_text(encoding="utf-8"))["entries"]
    n_inf = 0
    for sw, e in mined.items():
        k = norm(e.get("kitaita") or "")
        # Swahili loan verbs keep their final vowel (ku-fahamu, ku-rudi): the root keeps it too
        if e.get("confidence") in ("high", "derived") and re.fullmatch(r"ku[a-z]+[aeiu]", sw) \
                and re.fullmatch(r"ku[a-z]+[aeiu]", k):
            add(sw[2:-1] if sw.endswith("a") else sw[2:], k[2:-1] if k.endswith("a") else k[2:], "infinitive")
            n_inf += 1
    tables = build_lookup_tables(*load_pages())
    for sw, c in tables[1].items():
        if c.get("score", 0) >= MIN_DICE and c.get("kitaita_root"):
            add(sw, c["kitaita_root"], "pipeline", round(c["score"], 3))
    srcs = {}
    for e in table.values():
        srcs[e["source"]] = srcs.get(e["source"], 0) + 1
    OUT.write_text(json.dumps({"_meta": {"built_by": "Code_Experiments/rulebook/build_root_alignment.py",
                                         "date": "2026-11-03", "min_dice": MIN_DICE, "by_first_source": srcs,
                                         "inputs_sha256_16": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]
                                                              for f in ("kitaita_verb_roots.json",
                                                                        "kitaita_mined_lexicon_round1.json",
                                                                        "kitaita_verified_corrections.py")},
                                         "status": "corpus-derived except 'verified'; NOT speaker-verified"},
                               "roots": table}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(len(table), "Swahili roots;", srcs, "; infinitive pairs used:", n_inf)
    for r in ("pigan", "pik", "ongez", "ib", "angush", "fung", "som", "pat", "fany", "end", "fahamu", "ja", "l"):
        print(r, table.get(r))

"""
build_verb_roots.py

2026-11-02. Swahili verb root -> Kidaw'ida verb root(s), from round-1
roots_and_function_words (pos verb_root), for the version-B applicative step
(hf_space/rulebook_correct.py). Kidaw'ida roots are taken from the attested
infinitive when there is one (kulwa -> lw), else from the listed root with its
dashes and a trailing vowel removed. Low-confidence entries are left out.
Output: hf_space/kitaita_verb_roots.json (word pairs only). Not speaker-verified.
"""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = Path.home() / "languagemodels" / "mining" / "round1" / "roots_and_function_words.json"
OUT = ROOT / "hf_space" / "kitaita_verb_roots.json"

table = {}
for e in json.loads(SRC.read_text(encoding="utf-8"))["entries"]:
    if not (e.get("pos") or "").startswith("verb") or e.get("confidence") not in ("high", "medium"):
        continue
    sw = (e.get("swahili") or "").strip("-").lower()
    if not sw or not re.fullmatch(r"[a-z]+", sw):
        continue
    roots = []
    inf = (e.get("kitaita_infinitive") or "").lower().replace("'", "")
    if re.fullmatch(r"ku[a-z]+a", inf):
        roots.append(inf[2:-1])
    for k in e.get("kitaita") or []:
        k = k.strip("-").lower().replace("'", "")
        if re.fullmatch(r"[a-z]+", k):
            roots.append(k[:-1] if k.endswith("a") and len(k) > 2 else k)
    roots = [r for r in dict.fromkeys(roots) if len(r) >= 2]
    if roots:
        table[sw] = {"kitaita_roots": roots, "confidence": e["confidence"]}
OUT.write_text(json.dumps({"_meta": {"built_by": "Code_Experiments/rulebook/build_verb_roots.py", "date": "2026-11-02",
                                      "source_sha256_16": hashlib.sha256(SRC.read_bytes()).hexdigest()[:16],
                                      "status": "corpus-mined, NOT speaker-verified"},
                           "roots": table}, ensure_ascii=False, sort_keys=True), encoding="utf-8")
print(OUT, len(table), "Swahili roots;", "pigan ->", table.get("pigan"), "| fany ->", table.get("fany"))

"""
validate.py

2026-11-12, CLAUDE_rulebook.md §6: validation before freeze. Results -> rulebook/validation_results.json
(config + hashes). Dev = the held-out 10% of train (split_manifest.json dev_ids, seed 20261111), never
used to build rules, lexicon or the retrieval index.

1. Speaker-verified regeneration: every verified verb form / concord form the tools can express is
   regenerated from its features; every miss listed.
2. Round-trip on dev: Kidaw'ida verbs in dev whose shape the tools can read (prefix via
   kitaita_verb.parse_prefix, a known lexicon root, a known final) -> features -> generate_verb ->
   exact match, per slot type (TAM / NEG / EXT / FV) with n and Wilson CIs. The generator's ceiling.
3. Dictionary precision against speaker-verified pairs (held-out folds from bootstrap_lexicon.py)
   and coverage on dev by POS (lexicon_report.json).
4. Leakage: no excluded / dropped / dev pair in the retrieval index, rule examples or lexicon evidence.
5. Unit tests (agent_translate/tests).
6. Version B unchanged: hf_space tests and eval_version_b.py --set val give the committed numbers.
"""
from __future__ import annotations

import collections
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hf_space"))
from agent_translate import tools as T  # noqa: E402
from api.stats import rate_ci  # noqa: E402

PY = Path.home() / "languagemodels" / "bert-env" / "bin" / "python"
DEV = Path.home() / "languagemodels" / "rulebook_data" / "dev.jsonl"
OUT = HERE / "validation_results.json"


def sha16(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------- 1
def speaker_regeneration() -> dict:
    import kitaita_verified_corrections as V
    cases = [("wiluieghe (verified sentence)", "verb", dict(dav_root="lw", subj="wi", tense="rel_past_cleft",
                                                            exts=["applicative"]), "wiluieghe"),
             ("ne-re-kogh-e 1sg past (s84)", "verb", dict(dav_root="kogh", subj="1sg", tense="past"), "nerekoghe")]
    for (person, tense), marker in V.PERSON_SUBJECT_MARKERS_BY_TAM.items():
        if (person, tense) in (("1pl", "past"), ("2sg", "ka")):
            continue   # not verified (hypotheses in the source file)
        key = {"future": "future", "present": "present", "past": "past"}[tense]
        exp_prefix = marker + {"future": "cha", "present": "we", "past": "re"}[tense]
        cases.append((f"{person} {tense} SM {marker}", "prefix", dict(dav_root="kab", subj=person, tense=key), exp_prefix))
    for lab, a in V.PRESENT_SM_VOWEL_CONFIRMED.items():
        if lab == "gh":
            continue   # conflict recorded (speaker ghwa vs corpus gha)
        cases.append((f"present SM+a {lab} = {a}", "prefix", dict(dav_root="kab", subj=lab, tense="present"), a + "we"))
    for lab, tam in V.VERIFIED_PAST_TAM_BY_SM.items():
        cases.append((f"past TAM {lab} = -{tam}-", "prefix", dict(dav_root="kab", subj=lab, tense="past"), None, tam))
    for deixis, table in (("near", V.KITAITA_DEM_PROXIMAL_VERIFIED), ("mid", V.KITAITA_DEM_REFERENTIAL_VERIFIED),
                          ("far", V.KITAITA_DEM_DISTAL_VERIFIED)):
        for lab, f in table.items():
            cases.append((f"DEM {deixis} {lab}", "modifier", (lab, "x", f"dem_{deixis}"), f))
    for lab, f in V.KITAITA_ASSOCIATIVE_VERIFIED.items():
        cases.append((f"ASSOC {lab}", "modifier", (lab, "x", "assoc"), f))
    nclass = {"mungu": "u", "mti": "ghu", "nchi": "ji", "nyumba": "i_nasal", "siku": "ji", "kikombe": "chi",
              "kiti": "chi", "kitabu": "chi"}
    for (adj, noun), f in V.VERIFIED_ADJECTIVES.items():
        if adj == "kubwa" and noun in nclass:
            cases.append((f"kubwa {noun} = {f}", "modifier", (nclass[noun], "baa", "adj"), f))
    hits, misses = 0, []
    for c in cases:
        name, kind, arg, exp = c[0], c[1], c[2], c[3]
        if kind == "verb":
            got = T.generate_verb(arg).get("form")
            ok = got == exp
        elif kind == "prefix":
            got = T.generate_verb(arg).get("form") or ""
            ok = got.startswith(exp) if exp else (c[4] in got[:6])
        else:
            got = T.generate_modifier(*arg).get("form")
            ok = got == exp
        hits += ok
        if not ok:
            misses.append({"case": name, "expected": exp or f"-{c[4]}-", "got": got})
    return {"n": len(cases), "k": hits, "rate": rate_ci(hits, len(cases)), "misses": misses}


# ---------------------------------------------------------------- 2
INV_AFF = {(v[0], v[1], v[2], v[3]): k for k, v in T.TAM_TABLE.items() if k not in ("present_zero", "perfect_ie",
                                                                                    "rel_past", "rel_past_cleft")}
FINAL_KIND = [("ieghe", "perf", True), ("ireghe", "perf", True), ("ereghe", "perf", True), ("eghe", "perf", True),
              ("agha", "agha", False), ("ie", "perf", False), ("ire", "perf", False), ("ere", "perf", False),
              ("a", "a", False), ("e", "e", False)]
EXT_SURF = [("applicative", ("ir", "er", "i", "y")), ("passive", ("lw", "w")), ("causative", ("ish", "esh")),
            ("stative", ("ik", "ek", "k")), ("reciprocal", ("an", "n"))]


def read_dav_verb(word: str, roots: set) -> dict | None:
    """Recover generate_verb features from a reference Kidaw'ida verb: prefix (parse_prefix), known root,
    optional extension, final. None if the shape is not one the tools generate."""
    from kitaita_verb import parse_prefix
    best = None
    for j in range(1, len(word) - 2):
        for k in range(j + 2, len(word)):
            root = word[j:k]
            if root not in roots:
                continue
            tail = word[k:]
            ext, rest = [], tail
            for et, surfs in EXT_SURF:
                s = next((x for x in surfs if rest.startswith(x) and len(rest) > len(x)), None)
                if s:
                    ext, rest = [et], rest[len(s):]
                    break
            fk = next(((kind, ghe) for f, kind, ghe in FINAL_KIND if rest == f), None)
            if not fk:
                continue
            for rd in parse_prefix(word[:j]):
                if rd["om"] or rd["row"] is None:
                    continue
                cand = (len(root), rd, root, ext, fk)
                if best is None or cand[0] > best[0]:
                    best = cand
    if not best:
        return None
    _, rd, root, ext, (kind, ghe) = best
    subj = rd["row"].replace("c", "") if rd["row"].startswith("c") else rd["row"]
    subj = {"2": "wi", "3": "ghu", "9": "i", "5": "ji", "6": "ghi", "7": "chi", "8": "va", "10": "ri", "11": "lu",
            "15": "ku"}.get(subj, subj)
    if rd["neg"] == "nde":
        neg_key = {("bare", None, "agha"): "present", ("e", "re", "perf"): "past", ("bare", "chaa", "a"): "future",
                   ("bare", None, "e"): "not_yet"}.get((rd["variant"], rd["tam"], kind))
        if not neg_key:
            return None
        return {"dav_root": root, "subj": subj, "tense": neg_key, "neg": True, "exts": ext, "post_final": ghe,
                "_slot": "NEG"}
    if rd["neg"]:
        return None
    key = INV_AFF.get((rd["variant"], rd["tam"], kind, ghe if kind == "perf" else False))
    if not key:
        return None
    return {"dav_root": root, "subj": subj, "tense": key, "neg": False, "exts": ext, "post_final": ghe,
            "_slot": "EXT" if ext else ("FV" if kind == "perf" else "TAM")}


def round_trip(max_words: int = 4000) -> dict:
    roots = {c["lemma"] for e in T._lexicon().values() if e["pos"] == "V" for c in e["dav"] if len(c["lemma"]) >= 2}
    seen, per = set(), collections.defaultdict(lambda: [0, 0])
    misses = collections.defaultdict(list)
    for line in open(DEV, encoding="utf-8"):
        r = json.loads(line)
        for w in r["dav"].split():
            w = T._norm(w)
            if not w or w in seen or len(w) < 5:
                continue
            seen.add(w)
            feat = read_dav_verb(w, roots)
            if not feat:
                continue
            slot = feat.pop("_slot")
            got = T.generate_verb(feat).get("form")
            per[slot][0] += got == w
            per[slot][1] += 1
            if got != w and len(misses[slot]) < 8:
                misses[slot].append({"reference": w, "features": feat, "generated": got})
            if sum(v[1] for v in per.values()) >= max_words:
                break
    out = {s: {"k": k, "n": n, "rate": rate_ci(k, n)} for s, (k, n) in per.items()}
    tot = [sum(v[0] for v in per.values()), sum(v[1] for v in per.values())]
    out["all"] = {"k": tot[0], "n": tot[1], "rate": rate_ci(*tot)}
    return {"per_slot": out, "misses": misses, "n_dav_types_read": len(seen)}


# ---------------------------------------------------------------- 4
def leakage() -> dict:
    man = json.loads((HERE / "split_manifest.json").read_text())
    bad = set(man["excluded_ids"]) | set(man["dropped_train_ids"]) | set(man["dev_ids"])
    index_ids = [json.loads(l)["id"] for l in open(T.TRAIN, encoding="utf-8")]
    rule_ids = []
    for r in json.loads((HERE / "rules.json").read_text())["rules"]:
        for e in r.get("examples", []) + r.get("counterexamples", []):
            rule_ids.append(e.get("pair_id"))
        for e in (r.get("word_level") or {}).get("examples", []):
            rule_ids.append(e.get("pair_id"))
    lex_ids = [p for e in json.loads((HERE / "bootstrap_lexicon.json").read_text())["entries"]
               for p in e.get("pair_ids", [])]
    res = {"index_pairs": len(index_ids), "index_leaks": sorted(set(index_ids) & bad)[:10],
           "rule_example_ids": len(rule_ids), "rule_leaks": sorted(set(rule_ids) & bad)[:10],
           "lexicon_evidence_ids": len(lex_ids), "lexicon_leaks": sorted(set(lex_ids) & bad)[:10]}
    res["pass"] = not (res["index_leaks"] or res["rule_leaks"] or res["lexicon_leaks"])
    return res


# ---------------------------------------------------------------- 5 + 6
def run(cmd, cwd=ROOT, timeout=3600):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr)[-1500:]


def version_b_unchanged() -> dict:
    res_path = ROOT / "Code_Experiments" / "rulebook" / "eval_version_b_val_results.json"
    committed = json.loads(subprocess.run(["git", "show", "HEAD:Code_Experiments/rulebook/eval_version_b_val_results.json"],
                                          cwd=ROOT, capture_output=True, text=True).stdout)
    backup = res_path.with_suffix(".json.bak_validate")
    shutil.copy(res_path, backup)
    try:
        code, tail = run([str(PY), "Code_Experiments/rulebook/eval_version_b.py", "--set", "val"])
        new = json.loads(res_path.read_text()) if code == 0 else None
    finally:
        shutil.move(backup, res_path)    # restore the tracked file exactly

    def nums(d):
        return {k: (v["B_chrf"]["mean"], v["changed"]) for k, v in d["conditions"].items()} | {"A": d["A_chrf"]["mean"]}
    # compare the conditions both runs have; conditions the committed file predates are listed separately
    c, n = nums(committed), (nums(new) if new else {})
    shared = sorted(set(c) & set(n))
    diffs = {k: (c[k], n[k]) for k in shared if c[k] != n[k]}
    return {"exit_code": code, "same_numbers": new is not None and not diffs, "n_shared_conditions": len(shared),
            "diffs": diffs, "new_conditions_not_in_committed_file": sorted(set(n) - set(c)),
            "log_tail": None if new is not None and not diffs else tail}


def main():
    res = {"_meta": {"built_by": "rulebook/validate.py", "script_sha16": sha16(Path(__file__)),
                     "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                               text=True).stdout.strip(),
                     "hashes_sha256_16": {p: sha16(HERE / p) for p in ("rules.json", "lexicon.json",
                                          "bootstrap_lexicon.json", "dav_vocab.json", "split_manifest.json")}
                     | {"tools.py": sha16(ROOT / "agent_translate" / "tools.py"), "dev.jsonl": sha16(DEV),
                        "train.jsonl (retrieval index source)": sha16(T.TRAIN)}}}
    res["1_speaker_regeneration"] = speaker_regeneration()
    print("1", res["1_speaker_regeneration"]["k"], "/", res["1_speaker_regeneration"]["n"], flush=True)
    res["2_round_trip_dev"] = round_trip()
    print("2", json.dumps({k: (v["k"], v["n"]) for k, v in res["2_round_trip_dev"]["per_slot"].items()}), flush=True)
    boot = json.loads((HERE / "bootstrap_lexicon.json").read_text())["_meta"]
    lexr = json.loads((HERE / "lexicon_report.json").read_text())
    res["3_dictionary"] = {"heldout_speaker_seed_precision": boot.get("heldout_seed_check"),
                           "dev_coverage_swahili_tokens": lexr["dev_coverage_swahili_tokens"]}
    res["4_leakage"] = leakage()
    print("4", res["4_leakage"]["pass"], flush=True)
    code, tail = run([str(PY), "-m", "pytest", "-q", "agent_translate/tests"])
    res["5_unit_tests"] = {"exit_code": code, "tail": tail[-300:]}
    print("5", code, tail[-120:], flush=True)
    code, tail = run([str(PY), "-m", "pytest", "-q", "hf_space/tests", "-x", "-k", "not real_pipeline"])
    res["6a_hf_space_tests"] = {"exit_code": code, "tail": tail[-400:],
                                "note": "not run if fastapi is missing locally (no local env has it); hf_space/ code "
                                        "was not changed by this job"}
    print("6a", code, tail[-150:], flush=True)
    res["6b_version_b_val"] = version_b_unchanged()
    print("6b", res["6b_version_b_val"]["same_numbers"], flush=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

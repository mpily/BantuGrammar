"""
build_lexicon.py

2026-11-12, CLAUDE_rulebook.md §4. Merges the bootstrapped lexicon (bootstrap_lexicon.json: speaker
seeds, rule-book concord seeds, identical strings, loop pairs, corpus-gated agent pairs) into one
schema, keyed by Swahili lemma, candidates ranked and never collapsed:

  {"sw": "kondoo", "sw_forms": [...], "pos": "N",
   "dav": [{"lemma": "ngondi", "forms": ["ngondi"], "source": "speaker_verified", "evidence_count": 12,
            "domains": {"dav_swa": 9, "bible": 3}, "class": "i", "class_source": "speaker_verified",
            "plural": null, "senses": ["sheep"], "tier": "seed", "conflict": false}, ...],
   "source": <top candidate's>, "evidence_count": <top>, "domains": <top>, "conflict": <any>}

Source priority (CLAUDE_rulebook.md §1.4, extended): speaker_verified > corpus_attested (identical
strings, rule-book concord seeds, loop pairs with >= 2 shared pairs) > mined_alignment (other loop
pairs) > agent_interpretation (corpus-gated agent proposals; ranked last: highest audit error rate).
Noun class: speaker-verified table, else concord words after the noun on clean train (>= 3 votes,
top >= 60%), else the root-list agent's class. Coverage is reported on the dev slice (never used to
build anything here).
Outputs: rulebook/lexicon.json (word pairs, counts; no sentences), rulebook/lexicon_report.json.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))
import attest as A  # noqa: E402
import bootstrap_lexicon as B  # noqa: E402
import build_rules as R  # noqa: E402
import segment as G  # noqa: E402
import kitaita_verified_corrections as V  # noqa: E402
from api.stats import rate_ci  # noqa: E402
from rulebook_flagger import norm  # noqa: E402

OUT = HERE / "lexicon.json"
REPORT = HERE / "lexicon_report.json"
DEV = Path.home() / "languagemodels" / "rulebook_data" / "dev.jsonl"
PRIORITY = {"speaker_verified": 0, "corpus_attested": 1, "mined_alignment": 2, "agent_interpretation": 3}
CLASS_MIN_VOTES, CLASS_MIN_SHARE = 3, 0.6


def source_of(e: dict) -> str:
    s, t = e.get("source", ""), e.get("tier")
    if s == "speaker_verified":
        return "speaker_verified"
    if s == "agent_interpretation":
        return "agent_interpretation"
    if s in ("identical_string",) or s.startswith("rules.json"):
        return "corpus_attested"
    if t == "accepted" or (e.get("cooc", 0) >= 2 and t == "tentative" and e.get("elimination", 0) >= 2):
        return "corpus_attested"
    return "mined_alignment"


def main():
    lex = json.loads(B.OUT.read_text(encoding="utf-8"))
    sw_lem, dv_lem = B.lemmatizers()
    ps = A.pairs()
    sw_forms, dv_forms = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    idx_sw, idx_dv = collections.defaultdict(set), collections.defaultdict(set)
    for i, r in enumerate(ps):
        for w in r["sw"]:
            l = sw_lem.get(w, w)
            sw_forms[l][w] += 1
            idx_sw[l].add(i)
        for w in r["dv"]:
            l = dv_lem.get(w, w)
            dv_forms[l][w] += 1
            idx_dv[l].add(i)
    # noun class: speaker table, else concord votes on clean train, else root-list agent
    verified_cls = {norm(n): c for n, c in V.VERIFIED_SUBJECT_MARKERS.items()}
    vote_forms = collections.defaultdict(set)
    for lab, (_, row) in R.CLASSES.items():
        a0, e0 = R.forms(row, 1)[0], R.forms(row, 2)[0]
        for f in [a0, R.merge_a(a0, "ose")] + [(a0 if v == "a" else e0) + s for v, s, _ in R.POSS.values()]:
            vote_forms[f].add(lab)
    # rulebook-v2: classes whose modifier forms are all identical (gh / ghu; i / i_nasal) vote as one
    # group. v1 split each token between them, so neither could reach the 60% share: wundu, 1543 concord
    # tokens, stayed unclassed. Only the gh / ghu group is resolved (to ghu; gh is the one-noun class of
    # wukula); an i / i_nasal win stays unresolved, as in v1.
    forms_of = collections.defaultdict(set)
    for f, cs in vote_forms.items():
        for c in cs:
            forms_of[c].add(f)
    group = {c: frozenset(d for d in forms_of if forms_of[d] == forms_of[c]) for c in forms_of}
    RESOLVE = {frozenset({"gh", "ghu"}): "ghu"}
    votes = collections.defaultdict(collections.Counter)
    for r in ps:
        t = r["dv"]
        for j in range(len(t) - 1):
            cs = vote_forms.get(t[j + 1])
            if cs and len(t[j]) >= 3:
                gs = {group[c] for c in cs}
                for g in gs:
                    votes[t[j]][g] += 1 / len(gs)

    def voted_class(form: str):
        c = votes.get(form)
        if not c or sum(c.values()) < CLASS_MIN_VOTES:
            return None
        top, v = c.most_common(1)[0]
        if v / sum(c.values()) < CLASS_MIN_SHARE:
            return None
        return next(iter(top)) if len(top) == 1 else RESOLVE.get(top)
    wu_rule = next(r for r in json.loads((HERE / "rules.json").read_text())["rules"]
                   if r["id"] == "CLASS.wu_prefix.modifier_ghu")
    WU = re.compile(wu_rule["evidence"].get("pattern", r"^w'?u[^aeiou']"))

    def mod_class(form: str, cls: str | None):
        """Class for modifiers (adj, dem, poss, ose, num). Differs from the subject class only for wu- nouns."""
        if not WU.match(form):
            return cls, "same_as_class"
        v = voted_class(form)
        if v:
            return v, "corpus_concord_votes"
        return "ghu", f"rules.json:{wu_rule['id']}"
    agent_info = {}
    for f in sorted(G.AGENT_DIR.glob("segment*/out_agent*.jsonl")):
        for line in open(f, encoding="utf-8"):
            o = json.loads(line)
            if o.get("type") == "root":
                agent_info[(o["side"], norm(o["word"]))] = o
    meanings = collections.defaultdict(collections.Counter)
    for f in sorted(G.AGENT_DIR.glob("round*/out_agent*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") in ("proposal", "review") and o.get("meaning"):
                sw = o.get("sw_lemma") if o.get("type") == "review" else sw_lem.get(norm(o.get("sw_word") or ""), "")
                dv = o.get("dav_lemma") if o.get("type") == "review" else dv_lem.get(norm(o.get("dav_word") or ""), "")
                if sw and dv:
                    meanings[(sw, dv)][str(o["meaning"]).strip().lower()[:60]] += 1
    conflicts = {r["then"].get("default_class"): r for r in json.loads((HERE / "rules.json").read_text())["rules"]
                 if r["id"].startswith("NOUN.conflict.")}
    conflict_nouns = {r["if"]["kitaita_noun"]: r for r in conflicts.values()}

    def noun_class(form: str):
        if form in verified_cls:
            return verified_cls[form], "speaker_verified"
        v = voted_class(form)
        if v:
            return v, "corpus_concord_votes"
        a = agent_info.get(("dav", form))
        if a and a.get("noun_class"):
            return str(a["noun_class"]), "agent_root_list"
        return None, None

    vocab = collections.Counter(w for r in ps for w in r["dv"])
    swaps = [(r["if"]["sg_class"], r["then"]["pl_class"], r["if"]["prefix"], r["then"]["rewrite"].split(" -> ")[1])
             for r in json.loads((HERE / "rules.json").read_text())["rules"] if r["id"].startswith("PLURAL.")
             and r["id"] != "PLURAL.number_invariant"]

    def plural(form: str, cls: str | None):
        if form in V.NUMBER_INVARIANT_NOUNS:
            return form
        for sg, pl, pat, rep in swaps:
            if cls == sg:
                cand = re.sub(pat, rep, form, count=1)
                if cand != form and vocab.get(cand, 0) >= 1:
                    return cand
        return None

    by_sw = collections.defaultdict(list)
    for e in lex["entries"]:
        by_sw[e["sw_lemma"]].append(e)
    entries = []
    for swl, es in by_sw.items():
        is_verb = any(A.sw_parse(w) for w in sw_forms.get(swl, {}) )
        cands = []
        for e in es:
            dl = e["dav_lemma"]
            both = idx_sw[swl] & idx_dv[dl]
            dom = collections.Counter(ps[i]["src"] for i in both)
            forms = [w for w, _ in dv_forms.get(dl, collections.Counter()).most_common(3)] or [e.get("dav_word", dl)]
            cls, cls_src = (None, None) if is_verb else noun_class(forms[0])
            mcls, mcls_src = (None, None) if is_verb else mod_class(forms[0], cls)
            src = source_of(e)
            conf = forms[0] in conflict_nouns
            cands.append({"lemma": dl, "forms": forms, "source": src, "tier": e["tier"],
                          "evidence_count": len(both), "domains": {"dav_swa": dom.get("davswa", 0), "bible": dom.get("bible", 0)},
                          "class": cls, "class_source": cls_src, "mod_class": mcls, "mod_class_source": mcls_src,
                          "plural": None if is_verb else plural(forms[0], cls),
                          "senses": [m for m, _ in meanings.get((swl, dl), collections.Counter()).most_common(3)],
                          "conflict": conf,
                          **({"conflict_readings": conflict_nouns[forms[0]].get("readings")} if conf else {})})
        cands.sort(key=lambda c: (PRIORITY[c["source"]], -c["evidence_count"]))
        ai = agent_info.get(("sw", next(iter(sw_forms.get(swl, {swl: 1})))), {})
        pos = "V" if is_verb else (ai.get("pos") if ai else None) or "N"
        top = cands[0]
        entries.append({"sw": swl, "sw_forms": [w for w, _ in sw_forms.get(swl, collections.Counter()).most_common(4)],
                        "pos": pos, "dav": cands, "source": top["source"], "evidence_count": top["evidence_count"],
                        "domains": top["domains"], "conflict": any(c["conflict"] for c in cands),
                        "close_alternatives": len(cands) > 1 and cands[1]["evidence_count"] >= 0.5 * max(1, top["evidence_count"])})
    entries.sort(key=lambda e: e["sw"])
    if "--v2-patch" in sys.argv:
        # rulebook-v2 = v1 lexicon + new class / mod_class / plural only. A plain rebuild also moves ~8 entries'
        # forms and counts (tie order in the lemmatizers), which is not part of the v2 change.
        import subprocess
        base = json.loads(subprocess.run(["git", "show", "rulebook-v1:rulebook/lexicon.json"], cwd=HERE.parent,
                                         capture_output=True, text=True, check=True).stdout)["entries"]
        new_c = {(e["sw"], c["lemma"]): c for e in entries for c in e["dav"]}
        for e in base:
            for c in e["dav"]:
                n = new_c.get((e["sw"], c["lemma"]))
                if n is None or e["pos"] == "V":
                    c["mod_class"], c["mod_class_source"] = (None, None) if e["pos"] == "V" else (c["class"], "same_as_class")
                    continue
                if c["class"] is None and n["class"] is not None:      # the only class change v2 allows
                    c["class"], c["class_source"], c["plural"] = n["class"], n["class_source"], n["plural"]
                if c["class"] == n["class"]:
                    c["mod_class"], c["mod_class_source"] = n["mod_class"], n["mod_class_source"]
                else:
                    c["mod_class"], c["mod_class_source"] = mod_class(c["forms"][0], c["class"])
        entries = base

    # dev coverage by POS (dev never used above)
    csw, cdv = B.concord_words()
    known = {e["sw"] for e in entries}
    cov = collections.defaultdict(lambda: [0, 0])
    for line in open(DEV, encoding="utf-8"):
        r = json.loads(line)
        for w in A.toks(r["swa"]):
            if w in csw:
                pos = "CONCORD"
                ok = True
            else:
                l = sw_lem.get(w)
                if l is None:
                    p = A.sw_parse(w)
                    l = p["root"] if p and len(p["root"]) >= 3 else (G.sw_root(w) or w)
                p = A.sw_parse(w)
                pos = "V" if p else ("FUNC" if len(w) <= 3 else "N/other")
                ok = l in known
            cov[pos][0] += ok
            cov[pos][1] += 1
    coverage = {k: {"covered": a, "n": b, "rate": rate_ci(a, b)} for k, (a, b) in cov.items()}
    tot = [sum(v[0] for v in cov.values()), sum(v[1] for v in cov.values())]
    coverage["all"] = {"covered": tot[0], "n": tot[1], "rate": rate_ci(*tot)}
    meta = {"built_by": "rulebook/build_lexicon.py", "script_sha16": A.sha16(Path(__file__)),
            "bootstrap_lexicon_sha16": A.sha16(B.OUT), "rules_sha16": A.sha16(HERE / "rules.json"),
            "train_sha16": A.sha16(A.TRAIN), "priority": list(PRIORITY), "n_entries": len(entries),
            "n_candidates": sum(len(e["dav"]) for e in entries),
            "by_source": dict(collections.Counter(e["source"] for e in entries)),
            "by_pos": dict(collections.Counter(e["pos"] for e in entries)),
            "nouns_with_class": sum(1 for e in entries for c in e["dav"][:1] if c["class"]),
            "class_sources": dict(collections.Counter(c["class_source"] for e in entries for c in e["dav"][:1] if c["class"])),
            "close_alternatives": sum(e["close_alternatives"] for e in entries),
            "mod_class_differs_from_class": sum(1 for e in entries for c in e["dav"][:1]
                                                if c["mod_class"] and c["mod_class"] != c["class"]),
            "mod_class_sources": dict(collections.Counter(c["mod_class_source"] for e in entries for c in e["dav"][:1]
                                                          if c["mod_class_source"])),
            "rulebook_version": "v2 (2026-11-12): look-alike classes vote as one group (gh/ghu resolved to ghu) + wu- modifier class",
            "dev_coverage_swahili_tokens": coverage}
    OUT.write_text(json.dumps({"_meta": meta, "entries": entries}, ensure_ascii=False, indent=0))
    REPORT.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "dev_coverage_swahili_tokens"}, ensure_ascii=False))
    for k, v in coverage.items():
        print(k, v["covered"], "/", v["n"], round(v["rate"]["rate"], 3), [round(x, 3) for x in v["rate"]["ci95"]])


if __name__ == "__main__":
    main()

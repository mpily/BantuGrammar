"""
agent_round5_merge.py

2026-11-12. Scores the final dictionary review (agent round 5) and turns it into decisions for rulebook-v5
(build_lexicon.py --v5-patch applies them). The merge rules below were written before any agent output
was read.

Review items (agent_review_batches.py, one agent per entry):
  - controls (blinded speaker-verified entries): verdict counts with Wilson CI. If the upper CI of the
    "wrong" rate on them is above 20%, agent removals are not applied (agents reject good entries too often).
  - best = another existing candidate, verdict wrong/partly, confidence high/medium -> it moves to the top.
  - new_dav_lemma, verdict wrong, confidence high/medium -> new top candidate (source agent_interpretation)
    only with corpus support: pairs with the Swahili lemma and the Kidaw'ida word >= 1 (high) / >= 3 (medium),
    the same gate as earlier rounds (AGENT_MIN_COOC).
  - verdict wrong / not_vocabulary, nothing better, confidence high -> the first candidate is removed (the
    entry too if it was the only one); medium -> demoted below the others.
  - non-verbs: a full `dav_word` that is one of the candidate's forms, or attested >= 2 times in clean train,
    becomes forms[0] (what lookup_word returns as `word`: the idan -> idana problem).
  - speaker_verified candidates are never removed, demoted or outranked.
Connectives (agent_connective_batch.py, two agents on the same items):
  - only renderings both agents give (confidence >= medium) with >= 3 pairs where the Swahili phrase and the
    Kidaw'ida rendering co-occur; the best supported becomes the first candidate (pos FUNC), the current
    candidates follow. Disagreements are listed, not applied.

Writes rulebook/agent_round5_decisions.json (ids and words only, no corpus text) and
rulebook/agent_rounds/round5_score.json. Agent verdicts are an LLM's judgement, not speaker verification.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "hf_space"))
import attest as A  # noqa: E402
import bootstrap_lexicon as B  # noqa: E402
from api.stats import rate_ci  # noqa: E402

R5 = Path.home() / "languagemodels" / "rulebook_data" / "agents" / "round5"
CONF_OK = ("high", "medium")
MIN_SUPPORT = {"high": 1, "medium": 3}
CONTROL_MAX_WRONG_UPPER = 0.20


def load_jsonl(p: Path) -> tuple[list[dict], int]:
    rows, bad = [], 0
    if not p.exists():
        return rows, bad
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                bad += 1
    return rows, bad


def norm(s):
    return " ".join(A.toks(s or ""))


def main():
    boot = json.loads(B.OUT.read_text(encoding="utf-8"))
    swl, dvl = boot["lemmas"]["sw"], boot["lemmas"]["dav"]
    # always the frozen v4 lexicon (the working copy may already be a v5 build)
    import subprocess
    v4 = subprocess.run(["git", "show", "rulebook-v4:rulebook/lexicon.json"], cwd=HERE.parent, capture_output=True,
                        text=True, check=True).stdout
    lex = {e["sw"]: e for e in json.loads(v4)["entries"]}
    ps = A.pairs()
    idx_sw, idx_dvl, idx_dvw = (collections.defaultdict(set) for _ in range(3))
    dv_count = collections.Counter()
    for i, r in enumerate(ps):
        for w in set(r["sw"]):
            idx_sw[swl.get(w, w)].add(i)
        for w in set(r["dv"]):
            idx_dvl[dvl.get(w, w)].add(i)
            idx_dvw[w].add(i)
        dv_count.update(r["dv"])

    def dav_ids(word=None, lemma=None):
        s = set()
        if word:
            s |= idx_dvw.get(norm(word), set())
        if lemma:
            s |= idx_dvl.get(lemma, set())
        return s

    def domains(ids):
        return {"dav_swa": sum(ps[i]["src"] == "davswa" for i in ids), "bible": sum(ps[i]["src"] != "davswa" for i in ids)}

    # ------------------------------------------------------------------ review items
    batches, outs, bad_lines = {}, {}, 0
    for k in range(1, 11):
        b = json.loads((R5 / f"batch_agent{k}.json").read_text(encoding="utf-8"))
        batches.update({it["key"]: it for it in b["items"]})
        rows, bad = load_jsonl(R5 / f"out_agent{k}.jsonl")
        bad_lines += bad
        for r in rows:
            if r.get("key") in batches and r["key"] not in outs:
                outs[r["key"]] = r
    controls = json.loads((R5 / "controls.json").read_text(encoding="utf-8"))
    ctrl_rows = [outs[k] for k in controls if k in outs]
    cv = collections.Counter(r.get("verdict") for r in ctrl_rows)
    ctrl_wrong = rate_ci(cv["wrong"] + cv["not_vocabulary"], len(ctrl_rows))
    removals_ok = bool(ctrl_rows) and ctrl_wrong["ci95"][1] <= CONTROL_MAX_WRONG_UPPER

    decisions, skipped = [], collections.Counter()
    verdicts = collections.Counter()
    for key, r in outs.items():
        if key in controls:
            continue
        it, e = batches[key], lex.get(batches[key]["sw"])
        verdicts[(r.get("verdict"), r.get("confidence"))] += 1
        if not e or e["source"] == "speaker_verified":
            continue
        conf, verdict, top = r.get("confidence"), r.get("verdict"), e["dav"][0]
        lemmas = [c["lemma"] for c in e["dav"]]
        best = r.get("best")
        if verdict in ("wrong", "partly") and conf in CONF_OK and best in lemmas[1:]:
            decisions.append({"sw": e["sw"], "action": "promote", "lemma": best, "was": top["lemma"], "confidence": conf})
        elif verdict == "wrong" and conf in CONF_OK and r.get("new_dav_lemma"):
            word = r.get("dav_word") if e["pos"] != "V" else None
            ids = idx_sw[e["sw"]] & dav_ids(word, r["new_dav_lemma"])
            if len(ids) >= MIN_SUPPORT[conf]:
                decisions.append({"sw": e["sw"], "action": "add_top", "lemma": r["new_dav_lemma"],
                                  "word": norm(word) or None, "evidence_count": len(ids), "domains": domains(ids),
                                  "was": top["lemma"], "confidence": conf, "meaning": r.get("meaning")})
            else:
                skipped["new_word_without_support"] += 1
        elif verdict in ("wrong", "not_vocabulary") and not best and conf == "high":
            if removals_ok:
                decisions.append({"sw": e["sw"], "action": "remove_top", "was": top["lemma"], "verdict": verdict,
                                  "entry_removed": len(e["dav"]) == 1})
            else:
                skipped["removal_blocked_by_controls"] += 1
        elif verdict in ("wrong", "not_vocabulary") and not best and conf == "medium" and len(e["dav"]) > 1:
            decisions.append({"sw": e["sw"], "action": "demote_top", "was": top["lemma"]})
        # full word for non-verbs, on the candidate that ends up first
        w = norm(r.get("dav_word"))
        if e["pos"] != "V" and w and verdict in ("correct", "partly") and (best in (None, top["lemma"])):
            if w != (top.get("forms") or [None])[0] and (w in (top.get("forms") or []) or dv_count[w] >= 2):
                decisions.append({"sw": e["sw"], "action": "set_word", "lemma": top["lemma"], "word": w,
                                  "was": (top.get("forms") or [None])[0]})

    # ------------------------------------------------------------------ connectives
    cb = {it["key"]: it for it in json.loads((R5 / "batch_connectives.json").read_text(encoding="utf-8"))["items"]}
    ca, bad_a = load_jsonl(R5 / "out_connectives_A.jsonl")
    cbb, bad_b = load_jsonl(R5 / "out_connectives_B.jsonl")
    bad_lines += bad_a + bad_b

    def by_key(rows):
        d = collections.defaultdict(dict)
        for r in rows:
            if r.get("key") in cb and r.get("dav") and r.get("confidence") in CONF_OK:
                d[r["key"]].setdefault(norm(r["dav"]), (r.get("sense"), r["dav"].strip().lower()))
        return d
    da, db = by_key(ca), by_key(cbb)
    conn, conn_disagree = [], []
    for key, it in cb.items():
        ph = it["sw"].split()
        both = set(da.get(key, {})) & set(db.get(key, {}))
        sw_ids = {i for i, r in enumerate(ps) if any(r["sw"][j:j + len(ph)] == ph for j in range(len(r["sw"]) - len(ph) + 1))}
        sup = []
        for d in both:
            dt = d.split()
            ids = {i for i in sw_ids if any(ps[i]["dv"][j:j + len(dt)] == dt for j in range(len(ps[i]["dv"]) - len(dt) + 1))}
            if len(ids) >= 3:
                sup.append((len(ids), d, ids))
        if sup:
            sup.sort(reverse=True)
            conn.append({"sw": it["sw"], "action": "connective",
                         "renderings": [{"word": da[key][d][1], "evidence_count": n, "domains": domains(ids),
                                         "sense": da[key][d][0] or db[key][d][0]} for n, d, ids in sup]})
        elif da.get(key) or db.get(key):
            conn_disagree.append({"sw": it["sw"], "A": sorted(da.get(key, {})), "B": sorted(db.get(key, {})),
                                  "agreed_without_support": sorted(both)})

    score = {"round": 5, "review_items_answered": f"{len(outs)}/{len(batches)}", "bad_lines": bad_lines,
             "controls": {"n": len(ctrl_rows), "of": len(controls), "verdicts": dict(cv), "wrong_or_not_vocab": ctrl_wrong,
                          "removals_applied": removals_ok, "threshold_upper_ci": CONTROL_MAX_WRONG_UPPER},
             "verdicts_by_confidence": {f"{v}/{c}": n for (v, c), n in sorted(verdicts.items(), key=lambda x: -x[1])},
             "decisions": dict(collections.Counter(d["action"] for d in decisions)), "skipped": dict(skipped),
             "connectives": {"answered_A": len(set(r.get("key") for r in ca)), "answered_B": len(set(r.get("key") for r in cbb)),
                             "applied": len(conn), "disagreements": len(conn_disagree)}}
    (HERE / "agent_rounds").mkdir(exist_ok=True)
    (HERE / "agent_rounds" / "round5_score.json").write_text(json.dumps(score, ensure_ascii=False, indent=1))
    (HERE / "agent_round5_decisions.json").write_text(json.dumps(
        {"_meta": {"built_by": "rulebook/agent_round5_merge.py", "base": "rulebook-v4:rulebook/lexicon.json",
                   "outputs_sha16": {p.name: A.sha16(p) for p in sorted(R5.glob("out_*.jsonl"))},
                   "note": "agent judgement, not speaker verification"},
         "review": decisions, "connectives": conn, "connective_disagreements": conn_disagree},
        ensure_ascii=False, indent=1))
    print(json.dumps(score, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

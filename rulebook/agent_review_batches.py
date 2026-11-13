"""
agent_review_batches.py

2026-11-12. Final dictionary review (agent round 5, user request: "run through the dictionary one last
time"). Unlike rounds 1-4 (interpret unknown words + small audit), every entry of the frozen rulebook-v4
lexicon whose top candidate is not speaker-verified is shown to an agent with its ranked candidates and
clean-train evidence, most frequent Swahili lemmas first. Candidate sources are hidden (blind review).

Control: a random sample of speaker-verified entries is mixed in, looking like every other item. The agents'
"wrong" rate on them estimates how often an agent rejects a right entry (their keys are written to
controls.json in the round folder, which agents are told not to open).

Clean train only (attest.pairs); corpus text goes to ~/languagemodels/rulebook_data/agents/round5/ (not git).
Run: ~/languagemodels/bert-env/bin/python rulebook/agent_review_batches.py --agents 10
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attest as A  # noqa: E402
import bootstrap_lexicon as B  # noqa: E402

AGENT_DIR = Path.home() / "languagemodels" / "rulebook_data" / "agents"
MAXC = 220  # characters per example sentence


def clip(s: str) -> str:
    return s if len(s) <= MAXC else s[:MAXC] + " …"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=5)
    ap.add_argument("--agents", type=int, default=10)
    ap.add_argument("--controls-per-agent", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261112)
    a = ap.parse_args()
    rng = random.Random(a.seed + a.round)

    boot = json.loads(B.OUT.read_text(encoding="utf-8"))
    swl, dvl = boot["lemmas"]["sw"], boot["lemmas"]["dav"]
    lex = json.loads((HERE / "lexicon.json").read_text(encoding="utf-8"))
    ps = A.pairs()

    idx_sw, idx_dv = collections.defaultdict(set), collections.defaultdict(set)
    for i, r in enumerate(ps):
        for w in set(r["sw"]):
            idx_sw[swl.get(w, w)].add(i)
        for w in set(r["dv"]):
            idx_dv[dvl.get(w, w)].add(i)
            idx_dv["=" + w].add(i)  # surface form

    def dav_pairs(c):
        s = set(idx_dv[c["lemma"]])
        for f in c.get("forms") or []:
            s |= idx_dv["=" + f]
        return s

    def ex(i):
        return {"pair_id": ps[i]["id"], "corpus": ps[i]["src"], "sw": clip(ps[i]["swa"]), "dav": clip(ps[i]["dav"])}

    def shortest(ids, k):
        return [ex(i) for i in sorted(ids, key=lambda i: (len(ps[i]["sw"]), i))[:k]]

    def view(e, key):
        sw_ids = idx_sw[e["sw"]]
        cands, cov = [], set()
        for c in e["dav"][:5]:
            both = sw_ids & dav_pairs(c)
            cov |= both
            cands.append({"lemma": c["lemma"], "forms": (c.get("forms") or [])[:4], "evidence_count": c["evidence_count"],
                          "domains": c["domains"], "class": c.get("class"), "senses": c.get("senses") or [],
                          "examples": shortest(both, 2 if len(e["dav"]) == 1 else 1)})
        return {"key": key, "sw": e["sw"], "sw_forms": e.get("sw_forms", [])[:4], "pos": e["pos"],
                "n_train_pairs": len(sw_ids), "candidates": cands,
                "examples_without_any_candidate": shortest(sw_ids - cov, 2)}

    review = [e for e in lex["entries"] if e["source"] != "speaker_verified"]
    review.sort(key=lambda e: (-len(idx_sw[e["sw"]]), e["sw"]))
    spk = [e for e in lex["entries"] if e["source"] == "speaker_verified"]
    ctrl = rng.sample(spk, min(len(spk), a.agents * a.controls_per_agent))

    out = AGENT_DIR / f"round{a.round}"
    out.mkdir(parents=True, exist_ok=True)
    batches = [[] for _ in range(a.agents)]
    for n, e in enumerate(review):
        batches[n % a.agents].append(e)
    controls = {}
    for k in range(a.agents):
        items = [view(e, f"r{k + 1}-{j:04d}") for j, e in enumerate(batches[k])]
        for e in ctrl[k::a.agents]:
            pos = rng.randrange(len(items) + 1)
            key = f"r{k + 1}-c{len(controls):03d}"
            controls[key] = e["sw"]
            items.insert(pos, view(e, key))
        for j, it in enumerate(items):  # keys say nothing about control status
            old = it["key"]
            it["key"] = f"a{k + 1}-{j:04d}"
            if old in controls:
                controls[it["key"]] = controls.pop(old)
        (out / f"batch_agent{k + 1}.json").write_text(json.dumps({"round": a.round, "agent": k + 1, "items": items},
                                                                 ensure_ascii=False, indent=1))
    (out / "controls.json").write_text(json.dumps(controls, ensure_ascii=False, indent=1))
    print(json.dumps({"round": a.round, "review_entries": len(review), "controls": len(controls),
                      "per_agent": [len(b) for b in batches], "dir": str(out),
                      "lexicon_sha16": A.sha16(HERE / "lexicon.json")}))


if __name__ == "__main__":
    main()

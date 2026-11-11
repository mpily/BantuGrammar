"""
agent_batches.py

2026-11-11. Builds one agent round for the lexicon/rule loop (bootstrap_lexicon.py):
  - interpret: clean-train sentence pairs whose unknown words cover the most frequent
    still-unknown lemmas (greedy), each token shown with its lemma and known gloss;
  - review: entries to judge -- the ones the support check blocked (was the block
    right?), low-support tentative ones, and a RANDOM audit sample of accepted /
    tentative / agent entries (its error rate is the round-to-round convergence measure).
Clean train only. Corpus text goes to ~/languagemodels/rulebook_data/agents/round<N>/ (not git).
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--sentences-per-agent", type=int, default=60)
    ap.add_argument("--review-per-agent", type=int, default=50)
    ap.add_argument("--audit-per-agent", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20261111)
    a = ap.parse_args()
    rng = random.Random(a.seed + a.round)
    lex = json.loads(B.OUT.read_text(encoding="utf-8"))
    sw_lem = {**{w: w for w in []}, **lex["lemmas"]["sw"]}
    dv_lem = lex["lemmas"]["dav"]
    known = collections.defaultdict(dict)
    rev = collections.defaultdict(set)
    for e in lex["entries"]:
        known[e["sw_lemma"]][e["dav_lemma"]] = e
        rev[e["dav_lemma"]].add(e["sw_lemma"])
    ps = A.pairs()
    L = lambda w, m: m.get(w, w)  # noqa: E731

    # ---- interpret: greedy cover of frequent unknown lemmas
    unk_freq = collections.Counter()
    pair_unk = []
    for r in ps:
        us = {L(w, sw_lem) for w in r["sw"] if L(w, sw_lem) not in known}
        ud = {L(w, dv_lem) for w in r["dv"] if L(w, dv_lem) not in rev}
        pair_unk.append((us, ud))
        unk_freq.update({("sw", x) for x in us} | {("dav", x) for x in ud})
    cand = [i for i, (us, ud) in enumerate(pair_unk) if us and ud and len(ps[i]["sw"]) <= 14]
    covered, chosen = set(), []
    n_total = a.agents * a.sentences_per_agent
    while len(chosen) < n_total and cand:
        def gain(i):
            us, ud = pair_unk[i]
            return sum(unk_freq[k] for k in ({("sw", x) for x in us} | {("dav", x) for x in ud}) - covered)
        sample = rng.sample(cand, min(3000, len(cand)))
        best = max(sample, key=gain)
        if gain(best) == 0:
            break
        chosen.append(best)
        us, ud = pair_unk[best]
        covered |= {("sw", x) for x in us} | {("dav", x) for x in ud}
        cand.remove(best)

    def show(i):
        r = ps[i]
        sw = [{"w": w, "lemma": L(w, sw_lem), "known_dav": sorted(known.get(L(w, sw_lem), {}))[:3]} for w in r["sw"]]
        dv = [{"w": w, "lemma": L(w, dv_lem), "known_sw": sorted(rev.get(L(w, dv_lem), ()))[:3]} for w in r["dv"]]
        return {"pair_id": r["id"], "corpus": r["src"], "swahili": r["swa"], "kidawida": r["dav"],
                "sw_tokens": sw, "dav_tokens": dv}

    # ---- review
    idx_sw, idx_dv = collections.defaultdict(list), collections.defaultdict(list)
    for i, r in enumerate(ps):
        for w in set(r["sw"]):
            idx_sw[L(w, sw_lem)].append(i)
        for w in set(r["dv"]):
            idx_dv[L(w, dv_lem)].append(i)

    def entry_view(sw_l, dv_l, why, info=None):
        both = sorted(set(idx_sw[sw_l]) & set(idx_dv[dv_l]), key=lambda i: len(ps[i]["sw"]))[:3]
        only = sorted(set(idx_sw[sw_l]) - set(idx_dv[dv_l]), key=lambda i: len(ps[i]["sw"]))[:2]
        return {"sw_lemma": sw_l, "dav_lemma": dv_l, "why_reviewed": why,
                "stats": {k: (info or {}).get(k) for k in ("tier", "source", "support_sw", "support_dav", "n_sw",
                                                          "n_dav", "elimination", "votes")},
                "pairs_with_both": [{"pair_id": ps[i]["id"], "swahili": ps[i]["swa"], "kidawida": ps[i]["dav"]}
                                    for i in both],
                "pairs_with_sw_only": [{"pair_id": ps[i]["id"], "swahili": ps[i]["swa"], "kidawida": ps[i]["dav"]}
                                       for i in only]}

    blocked = [tuple(x.split("=", 1)) for x in lex["_meta"].get("blocked_as_errors", [])]
    rng.shuffle(blocked)
    nonseed = [e for e in lex["entries"] if e["tier"] in ("accepted", "tentative", "agent")]
    low = sorted([e for e in nonseed if e["tier"] == "tentative" and e.get("n_sw", 0) >= 3],
                 key=lambda e: e.get("support_sw", 0))
    audit = rng.sample(nonseed, min(len(nonseed), a.agents * a.audit_per_agent))
    n_rev = a.agents * (a.review_per_agent - a.audit_per_agent)
    review = [entry_view(x, y, "blocked_by_support_check") for x, y in blocked[: n_rev // 2]]
    review += [entry_view(e["sw_lemma"], e["dav_lemma"], "low_support_tentative", e) for e in low[: n_rev - len(review)]]
    audits = [entry_view(e["sw_lemma"], e["dav_lemma"], "random_audit", e) for e in audit]

    out = AGENT_DIR / f"round{a.round}"
    out.mkdir(parents=True, exist_ok=True)
    for k in range(a.agents):
        batch = {"round": a.round, "agent": k + 1,
                 "interpret": [show(i) for i in chosen[k::a.agents]],
                 "review": review[k::a.agents] + audits[k::a.agents]}
        (out / f"batch_agent{k + 1}.json").write_text(json.dumps(batch, ensure_ascii=False, indent=1))
    print(json.dumps({"round": a.round, "sentences": len(chosen), "review": len(review), "audit": len(audits),
                      "unknown_lemmas_covered": len(covered), "dir": str(out)}))


if __name__ == "__main__":
    main()

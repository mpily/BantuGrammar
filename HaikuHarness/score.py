"""
score.py -- score a pilot run: chrF and exact match per arm, the paired C - E difference,
tool use, unknown words and cost. Numbers only (no sentences) go to
HaikuHarness/results/<run>.json, which is safe to commit.

chrF and the CIs reuse hf_space/api/stats.py (the Space's Progress tab uses the same code).
Run:  ~/languagemodels/bert-env/bin/python HaikuHarness/score.py pilot1
"""
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "hf_space"))

from api.stats import chrf, mean_ci, rate_ci   # noqa: E402
from api.util import normalize_kitaita          # noqa: E402

DATA = Path.home() / "languagemodels" / "rulebook_data" / "pilot"
PRICE_PER_MTOK = {"claude-haiku-5-5": {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write": 0.125}}


def load(path: Path) -> dict:
    return {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))} if path.exists() else {}


def cost(usage: dict, model: str) -> float:
    p = PRICE_PER_MTOK.get(model)
    if not p or not usage:
        return 0.0
    return (usage.get("input_tokens", 0) * p["in"] + usage.get("output_tokens", 0) * p["out"]
            + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
            + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]) / 1e6


def ref_in_retrieval(row: dict) -> bool:
    """True if any retrieve_examples result in the transcript has the reference as its Kidaw'ida side."""
    ref = row["reference"].strip()
    for m in row.get("transcript", []):
        if m["role"] != "user":
            continue
        for b in m["content"]:
            try:
                o = json.loads(b["content"])
            except (TypeError, KeyError, ValueError):
                continue
            if any(e.get("kidawida", "").strip() == ref for e in o.get("examples", [])):
                return True
    return False


def score_arm(rows: dict, model: str) -> dict:
    ch, exact, missing, unk_tok, n_tok = [], 0, 0, 0, 0
    tools = collections.Counter()
    statuses = collections.Counter()
    for r in rows.values():
        hyp = r.get("translation")
        if hyp is None:
            missing += 1
            hyp = ""
        ch.append(chrf(hyp, r["reference"]))
        exact += normalize_kitaita(hyp) == normalize_kitaita(r["reference"])
        toks = hyp.split()
        n_tok += len(toks)
        unk_tok += sum(t.startswith("[?") for t in toks)
        for c in r.get("tool_calls", []):
            tools[c["name"]] += 1
            statuses[c["status"]] += 1
    n = len(rows)
    return {"n": n, "chrf": mean_ci(ch), "exact_match": rate_ci(exact, n),
            "no_translation_tag": missing, "unknown_word_share": rate_ci(unk_tok, n_tok),
            "stop_reasons": dict(collections.Counter(r["stop_reason"] for r in rows.values())),
            "requests_per_sentence": mean_ci([r["n_requests"] for r in rows.values()]),
            "tool_calls_per_sentence": mean_ci([len(r.get("tool_calls", [])) for r in rows.values()]),
            "tool_calls_by_name": dict(tools), "tool_status": dict(statuses),
            "cost_usd": round(sum(cost(r.get("usage"), model) for r in rows.values()), 4),
            "seconds_per_sentence": mean_ci([r.get("seconds", 0) for r in rows.values()])}


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "pilot1"
    d = DATA / run
    config = json.loads((d / "config.json").read_text())
    arms = {a: load(d / f"arm_{a}.jsonl") for a in config["arms"]}
    out = {"run": run, "config": config, "scored_by": "HaikuHarness/score.py", "metric": "chrF (char 6-gram, beta 2; hf_space/api/stats.py)",
           "arms": {a: score_arm(rows, config["model"]) for a, rows in arms.items()}}
    if "C" in arms and "E" in arms:
        both = sorted(set(arms["C"]) & set(arms["E"]))
        diffs = [chrf(arms["C"][i].get("translation") or "", arms["C"][i]["reference"])
                 - chrf(arms["E"][i].get("translation") or "", arms["E"][i]["reference"]) for i in both]
        out["paired_C_minus_E_chrf"] = mean_ci(diffs)
        # sentences where arm C's retrieval returned the reference itself (dev is not near-dup filtered
        # against train, unlike the test sets): report the comparison with and without them
        copied = {i for i in both if ref_in_retrieval(arms["C"][i])}
        rest = [i for i in both if i not in copied]
        out["ref_in_retrieval"] = {"n": len(copied), "ids": sorted(copied)}
        out["without_ref_in_retrieval"] = {
            "n": len(rest),
            **{f"chrf_{a}": mean_ci([chrf(arms[a][i].get("translation") or "", arms[a][i]["reference"]) for i in rest])
               for a in ("C", "E")},
            "paired_C_minus_E_chrf": mean_ci([d for i, d in zip(both, diffs) if i not in copied])}
        for src in ("davswa", "bible"):
            ids = [i for i in both if arms["C"][i]["src"] == src]
            out.setdefault("by_corpus", {})[src] = {
                "n": len(ids),
                **{f"chrf_{a}": mean_ci([chrf(arms[a][i].get("translation") or "", arms[a][i]["reference"]) for i in ids])
                   for a in ("C", "E")},
                "paired_C_minus_E_chrf": mean_ci([d for i, d in zip(both, diffs) if i in ids])}
    out["sentences_expected"] = config["n"]
    out["note"] = ("Pilot on the rule-book dev slice, not the fresh test set. n < 8 per stratum: no selection "
                   "or claims (CLAUDE.md rule 5).")
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / f"{run}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for a, s in out["arms"].items():
        c = s["chrf"]
        print(f"arm {a}: n={s['n']}  chrF={c['mean'] if c['mean'] is None else round(c['mean'], 2)} "
              f"CI={c['ci95']}  exact={s['exact_match']['rate']:.3f}  tools/sent={s['tool_calls_per_sentence']['mean']}  "
              f"cost=${s['cost_usd']}  stops={s['stop_reasons']}")
    if "paired_C_minus_E_chrf" in out:
        print("C - E chrF (paired):", out["paired_C_minus_E_chrf"])
        for k, v in out["by_corpus"].items():
            print(f"  {k}: n={v['n']} C={v['chrf_C']['mean']:.2f} E={v['chrf_E']['mean']:.2f} C-E={v['paired_C_minus_E_chrf']}")
        w = out["without_ref_in_retrieval"]
        print(f"  without {out['ref_in_retrieval']['n']} ref-in-retrieval: n={w['n']} C={w['chrf_C']['mean']:.2f} "
              f"E={w['chrf_E']['mean']:.2f} C-E={w['paired_C_minus_E_chrf']}")
    print(f"wrote {res / (run + '.json')}")


if __name__ == "__main__":
    main()

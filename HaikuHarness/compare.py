"""
compare.py -- paired comparison of one arm across two pilot runs on the same sentences
(one prompt change per run pair, CLAUDE.md rule 9).

Run:  ~/languagemodels/bert-env/bin/python HaikuHarness/compare.py pilot1_v2 pilot2_rephrase [--arm C]
Writes HaikuHarness/results/compare_<a>_vs_<b>.json (numbers only).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score import DATA, chrf, load, mean_ci, rate_ci, ref_in_retrieval  # noqa: E402


def unknown(rows):
    toks = [t for r in rows for t in (r.get("translation") or "").split()]
    return rate_ci(sum(t.startswith("[?") for t in toks), len(toks))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    arm = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "C"
    a_name, b_name = args[0], args[1]
    a, b = load(DATA / a_name / f"arm_{arm}.jsonl"), load(DATA / b_name / f"arm_{arm}.jsonl")
    ca, cb = (json.loads((DATA / n / "config.json").read_text()) for n in (a_name, b_name))
    ids = sorted(set(a) & set(b))
    s = {i: (chrf(a[i].get("translation") or "", a[i]["reference"]), chrf(b[i].get("translation") or "", b[i]["reference"]))
         for i in ids}
    copied = {i for i in ids if ref_in_retrieval(a[i]) or ref_in_retrieval(b[i])}

    def block(sel):
        return {"n": len(sel), f"chrf_{a_name}": mean_ci([s[i][0] for i in sel]), f"chrf_{b_name}": mean_ci([s[i][1] for i in sel]),
                "paired_b_minus_a": mean_ci([s[i][1] - s[i][0] for i in sel]),
                "b_better": sum(s[i][1] > s[i][0] for i in sel), "a_better": sum(s[i][1] < s[i][0] for i in sel)}
    out = {"arm": arm, "a": a_name, "b": b_name, "same_sentences": ca["ids"] == cb["ids"],
           "prompt_sha16": {a_name: ca["prompt_sha16"], b_name: cb["prompt_sha16"]},
           "rulebook_manifest_sha16": {a_name: ca["rulebook_manifest_sha16"], b_name: cb["rulebook_manifest_sha16"]},
           "all": block(ids), "without_ref_in_retrieval": block([i for i in ids if i not in copied]),
           "by_corpus": {src: block([i for i in ids if a[i]["src"] == src]) for src in ("davswa", "bible")},
           "unknown_word_share": {a_name: unknown(a.values()), b_name: unknown(b.values())},
           "tool_calls_per_sentence": {n: mean_ci([len(r.get("tool_calls", [])) for r in x.values()]) for n, x in ((a_name, a), (b_name, b))}}
    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / f"compare_{a_name}_vs_{b_name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k in ("all", "without_ref_in_retrieval"):
        v = out[k]
        p = v["paired_b_minus_a"]
        print(f"{k:26} n={v['n']}  {a_name}={v[f'chrf_{a_name}']['mean']:.2f}  {b_name}={v[f'chrf_{b_name}']['mean']:.2f}  "
              f"b-a={p['mean']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]  b better {v['b_better']}, a better {v['a_better']}")
    for src, v in out["by_corpus"].items():
        p = v["paired_b_minus_a"]
        print(f"  {src:24} n={v['n']}  b-a={p['mean']:+.2f} [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}]")
    for n, u in out["unknown_word_share"].items():
        print(f"  [?word] share {n}: {u['k']}/{u['n']} = {u['rate']:.3f}")
    print("  tool calls/sentence:", {n: round(v["mean"], 1) for n, v in out["tool_calls_per_sentence"].items()})


if __name__ == "__main__":
    main()

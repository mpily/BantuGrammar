"""
run_pilot.py -- the 50-sentence pilot: arms C (tools) and E (zero-shot) on the same sentences.

Sentences: a fixed random sample of the rule-book DEV slice (held out from building the rule
book, lexicon and retrieval index; NOT the fresh test set, which is still to be chosen).
Output (corpus text, so outside git): ~/languagemodels/rulebook_data/pilot/<run>/
    config.json      model, prompt / schema / rule-book hashes, sentence IDs
    arm_C.jsonl      one line per sentence (resumable: rerunning skips IDs already done)
    arm_E.jsonl
Then: score.py <run>.

Run:  ~/languagemodels/bert-env/bin/python HaikuHarness/run_pilot.py --n 50 --run pilot1
      ... --dry-run --n 3 --arms C   (offline, fake model; no cost)
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

import agent_loop
import one_call
import prompts

DATA = Path.home() / "languagemodels" / "rulebook_data"
DEV = DATA / "dev.jsonl"
SEED = 20261112


def sha16(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def sample(n: int) -> list[dict]:
    """Half dav_swa, half Bible (the dev slice is 62% Bible, whose verses are much longer)."""
    rows = [json.loads(l) for l in open(DEV, encoding="utf-8")]
    rng = random.Random(SEED)
    dav = rng.sample([r for r in rows if r["src"] == "davswa"], (n + 1) // 2)
    bib = rng.sample([r for r in rows if r["src"] == "bible"], n // 2)
    return [x for pair in zip(dav, bib) for x in pair] + dav[len(bib):]


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8") if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--arms", default="C,E")
    ap.add_argument("--model", default=one_call.MODEL)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run = a.run + ("_dryrun" if a.dry_run else "")
    out = DATA / "pilot" / run
    out.mkdir(parents=True, exist_ok=True)
    rows = sample(a.n)

    import anthropic
    config = {"run": run, "model": a.model, "effort": one_call.EFFORT, "max_tokens": one_call.MAX_TOKENS,
              "max_requests_C": agent_loop.MAX_REQUESTS, "arms": a.arms.split(","), "dry_run": a.dry_run,
              "n": a.n, "seed": SEED, "sample_source": "rulebook dev slice (seed 20261111), stratified half davswa / half bible", "dev_sha16": sha16(DEV),
              "ids": [r["id"] for r in rows], "prompt_sha16": prompts.prompt_sha16(),
              "tool_schemas_sha16": sha16(PROJECT / "agent_translate" / "tool_schemas.json"),
              "tools_py_sha16": sha16(PROJECT / "agent_translate" / "tools.py"),
              "rulebook_manifest_sha16": sha16(PROJECT / "rulebook" / "MANIFEST.json"),
              "harness_sha16": {f.name: sha16(f) for f in sorted(HERE.glob("*.py"))},
              "anthropic_sdk": anthropic.__version__}
    cfg_path = out / "config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        if old["ids"] != config["ids"] or old["prompt_sha16"] != config["prompt_sha16"] or old["model"] != config["model"]:
            sys.exit(f"{cfg_path} exists with different sentences/prompt/model: use a new --run name")
    cfg_path.write_text(json.dumps(config, indent=1))

    client = agent_loop.ScriptedClient() if a.dry_run else anthropic.Anthropic()
    for arm in config["arms"]:
        path = out / f"arm_{arm}.jsonl"
        skip = done_ids(path)
        for i, r in enumerate(rows):
            if r["id"] in skip:
                continue
            t0 = time.time()
            try:
                if arm == "C":
                    res = agent_loop.translate_with_tools(r["swa"], client=client, model=a.model)
                else:
                    if a.dry_run:
                        res = {"arm": "E", "sw": r["swa"], "translation": r["swa"], "reply": "", "stop_reason": "dry_run",
                               "n_requests": 0, "usage": {}, "tool_calls": []}
                    else:
                        res = one_call.translate_zero_shot(r["swa"], client=client, model=a.model)
            except Exception as e:      # log and move on; the row stays missing and is retried on rerun
                print(f"[{arm} {i+1}/{len(rows)}] {r['id']} ERROR {type(e).__name__}: {e}")
                continue
            res.update({"id": r["id"], "src": r["src"], "reference": r["dav"], "seconds": round(time.time() - t0, 1)})
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False, default=str) + "\n")
            print(f"[{arm} {i+1}/{len(rows)}] {r['id']} {res['stop_reason']} "
                  f"req={res['n_requests']} tools={len(res['tool_calls'])} -> {res['translation']}")
    print(f"\nwrote {out}. Next: score.py {run}")


if __name__ == "__main__":
    main()

"""
eval_subject_probe_on_draft.py

2026-11-03. User request: apply the relation probes to the Kidaw'ida draft
itself (not to the Swahili source) to find each verb's subject, then correct
the subject marker (rule-book step `subject_probe`), on the 500-sentence sets
already decoded. No pairing of draft verbs with Swahili verbs; the class comes
from the noun as written in the draft (subject_class_resolver); the Swahili
clauses are used only for number when the noun form has one form for both.

Probes: saved_identification_probes unchanged (Swahili-trained, NLLB-1.3B
encoder, draft encoded as swh_Latn), identify_relations per sentence with the
draft's noun and verb candidates (rulebook_correct.draft_candidates).
Probe output is cached per sentence (resumable): ~/languagemodels/
arm2_draftprobe_<set>.jsonl. Scored like eval_version_b.py.

Both sets have had subject-step errors inspected (test_novel 2026-11-03 cont.,
test_novel2 cont. 1), so neither is a clean test for subject steps.
Usage: python eval_subject_probe_on_draft.py --set test_novel2
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
sys.path.insert(0, str(Path(__file__).parent))
import os  # noqa: E402

os.chdir(ROOT / "hf_space")
from eval_version_b import boot, score, wilson  # noqa: E402
from rulebook_correct import DEFAULT_STEPS, correct, draft_candidates, draft_words  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["test_novel", "test_novel2", "test_novel3"])
    a = ap.parse_args()
    lm = Path.home() / "languagemodels"
    rows = [json.loads(l) for l in open(lm / f"arm2_nbest_{a.set}.jsonl", encoding="utf-8")]
    clauses = {d["swa"]: d["clauses"] for d in map(json.loads, open(lm / f"arm2_clauses_{a.set}.jsonl", encoding="utf-8"))}
    cache_path = lm / f"arm2_draftprobe_{a.set}.jsonl"
    cache = {}
    if cache_path.exists():
        cache = {d["id"]: d for d in map(json.loads, open(cache_path, encoding="utf-8"))}
    todo = [r for r in rows if r["id"] not in cache]
    if todo:
        from nllb_probe_identification import identify_relations
        with open(cache_path, "a", encoding="utf-8") as f:
            for k, r in enumerate(todo):
                words = draft_words(r["swa"], r["nbest"][0]["hyp"])
                nouns, verbs = draft_candidates(words)
                subj, note = {}, None
                if nouns and verbs:
                    try:
                        res = identify_relations(words, nouns, sorted(verbs), [])
                        subj = {str(v): x.get("subject_idx") for v, x in (res.verb_results or {}).items()}
                        note = "; ".join(res.notes or []) or None
                    except Exception as e:  # logged, counted as no subjects
                        note = f"{type(e).__name__}: {e}"
                d = {"id": r["id"], "n_nouns": len(nouns), "n_verbs": len(verbs), "subjects": subj, "note": note}
                cache[r["id"]] = d
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                f.flush()
                if k % 50 == 0:
                    print(k, len(todo), flush=True)

    base = [score(r["nbest"][0]["hyp"], r["ref"]) for r in rows]
    out = {"set": a.set, "n": len(rows), "coverage": {
        "drafts_with_noun_and_verb": sum(1 for r in rows if cache[r["id"]]["n_nouns"] and cache[r["id"]]["n_verbs"]),
        "verbs": sum(cache[r["id"]]["n_verbs"] for r in rows),
        "verbs_with_probe_subject": sum(sum(v is not None for v in cache[r["id"]]["subjects"].values()) for r in rows)},
        "conditions": {}}
    examples = {}
    conds = {"subject_probe": ("subject_probe",), "subject (order pairing)": ("subject",),
             "subject_checked": ("subject_checked",),
             "defaults (Space now)": DEFAULT_STEPS,
             "defaults + subject_checked": DEFAULT_STEPS + ("subject_checked",)}
    for name, steps in conds.items():
        diffs, ch_d, ch, better, worse, ex = [], [], 0, 0, 0, []
        for r, b0 in zip(rows, base):
            ds = {int(k): v for k, v in cache[r["id"]]["subjects"].items() if v is not None}
            res = correct(r["swa"], [c["hyp"] for c in r["nbest"]], steps, clauses=clauses.get(r["swa"]),
                          draft_subjects=ds)
            b = score(res["kitaita"], r["ref"])
            diffs.append(b - b0)
            if res["kitaita"] != r["nbest"][0]["hyp"]:
                ch += 1
                ch_d.append(b - b0)
                better += b > b0
                worse += b < b0
                ex.append({"swa": r["swa"], "A": r["nbest"][0]["hyp"], "B": res["kitaita"], "ref": r["ref"],
                           "changes": res["changes"], "chrf_diff": round(b - b0, 1)})
        out["conditions"][name] = {"changed": ch, "changed_share": wilson(ch, len(rows)),
                                   "paired_B_minus_A_all": boot(diffs), "paired_B_minus_A_changed_only": boot(ch_d),
                                   "changed_better": better, "changed_worse": worse,
                                   "changed_same": ch - better - worse}
        examples[name] = ex
        print(name, json.dumps(out["conditions"][name]), flush=True)
    print("coverage", out["coverage"])
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    out["config"] = {"draftprobe_cache_sha256_16": h(cache_path),
                     "files_sha256_16": {f: h(ROOT / "hf_space" / f) for f in
                                         ("rulebook_correct.py", "subject_class_resolver.py",
                                          "kitaita_subject_markers.json", "kitaita_bible_class_decisions.json")}}
    (lm / f"eval_subject_probe_{a.set}_examples.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False))
    out["examples_file"] = f"~/languagemodels/eval_subject_probe_{a.set}_examples.json (not in git: corpus text)"
    (Path(__file__).parent / f"eval_subject_probe_{a.set}_results.json").write_text(json.dumps(out, indent=1))

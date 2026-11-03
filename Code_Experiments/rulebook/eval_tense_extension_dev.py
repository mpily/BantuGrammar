"""
eval_tense_extension_dev.py

2026-11-03. Development check of the rule-book `tense` and `extension` steps
against the agent audit of test_novel2 (~/languagemodels/audit/round_a, 500
drafts, each counted once from its main auditor). This set is DEVELOPMENT data
(its errors were inspected); the clean score is on test_novel3.

For each step: edits made; edits on a word the audit flagged as an ending error
(and whether the new form equals the auditor's expected form); edits on words
the audit did not flag (possible false positives, or audit misses); audited
ending errors left unfixed (recall); chrF B - A on the 500.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "hf_space"))
sys.path.insert(0, str(Path(__file__).parent))
from eval_version_b import boot, score, wilson  # noqa: E402
from rulebook_correct import correct  # noqa: E402
from rulebook_flagger import norm  # noqa: E402

A = Path.home() / "languagemodels" / "audit" / "round_a"

if __name__ == "__main__":
    items = {k: [json.loads(l) for l in open(A / f"items_agent{k}.jsonl")] for k in range(1, 5)}
    aud = {k: {d["id"]: d for d in map(json.loads, open(A / f"audit_agent{k}.jsonl"))} for k in range(1, 5)}
    rows = [(x, aud[k][x["id"]]) for k in range(1, 5) for x in items[k][:125]]
    out, examples = {}, {}
    for name, steps in (("tense", ("tense",)), ("extension", ("extension",)), ("tense+extension", ("tense", "extension"))):
        kinds = ("ending_tam",) if name == "tense" else ("ending_extension",) if name == "extension" \
            else ("ending_tam", "ending_extension")
        n_edit = on_flagged = exact = unflagged = 0
        flagged_total = sum(1 for _, a in rows for e in a.get("errors", []) if e.get("type") in kinds)
        fixed_words = 0
        diffs, ex = [], []
        for x, a in rows:
            r = correct(x["swahili"], [x["draft"]], steps)
            diffs.append(score(r["kitaita"], x["reference"]) - score(x["draft"], x["reference"]))
            errs = {norm(e.get("draft_word", "")): e for e in a.get("errors", []) if e.get("type") in kinds}
            any_errs = {norm(e.get("draft_word", "")) for e in a.get("errors", [])}
            for kind, desc in r["changes"]:
                n_edit += 1
                old, new = desc.split(" (")[0].split(" -> ")
                e = errs.get(old)
                tag = "unflagged"
                if e:
                    on_flagged += 1
                    fixed_words += 1
                    tag = "on audited error"
                    if e.get("expected") and norm(e["expected"].split()[0]) == new:
                        exact += 1
                        tag += ", = expected"
                elif old in any_errs:
                    tag = "audited, other error type"
                else:
                    unflagged += 1
                ex.append({"id": x["id"], "swa": x["swahili"], "draft": x["draft"], "ref": x["reference"],
                           "change": desc, "audit": tag, "audit_expected": (e or {}).get("expected")})
        out[name] = {"edits": n_edit, "on_audited_error": on_flagged, "equals_auditor_expected": exact,
                     "on_unflagged_words": unflagged,
                     "audited_errors_of_this_type": flagged_total,
                     "recall": wilson(fixed_words, flagged_total),
                     "chrf_B_minus_A": boot(diffs)}
        examples[name] = ex
        print(name, json.dumps(out[name]))
    (Path.home() / "languagemodels" / "eval_tense_extension_dev_examples.json").write_text(
        json.dumps(examples, indent=1, ensure_ascii=False))
    (Path(__file__).parent / "eval_tense_extension_dev_results.json").write_text(json.dumps(
        {"_meta": {"date": "2026-11-03", "set": "test_novel2 (development; audited)",
                   "examples_file": "~/languagemodels/eval_tense_extension_dev_examples.json (not in git)"},
         "results": out}, indent=1))

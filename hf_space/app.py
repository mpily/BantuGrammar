# -*- coding: utf-8 -*-
"""
app.py -- Hugging Face Space front end for the Kitaita subspace-transfer
pipeline (see README.md for the project this Space showcases).

Loads NLLB (facebook/nllb-200-distilled-1.3B) and this project's trained
relation/POS probes ONCE at startup, then serves translate_sentence_
with_probes() / translate_sentence() -- the exact same functions used
throughout development, not a reimplementation. Ported from
kitaita_local_webapp.py (a stdlib http.server version of the same idea)
to Gradio for Spaces hosting.

2026-10-20: added a "give me a sentence to test" + rate/correct feedback
loop, direct user request -- a native speaker should be able to pull a
real sentence, see what this pipeline does with it, and tell us whether
it's right without needing to construct their own test cases. Feedback
(rating + optional corrected translation) is pushed to a private HF
Dataset (mpily/kitaita-translation-feedback) rather than written to local
disk: a free/cpu-basic Space's filesystem is wiped on every restart/sleep,
so anything written locally would silently vanish -- a dataset repo is
the one thing on the Hub that actually survives that.
"""
from __future__ import annotations

import json
import os
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

from kitaita_e2e_pipeline import translate_sentence_with_probes, translate_sentence
from kitaita_sentence_generator import load_pages, build_lookup_tables
from kitaita_eval_metrics import (
    GoldSlot, compute_agreement_accuracy, aggregate_agreement_accuracy,
    compute_vocabulary_choice, aggregate_vocabulary_choice, compute_stability,
)

EXAMPLES = [
    ["Mbwa mkubwa aliona Kijana mdogo akitembea.", "Probes (full pipeline)"],
    ["Mwalimu alisoma kitabu kisha akalala.", "Probes (full pipeline)"],
    ["Kitabu cha mwalimu kilikuwa kikubwa.", "Probes (full pipeline)"],
    ["Askari alimwona mwizi.", "Probes (full pipeline)"],
]

FEEDBACK_DATASET_REPO = "mpily/kitaita-translation-feedback"

# 2026-10-26: direct user request ("use the mined storybook sentences
# mostly, mine more, increase to about 1000") -- REPLACES the 2026-10-20
# version of this file, which was 100% Bible-register (the exact register
# mismatch this project moved its own free-text testing away from,
# 2026-10-22). Now 999 sentences mined directly from ALL 40 Swahili
# stories in the African Storybook Initiative's open mirror
# (github.com/global-asp/storybooks-swahili, CC-BY/CC-BY-NC 3.0 per
# story -- see Code_Experiments/mine_african_storybook.py) plus a 60-
# sentence Bible top-up (same source as before, `bible collection/data/
# parallel_full_suv/parallel.jsonl`) for register diversity -- ~94%
# storybook by design, not an approximation. Real sentences a speaker
# would recognize, not synthetic ones, without shipping either source
# corpus's full size to the Space. This Space does NOT do live internet
# fetching at request time (no reliable way to guarantee that inside a
# Space, and it would make every click dependent on a 3rd-party site
# being up) -- flagged here rather than silently narrower than asked.
_SENTENCES_PATH = Path(__file__).parent / "test_sentences.json"
try:
    TEST_SENTENCES = json.loads(_SENTENCES_PATH.read_text(encoding="utf-8"))
except Exception:
    TEST_SENTENCES = []

print("Loading vocabulary/grammar tables...", flush=True)
_tables = build_lookup_tables(*load_pages())
print("Pre-warming NLLB + trained probes (one-time cost)...", flush=True)
from nllb_probe_identification import load_probes  # noqa: E402

load_probes()
print("Ready.", flush=True)


def translate(text: str, mode: str):
    text = (text or "").strip()
    if not text:
        return "", "", ""
    fn = translate_sentence_with_probes if mode.startswith("Probes") else translate_sentence
    try:
        r = fn(text, _tables)
    except Exception as e:  # keep the Space up even if one input trips a bug
        return "", "", f"SERVER ERROR: {type(e).__name__}: {e}"
    sentence = r.get("kitaita_sentence") or ""
    error = r.get("error")
    flags = r.get("flags")
    if flags is None:
        d = r.get("decomposition")
        flags = getattr(d, "flags", []) if d is not None else []
    trace = r.get("trace", [])
    flags_text = "\n".join(f"- {f}" for f in flags) if flags else "(none)"
    trace_text = "\n".join(f"- {t}" for t in trace) if trace else "(none)"
    # 2026-10-20: direct user report -- a verbless input (e.g. "Siku zote za
    # maisha yangu.") produces a real, often fully correct best-effort
    # translation (every noun/particle/possessive/quantifier still gets
    # substituted independently; only VERB agreement needs a clause to
    # compute), but `error` was always prefixed with "ERROR:" regardless,
    # reading as an outright refusal even when the line above it was right.
    # Only show the ERROR prefix when there's genuinely nothing to show;
    # otherwise the detail belongs in the Flags panel, where it already is.
    if not sentence:
        out = f"(no output)\n\nERROR: {error}" if error else "(no output)"
    else:
        out = sentence
    return out, flags_text, trace_text


def random_test_sentence():
    """Direct user request: a button that hands the user a real Swahili
    sentence to try, instead of them having to think one up. Also clears
    the previous feedback status so an old "thanks, logged" message
    doesn't linger next to a brand-new, not-yet-rated sentence."""
    if not TEST_SENTENCES:
        return gr.update(), ""
    choice = random.choice(TEST_SENTENCES)
    return choice["sw"], ""


def submit_feedback(swahili: str, kitaita_output: str, mode: str, rating: str, correction: str):
    """Logs ONE rating (+ optional corrected translation) as its own small
    JSON file in a private HF Dataset -- one file per submission (not a
    shared CSV/JSON Claude read-modifies-writes) so two people submitting
    at the same moment can never clobber each other, and a failed/partial
    upload never corrupts anything else already logged. Requires an
    HF_TOKEN secret on THIS Space with write access to the dataset repo
    (Space Settings -> Variables and secrets) -- without it, this fails
    closed with a clear message rather than silently losing the feedback
    or crashing the whole Space for every other visitor."""
    swahili = (swahili or "").strip()
    if not swahili:
        return "Nothing to log yet -- translate a sentence first."
    if not rating:
        return "Pick a rating (1-5) before submitting."
    token = os.environ.get("HF_TOKEN")
    if not token:
        return ("Feedback was NOT saved: this Space has no HF_TOKEN secret configured, so it "
                "can't write to the feedback dataset. (Space owner: add one under Settings -> "
                "Variables and secrets.)")
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "swahili_input": swahili,
        "kitaita_output": (kitaita_output or "").strip(),
        "mode": mode,
        "rating": rating,
        "native_speaker_correction": (correction or "").strip() or None,
    }
    # 2026-10-26: direct user request ("add summary charts for all three
    # [evaluation] tests that automatically update") -- the 3 metrics
    # (kitaita_eval_metrics.py) need the pipeline's own structured
    # CONCORD-SLOT ledger (Decomposition.slots) to compare against a
    # native speaker's correction, not just the holistic rating this
    # form already captured. Re-running translation here (same swahili+
    # mode already submitted) is the least invasive way to get that --
    # no change to the existing Translate button/UI, and feedback
    # submission is rare enough (one click per rated sentence, not per
    # keystroke) that the repeated inference cost is a non-issue. Never
    # blocks the feedback save on this -- if it fails, the rating/
    # correction above is still logged, just without slot-level detail
    # for the dashboard to use.
    try:
        fn = translate_sentence_with_probes if mode.startswith("Probes") else translate_sentence
        r2 = fn(swahili, _tables)
        d2 = r2.get("decomposition")
        if d2 is not None:
            record["slots"] = [{"token_idx": s.token_idx, "slot_type": s.slot_type,
                                 "predicted_form": s.predicted_form} for s in d2.slots]
            # vocabulary choices: content-word substitutions NOT already a
            # concord slot (see kitaita_eval_metrics.py's own "slots are
            # concord, not vocabulary" scoping) -- d.secondary_nouns is
            # this project's own existing ledger for exactly that, reused
            # rather than re-derived.
            slot_idxs = {s.token_idx for s in d2.slots}
            record["vocab_choices"] = [{"token_idx": idx, "kitaita": kit}
                                        for idx, _lemma, kit in d2.secondary_nouns
                                        if kit and idx not in slot_idxs]
        else:
            record["slots"], record["vocab_choices"] = [], []
    except Exception:
        record["slots"], record["vocab_choices"] = [], []
    try:
        from huggingface_hub import HfApi
        import io
        api = HfApi(token=token)
        path_in_repo = f"feedback/{record['timestamp'].replace(':', '-')}_{record['id']}.json"
        api.upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(record, ensure_ascii=False, indent=1).encode("utf-8")),
            path_in_repo=path_in_repo,
            repo_id=FEEDBACK_DATASET_REPO,
            repo_type="dataset",
        )
    except Exception as e:
        return f"Feedback was NOT saved -- upload failed: {type(e).__name__}: {e}"
    return "Thanks -- logged for review." + (" (your correction was included.)" if record["native_speaker_correction"] else "")


# --------------------------------------------------------------------------- progress dashboard
# 2026-10-26: direct user request -- "add summary charts for all three
# [evaluation] tests that automatically update as people continue to
# test." Reuses kitaita_eval_metrics.py directly (the SAME module
# validated in kitaita_eval_demo.py) -- this is not a parallel/ad-hoc
# metric implementation.
#
# The one thing this dashboard has to invent, because the feedback form
# only ever collects a holistic corrected SENTENCE (not a per-slot
# correction -- that would need a much heavier submission UI, not asked
# for here): an APPROXIMATE alignment between the pipeline's output and
# a native speaker's correction, position by position. This is only
# reasonable because of how this project's own assembly works --
# `assemble_in_order`/`assemble_multi_clause` are explicitly POSITION-
# PRESERVING (each Swahili token's slot keeps its own position in the
# Kitaita output), so word i of the output and word i of a same-length
# correction really do correspond to the same original slot, unlike a
# generic MT alignment problem. When lengths differ (word insertion/
# deletion/reordering in the correction), that entry is EXCLUDED from
# the slot/vocabulary charts entirely rather than guessed -- it still
# counts toward the plain rating-count/mean-rating numbers, which need
# no alignment at all.
@dataclass
class _PredSlot:
    token_idx: int
    slot_type: str
    predicted_form: str | None


def _load_all_feedback() -> list[dict]:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        files = [f for f in api.list_repo_files(FEEDBACK_DATASET_REPO, repo_type="dataset")
                 if f.startswith("feedback/") and f.endswith(".json")]
        records = []
        for f in files:
            try:
                path = api.hf_hub_download(FEEDBACK_DATASET_REPO, f, repo_type="dataset")
                records.append(json.loads(Path(path).read_text(encoding="utf-8")))
            except Exception:
                continue  # one unreadable file never blocks the rest
        return records
    except Exception:
        return []


def compute_dashboard_data() -> dict:
    records = _load_all_feedback()
    ratings = [int(r["rating"]) for r in records if str(r.get("rating") or "").isdigit()]

    agreement_per_sentence, vocab_per_sentence = [], []
    for r in records:
        corr = (r.get("native_speaker_correction") or "").strip()
        pred_text = (r.get("kitaita_output") or "").strip()
        if not corr or not pred_text:
            continue
        # strip trailing sentence punctuation per token -- `d.secondary_
        # nouns`'s own kitaita word (what `vocab_choices`/slot predicted_
        # forms store) never carries it, but a token recovered by
        # splitting the FULL correction/output STRING does whenever that
        # word lands at the sentence-final position; found directly while
        # testing with constructed examples -- a legitimate vocabulary
        # match at the last word position otherwise reads as a mismatch
        # purely because one side has "word" and the other "word.".
        strip_punct = lambda w: w.strip(".,!?;:“”\"'")
        pred_tokens = [strip_punct(w) for w in pred_text.split()]
        corr_tokens = [strip_punct(w) for w in corr.split()]
        if len(pred_tokens) != len(corr_tokens):
            continue  # not position-alignable -- excluded, not guessed (see module comment)

        pred_slots, gold_slots = [], []
        for s in r.get("slots") or []:
            idx = s.get("token_idx")
            if idx is None or idx >= len(corr_tokens):
                continue
            corr_word = corr_tokens[idx]
            predicted_form = s.get("predicted_form")
            # subject_sm's predicted_form is a bare PREFIX (e.g. "u"), every
            # other slot type is a whole standalone word -- see SlotRecord's
            # own docstring (kitaita_e2e_pipeline.py) for why.
            if s.get("slot_type") == "subject_sm":
                matches = bool(predicted_form) and corr_word.startswith(predicted_form)
            else:
                matches = predicted_form == corr_word
            pred_slots.append(_PredSlot(idx, s.get("slot_type"), predicted_form))
            gold_slots.append(GoldSlot(idx, s.get("slot_type"), needs_rewrite=True,
                                        correct_form=predicted_form if matches else corr_word))
        if gold_slots:
            agreement_per_sentence.append(compute_agreement_accuracy(pred_slots, gold_slots))

        vocab = r.get("vocab_choices") or []
        pred_words = {v["token_idx"]: v["kitaita"] for v in vocab if v.get("token_idx", -1) < len(corr_tokens)}
        native_choices = {idx: [corr_tokens[idx]] for idx in pred_words}
        if pred_words:
            vocab_per_sentence.append(compute_vocabulary_choice(pred_words, native_choices))

    agreement_agg = aggregate_agreement_accuracy(agreement_per_sentence) if agreement_per_sentence else None
    vocab_agg = aggregate_vocabulary_choice(vocab_per_sentence) if vocab_per_sentence else None
    stability_rows = [{"sentence": "", "n_slots": a["scored_slots"], "accuracy": a["accuracy"]}
                       for a in agreement_per_sentence]
    stability = compute_stability(stability_rows) if stability_rows else None

    return {
        "n_feedback": len(records),
        "n_ratings": len(ratings),
        "mean_rating": (sum(ratings) / len(ratings)) if ratings else None,
        "n_alignable": len(agreement_per_sentence),
        "agreement": agreement_agg,
        "vocabulary": vocab_agg,
        "stability": stability,
    }


_LABEL_ORDER = ["correct", "wrong_rewrite", "missed_slot", "spurious_rewrite"]


def refresh_dashboard():
    """One function, called on page load AND every 60s (see demo.load
    below) -- feeds all 4 dashboard outputs together so they always
    reflect the SAME snapshot of feedback data, never 3 separately-timed
    reads that could show inconsistent totals between charts."""
    data = compute_dashboard_data()

    summary = (f"**{data['n_feedback']}** feedback submissions logged so far "
               f"({data['n_ratings']} rated"
               + (f", mean rating {data['mean_rating']:.2f}/5" if data["mean_rating"] is not None else "")
               + f"); **{data['n_alignable']}** had a same-length native-speaker correction usable "
                 f"for the agreement/vocabulary charts below (see this file's own comment for why "
                 f"length mismatches are excluded rather than guessed).")

    agg = data["agreement"]
    if agg and agg["scored_slots"]:
        counts = agg["counts"]
        agreement_df = pd.DataFrame({"label": _LABEL_ORDER,
                                      "count": [counts.get(l, 0) for l in _LABEL_ORDER]})
    else:
        agreement_df = pd.DataFrame({"label": ["(no data yet)"], "count": [0]})

    vocab = data["vocabulary"]
    if vocab and vocab["scored_positions"]:
        vocab_df = pd.DataFrame({"metric": ["mean native-speaker support"],
                                  "value": [vocab["mean_support"]]})
    else:
        vocab_df = pd.DataFrame({"metric": ["(no data yet)"], "value": [0]})

    stab = data["stability"]
    if stab and stab["curve"]:
        stability_df = pd.DataFrame(stab["curve"])
    else:
        stability_df = pd.DataFrame({"n_slots": [0], "mean_accuracy": [0], "n_sentences": [0]})

    return summary, agreement_df, vocab_df, stability_df


with gr.Blocks(title="Kitaita Translator") as demo:
    with gr.Tabs():
        with gr.Tab("Translator"):
            gr.Markdown(
                "# Kitaita Translator\n"
                "Swahili &rarr; Kitaita (Kidaw'ida, ISO `dav`), via NLLB's Swahili output "
                "run through a probe-guided rule-based concord pipeline -- see this Space's "
                "README for the full methodology, what's validated, and what's still open. "
                "**Probes** mode runs the real NLLB encoder + 5 trained relation/POS probes "
                "(the full pipeline this project validates); **heuristic** mode is a plain "
                "adjacency-based decomposer kept for comparison."
            )
            with gr.Row():
                inp = gr.Textbox(label="Swahili sentence", lines=2,
                                  placeholder="e.g. Mbwa mkubwa aliona Kijana mdogo akitembea.")
                mode = gr.Radio(["Probes (full pipeline)", "Heuristic (for comparison)"],
                                 value="Probes (full pipeline)", label="Mode")
            with gr.Row():
                btn = gr.Button("Translate", variant="primary")
                sample_btn = gr.Button("🎲 Give me a sentence to test")
            out = gr.Textbox(label="Kitaita output", lines=2)
            with gr.Accordion("Flags (identification/translation caveats on this sentence)", open=False):
                flags_box = gr.Markdown()
            with gr.Accordion("Evidence trail (full per-step trace)", open=False):
                trace_box = gr.Markdown()

            gr.Markdown("### Is this translation right?")
            gr.Markdown(
                "If you speak Kitaita: rate the output above and, if it's wrong, tell us what it "
                "should have been. This is logged for review, not applied automatically -- it's exactly "
                "how this project finds the vocabulary and rules it's still missing. Your correction "
                "also feeds the **📊 Progress** tab's charts."
            )
            with gr.Row():
                rating = gr.Radio(["1", "2", "3", "4", "5"], label="Rating (1 = way off, 5 = correct)")
                correction = gr.Textbox(label="Correct Kitaita translation (optional)", lines=2,
                                         placeholder="If the output above is wrong, what should it be?")
            feedback_btn = gr.Button("Submit feedback")
            feedback_status = gr.Markdown()

            btn.click(translate, inputs=[inp, mode], outputs=[out, flags_box, trace_box])
            inp.submit(translate, inputs=[inp, mode], outputs=[out, flags_box, trace_box])
            sample_btn.click(random_test_sentence, inputs=[], outputs=[inp, feedback_status])
            feedback_btn.click(submit_feedback, inputs=[inp, out, mode, rating, correction], outputs=[feedback_status])
            gr.Examples(examples=EXAMPLES, inputs=[inp, mode])

        with gr.Tab("📊 Progress"):
            gr.Markdown(
                "# Evaluation progress\n"
                "Live charts for this project's 3 evaluation metrics, computed directly from "
                "**Submit feedback** entries on the Translator tab (a native-speaker correction is "
                "what makes a submission usable here -- a rating alone only counts toward the "
                "summary line). Refreshes automatically every 60s, or click Refresh now."
            )
            dash_summary = gr.Markdown()
            refresh_btn = gr.Button("🔄 Refresh now")
            with gr.Row():
                agreement_plot = gr.BarPlot(x="label", y="count", title="1. Agreement accuracy "
                                             "(concord slots, by outcome)", height=300)
                vocab_plot = gr.BarPlot(x="metric", y="value", title="2. Vocabulary choice "
                                         "(mean native-speaker support)", y_lim=[0, 1], height=300)
            stability_plot = gr.BarPlot(x="n_slots", y="mean_accuracy", title="3. Stability "
                                         "(agreement accuracy vs. agreement-slot count per sentence)",
                                         y_lim=[0, 1], height=300)

            dash_outputs = [dash_summary, agreement_plot, vocab_plot, stability_plot]
            refresh_btn.click(refresh_dashboard, inputs=[], outputs=dash_outputs)
            demo.load(refresh_dashboard, inputs=[], outputs=dash_outputs)
            # 2026-10-26: Gradio 6's auto-refresh mechanism is gr.Timer (the
            # `every=` kwarg on .load()/.click() from older Gradio versions
            # was removed, confirmed directly -- TypeError on import until
            # this was fixed) -- a real, ticking component wired to the same
            # refresh function, not a cosmetic difference.
            dash_timer = gr.Timer(60)
            dash_timer.tick(refresh_dashboard, inputs=[], outputs=dash_outputs)

if __name__ == "__main__":
    demo.launch()

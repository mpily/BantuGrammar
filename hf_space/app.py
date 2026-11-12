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
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import pandas as pd

from kitaita_e2e_pipeline import translate_sentence_with_probes, translate_sentence
from kitaita_sentence_generator import load_pages, build_lookup_tables

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
# 2026-11-02: system A of the two-system comparison (fine-tuned NLLB, arm 2),
# loaded in the background so the rule-based pipeline is usable immediately.
import finetuned_mt  # noqa: E402

finetuned_mt.start_loading()
# 2026-11-12, user decision: Translation 2 is Claude Haiku + the rule-book tools (study arm C), replacing
# fine-tuned + rule book. Paid API: daily cap + per-visitor limit inside haiku_mt.
import haiku_mt  # noqa: E402

haiku_mt.start()
print("Ready.", flush=True)


FLAG_LABELS = {"unknown": "not in the model's training vocabulary",
               "swahili": "copied from the Swahili source",
               "repeat": "repetition", "name": "proper name (kept)"}


CHANGE_LABELS = {"rerank": "agreement: picked another model candidate",
                 "fill": "copied Swahili replaced from the mined lexicon",
                 "unknown": "unknown word replaced by the Swahili source word",
                 "repeat": "repetition removed",
                 "applicative": "applicative ('do for') added to match the Swahili verb",
                 "verified": "replaced by a native-speaker-verified translation",
                 "subject": "verb subject marker matched to its subject's class (pipeline analysis)",
                 "tense": "verb tense/negation matched to the Swahili verb",
                 "extension": "verb extension (passive, applicative, causative, stative, reciprocal) added"}


def translate_finetuned(text: str, clauses=None):
    """Fine-tuned NLLB draft (A), the rule-book version (B), and what the
    rule book flagged / changed. clauses: the rule-based pipeline's per-verb
    subject analysis (probe path), used by the rule book's subject step."""
    r = finetuned_mt.translate(text, clauses=clauses)
    if r["kitaita"] is None:
        msg = f"(fine-tuned model {r['status']})"
        return msg, msg, ""
    flagged = [f"- **{t}**: {FLAG_LABELS.get(c, c)}" for t, c in r["flags"]]
    b = r["version_b"]
    changes = [f"- {CHANGE_LABELS.get(k, k)}: {v}" for k, v in b["changes"]]
    notes = [f"- {n}" for n in b.get("notes", [])]
    report = ("**Flagged in the draft (A):**\n" + ("\n".join(flagged) if flagged else "(nothing)")
              + "\n\n**Changed by the rule book (B):**\n" + ("\n".join(changes) if changes else "(nothing; B = A)")
              + ("\n\n**Checked, not changed:**\n" + "\n".join(notes) if notes else ""))
    return r["kitaita"], b["kitaita"], report


def haiku_report(h: dict) -> str:
    """Not for blind raters: how translation 2 was built."""
    if h["kitaita"] is None:
        return f"**Translation 2 (Haiku + tools):** {h['status']}"
    calls = [f"- `{c['name']}({json.dumps(c['input'], ensure_ascii=False)[:90]})` -> {c['status']}" for c in h["tool_calls"]]
    return (f"**Translation 2 = Claude Haiku + rule-book tools ({haiku_mt.rulebook_tag()}).** "
            f"{len(h['tool_calls'])} tool calls, {h.get('n_requests', 0)} model requests, est. ${h['usd']:.4f}; "
            f"retrieval corpus: {h['corpus']}. `[?word]` = no Kidaw'ida found, Swahili kept.\n\n"
            + (f"**Model's notes:** {h['notes']}\n\n" if h["notes"] else "")
            + "**Tool calls:**\n" + "\n".join(calls))


def translate(text: str, mode: str, request: gr.Request = None):
    """Generator: shows translation 1 (fine-tuned) first, then translation 2 (Haiku + tools, ~30 s),
    then the research pipeline in its accordion."""
    text = (text or "").strip()
    if not text:
        yield "", "", "", "", "", ""
        return
    ft = finetuned_mt.translate(text)
    ft_out = ft["kitaita"] if ft["kitaita"] is not None else f"(fine-tuned model {ft['status']})"
    yield "", "", "", ft_out, "(translating... about 30 seconds)", ""
    h = haiku_mt.translate(text, haiku_mt.visitor_id(request))
    b_out = h["kitaita"] if h["kitaita"] is not None else f"({h['status']})"
    report = haiku_report(h)
    yield "", "", "", ft_out, b_out, report
    fn = translate_sentence_with_probes if mode.startswith("Probes") else translate_sentence
    try:
        r = fn(text, _tables)
    except Exception as e:  # keep the Space up even if one input trips a bug
        r = None
        err = f"SERVER ERROR: {type(e).__name__}: {e}"
    if r is None:
        yield "", "", err, ft_out, b_out, report
        return
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
    yield out, flags_text, trace_text, ft_out, b_out, report


def random_test_sentence():
    """Direct user request: a button that hands the user a real Swahili
    sentence to try, instead of them having to think one up. Also clears
    the previous feedback status so an old "thanks, logged" message
    doesn't linger next to a brand-new, not-yet-rated sentence."""
    if not TEST_SENTENCES:
        return gr.update(), ""
    choice = random.choice(TEST_SENTENCES)
    return choice["sw"], ""


def submit_ab_feedback(swahili: str, a_output: str, b_output: str, rule_book_report: str,
                       rating_a: str, rating_b: str, correction: str):
    """2026-11-02: one record per A/B rating, written to feedback_ab/ in the same private dataset.
    2026-11-12: B is Haiku + tools (system_b "haiku+tools:..."); older records have
    system_b "finetuned+rulebook" and the Progress tab keeps the two pairings apart."""
    swahili = (swahili or "").strip()
    if not swahili or not (a_output or "").strip():
        return "Nothing to log yet -- translate a sentence first."
    if not rating_a and not rating_b:
        return "Rate translation 1 and/or 2 (1-5) before submitting."
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "Feedback was NOT saved: this Space has no HF_TOKEN secret configured."
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "swahili_input": swahili,
        "system_a": "finetuned:" + str(finetuned_mt._CFG.get("model_repo")),
        "system_b": haiku_mt.system_label(),
        "a_output": (a_output or "").strip(),
        "b_output": (b_output or "").strip(),
        "b_equals_a": (a_output or "").strip() == (b_output or "").strip(),
        "rule_book_report": rule_book_report,
        "blind": True,  # 2026-11-03: labels hidden; translation 1 = A, translation 2 = B (fixed order)
        "rating_a": rating_a or None,
        "rating_b": rating_b or None,
        "native_speaker_correction": (correction or "").strip() or None,
    }
    try:
        from huggingface_hub import HfApi
        import io
        HfApi(token=token).upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(record, ensure_ascii=False, indent=1).encode("utf-8")),
            path_in_repo=f"feedback_ab/{record['timestamp'].replace(':', '-')}_{record['id']}.json",
            repo_id=FEEDBACK_DATASET_REPO, repo_type="dataset")
    except Exception as e:
        return f"Feedback was NOT saved -- upload failed: {type(e).__name__}: {e}"
    return "Thanks -- logged for review." + (" (your correction was included.)" if record["native_speaker_correction"] else "")


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
# 2026-11-09, user decision: the Progress tab tracks only the two systems being
# compared (fine-tuned, fine-tuned + rule book). The rule-based pipeline's
# metrics and concord-slot charts were removed; its feedback (feedback/) is
# still saved but not shown.
def _load_all_feedback(folder: str = "feedback") -> list[dict]:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        files = [f for f in api.list_repo_files(FEEDBACK_DATASET_REPO, repo_type="dataset")
                 if f.startswith(folder + "/") and f.endswith(".json")]
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


# 2026-11-12: translation 2 changed from fine-tuned + rule book to Haiku + tools; ratings are kept per pairing
_PAIRINGS = [("haiku", "Haiku + tools"), ("finetuned+rulebook", "fine-tuned + rule book (until 2026-11-12)")]


def _pairing(record: dict) -> str:
    return "haiku" if str(record.get("system_b") or "").startswith("haiku") else "finetuned+rulebook"


def compute_all_feedback_overview() -> tuple[str, pd.DataFrame]:
    """Blind A/B feedback (feedback_ab/, the Translator tab's Submit feedback
    button). Ratings stay labelled by system, never averaged into one mean."""
    from api.stats import mean_ci, rate_ci, chrf
    from api.util import normalize_kitaita
    ab = _load_all_feedback("feedback_ab")

    # (system, rating or None, output, correction or None) -- one row per rated output
    rows, systems = [], []
    for key, label in _PAIRINGS:
        t1, t2 = f"Translation 1 (fine-tuned), rated next to {label}", f"Translation 2 ({label})"
        systems += [t1, t2]
        for r in ab:
            if _pairing(r) == key:
                corr = r.get("native_speaker_correction")
                rows.append((t1, r.get("rating_a"), r.get("a_output"), corr))
                rows.append((t2, r.get("rating_b"), r.get("b_output"), corr))

    all_recs = ab
    n_sent = len({(r.get("swahili_input") or "").strip().lower() for r in all_recs})
    n_corr = sum(1 for r in all_recs if r.get("native_speaker_correction"))
    lines = [f"**{len(all_recs)}** feedback submissions, covering **{n_sent}** different Swahili "
             f"sentences; **{n_corr}** include a native-speaker correction.", "",
             "| System | Ratings | Mean rating (95% CI) | Corrections | Match to correction (chrF, 0-100) | Exactly matches correction |",
             "|---|---|---|---|---|---|"]
    hist = []
    for sysname in systems:
        sr = [x for x in rows if x[0] == sysname]
        ratings = [int(x[1]) for x in sr if str(x[1] or "").isdigit()]
        m = mean_ci(ratings)
        if m["n"] == 0:
            mean_txt = "--"
        else:
            mean_txt = f"{m['mean']:.2f}" + (f" ({m['ci95'][0]:.2f}-{m['ci95'][1]:.2f})" if m["ci95"] else "")
            mean_txt += " low n" if m["low_n"] else ""
        with_corr = [x for x in sr if (x[3] or "").strip() and (x[2] or "").strip()]
        c = mean_ci([chrf(x[2], x[3]) for x in with_corr])
        chrf_txt = "--" if c["n"] == 0 else f"{c['mean']:.1f}" + (" low n" if c["low_n"] else "")
        ex = rate_ci(sum(1 for x in with_corr if normalize_kitaita(x[2]) == normalize_kitaita(x[3])), len(with_corr))
        ex_txt = "--" if ex["n"] == 0 else f"{ex['k']}/{ex['n']}"
        lines.append(f"| {sysname} | {m['n']} | {mean_txt} | {len(with_corr)} | {chrf_txt} | {ex_txt} |")
        for star in range(1, 6):
            hist.append({"stars": str(star), "count": ratings.count(star), "system": sysname})

    for key, label in _PAIRINGS:
        pairs = [(int(r["rating_a"]), int(r["rating_b"])) for r in ab if _pairing(r) == key and not r.get("b_equals_a")
                 and str(r.get("rating_a") or "").isdigit() and str(r.get("rating_b") or "").isdigit()]
        if pairs:
            better = rate_ci(sum(1 for a, b in pairs if b > a), len(pairs))
            ci = better["ci95"]
            lines.append(f"\n**{label}:** where translations 1 and 2 differed and both were rated, translation 2 "
                         f"was rated higher in {better['k']}/{better['n']} (95% CI {ci[0]:.2f}-{ci[1]:.2f})"
                         + (", low n." if better["low_n"] else "."))
        else:
            lines.append(f"\n**{label}:** no submission yet where translations 1 and 2 differed and both were rated.")
    return "\n".join(lines), pd.DataFrame(hist)


def refresh_dashboard():
    """Called on page load, on Refresh now, and every 60s."""
    return compute_all_feedback_overview()


with gr.Blocks(title="Kitaita Translator") as demo:
    with gr.Tabs():
        with gr.Tab("Translator"):
            gr.Markdown(
                "# Kitaita Translator\n"
                "Swahili &rarr; Kitaita (Kidaw'ida, ISO `dav`). Two machine translations of the same "
                "sentence, side by side. Translation 2 takes about 30 seconds. `[?word]` marks a word "
                "with no Kitaita found (the Swahili is kept).\n\n"
                "Both are machine translation awaiting native-speaker review."
            )
            inp = gr.Textbox(label="Swahili sentence", lines=2,
                             placeholder="e.g. Mbwa mkubwa aliona Kijana mdogo akitembea.")
            with gr.Row():
                btn = gr.Button("Translate", variant="primary")
                sample_btn = gr.Button("🎲 Give me a sentence to test")
            with gr.Row():
                # 2026-11-03, user request: blind comparison -- no system names on the page.
                # Translation 1 = fine-tuned NLLB (A), Translation 2 = Haiku + tools (B, since 2026-11-12);
                # submit_ab_feedback stores the ratings under A and B.
                ft_out = gr.Textbox(label="Translation 1", lines=2)
                b_out = gr.Textbox(label="Translation 2", lines=2)

            gr.Markdown("### Are these translations right?")
            gr.Markdown(
                "If you speak Kitaita: rate **each translation** separately and, if both are wrong, write "
                "the correct translation. Logged for review, not applied automatically."
            )
            with gr.Row():
                rating_a = gr.Radio(["1", "2", "3", "4", "5"], label="Rating for translation 1 (1 = way off, 5 = correct)")
                rating_b = gr.Radio(["1", "2", "3", "4", "5"], label="Rating for translation 2 (1 = way off, 5 = correct)")
            correction = gr.Textbox(label="Correct Kitaita translation (optional)", lines=2,
                                    placeholder="If neither output is right, what should it be?")
            feedback_btn = gr.Button("Submit feedback")
            feedback_status = gr.Markdown()

            with gr.Accordion("Research: the earlier rule-based pipeline (NLLB probes + concord rewriting)",
                              open=False):
                gr.Markdown(
                    "Translates directly from Swahili with dictionaries and concord rules; kept for "
                    "research. **Probes** mode uses the NLLB encoder + trained relation probes; "
                    "**heuristic** mode is an adjacency-based decomposer."
                )
                mode = gr.Radio(["Probes (full pipeline)", "Heuristic (for comparison)"],
                                value="Probes (full pipeline)", label="Mode")
                out = gr.Textbox(label="Rule-based pipeline output", lines=2)
                with gr.Accordion("Flags (identification/translation caveats on this sentence)", open=False):
                    flags_box = gr.Markdown()
                with gr.Accordion("Evidence trail (full per-step trace)", open=False):
                    trace_box = gr.Markdown()
                with gr.Row():
                    rating = gr.Radio(["1", "2", "3", "4", "5"], label="Rating for the rule-based output")
                    rb_correction = gr.Textbox(label="Correct Kitaita translation (optional)", lines=2)
                rb_feedback_btn = gr.Button("Submit feedback on the rule-based output")
                rb_feedback_status = gr.Markdown()
                # names the systems: kept out of sight for blind raters
                with gr.Accordion("How translation 2 was built (names the systems -- not for blind raters)",
                                  open=False):
                    ft_flags_box = gr.Markdown()

            outputs = [out, flags_box, trace_box, ft_out, b_out, ft_flags_box]
            btn.click(translate, inputs=[inp, mode], outputs=outputs)
            inp.submit(translate, inputs=[inp, mode], outputs=outputs)
            sample_btn.click(random_test_sentence, inputs=[], outputs=[inp, feedback_status])
            feedback_btn.click(submit_ab_feedback,
                               inputs=[inp, ft_out, b_out, ft_flags_box, rating_a, rating_b, correction],
                               outputs=[feedback_status])
            rb_feedback_btn.click(submit_feedback, inputs=[inp, out, mode, rating, rb_correction],
                                  outputs=[rb_feedback_status])
            gr.Examples(examples=EXAMPLES, inputs=[inp, mode])

        with gr.Tab("📊 Progress"):
            gr.Markdown(
                "# Evaluation progress\n"
                "Native-speaker ratings of translation 1 (fine-tuned) and translation 2 (Haiku + tools since "
                "2026-11-12; before that fine-tuned + rule book, shown separately), from the Translator tab's "
                "**Submit feedback** button. "
                "Refreshes automatically every 60s, or click Refresh now."
            )
            refresh_btn = gr.Button("🔄 Refresh now")
            overview_md = gr.Markdown()
            ratings_plot = gr.BarPlot(x="stars", y="count", color="system",
                                      title="Ratings by system (1 = way off, 5 = correct)", height=300)

            dash_outputs = [overview_md, ratings_plot]
            refresh_btn.click(refresh_dashboard, inputs=[], outputs=dash_outputs)
            demo.load(refresh_dashboard, inputs=[], outputs=dash_outputs)
            # 2026-10-26: Gradio 6's auto-refresh mechanism is gr.Timer (the
            # `every=` kwarg on .load()/.click() from older Gradio versions
            # was removed, confirmed directly -- TypeError on import until
            # this was fixed) -- a real, ticking component wired to the same
            # refresh function, not a cosmetic difference.
            dash_timer = gr.Timer(60)
            dash_timer.tick(refresh_dashboard, inputs=[], outputs=dash_outputs)

# 2026-10-26 (hackathon Space API, CLAUDE.md "Space API" section): the REST
# API (api/ package) and this Gradio UI share ONE process and ONE loaded
# model -- Gradio is mounted onto the FastAPI app at "/", so /api/* and
# /at/* are served alongside the existing tabs. The API only wraps the
# pipeline (same translate_sentence_with_probes/_tables as the UI above).
from api.server import build_app  # noqa: E402

app = build_app(_tables)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

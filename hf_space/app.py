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
import re
import random
import time
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
# 2026-11-14, user request ("news sources as well as the storybooks"): rebuilt by
# scripts/build_test_sentences.py -- 400 storybook, 500 news (Tanzanian newspapers + MasakhaNEWS: business,
# health, politics, religion, sports, technology, entertainment), 200 Swahili Wikipedia; Bible top-up dropped
# (it is in the fine-tuned model's training data); every sentence checked against the training data.
# The button picks a register first, then a sentence, so each register comes up equally often.
try:
    TEST_SENTENCES = json.loads(_SENTENCES_PATH.read_text(encoding="utf-8"))
except Exception:
    TEST_SENTENCES = []
_BY_REGISTER: dict[str, list[dict]] = {}
for _s in TEST_SENTENCES:
    _BY_REGISTER.setdefault(_s.get("register", "other"), []).append(_s)
_SENTENCE_INFO = {" ".join(_s["sw"].split()): _s for _s in TEST_SENTENCES}

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
import testers  # noqa: E402

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


_UNKNOWN_MARK = re.compile(r"\[\?([^\]]+)\]")


def unmark(text: str) -> str:
    """2026-11-12, user request: show Swahili words with no Kitaita as plain words, without [?...] brackets."""
    return _UNKNOWN_MARK.sub(r"\1", text or "")


def haiku_report(c: dict, e: dict) -> str:
    """Not for blind raters: how arms C (Haiku + tools) and E (Haiku zero-shot) were built."""
    if c["kitaita"] is None:
        head = f"**Arm C (Haiku + tools):** {c['status']}"
        calls = []
    else:
        head = (f"**Arm C = Claude Haiku + rule-book tools ({haiku_mt.rulebook_tag()}).** "
                f"{len(c['tool_calls'])} tool calls, {c.get('n_requests', 0)} model requests, est. ${c['usd']:.4f}; "
                f"retrieval corpus: {c['corpus']}. Marked by the model as having no Kidaw'ida (Swahili kept): "
                f"{', '.join(_UNKNOWN_MARK.findall(c['kitaita'])) or 'none'}.")
        calls = [f"- `{x['name']}({json.dumps(x['input'], ensure_ascii=False)[:90]})` -> {x['status']}"
                 for x in c["tool_calls"]]
    tail = (f"\n\n**Arm E = Claude Haiku zero-shot (no tools, no dictionary)**: {e['status']}"
            + (f", est. ${e['usd']:.4f}" if e.get("usd") else "")
            + "\n\n**Arm A = fine-tuned NLLB-600M.**")
    notes = f"\n\n**Model's notes (arm C):** {c['notes']}" if c.get("notes") else ""
    return head + notes + tail + ("\n\n**Tool calls (arm C):**\n" + "\n".join(calls) if calls else "")


ARMS = [("A", "fine-tuned NLLB"), ("C", "Haiku + tools"), ("E", "Haiku zero-shot")]
ARM_KEYS = [k for k, _ in ARMS]


def order_key(order: list[str]) -> str:
    """Researcher view: which arm sits behind each displayed translation this time."""
    names = dict(ARMS)
    return "**Display order this time:** " + ", ".join(
        f"Translation {i} = {k} ({names[k]})" for i, k in enumerate(order, 1))


def translate(text: str, request: gr.Request = None):
    """Translate one Swahili sentence three ways."""
    # (Docstrings of UI callbacks are published in the page config, so system details live in comments.)
    # Generator, the three study arms: A = fine-tuned NLLB, C = Haiku + tools, E = Haiku zero-shot.
    # 2026-11-15, user request (before inviting testers): the arms are shown in a fresh random order for
    # every sentence (`order[i]` = the arm in Translation i+1), and all three appear together when the
    # slowest is done -- showing the fast arm first would reveal which slot it is.
    # 2026-11-12, user decision: the probe pipeline is no longer run or shown here (it confused testers);
    # /api/translate still serves it.
    # Yields (translation 1, 2, 3, order, researcher report).
    text = (text or "").strip()
    if not text:
        yield "", "", "", [], ""
        return
    order = random.sample(ARM_KEYS, len(ARM_KEYS))
    wait = "(translating... up to about 30 seconds)"
    yield wait, wait, wait, order, ""
    ft = finetuned_mt.translate(text)
    a_out = ft["kitaita"] if ft["kitaita"] is not None else f"(fine-tuned model {ft['status']})"
    why = haiku_mt.precheck(text, haiku_mt.visitor_id(request))
    if why:
        outs = {"A": a_out, "C": f"({why})", "E": f"({why})"}
        report = order_key(order) + f"\n\n**Arms C and E (Haiku):** {why}"
    else:
        e = haiku_mt.translate_zero_shot(text)
        c = haiku_mt.translate(text, checked=True)
        outs = {"A": a_out,
                "C": unmark(c["kitaita"]) if c["kitaita"] is not None else f"({c['status']})",
                "E": unmark(e["kitaita"]) if e["kitaita"] is not None else f"({e['status']})"}
        report = order_key(order) + "\n\n" + haiku_report(c, e)
    yield outs[order[0]], outs[order[1]], outs[order[2]], order, report


def _tester_of(phone: str) -> str | None:
    """Salted tester id for a phone box value, or None (empty / invalid number / no salt)."""
    p = testers.normalize_phone(phone or "")
    if not p:
        return None
    try:
        return testers.tester_id(p)
    except RuntimeError:
        return None


def random_test_sentence(seen: list | None = None, phone: str = ""):
    """Direct user request: a button that hands the user a real Swahili
    sentence to try, instead of them having to think one up. Also clears
    the previous feedback status so an old "thanks, logged" message
    doesn't linger next to a brand-new, not-yet-rated sentence.
    2026-11-17, user request: no repeats -- skips sentences this browser was already given (`seen`, kept in
    the browser) and, for a paid tester, sentences they already rated (loaded in the background on first use).
    Returns (sentence, status, updated seen)."""
    seen = list(seen or [])
    if not TEST_SENTENCES:
        return gr.update(), "", seen
    skip = set(seen)
    tid = _tester_of(phone)
    if tid and os.environ.get("HF_TOKEN"):
        testers.load_rated_in_background(tid, os.environ["HF_TOKEN"], FEEDBACK_DATASET_REPO)
        skip |= testers.rated_if_loaded(tid)
    regs = sorted(_BY_REGISTER)
    random.shuffle(regs)
    for reg in regs:   # register first (even mix), then an unseen sentence in it
        fresh = [x for x in _BY_REGISTER[reg] if testers.norm_sentence(x["sw"]) not in skip]
        if fresh:
            choice = random.choice(fresh)
            break
    else:              # everything seen: start over
        seen = []
        choice = random.choice(_BY_REGISTER[random.choice(sorted(_BY_REGISTER))])
    return choice["sw"], "", seen + [testers.norm_sentence(choice["sw"])]


def submit_ab_feedback(swahili: str, out1: str, out2: str, out3: str, order: list, report: str,
                       rating1: str, rating2: str, rating3: str, correction: str, phone: str = ""):
    """Save one blind rating of the three translations."""
    # One record per blind rating of the three arms, in feedback_ab/ of the private dataset.
    # 2026-11-15: the arms are shown in a random order per sentence; `order[i]` is the arm in Translation
    # i+1, so slot outputs/ratings are mapped back to arms A, C, E here and `display_order` records the order.
    # Legacy fields a_*/b_* are kept so older readers work (b = C). On success the form is cleared so the
    # tester knows to go on (user request).
    # 2026-11-13 (paid testers, testers.py): the record carries only a salted hash of the tester's phone number;
    # the number itself goes to the private testers dataset. The phone box is not cleared.
    keep = (gr.update(),) * 10
    swahili = (swahili or "").strip()
    slots = [(out1 or "").strip(), (out2 or "").strip(), (out3 or "").strip()]
    if not swahili or not any(slots) or sorted(order or []) != sorted(ARM_KEYS):
        return ("Nothing to log yet -- translate a sentence first.",) + keep
    if not (rating1 or rating2 or rating3):
        return ("Rate at least one translation (1-5) before submitting.",) + keep
    token = os.environ.get("HF_TOKEN")
    if not token:
        return ("Feedback was NOT saved: this Space has no HF_TOKEN secret configured.",) + keep
    tid, phone_n = None, None
    if (phone or "").strip():
        phone_n = testers.normalize_phone(phone)
        if not phone_n:
            return ("Feedback was NOT saved: the phone number does not look right (e.g. 0712345678 or "
                    "+254712345678). Fix it, or clear the box to submit without payment.",) + keep
        try:
            tid = testers.tester_id(phone_n)
        except RuntimeError:
            return ("Feedback was NOT saved: payments are not set up on this Space (no PHONE_HASH_SALT secret).",) + keep
    outs = dict(zip(order, slots))
    ratings = dict(zip(order, [rating1 or None, rating2 or None, rating3 or None]))
    # 2026-11-17: one click was saved up to 6 times (clicks queued during the upload); drop an identical
    # resubmission within testers.DUPLICATE_WINDOW_S and say it is already saved.
    dup_key = (tid or "anon", testers.norm_sentence(swahili), tuple(sorted(ratings.items())),
               (correction or "").strip())
    if testers.is_duplicate_submit(dup_key, time.time()):
        return ("### ✅ Already saved -- this submission was received a moment ago.\n"
                "Type or pick the next sentence to continue.", "", "", "", "", [], "", None, None, None, "")
    systems = {"A": "finetuned:" + str(finetuned_mt._CFG.get("model_repo")), "C": haiku_mt.system_label(),
               "E": haiku_mt.zero_shot_label()}
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "swahili_input": swahili,
        # where a "Give me a sentence" sentence came from (None = typed by the tester)
        "test_sentence": {k: _SENTENCE_INFO[" ".join(swahili.split())].get(k) for k in ("id", "register", "source")}
                         if " ".join(swahili.split()) in _SENTENCE_INFO else None,
        "arms": {k: {"system": systems[k], "output": outs[k], "rating": ratings[k]} for k in outs},
        "display_order": list(order),   # arm shown as translation 1, 2, 3 (random per sentence, labels hidden)
        "system_a": systems["A"], "system_b": systems["C"], "a_output": outs["A"], "b_output": outs["C"],
        "b_equals_a": outs["A"] == outs["C"], "rating_a": ratings["A"], "rating_b": ratings["C"],
        "report": report,
        "blind": True,
        "native_speaker_correction": (correction or "").strip() or None,
        "tester_id": tid,   # salted hash only; never the number
        "complete_for_payment": testers.is_complete(ratings, correction),
    }
    tag = f"_{tid}_{'complete' if record['complete_for_payment'] else 'partial'}" if tid else ""
    try:
        from huggingface_hub import HfApi
        import io
        HfApi(token=token).upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(record, ensure_ascii=False, indent=1).encode("utf-8")),
            path_in_repo=f"feedback_ab/{record['timestamp'].replace(':', '-')}{tag}_{record['id']}.json",
            repo_id=FEEDBACK_DATASET_REPO, repo_type="dataset")
    except Exception as e:
        testers.forget_submit(dup_key)
        return (f"Feedback was NOT saved -- upload failed: {type(e).__name__}: {e}",) + keep
    msg = ("### ✅ Thank you! Your feedback was saved."
           + (" (Your correction was included.)" if record["native_speaker_correction"] else "")
           + "\nThe form has been cleared: type or pick the next sentence to continue.")
    if tid:
        try:
            testers.register(phone_n, tid, token)
            # 2026-11-17: count unique sentences, as the payout script does (load includes the file just saved)
            testers.load_rated(tid, token, FEEDBACK_DATASET_REPO)
            testers.record_rated(tid, swahili, record["complete_for_payment"])
            n = testers.unique_complete(tid)
            msg += (f"\n\n**Paid testing:** {n} complete submission(s) so far (payment every {testers.PAY_EVERY}; "
                    "each sentence counts once).")
        except Exception as e:
            msg += f"\n\n⚠️ Saved, but your phone number could not be registered for payment ({type(e).__name__}); tell the organiser."
        if not record["complete_for_payment"]:
            msg += ("\n\nThis one does **not** count for payment: a paid submission rates all three translations "
                    "**and** gives your own Kitaita translation.")
    # cleared: input, 3 outputs, order, report, 3 ratings, correction
    return (msg, "", "", "", "", [], "", None, None, None, "")


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


# 2026-11-12, user decision: the Progress tab tracks only the three study arms -- A (fine-tuned NLLB),
# C (Haiku + tools), E (Haiku zero-shot): number of ratings, mean rating, and chrF against native-speaker
# corrections. Ratings of the retired "fine-tuned + rule book" translation are not shown.
def _arm_rows(record: dict) -> list[tuple]:
    """(arm, rating, output, correction) for the A / C / E outputs in one feedback record."""
    corr = record.get("native_speaker_correction")
    if "arms" in record:
        return [(k, v.get("rating"), v.get("output"), corr) for k, v in record["arms"].items() if k in ("A", "C", "E")]
    rows = [("A", record.get("rating_a"), record.get("a_output"), corr)]
    if str(record.get("system_b") or "").startswith("haiku+tools"):
        rows.append(("C", record.get("rating_b"), record.get("b_output"), corr))
    return rows


def compute_all_feedback_overview() -> tuple[str, pd.DataFrame]:
    from api.stats import mean_ci, chrf
    ab = _load_all_feedback("feedback_ab")
    rows = [x for r in ab for x in _arm_rows(r)]
    lines = ["| Arm | Ratings | Mean rating (95% CI) | Corrections | Mean chrF vs correction (95% CI) |",
             "|---|---|---|---|---|"]
    hist = []
    for arm, name in ARMS:
        sr = [x for x in rows if x[0] == arm]
        ratings = [int(x[1]) for x in sr if str(x[1] or "").isdigit()]
        m = mean_ci(ratings)
        def fmt(v, digits, lo, hi):
            """mean (95% CI clipped to the scale's range: at low n the t interval runs past it)"""
            if v["n"] == 0:
                return "--"
            ci = v["ci95"]
            t = f"{v['mean']:.{digits}f}" + (f" ({max(lo, ci[0]):.{digits}f}-{min(hi, ci[1]):.{digits}f})" if ci else "")
            return t + (" low n" if v["low_n"] else "")
        with_corr = [x for x in sr if (x[3] or "").strip() and (x[2] or "").strip() and not str(x[2]).startswith("(")]
        c = mean_ci([chrf(x[2], x[3]) for x in with_corr])
        lines.append(f"| {arm}: {name} | {m['n']} | {fmt(m, 2, 1, 5)} | {c['n']} | {fmt(c, 1, 0, 100)} |")
        for star in range(1, 6):
            hist.append({"stars": str(star), "count": ratings.count(star), "system": f"{arm}: {name}"})
    n_sent = len({(r.get("swahili_input") or "").strip().lower() for r in ab if _arm_rows(r)})
    head = (f"Ratings from **{n_sent}** different Swahili sentences. Mean rating: 1 = way off, 5 = correct. "
            "chrF (0-100) compares each arm's output with the native speaker's correction, when one was given. "
            "\"low n\" = fewer than 5.\n\n")
    return head + "\n".join(lines), pd.DataFrame(hist)


def refresh_dashboard():
    """Called on page load, on Refresh now, and every 60s."""
    return compute_all_feedback_overview()


def build_translator_tab(blocks: gr.Blocks, researcher: bool) -> None:
    """Translator tab contents. researcher=False: the public, blind page for testers;
    researcher=True: the same page plus the report naming each system (password-protected)."""
    gr.Markdown(
        "# Kitaita Translator\n"
        "Swahili &rarr; Kitaita (Kidaw'ida, ISO `dav`). Three machine translations of the same "
        "sentence, in a different order each time. They take up to 30 seconds.\n\n"
        "All three are machine translation awaiting native-speaker review.\n\n"
        "<small>Test sentences: African Storybook Initiative (CC BY / CC BY-NC), Swahili News Dataset "
        "(CC BY 4.0), MasakhaNEWS (AFL-3.0), Swahili Wikipedia (CC BY-SA 3.0).</small>"
    )
    # 2026-11-13, paid testers (testers.py): the number stays in this browser (BrowserState) across
    # submits and reloads; feedback stores only its salted hash.
    with gr.Accordion("Paid testers: your phone number", open=True):
        phone = gr.Textbox(label="Phone number (M-Pesa) -- leave empty if you are not a paid tester",
                           placeholder="e.g. 0712345678", max_lines=1)
        gr.Markdown(
            "Used only to pay you. It is stored privately, separately from your ratings (which carry only "
            "a code), and deleted after payment. A submission counts when you rate **all three** "
            f"translations **and** write your own Kitaita translation; payment every {testers.PAY_EVERY} "
            "complete submissions. Each sentence counts once.")
        phone_store = gr.BrowserState("", storage_key="kitaita_tester_phone")
    seen_store = gr.BrowserState([], storage_key="kitaita_seen_sentences")   # sentences this browser was given
    inp = gr.Textbox(label="Swahili sentence", lines=2,
                     placeholder="e.g. Mama alimpa mtoto maji ya kunywa.")
    with gr.Row():
        btn = gr.Button("Translate", variant="primary")
        sample_btn = gr.Button("🎲 Give me a sentence to test")
    with gr.Row():
        # blind comparison: no system names on the page, and the arm behind each slot is drawn at random
        # per sentence (translate); `order` is server-side state, submit_ab_feedback maps slots back to arms.
        out1 = gr.Textbox(label="Translation 1", lines=2)
        out2 = gr.Textbox(label="Translation 2", lines=2)
        out3 = gr.Textbox(label="Translation 3", lines=2)
    order = gr.State([])

    gr.Markdown("### Are these translations right?")
    gr.Markdown(
        "If you speak Kitaita: rate **each translation** separately and, if none is right, write "
        "the correct translation. Logged for review, not applied automatically."
    )
    with gr.Row():
        rating1 = gr.Radio(["1", "2", "3", "4", "5"], label="Translation 1 (1 = way off, 5 = correct)")
        rating2 = gr.Radio(["1", "2", "3", "4", "5"], label="Translation 2 (1 = way off, 5 = correct)")
        rating3 = gr.Radio(["1", "2", "3", "4", "5"], label="Translation 3 (1 = way off, 5 = correct)")
    correction = gr.Textbox(label="Your Kitaita translation (required for paid testers)", lines=2,
                            placeholder="How would you say this sentence in Kitaita?")
    feedback_btn = gr.Button("Submit feedback", variant="primary")
    feedback_status = gr.Markdown()

    # names the systems. 2026-11-15, user request: only on the password-protected /research page; on the
    # public page the report is server-side state (never sent to the browser) and still saved with feedback.
    if researcher:
        with gr.Accordion("For researchers: which system made each translation", open=True):
            report_box = gr.Markdown()
    else:
        report_box = gr.State("")

    outputs = [out1, out2, out3, order, report_box]
    btn.click(translate, inputs=[inp], outputs=outputs)
    inp.submit(translate, inputs=[inp], outputs=outputs)
    sample_btn.click(random_test_sentence, inputs=[seen_store, phone], outputs=[inp, feedback_status, seen_store])
    # Submit is disabled while saving (2026-11-17: queued extra clicks were saved as extra submissions)
    feedback_btn.click(lambda: gr.update(interactive=False), None, feedback_btn, queue=False).then(
        submit_ab_feedback,
        inputs=[inp, out1, out2, out3, order, report_box, rating1, rating2, rating3, correction, phone],
        outputs=[feedback_status, inp, out1, out2, out3, order, report_box,
                 rating1, rating2, rating3, correction]).then(
        lambda: gr.update(interactive=True), None, feedback_btn, queue=False)
    gr.Examples(examples=[[e[0]] for e in EXAMPLES], inputs=[inp])
    blocks.load(lambda s: s or "", inputs=[phone_store], outputs=[phone])
    phone.change(lambda p: p, inputs=[phone], outputs=[phone_store])


def build_progress_tab(blocks: gr.Blocks) -> None:
    """Progress dashboard. Names the systems and their scores, so research page only (2026-11-15)."""
    gr.Markdown(
        "# Evaluation progress\n"
        "Native-speaker ratings of the three systems being tested -- **A**: fine-tuned NLLB, "
        "**C**: Haiku + rule-book tools, **E**: Haiku zero-shot -- from the Translator tab's "
        "**Submit feedback** button (testers see them in a random order, unnamed). "
        "Refreshes automatically every 60s, or click Refresh now."
    )
    refresh_btn = gr.Button("🔄 Refresh now")
    overview_md = gr.Markdown()
    ratings_plot = gr.BarPlot(x="stars", y="count", color="system",
                              title="Ratings by arm (1 = way off, 5 = correct)", height=300)

    dash_outputs = [overview_md, ratings_plot]
    refresh_btn.click(refresh_dashboard, inputs=[], outputs=dash_outputs)
    blocks.load(refresh_dashboard, inputs=[], outputs=dash_outputs)
    # Gradio 6: auto-refresh is gr.Timer (the old `every=` kwarg was removed)
    dash_timer = gr.Timer(60)
    dash_timer.tick(refresh_dashboard, inputs=[], outputs=dash_outputs)


# Public page (testers): blind Translator only. 2026-11-15, user request: the researcher report and the
# Progress tab (both name the systems) moved to the password-protected /research page.
with gr.Blocks(title="Kitaita Translator") as demo:
    build_translator_tab(demo, researcher=False)

with gr.Blocks(title="Kitaita Research") as research_demo:
    with gr.Tabs():
        with gr.Tab("Translator (with system names)"):
            build_translator_tab(research_demo, researcher=True)
        with gr.Tab("📊 Progress"):
            build_progress_tab(research_demo)

# 2026-10-26 (hackathon Space API, CLAUDE.md "Space API" section): the REST
# API (api/ package) and this Gradio UI share ONE process and ONE loaded
# model -- Gradio is mounted onto the FastAPI app at "/", so /api/* and
# /at/* are served alongside the existing tabs. The API only wraps the
# pipeline (same translate_sentence_with_probes/_tables as the UI above).
from api.server import build_app  # noqa: E402

app = build_app(_tables)
# /research must be mounted before "/", which would otherwise catch it. Login: user "researcher", password =
# Space secret RESEARCHER_PASSWORD; without the secret the page is not served at all.
if os.environ.get("RESEARCHER_PASSWORD"):
    app = gr.mount_gradio_app(app, research_demo, path="/research",
                              auth=("researcher", os.environ["RESEARCHER_PASSWORD"]),
                              auth_message="Kitaita research page (system names). Not for testers.")
else:
    print("RESEARCHER_PASSWORD not set: the /research page is disabled.", flush=True)
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))

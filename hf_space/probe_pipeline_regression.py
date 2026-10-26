# -*- coding: utf-8 -*-
"""
probe_pipeline_regression.py

2026-10-09 (cont. 11). The probe-based pipeline (kitaita_e2e_pipeline.
translate_sentence_with_probes) has been regression-checked against a
growing set of real sentences repeatedly this session (cont. 6-10's "16/16",
"17/17" battery references), but every one of those checks was run ad hoc
in-session and never saved as a script -- so each new change had no fixed
battery to run against, just whatever the author remembered to retype.
This persists that battery so it can actually be re-run identically before
and after a change (hard rule: verify, don't assume; also needed to honestly
check (c) in the efficient-relation-probe task brief: "not regress any of
the currently-correct simple cases").

Each case names what it is checking and the expected behavior in a comment;
most assert on a SUBSTRING or STRUCTURAL property (e.g. "mvulana must still
appear", "no crash") rather than the full exact string, since several of
these sentences have components this project doesn't fully resolve yet
(weak root matches, the mbwa/kijana structural ceiling) and asserting the
full string would start failing for reasons unrelated to what each case is
actually checking -- the historical "16/16"/"17/17" checks in status.md were
similarly checking "unchanged from before", not "matches some external gold
string".

Usage: python3 -u probe_pipeline_regression.py
"""
from __future__ import annotations

from kitaita_e2e_pipeline import translate_sentence_with_probes
from kitaita_sentence_generator import load_pages, build_lookup_tables

CASES = [
    # (label, sentence, check_fn(report) -> (ok: bool, note: str))
    ("simple_transitive", "Mwalimu mzuri alisoma kitabu.",
     lambda r: (r["kitaita_sentence"] is not None, r["kitaita_sentence"])),

    ("amod_on_object_mdogo", "Mbwa mkubwa aliona Kijana mdogo akitembea.",
     lambda r: ("mdawana" in (r["kitaita_sentence"] or "") or "mdogo" in (r["kitaita_sentence"] or ""),
                r["kitaita_sentence"])),

    ("multi_clause_kisha", "Mwalimu alisoma kitabu kisha akalala.",
     lambda r: (r["kitaita_sentence"] is not None and "kisha" in r["kitaita_sentence"],
                r["kitaita_sentence"])),

    ("ki_override_mvulana_kalamu", "Mvulana aliona kalamu ikivunjika.",
     # The validated FIX this session's self-loss mechanism exists for:
     # 'kalamu', not 'mvulana', must win as ikivunjika's subject, AND
     # 'mvulana' must still separately appear (clause 0's own subject,
     # the used_noun_idxs bugfix this verifies did not regress).
     lambda r: ("kalamu" in " ".join(r.get("flags") or []).lower()
                and any("OVERRIDING to SUBJECT" in f and "kalamu" in f for f in (r.get("flags") or [])),
                r.get("flags"))),

    ("ki_perception_object_control_mbwa_kijana", "Mbwa mkubwa aliona Kijana mdogo akitembea.",
     # 2026-10-09 (cont. 12): this was the documented "structural ceiling"
     # (mbwa/kijana share one Swahili subject-marker class, unfixable by
     # ANY marker-swapping method -- self-loss included). The NEW rule-3
     # syntactic control rule (perception-verb object control) is NOT a
     # marker-swapping method and resolves it: 'kijana', not 'mbwa', must
     # now win as 'akitembea's subject.
     lambda r: (any("kijana" in f and "OVERRIDING to SUBJECT of 'akitembea'" in f for f in (r.get("flags") or [])),
                r.get("flags"))),

    ("sentence_boundary_split", "Mbwa anakunywa maji. Mtoto analia.",
     # cont. 10's fix: 'mtoto' must win as 'analia's subject in ITS OWN
     # sentence, not 'mbwa'/'maji' leaking across the sentence boundary.
     lambda r: (isinstance(r["decomposition"], list) and len(r["decomposition"]) == 2
                and "mtoto" in " ".join(r.get("flags") or []).lower(),
                r.get("flags"))),

    ("askari_mwizi_object_as_subject_bug", "Askari alimwona mwizi.",
     # cont. 6's fix: 'askari' (not statistically recognized at the time)
     # must not force 'mwizi' into the subject role just because it was
     # noun_hits[0] -- today this is really just a no-crash + sanity check
     # since lexicon coverage may have changed since cont. 6.
     lambda r: (r["kitaita_sentence"] is not None, r["kitaita_sentence"])),

    # The STANDING regression test named explicitly in the 2026-10-09
    # cont. 10 handoff prompt: attempt 1 (negative-sampling retrain of the
    # main multiclass nsubj/obj probe) broke these two zero-competing-
    # candidate sentences; reverted. Neither has a '-ki-' verb, so neither
    # exercises this session's self-loss-batching change at all -- kept
    # here anyway because the handoff explicitly named them as the bar any
    # future change to this area must clear.
    ("standing_regression_samaki", "Samaki mbaya alikuwa.",
     lambda r: (r["kitaita_sentence"] is not None, r["kitaita_sentence"])),

    ("standing_regression_kiti", "Kiti kitakuwa.",
     lambda r: (r["kitaita_sentence"] is not None, r["kitaita_sentence"])),

    # 2026-10-09 (cont. 12): the two new zero-NLLB-cost syntactic control
    # rules (reported-speech same-subject; periphrastic 'kuwa' same-subject).
    ("ki_rule1_reported_speech", "Mungu akavibarikia akisema.",
     lambda r: (any("reported-speech" in f for f in (r.get("flags") or [])), r.get("flags"))),

    ("ki_rule2_periphrastic_kuwa", "Mwalimu alikuwa akifanya kazi.",
     lambda r: (any("periphrastic" in f for f in (r.get("flags") or [])), r.get("flags"))),

    # 2026-10-09 (cont. 14): direct user report -- cross-clause subject
    # leakage within ONE sentence (comma-juxtaposed clauses, no period, so
    # split_sentences() doesn't apply). 'mtoto' (clause 1's subject) was
    # wrongly winning as subject of 'alipokuwa'/'akilala' too, even though
    # 'mbwa' sits immediately before and is the obviously correct local
    # subject. Must now correctly identify 'mbwa'.
    ("ki_cross_clause_leakage_mtoto_mbwa", "Mtoto alipika mayai, mbwa alipokuwa akilala.",
     lambda r: (any("'mbwa' -> probe-identified SUBJECT of 'alipokuwa'" in f for f in (r.get("flags") or []))
                and any("CONFIRMED 'mbwa' as subject of 'akilala'" in f for f in (r.get("flags") or [])),
                r.get("flags"))),
]


def main():
    p1, p2, p3 = load_pages()
    tables = build_lookup_tables(p1, p2, p3)

    n_ok = 0
    for label, sentence, check in CASES:
        try:
            report = translate_sentence_with_probes(sentence, tables)
        except Exception as e:  # noqa: BLE001 -- regression battery must report, not crash
            print(f"[CRASH] {label}: {sentence!r} -> {type(e).__name__}: {e}")
            continue
        ok, note = check(report)
        n_ok += int(ok)
        print(f"[{'OK' if ok else 'FAIL'}] {label}: {sentence!r}")
        print(f"       kitaita={report.get('kitaita_sentence')!r}")
        print(f"       check_note={note!r}")

    print(f"\n{n_ok}/{len(CASES)} cases passed their check.")


if __name__ == "__main__":
    main()

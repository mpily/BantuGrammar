"""Round-1 mined-lexicon fallback tier (kitaita_mined_lexicon.py). Pure Python:
no torch, no model, so these run in the Space sync workflow too."""
from types import SimpleNamespace

import pytest

import kitaita_mined_lexicon as ml
from api.translator import adapt_report

CLASS_DEPENDENT = ["yangu", "zake", "wake", "hilo", "huyo", "ya", "za", "la", "cha", "wote"]


@pytest.fixture(autouse=True)
def default_settings():
    saved = (ml.MIN_CONFIDENCE, ml.PREFER_DAVSWA_FORM)
    yield
    ml.MIN_CONFIDENCE, ml.PREFER_DAVSWA_FORM = saved


def decomposition(tokens, passthrough, secondary=()):
    return SimpleNamespace(tokens=tokens, passthrough=list(passthrough), secondary_nouns=list(secondary),
                           flags=[], slots=[], subject_class=None, mined_fills=[])


def test_table_carries_provenance_and_is_unverified():
    meta = ml._PATH.read_text(encoding="utf-8")
    assert "NOT speaker-verified" in meta
    e = ml.mined_lookup("kweli")
    assert e and e["kitaita"] and e["confidence"] in ml.CONF_RANK


@pytest.mark.parametrize("word", CLASS_DEPENDENT)
def test_class_dependent_words_are_never_filled(word):
    # their Kidaw'ida form depends on the controlling noun's Kidaw'ida class
    assert ml.mined_lookup(word, "derived") is None


def test_confidence_floor_and_off_switch():
    ml.MIN_CONFIDENCE = None
    assert ml.mined_lookup("kweli") is None
    derived = next(k for k, v in ml._table().items() if v["confidence"] == "derived")
    assert ml.mined_lookup(derived, "high") is None
    assert ml.mined_lookup(derived, "derived") is not None


def test_fill_moves_passthrough_and_flags_it():
    d = decomposition(["mtoto", "kweli", "zzqx"], [(1, "kweli"), (2, "zzqx")])
    ml.apply_mined_lexicon_fallback(d, "derived")
    assert d.passthrough == [(2, "zzqx")]
    assert [(i, sw) for i, sw, _, _ in d.mined_fills] == [(1, "kweli")]
    assert any("NOT speaker-verified" in f for f in d.flags)


def test_secondary_noun_without_kitaita_gets_mined_form_but_existing_form_is_kept():
    d = decomposition(["a", "b"], [], [(0, "mtoto", None), (1, "kweli", "EXISTING")])
    ml.apply_mined_lexicon_fallback(d, "derived")
    assert d.secondary_nouns[0][2] == ml.mined_lookup("mtoto", "derived")["kitaita"]
    assert d.secondary_nouns[1][2] == "EXISTING"  # never overrides an earlier tier


def test_api_no_longer_lists_filled_words_as_unknown():
    d = decomposition(["kweli", "zzqx"], [(0, "kweli"), (1, "zzqx")])
    ml.apply_mined_lexicon_fallback(d, "derived")
    kit = ml.mined_lookup("kweli", "derived")["kitaita"]
    out = adapt_report({"kitaita_sentence": f"{kit} zzqx.", "decomposition": d, "flags": []})
    assert out["unknown_words"] == ["zzqx"]
    assert "[?zzqx]" in out["kitaita_marked"] and f"[?{kit}]" not in out["kitaita_marked"]

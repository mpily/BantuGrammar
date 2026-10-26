"""Runs the adapter against the REAL pipeline in heuristic mode (no NLLB
download needed). Skipped where torch isn't installed."""
import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="module")
def real():
    from kitaita_sentence_generator import build_lookup_tables, load_pages
    from api.translator import pipeline_translator
    return pipeline_translator(build_lookup_tables(*load_pages()), pipeline_version="t")


def test_passthrough_swahili_is_reported_unknown(real):
    out, _, _ = real.translate("Nyoka aja.", "heuristic")
    assert "aja" in out["unknown_words"]
    assert "[?aja]" in out["kitaita_marked"]
    assert out["vocabulary_gaps"] >= 1


def test_slots_are_exposed(real):
    out, _, _ = real.translate("Kitabu cha mwalimu kilikuwa kikubwa.", "heuristic")
    assert out["kitaita"]
    assoc = [s for s in out["slots"] if s["slot_type"] == "associative"]
    assert assoc and assoc[0]["controller"].lower() == "kitabu" and assoc[0]["relation"] == "nmod"

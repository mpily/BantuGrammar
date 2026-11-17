"""Paid testers (testers.py, 2026-11-13): phone normalisation, hashing, the complete-submission rule."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import testers  # noqa: E402


@pytest.mark.parametrize("raw,norm", [
    ("0712345678", "+254712345678"), ("0112 345 678", "+254112345678"), ("712345678", "+254712345678"),
    ("254712345678", "+254712345678"), ("+254 712-345-678", "+254712345678"), ("00254712345678", "+254712345678"),
    ("+255712345678", "+255712345678"),
])
def test_normalize_phone(raw, norm):
    assert testers.normalize_phone(raw) == norm


@pytest.mark.parametrize("raw", ["", "12345", "0812345678", "+2547123", "hello", "+254812345678"])
def test_normalize_phone_rejects(raw):
    assert testers.normalize_phone(raw) is None


def test_tester_id_salted_and_stable(monkeypatch):
    monkeypatch.setenv("PHONE_HASH_SALT", "s1")
    a = testers.tester_id("+254712345678")
    assert a == testers.tester_id("+254712345678") and a.startswith("ph_") and "712345678" not in a
    monkeypatch.setenv("PHONE_HASH_SALT", "s2")
    assert testers.tester_id("+254712345678") != a
    monkeypatch.setenv("PHONE_HASH_SALT", "")
    with pytest.raises(RuntimeError):
        testers.tester_id("+254712345678")


def test_is_complete():
    full = {"A": "3", "C": "4", "E": "1"}
    assert testers.is_complete(full, "Mndu wacha")
    assert not testers.is_complete(full, "  ")
    assert not testers.is_complete({**full, "E": None}, "Mndu wacha")


def test_complete_count():
    files = ["feedback_ab/t1_ph_aa_complete_x.json", "feedback_ab/t2_ph_aa_partial_y.json",
             "feedback_ab/t3_ph_bb_complete_z.json", "feedback_ab/t4_old.json"]
    assert testers.complete_count(files, "ph_aa") == 1


def test_norm_sentence_shared_key():
    assert testers.norm_sentence("  Walifanya kazi  zao.") == testers.norm_sentence("walifanya kazi zao")


def test_duplicate_submit_window_and_forget():
    k = ("ph_x", "sentence", (("A", "4"),), "corr")
    assert not testers.is_duplicate_submit(k, 1000.0)
    assert testers.is_duplicate_submit(k, 1010.0)                 # second click 10 s later: dropped
    assert not testers.is_duplicate_submit(k, 1000.0 + testers.DUPLICATE_WINDOW_S + 11)   # window passed
    testers.forget_submit(k)
    assert not testers.is_duplicate_submit(k, 2000.0)             # failed save forgotten: retry goes through


def test_unique_complete_counts_each_sentence_once(tmp_path, monkeypatch):
    import json
    tid = "ph_test_unique"
    d = tmp_path / "feedback_ab"
    d.mkdir()
    full = {a: {"rating": "4"} for a in "ACE"}
    for i, (sw, arms, corr) in enumerate([("Habari yako.", full, "x"), ("habari yako", full, "y"),
                                          ("Nyumba", full, None), ("Mti", full, "z")]):
        (d / f"t{i}_{tid}_x.json").write_text(json.dumps({"swahili_input": sw, "arms": arms,
                                                           "native_speaker_correction": corr}))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *a, **k: str(tmp_path))
    testers.record_rated(tid, "Mti", True)          # recorded before loading: load must still happen
    testers.load_rated(tid, "tok", "repo")
    assert testers.unique_complete(tid) == 2        # habari yako (twice) + mti; nyumba not complete
    assert "nyumba" in testers.rated_if_loaded(tid)
    testers.record_rated(tid, "Nyumba.", True)
    assert testers.unique_complete(tid) == 3

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

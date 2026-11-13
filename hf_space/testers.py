"""
testers.py -- paid testers on the Translator tab (2026-11-13, user decision).

The user got funding to pay native-speaker testers. Each tester enters a phone number once (it stays in their
browser); every submission carries only a salted hash of it. User decisions (2026-11-13):
  - storage: the raw number goes ONLY to a separate private dataset (TESTERS_DATASET_REPO, one file per tester:
    number <-> hash). The feedback dataset never contains a number. This is a dated exception to CLAUDE.md
    "never store raw phone numbers"; delete the numbers after the final payment.
  - paid unit: a COMPLETE submission = all three translations rated AND the tester's own Kitaita translation.
    Payment once per 100 complete submissions (scripts/tester_payouts.py). Each Swahili sentence counts once
    per tester.
The hash is the same as the SMS API's (api.util.hash_identifier, prefix "ph", PHONE_HASH_SALT), so one person
is one tester across channels.
"""
from __future__ import annotations

import io
import json
import os
import re
import threading
from datetime import datetime, timezone

TESTERS_DATASET_REPO = os.environ.get("TESTERS_DATASET_REPO", "mpily/bantugrammar-testers")
PAY_EVERY = 100
_registered: set[str] = set()
_lock = threading.Lock()


def normalize_phone(raw: str) -> str | None:
    """Kenyan numbers to +254XXXXXXXXX (07.., 01.., 7.., 254.., +254..); other +<country> numbers kept as
    + and 8-15 digits. None if it does not look like a phone number."""
    d = re.sub(r"[^\d+]", "", raw or "")
    if d.startswith("00"):
        d = "+" + d[2:]
    if re.fullmatch(r"0[17]\d{8}", d):
        return "+254" + d[1:]
    if re.fullmatch(r"[17]\d{8}", d):
        return "+254" + d
    if re.fullmatch(r"254[17]\d{8}", d):
        return "+" + d
    if re.fullmatch(r"\+254[17]\d{8}", d):
        return d
    if re.fullmatch(r"\+[1-9]\d{7,14}", d) and not d.startswith("+254"):
        return d
    return None


def tester_id(phone: str) -> str:
    """Salted hash of a normalized number (raises if PHONE_HASH_SALT is not set)."""
    from api.util import hash_identifier
    return hash_identifier(phone, os.environ.get("PHONE_HASH_SALT", ""), "ph")


def is_complete(ratings: dict, correction: str | None) -> bool:
    return all(ratings.get(k) for k in ("A", "C", "E")) and bool((correction or "").strip())


def register(phone: str, tid: str, token: str) -> None:
    """Write number <-> hash to the private testers dataset (once per tester per process)."""
    with _lock:
        if tid in _registered:
            return
    from huggingface_hub import HfApi
    rec = {"tester_id": tid, "phone": phone, "registered": datetime.now(timezone.utc).isoformat(),
           "consent": "entered on the Translator tab under the payment notice"}
    api = HfApi(token=token)
    api.create_repo(TESTERS_DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)
    path = f"testers/{tid}.json"
    if path not in set(api.list_repo_files(TESTERS_DATASET_REPO, repo_type="dataset")):
        api.upload_file(path_or_fileobj=io.BytesIO(json.dumps(rec, indent=1).encode()), path_in_repo=path,
                        repo_id=TESTERS_DATASET_REPO, repo_type="dataset")
    with _lock:
        _registered.add(tid)


def complete_count(files: list[str], tid: str) -> int:
    """Complete submissions of one tester, from feedback file names (feedback_ab/<ts>_<tid>_complete_<id>.json).
    An upper bound: the payout script also counts each Swahili sentence once."""
    return sum(1 for f in files if f"_{tid}_complete_" in f)

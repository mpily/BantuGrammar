"""Feedback layer: append-only JSONL shards, pushed to a private HF Dataset.

The Space disk is not persistent. Each process writes its OWN shard per
table (`data/<table>/<run_id>.jsonl`) into a local buffer folder that
huggingface_hub.CommitScheduler pushes every few minutes (and once more on
shutdown). On startup every existing shard is downloaded and replayed into
memory. Per-run shards mean a restart can never overwrite earlier data, even
if the startup download failed.

Tables:
  translations   one row per /api/translate call (or SMS/USSD translation)
  feedback       one row per rating/correction
  verified_pairs one row per promotion event (latest per translation wins)
  sessions       SMS/USSD conversation state per hashed phone (latest wins)
  runs           one manifest per process: resolved config + file hashes
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from . import promotion as promo
from .stats import compute_progress
from .util import new_id, normalize_input, parse_iso, utcnow_iso

log = logging.getLogger("api.store")
TABLES = ("translations", "feedback", "verified_pairs", "sessions", "runs", "word_votes")


class Store:
    def __init__(self, local_dir: str | Path, run_id: str | None = None, *,
                 min_distinct_speakers: int = 2, trusted_reviewer_approval: bool = True,
                 low_n_threshold: int = 5, min_rooms_to_verify: int = 2,
                 history_loader: Callable[[], Path | None] | None = None,
                 scheduler_factory: Callable[[Path], object] | None = None):
        self.run_id = run_id or new_id("run")
        self.root = Path(local_dir)
        self.upload_dir = self.root / "upload"
        self.min_distinct_speakers = min_distinct_speakers
        self.trusted_reviewer_approval = trusted_reviewer_approval
        self.low_n_threshold = low_n_threshold
        self.min_rooms_to_verify = min_rooms_to_verify
        self.word_votes: list[dict] = []
        self._lock = threading.RLock()
        self.translations: dict[str, dict] = {}
        self.feedback: list[dict] = []
        self.feedback_by_tr: dict[str, list[dict]] = {}
        self.verified: dict[str, dict] = {}          # translation_id -> pair row
        self.sessions: dict[str, dict] = {}
        self.persistence = "local_only"
        self.history_error: str | None = None

        for t in TABLES:
            (self.upload_dir / "data" / t).mkdir(parents=True, exist_ok=True)

        # replay earlier runs: the downloaded snapshot (if any) plus anything
        # still sitting in the local buffer from this container
        sources = [self.upload_dir / "data"]
        if history_loader is not None:
            try:
                snap = history_loader()
                if snap is not None:
                    sources.append(Path(snap) / "data")  # local copy of a shard wins (newer)
            except Exception as e:  # keep serving; new shards can't clobber old ones
                self.history_error = f"{type(e).__name__}: {e}"
                log.warning("could not load feedback history: %s", self.history_error)
        self._replay(sources)

        self.scheduler = None
        if scheduler_factory is not None:
            try:
                self.scheduler = scheduler_factory(self.upload_dir)
                self.persistence = "hf_dataset"
            except Exception as e:
                self.persistence = f"local_only (dataset sync unavailable: {type(e).__name__})"
                log.warning("CommitScheduler unavailable: %s", e)

    # ------------------------------------------------------------------ io
    def _replay(self, sources: list[Path]) -> None:
        seen: set[tuple[str, str]] = set()
        for src in sources:
            for table in ("translations", "feedback", "verified_pairs", "sessions", "word_votes"):
                for f in sorted((src / table).glob("*.jsonl")) if (src / table).exists() else []:
                    if (table, f.name) in seen:
                        continue  # same shard present both remotely and locally
                    seen.add((table, f.name))
                    for line in f.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                self._apply(table, json.loads(line))
                            except Exception:
                                log.warning("skipping unreadable row in %s", f)
        for rows in self.feedback_by_tr.values():
            rows.sort(key=lambda r: r["created_at"])
        self.feedback.sort(key=lambda r: r["created_at"])

    def _apply(self, table: str, row: dict) -> None:
        if table == "translations":
            self.translations[row["id"]] = row
        elif table == "feedback":
            self.feedback.append(row)
            self.feedback_by_tr.setdefault(row["translation_id"], []).append(row)
        elif table == "verified_pairs":
            prev = self.verified.get(row["translation_id"])
            if prev is None or row["verified_at"] >= prev["verified_at"]:
                self.verified[row["translation_id"]] = row
        elif table == "word_votes":
            self.word_votes.append(row)
        elif table == "sessions":
            prev = self.sessions.get(row["phone_hash"])
            if prev is None or row["updated_at"] >= prev["updated_at"]:
                self.sessions[row["phone_hash"]] = row

    def _append(self, table: str, row: dict) -> None:
        path = self.upload_dir / "data" / table / f"{self.run_id}.jsonl"
        line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        lock = getattr(self.scheduler, "lock", None) or threading.Lock()
        with lock, path.open("a", encoding="utf-8") as fh:
            fh.write(line)

    def write_manifest(self, manifest: dict) -> None:
        with self._lock:
            self._append("runs", {"run_id": self.run_id, "started_at": utcnow_iso(), **manifest})

    def flush(self, timeout: float = 60) -> bool:
        """Push the buffer now (called on shutdown). Returns True on success."""
        if self.scheduler is None:
            return False
        try:
            self.scheduler.trigger().result(timeout=timeout)
            return True
        except Exception as e:
            log.error("final feedback flush failed: %s", e)
            return False

    def stop(self) -> None:
        if self.scheduler is not None and hasattr(self.scheduler, "stop"):
            self.scheduler.stop()

    # ------------------------------------------------------------------ translations
    def add_translation(self, row: dict) -> dict:
        row = {"id": new_id("tr"), "created_at": utcnow_iso(), "run_id": self.run_id, **row}
        with self._lock:
            self._apply("translations", row)
            self._append("translations", row)
        return row

    def get_translation(self, tid: str) -> dict | None:
        return self.translations.get(tid)

    def translation_status(self, tid: str) -> str:
        tr = self.translations.get(tid)
        if tr is None:
            return "unknown"
        if tid in self.verified:
            return "verified"
        if any(fb.get("native_output") for fb in self.feedback_by_tr.get(tid, [])):
            return "speaker_corrected"
        return "machine" if tr.get("kitaita") else "unknown"

    def verified_for_source(self, source: str) -> dict | None:
        key = normalize_input(source)
        best = None
        for row in self.verified.values():
            if normalize_input(row["swahili"]) == key and (best is None or row["verified_at"] > best["verified_at"]):
                best = row
        return best

    # ------------------------------------------------------------------ feedback
    def add_feedback(self, row: dict) -> tuple[dict, bool]:
        """Stores one feedback row and re-evaluates the promotion rule for its
        translation. Returns (row, newly_promoted)."""
        with self._lock:
            tr = self.translations.get(row["translation_id"])
            if tr is None:
                raise KeyError(row["translation_id"])
            row = {"id": new_id("fb"), "created_at": utcnow_iso(), "run_id": self.run_id, **row}
            self._apply("feedback", row)
            self._append("feedback", row)
            p = promo.evaluate(self.feedback_by_tr[tr["id"]], tr.get("kitaita"),
                               bool(tr.get("unknown_words")), self.min_distinct_speakers,
                               self.trusted_reviewer_approval)
            prev = self.verified.get(tr["id"])
            promoted = p.status == "verified" and (prev is None or prev["kitaita"] != p.kitaita)
            if promoted:
                pair = {
                    "id": new_id("vp"), "translation_id": tr["id"], "swahili": tr["swahili_mt"],
                    "kitaita": p.kitaita, "kind": promo.pair_kind(tr["swahili_mt"]),
                    "noun_class": tr.get("noun_class") if promo.pair_kind(tr["swahili_mt"]) == "word" else None,
                    "level": promo.pair_level(tr["swahili_mt"]), "verified_at": utcnow_iso(),
                    "method": p.method, "n_speakers": p.n_speakers,
                }
                self._apply("verified_pairs", pair)
                self._append("verified_pairs", pair)
            return row, promoted

    def verified_pairs(self, since: datetime | None = None, kind: str | None = None) -> list[dict]:
        rows = sorted(self.verified.values(), key=lambda r: r["verified_at"])
        if since is not None:
            rows = [r for r in rows if parse_iso(r["verified_at"]) > since]
        if kind is not None:
            rows = [r for r in rows if r["kind"] == kind]
        return rows

    def progress(self) -> dict:
        with self._lock:
            return compute_progress(dict(self.translations), list(self.feedback), dict(self.verified),
                                    self.low_n_threshold)

    # ------------------------------------------------------------------ sms/ussd sessions
    def get_session(self, phone_hash: str, ttl_minutes: float) -> dict | None:
        s = self.sessions.get(phone_hash)
        if s is None:
            return None
        if datetime.now(timezone.utc) - parse_iso(s["updated_at"]) > timedelta(minutes=ttl_minutes):
            return {**s, "state": "idle"}
        return s

    def set_session(self, phone_hash: str, state: str, last_translation_id: str | None) -> dict:
        row = {"phone_hash": phone_hash, "state": state, "last_translation_id": last_translation_id,
               "updated_at": utcnow_iso()}
        with self._lock:
            self._apply("sessions", row)
            self._append("sessions", row)
        return row

    # ------------------------------------------------------------------ game: player-accepted words
    def add_word_vote(self, row: dict) -> dict:
        """One vote outcome from a game room (kind 'room_vote') or a trusted
        reviewer decision (kind 'reviewer'). Room ids are random per room; no
        nicknames or player ids are stored."""
        row = {"id": new_id("wv"), "created_at": utcnow_iso(), "run_id": self.run_id, **row}
        row["word"] = normalize_input(row["word"]).lower()
        with self._lock:
            self._apply("word_votes", row)
            self._append("word_votes", row)
        return row

    def player_words(self) -> list[dict]:
        """Words players voted to accept. status 'verified' when accepted in
        >= min_rooms_to_verify different rooms or approved by a trusted reviewer;
        otherwise 'player_accepted'. A reviewer rejection removes a word."""
        by_word: dict[str, dict] = {}
        for r in sorted(self.word_votes, key=lambda r: r["created_at"]):
            w = by_word.setdefault(r["word"], {"rooms": set(), "reviewer": None, "first": None, "verified_at": None,
                                               "rejected_rooms": set()})
            if r.get("kind") == "reviewer":
                w["reviewer"] = r["accepted"]
                if r["accepted"] and w["verified_at"] is None:
                    w["verified_at"] = r["created_at"]
            elif r.get("accepted"):
                w["rooms"].add(r["room_id"])
                w["first"] = w["first"] or r["created_at"]
                if len(w["rooms"]) >= self.min_rooms_to_verify and w["verified_at"] is None:
                    w["verified_at"] = r["created_at"]
            else:
                w["rejected_rooms"].add(r["room_id"])
        out = []
        for word, w in sorted(by_word.items()):
            if w["reviewer"] is False or (not w["rooms"] and not w["reviewer"]):
                continue
            verified = bool(w["reviewer"]) or len(w["rooms"]) >= self.min_rooms_to_verify
            out.append({"word": word, "status": "verified" if verified else "player_accepted",
                        "n_rooms": len(w["rooms"]), "n_rooms_rejected": len(w["rejected_rooms"]),
                        "first_accepted_at": w["first"], "verified_at": w["verified_at"] if verified else None})
        return out

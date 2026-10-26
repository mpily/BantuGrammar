"""Resolved configuration: config.yaml + environment secrets."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class Settings:
    api_version: str = "1"
    min_distinct_speakers: int = 2
    trusted_reviewer_approval: bool = True
    dataset_repo: str = "mpily/bantugrammar-feedback"
    commit_every_minutes: float = 5
    local_dir: str = "api_data"
    default_mode: str = "probes"
    lru_cache_size: int = 512
    max_chars: int = 1000
    low_n_threshold: int = 5
    allow_origins: list[str] = field(default_factory=lambda: ["https://mpily.github.io"])
    at_environment: str = "sandbox"
    at_username: str = "sandbox"
    sms_state_ttl_minutes: float = 60
    ussd_budget_seconds: float = 1.5
    game: dict = field(default_factory=dict)  # config.yaml `game` section (api/game.py GameConfig)
    # secrets (from env only; never logged)
    hf_token: str | None = field(default=None, repr=False)
    phone_hash_salt: str | None = field(default=None, repr=False)
    at_api_key: str | None = field(default=None, repr=False)
    at_sender_id: str | None = None
    at_webhook_token: str | None = field(default=None, repr=False)
    trusted_reviewer_keys: frozenset[str] = field(default_factory=frozenset, repr=False)

    SECRET_FIELDS = ("hf_token", "phone_hash_salt", "at_api_key", "at_webhook_token", "trusted_reviewer_keys")

    def public_dict(self) -> dict:
        """Resolved config with secrets replaced by presence booleans (logged
        into every run manifest, CLAUDE.md hard rule #4)."""
        d = asdict(self)
        for k in self.SECRET_FIELDS:
            d[k] = bool(d[k])
        return d

    def config_hash(self) -> str:
        blob = json.dumps(self.public_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def load_settings(path: str | Path | None = None, env: dict | None = None) -> Settings:
    env = os.environ if env is None else env
    path = Path(path or env.get("API_CONFIG") or DEFAULT_CONFIG_PATH)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    raw = raw or {}
    p, s, t = raw.get("promotion", {}), raw.get("storage", {}), raw.get("translate", {})
    at = raw.get("africas_talking", {})
    keys = env.get("TRUSTED_REVIEWER_KEYS", "")
    return Settings(
        api_version=str(raw.get("api_version", "1")),
        min_distinct_speakers=int(p.get("min_distinct_speakers", 2)),
        trusted_reviewer_approval=bool(p.get("trusted_reviewer_approval", True)),
        dataset_repo=env.get("FEEDBACK_DATASET_REPO") or s.get("dataset_repo", "mpily/bantugrammar-feedback"),
        commit_every_minutes=float(s.get("commit_every_minutes", 5)),
        local_dir=env.get("API_DATA_DIR") or s.get("local_dir", "api_data"),
        default_mode=t.get("default_mode", "probes"),
        lru_cache_size=int(t.get("lru_cache_size", 512)),
        max_chars=int(t.get("max_chars", 1000)),
        low_n_threshold=int(raw.get("progress", {}).get("low_n_threshold", 5)),
        allow_origins=list(raw.get("cors", {}).get("allow_origins", ["https://mpily.github.io"])),
        at_environment=env.get("AT_ENVIRONMENT") or at.get("environment", "sandbox"),
        at_username=env.get("AT_USERNAME") or at.get("username", "sandbox"),
        sms_state_ttl_minutes=float(at.get("sms_state_ttl_minutes", 60)),
        ussd_budget_seconds=float(at.get("ussd_budget_seconds", 1.5)),
        game=dict(raw.get("game", {}) or {}),
        hf_token=env.get("HF_TOKEN") or None,
        phone_hash_salt=env.get("PHONE_HASH_SALT") or None,
        at_api_key=env.get("AT_API_KEY") or None,
        at_sender_id=env.get("AT_SENDER_ID") or None,
        at_webhook_token=env.get("AT_WEBHOOK_TOKEN") or None,
        trusted_reviewer_keys=frozenset(k.strip() for k in keys.split(",") if k.strip()),
    )

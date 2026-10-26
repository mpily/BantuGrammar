"""IDs, text normalization, hashing, SMS text helpers."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import time
import unicodedata
from datetime import datetime, timezone

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_id(prefix: str) -> str:
    """ULID-style id: 48-bit ms timestamp + 80 random bits, Crockford base32
    (sortable by creation time), e.g. tr_01H...."""
    n = (int(time.time() * 1000) << 80) | int.from_bytes(os.urandom(10), "big")
    chars = []
    for _ in range(26):
        chars.append(_CROCKFORD[n & 31])
        n >>= 5
    return f"{prefix}_{''.join(reversed(chars))}"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


_QUOTES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "“": '"', "”": '"'})


def normalize_input(text: str) -> str:
    """Cache key for translation input. Case is KEPT: the pipeline uses
    capitalization (proper-name detection), so lowercasing could change its
    output."""
    text = unicodedata.normalize("NFC", text or "").translate(_QUOTES)
    return re.sub(r"\s+", " ", text).strip()


def normalize_kitaita(text: str | None) -> str:
    """Normalization used by the promotion rule and exact-match stats:
    NFC, straight apostrophes, lowercase, collapsed whitespace, no
    sentence-final punctuation."""
    t = normalize_input(text or "").lower()
    return t.rstrip(" .!?;:,")


def hash_identifier(value: str, salt: str, prefix: str) -> str:
    """Salted HMAC-SHA256. Raw value is never stored."""
    if not salt:
        raise RuntimeError("PHONE_HASH_SALT secret is not set; refusing to hash without a salt")
    digest = hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()
    return f"{prefix}_{digest[:24]}"


def normalize_phone(phone: str) -> str:
    digits = re.sub(r"[^\d+]", "", phone or "")
    if digits.startswith("00"):
        digits = "+" + digits[2:]
    return digits


GSM7_BASIC = set(
    "@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?"
    "¡ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà"
)
GSM7_EXT = set("^{}\\[~]|€")  # each costs 2 septets


def to_gsm7(text: str) -> str:
    """Replace common non-GSM-7 characters (curly quotes, dashes, ellipsis)
    so a message isn't forced into UCS-2 (70-char segments)."""
    text = text.translate(_QUOTES).replace("—", "-").replace("–", "-").replace("…", "...")
    return "".join(c if (c in GSM7_BASIC or c in GSM7_EXT) else
                   (unicodedata.normalize("NFKD", c).encode("ascii", "ignore").decode() or "?")
                   for c in text)


def gsm7_length(text: str) -> int:
    return sum(2 if c in GSM7_EXT else 1 for c in text)


def fit_sms(main: str, suffix: str = "", limit: int = 160) -> str:
    """GSM-7-safe `main + suffix`, truncating `main` (with '...') so the whole
    message fits one SMS when possible. The suffix is dropped before `main`
    is cut below 40 chars."""
    main, suffix = to_gsm7(main), to_gsm7(suffix)
    if gsm7_length(main + suffix) <= limit:
        return main + suffix
    budget = limit - gsm7_length(suffix)
    if budget < 40:
        suffix, budget = "", limit
    if gsm7_length(main) > budget:
        out = ""
        for c in main:
            if gsm7_length(out + c) > budget - 3:
                break
            out += c
        main = out.rstrip() + "..."
    return main + suffix

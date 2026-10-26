"""Africa's Talking SMS + USSD conversation logic (sandbox by default).

Phone numbers are hashed (salted HMAC) on arrival and never stored or
logged raw. The raw number is held only in memory long enough to send the
reply SMS.
"""
from __future__ import annotations

import logging
import re
from datetime import date

from .util import fit_sms, normalize_input

log = logging.getLogger("api.at")

SMS_PROMPT = " Reply 1-5 to rate, or send the correct Kitaita"
SMS_MACHINE_TAG = " (machine, unverified)."
FORCE_TRANSLATE_RE = re.compile(r"^(t|tafsiri)\s+", re.IGNORECASE)
USSD_MENU = "CON Kitaita translator\n1 Translate\n2 Daily word\n3 Rate last translation"


class ATClient:
    """Minimal Africa's Talking SMS sender (REST, no SDK dependency)."""

    def __init__(self, username: str, api_key: str | None, environment: str = "sandbox",
                 sender_id: str | None = None, http=None):
        self.username = username
        self.api_key = api_key
        self.sender_id = sender_id
        host = "api.sandbox.africastalking.com" if environment == "sandbox" else "api.africastalking.com"
        self.url = f"https://{host}/version1/messaging"
        self._http = http

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.username)

    def send_sms(self, to: str, message: str) -> bool:
        if not self.configured:
            log.warning("AT_API_KEY not set; SMS not sent")
            return False
        data = {"username": self.username, "to": to, "message": message}
        if self.sender_id:
            data["from"] = self.sender_id
        try:
            if self._http is not None:
                r = self._http.post(self.url, data=data, headers=self._headers())
            else:
                import httpx
                r = httpx.post(self.url, data=data, headers=self._headers(), timeout=10)
            ok = r.status_code < 300
            if not ok:
                log.error("AT SMS send failed: HTTP %s", r.status_code)
            return ok
        except Exception as e:
            log.error("AT SMS send failed: %s", type(e).__name__)
            return False

    def _headers(self) -> dict:
        return {"apiKey": self.api_key or "", "Accept": "application/json"}


def sms_reply_for_translation(tr: dict) -> str:
    if tr.get("kitaita"):
        body = tr.get("kitaita_marked") or tr["kitaita"]
        return fit_sms(body.rstrip("."), SMS_MACHINE_TAG + SMS_PROMPT)
    return fit_sms("Sorry, no Kitaita translation is known for this yet (unknown).", "")


class Conversations:
    """SMS and USSD flows. `translate_fn(text, channel) -> translation row`
    (stored, with id); `cached_fn(text) -> translation row | None` must
    never run the model (USSD's < 2 s budget)."""

    def __init__(self, store, translate_fn, cached_fn, feedback_fn, hash_phone, at: ATClient,
                 ttl_minutes: float = 60):
        self.store = store
        self.translate_fn = translate_fn
        self.cached_fn = cached_fn
        self.feedback_fn = feedback_fn
        self.hash_phone = hash_phone
        self.at = at
        self.ttl = ttl_minutes

    # ------------------------------------------------------------------ SMS
    def handle_sms(self, phone: str, text: str) -> str:
        """Returns the reply text (also sent via AT by the caller)."""
        ph = self.hash_phone(phone)
        msg = normalize_input(text)
        sess = self.store.get_session(ph, self.ttl) or {"state": "idle", "last_translation_id": None}
        last_id = sess.get("last_translation_id")
        awaiting = sess.get("state") == "awaiting_feedback" and last_id

        forced = FORCE_TRANSLATE_RE.match(msg)
        if forced:
            msg = msg[forced.end():]
        elif awaiting and re.fullmatch(r"[1-5]", msg):
            self.feedback_fn(last_id, ph, stars=int(msg), native_output=None, channel="sms")
            self.store.set_session(ph, "idle", last_id)
            return fit_sms("Asante! Rating saved. Send Swahili to translate.")
        elif awaiting and msg:
            last = self.store.get_translation(last_id) or {}
            if normalize_input(last.get("swahili_mt", "")).lower() != msg.lower():
                self.feedback_fn(last_id, ph, stars=None, native_output=msg, channel="sms")
                self.store.set_session(ph, "idle", last_id)
                return fit_sms("Asante! Your Kitaita correction was saved for review. Send Swahili to translate.")

        if not msg:
            return fit_sms("Send a Swahili sentence to get it in Kitaita.")
        tr = self.translate_fn(msg, "sms")
        self.store.set_session(ph, "awaiting_feedback", tr["id"])
        return sms_reply_for_translation(tr)

    # ------------------------------------------------------------------ USSD
    def handle_ussd(self, phone: str, text: str, schedule_background) -> str:
        """`schedule_background(fn, *args)` runs fn after the response is sent."""
        ph = self.hash_phone(phone)
        text = text or ""
        if text == "":
            return USSD_MENU
        choice, _, rest = text.partition("*")

        if choice == "1":
            if not rest.strip():
                return "CON Enter a Swahili sentence:"
            sentence = normalize_input(rest.replace("*", " "))
            hit = self.cached_fn(sentence)
            if hit is not None:
                tr = self.translate_fn(sentence, "ussd")  # cache hit: no model run
                self.store.set_session(ph, "awaiting_feedback", tr["id"])
                body = tr.get("kitaita_marked") or "unknown"
                return f"END {body}\n(machine translation, awaiting native-speaker review)\nDial again, 3 to rate."
            schedule_background(self._translate_and_sms, phone, sentence)
            return ("END Translating... the result will arrive by SMS."
                    if self.at.configured else "END Translating... dial again in a minute (option 1).")

        if choice == "2":
            words = sorted(self.store.verified_pairs(kind="word"), key=lambda r: r["id"])
            if not words:
                return "END No speaker-verified words yet. Check back soon!"
            w = words[date.today().toordinal() % len(words)]
            return f"END Word of the day:\n{w['swahili']} = {w['kitaita']}\n(verified by Kitaita speakers)"

        if choice == "3":
            sess = self.store.get_session(ph, 24 * 60) or {}
            tid = sess.get("last_translation_id")
            tr = self.store.get_translation(tid) if tid else None
            if tr is None:
                return "END No translation to rate yet. Choose 1 to translate."
            if not rest:
                shown = (tr.get("kitaita_marked") or "unknown")[:80]
                return f"CON Rate: {shown}\nEnter 1-5:"
            if re.fullmatch(r"[1-5]", rest.strip()):
                self.feedback_fn(tid, ph, stars=int(rest.strip()), native_output=None, channel="ussd")
                return "END Asante! Rating saved."
            return "END Invalid rating. Please use 1-5."

        return "END Invalid choice."

    def _translate_and_sms(self, phone: str, sentence: str) -> None:
        tr = self.translate_fn(sentence, "ussd")
        self.store.set_session(self.hash_phone(phone), "awaiting_feedback", tr["id"])
        self.at.send_sms(phone, sms_reply_for_translation(tr))

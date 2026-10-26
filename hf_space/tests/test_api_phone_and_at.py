import pytest

from api.util import fit_sms, gsm7_length, hash_identifier, normalize_phone, to_gsm7
from conftest import SALT

PHONE = "+254711000111"


def test_phone_hash_is_salted_deterministic_and_opaque():
    h = hash_identifier(normalize_phone(PHONE), SALT, "ph")
    assert h == hash_identifier(normalize_phone("+254 711-000-111"), SALT, "ph")
    assert h != hash_identifier(normalize_phone(PHONE), "other-salt", "ph")
    assert h.startswith("ph_") and "711000111" not in h


def test_hash_refuses_without_salt():
    with pytest.raises(RuntimeError):
        hash_identifier(PHONE, "", "ph")


def test_at_endpoints_refuse_without_salt(client, settings):
    settings.phone_hash_salt = None
    assert client.post("/at/sms", data={"from": PHONE, "text": "x"}).status_code == 503
    assert client.post("/at/ussd", data={"phoneNumber": PHONE, "text": ""}).status_code == 503


def test_webhook_token(client, settings):
    settings.at_webhook_token = "tok"
    assert client.post("/at/ussd", data={"phoneNumber": PHONE, "text": ""}).status_code == 403
    assert client.post("/at/ussd?token=tok", data={"phoneNumber": PHONE, "text": ""}).status_code == 200


def test_gsm7_helpers():
    assert to_gsm7("Kidaw’ida — ok") == "Kidaw'ida - ok"
    msg = fit_sms("a" * 300, " Reply 1-5")
    assert gsm7_length(msg) <= 160 and msg.endswith(" Reply 1-5")
    assert gsm7_length(fit_sms("short", " tail")) == 10


def test_sms_translate_rate_correct_flow(client, at, store):
    client.post("/at/sms", data={"from": PHONE, "text": "Nyoka aja"})
    to, msg = at.sent[-1]
    assert to == PHONE and msg.startswith("Choka yacha") and "Reply 1-5" in msg and len(msg) <= 160
    assert "machine" in msg
    client.post("/at/sms", data={"from": PHONE, "text": "4"})
    assert "Rating saved" in at.sent[-1][1]
    assert store.feedback[-1]["stars"] == 4 and store.feedback[-1]["channel"] == "sms"
    # next message is a new translation, then a correction
    client.post("/at/sms", data={"from": PHONE, "text": "Nyoka aja"})
    client.post("/at/sms", data={"from": PHONE, "text": "Choka icha"})
    assert "correction was saved" in at.sent[-1][1]
    assert store.feedback[-1]["native_output"] == "Choka icha" and store.feedback[-1]["stars"] is None
    # state lives in the store (persisted), not only in memory
    assert any(s["state"] == "idle" for s in store.sessions.values())


def test_sms_resending_source_is_a_new_translation_not_a_correction(client, at, store):
    client.post("/at/sms", data={"from": PHONE, "text": "Nyoka aja"})
    client.post("/at/sms", data={"from": PHONE, "text": "Nyoka aja"})
    assert store.feedback == [] and len(store.translations) == 2
    client.post("/at/sms", data={"from": PHONE, "text": "T Habari gani"})
    assert store.feedback == [] and "[?Habari]" in at.sent[-1][1]


def test_ussd_menu_and_cached_translation(client, at):
    menu = client.post("/at/ussd", data={"phoneNumber": PHONE, "text": "", "sessionId": "s1"}).text
    assert menu.startswith("CON") and "1 Translate" in menu and "2 Daily word" in menu and "3 Rate" in menu
    assert client.post("/at/ussd", data={"phoneNumber": PHONE, "text": "1"}).text.startswith("CON")
    # not cached yet: answered immediately, result goes by SMS
    r = client.post("/at/ussd", data={"phoneNumber": PHONE, "text": "1*Nyoka aja"}).text
    assert r.startswith("END") and "SMS" in r
    assert at.sent and at.sent[-1][1].startswith("Choka yacha")
    # now cached: answered inline
    r = client.post("/at/ussd", data={"phoneNumber": PHONE, "text": "1*Nyoka aja"}).text
    assert r.startswith("END Choka yacha") and "machine" in r


def test_ussd_rate_last_and_daily_word(client, store):
    assert "No translation" in client.post("/at/ussd", data={"phoneNumber": PHONE, "text": "3"}).text
    assert "No speaker-verified words" in client.post("/at/ussd", data={"phoneNumber": PHONE, "text": "2"}).text
    client.post("/at/sms", data={"from": PHONE, "text": "Nyoka"})
    assert client.post("/at/ussd", data={"phoneNumber": PHONE, "text": "3"}).text.startswith("CON Rate: Choka")
    assert "Invalid" in client.post("/at/ussd", data={"phoneNumber": PHONE, "text": "3*9"}).text
    assert "Asante" in client.post("/at/ussd", data={"phoneNumber": PHONE, "text": "3*5"}).text
    tid = store.feedback[-1]["translation_id"]
    for s in ("a", "b"):
        client.post("/api/feedback", json={"translation_id": tid, "native_output": "Choka", "speaker_id": s})
    word = client.post("/at/ussd", data={"phoneNumber": PHONE, "text": "2"}).text
    assert word.startswith("END") and "Nyoka = Choka" in word and "verified" in word

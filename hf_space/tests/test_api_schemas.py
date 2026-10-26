import pytest
from pydantic import ValidationError

from api.schemas import FeedbackRequest, TranslateRequest


def test_translate_request_defaults_and_validation():
    r = TranslateRequest(text="Nyoka aja")
    assert (r.src, r.tgt, r.mode) == ("swh", "dav", None)
    for bad in [{"text": ""}, {"text": "   "}, {"text": "x", "src": "eng"}, {"text": "x", "tgt": "swh"},
                {"text": "x", "extra": 1}, {"text": "x" * 1001}, {"text": "x", "mode": "steering"}]:
        with pytest.raises(ValidationError):
            TranslateRequest(**bad)


@pytest.mark.parametrize("stars", [0, 6, -1])
def test_feedback_stars_bounds(stars):
    with pytest.raises(ValidationError):
        FeedbackRequest(translation_id="tr_x", stars=stars, speaker_id="s")


def test_feedback_needs_rating_or_correction():
    with pytest.raises(ValidationError):
        FeedbackRequest(translation_id="tr_x", speaker_id="s")
    with pytest.raises(ValidationError):
        FeedbackRequest(translation_id="tr_x", speaker_id="s", native_output="   ")
    assert FeedbackRequest(translation_id="tr_x", speaker_id="s", stars=5).native_output is None
    assert FeedbackRequest(translation_id="tr_x", speaker_id="s", native_output=" Choka ").native_output == "Choka"


def test_feedback_channel_enum():
    with pytest.raises(ValidationError):
        FeedbackRequest(translation_id="tr_x", speaker_id="s", stars=3, channel="email")


def test_translate_response_shape(client):
    r = client.post("/api/translate", json={"text": "Nyoka aja", "src": "swh", "tgt": "dav"})
    assert r.status_code == 200
    body = r.json()
    assert body["api_version"] == "1"
    assert body["id"].startswith("tr_")
    assert body["kitaita"] == "Choka yacha."
    assert body["status"] == "machine"
    assert "awaiting native-speaker review" in body["label"]
    assert body["pipeline_version"] == "test-sha"
    assert isinstance(body["latency_ms"], int)
    assert body["slots"] == [{"slot": "aja", "controller": "Nyoka", "relation": "nsubj", "rewritten_to": "ya",
                              "verified_form": False, "slot_type": "subject_sm", "confidence": "statistical",
                              "token_index": 1, "sentence_index": 0}]


def test_unknown_words_are_marked_not_presented_as_kitaita(client):
    body = client.post("/api/translate", json={"text": "Habari gani"}).json()
    assert body["unknown_words"] == ["Habari", "gani"]
    assert body["kitaita_marked"] == "[?Habari] [?gani]."
    assert body["status"] == "machine"


def test_pipeline_error_returns_unknown_not_500(client):
    body = client.post("/api/translate", json={"text": "BOOM"}).json()
    assert body["kitaita"] is None and body["status"] == "unknown"
    assert any("SERVER ERROR" in f for f in body["flags"])


def test_errors_carry_api_version(client):
    r = client.post("/api/translate", json={"text": ""})
    assert r.status_code == 422 and r.json()["api_version"] == "1"


def test_lru_cache_hits_on_normalized_input(client, calls):
    a = client.post("/api/translate", json={"text": "Nyoka aja"}).json()
    b = client.post("/api/translate", json={"text": "  Nyoka   aja "}).json()
    assert calls == ["Nyoka aja"]
    assert not a["cached"] and b["cached"]
    assert a["id"] != b["id"]  # each call is its own rateable translation


def test_cors_allows_only_app_origin_and_localhost(client):
    def preflight(origin):
        return client.options("/api/translate", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
    assert preflight("https://mpily.github.io").headers.get("access-control-allow-origin") == "https://mpily.github.io"
    assert preflight("http://localhost:5173").headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in preflight("https://evil.example").headers

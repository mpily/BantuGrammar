from api.promotion import evaluate, pair_kind, pair_level


def fb(speaker, out=None, stars=None, approve=False, trusted=False):
    return {"speaker_id": speaker, "native_output": out, "stars": stars, "approve": approve,
            "trusted_reviewer": trusted}


def test_machine_when_no_corrections():
    assert evaluate([fb("a", stars=5)], "Choka yacha.", False).status == "machine"


def test_single_correction_is_speaker_corrected():
    assert evaluate([fb("a", "Choka yacha")], "x", False).status == "speaker_corrected"


def test_two_distinct_speakers_identical_normalized_output_verifies():
    p = evaluate([fb("a", "Choka yacha."), fb("b", "  choka   YACHA ")], "x", False)
    assert p.status == "verified" and p.method == "consensus" and p.n_speakers == 2
    assert p.kitaita == "Choka yacha."  # first submitted surface form


def test_curly_apostrophe_normalizes():
    assert evaluate([fb("a", "Kidaw’ida"), fb("b", "kidaw'ida")], "x", False).status == "verified"


def test_same_speaker_twice_does_not_verify():
    assert evaluate([fb("a", "Choka yacha"), fb("a", "Choka yacha")], "x", False).status == "speaker_corrected"


def test_different_outputs_do_not_verify():
    assert evaluate([fb("a", "Choka yacha"), fb("b", "Choka icha")], "x", False).status == "speaker_corrected"


def test_threshold_is_configurable():
    fbs = [fb("a", "y"), fb("b", "y")]
    assert evaluate(fbs, "x", False, min_distinct_speakers=3).status == "speaker_corrected"
    assert evaluate(fbs + [fb("c", "y")], "x", False, min_distinct_speakers=3).status == "verified"


def test_trusted_reviewer_approval():
    p = evaluate([fb("r", "Choka yacha", approve=True, trusted=True)], "x", False)
    assert p.status == "verified" and p.method == "reviewer" and p.kitaita == "Choka yacha"
    # approving the machine output as-is
    assert evaluate([fb("r", approve=True, trusted=True)], "Choka yacha.", False).kitaita == "Choka yacha."
    # ...but not when the machine output contains unknown (passed-through) words
    assert evaluate([fb("r", approve=True, trusted=True)], "Choka aja.", True).status == "machine"
    # untrusted approve flag is ignored
    assert evaluate([fb("r", "y", approve=True, trusted=False)], "x", False).status == "speaker_corrected"
    # reviewer approval can be switched off in config
    assert evaluate([fb("r", "y", approve=True, trusted=True)], "x", False,
                    trusted_reviewer_approval=False).status == "speaker_corrected"


def test_kind_and_level():
    assert pair_kind("Nyoka.") == "word" and pair_level("Nyoka") == 1
    assert pair_kind("Nyoka aja") == "sentence" and pair_level("Nyoka aja") == 2
    assert pair_level("a b c d e f") == 3


def test_feedback_endpoint_promotes(client):
    tid = client.post("/api/translate", json={"text": "Nyoka aja"}).json()["id"]
    r1 = client.post("/api/feedback", json={"translation_id": tid, "stars": 2, "native_output": "Choka icha",
                                            "speaker_id": "spk_a"}).json()
    assert r1["translation_status"] == "speaker_corrected" and not r1["promoted"]
    r2 = client.post("/api/feedback", json={"translation_id": tid, "stars": 3, "native_output": "choka icha.",
                                            "speaker_id": "spk_b"}).json()
    assert r2["translation_status"] == "verified" and r2["promoted"]


def test_feedback_unknown_translation_404(client):
    r = client.post("/api/feedback", json={"translation_id": "tr_nope", "stars": 3, "speaker_id": "s"})
    assert r.status_code == 404 and r.json()["api_version"] == "1"


def test_approve_requires_reviewer_key(client):
    tid = client.post("/api/translate", json={"text": "Nyoka aja"}).json()["id"]
    body = {"translation_id": tid, "native_output": "Choka icha", "speaker_id": "rev", "approve": True}
    assert client.post("/api/feedback", json=body).status_code == 403
    assert client.post("/api/feedback", json=body, headers={"X-Reviewer-Key": "wrong"}).status_code == 403
    r = client.post("/api/feedback", json=body, headers={"X-Reviewer-Key": "rev-key"}).json()
    assert r["translation_status"] == "verified"


def test_speaker_name_only_with_consent(client, store):
    tid = client.post("/api/translate", json={"text": "Nyoka aja"}).json()["id"]
    client.post("/api/feedback", json={"translation_id": tid, "stars": 4, "speaker_id": "a", "speaker_name": "Mama A"})
    client.post("/api/feedback", json={"translation_id": tid, "stars": 4, "speaker_id": "b", "speaker_name": "Baba B",
                                       "name_consent": True})
    names = [f["speaker_name"] for f in store.feedback]
    assert names == [None, "Baba B"]
    assert all(f["speaker_id"].startswith("spk_") and f["speaker_id"] not in ("a", "b") for f in store.feedback)

import json

ALLOWED = {"id", "swahili", "kitaita", "kind", "noun_class", "level", "verified_at", "status"}


def _verify(client, text, form, speakers=("a", "b")):
    tid = client.post("/api/translate", json={"text": text}).json()["id"]
    for s in speakers:
        client.post("/api/feedback", json={"translation_id": tid, "stars": 4, "native_output": form,
                                           "speaker_id": s, "comment": "SECRET COMMENT",
                                           "speaker_name": "Named Person", "name_consent": True})
    return tid


def test_only_verified_rows_and_no_speaker_data(client):
    _verify(client, "Nyoka aja", "Choka icha")                 # verified
    _verify(client, "Habari gani", "Something", ("a",))        # speaker_corrected only
    client.post("/api/translate", json={"text": "Mtu"})        # machine only
    r = client.get("/api/verified_pairs")
    assert r.status_code == 200
    pairs = r.json()["pairs"]
    assert len(pairs) == 1
    p = pairs[0]
    assert set(p) == ALLOWED and p["status"] == "verified"
    assert p["swahili"] == "Nyoka aja" and p["kitaita"] == "Choka icha" and p["kind"] == "sentence"
    raw = r.text
    for leak in ("SECRET COMMENT", "Named Person", "spk_", "ph_", "speaker", "translation_id", "n_speakers"):
        assert leak not in raw, leak


def test_kind_filter_and_noun_class(client):
    _verify(client, "Nyoka aja", "Choka icha")
    _verify(client, "Nyoka", "Choka")
    words = client.get("/api/verified_pairs?kind=word").json()["pairs"]
    assert [w["kitaita"] for w in words] == ["Choka"]
    assert words[0]["noun_class"] == "ANIMATE-SG" and words[0]["level"] == 1
    assert len(client.get("/api/verified_pairs?kind=sentence").json()["pairs"]) == 1
    assert client.get("/api/verified_pairs?kind=phrase").status_code == 422


def test_since_filter(client):
    _verify(client, "Nyoka aja", "Choka icha")
    assert len(client.get("/api/verified_pairs?since=2000-01-01T00:00:00Z").json()["pairs"]) == 1
    assert client.get("/api/verified_pairs?since=2999-01-01T00:00:00Z").json()["pairs"] == []
    assert client.get("/api/verified_pairs?since=yesterday").status_code == 422


def test_verified_alternative_on_translate(client):
    _verify(client, "Nyoka aja", "Choka icha")
    body = client.post("/api/translate", json={"text": "Nyoka aja"}).json()
    assert body["status"] == "machine"   # the machine output is still labelled machine
    assert body["verified_alternative"]["kitaita"] == "Choka icha"
    assert body["verified_alternative"]["status"] == "verified"


def test_persisted_shards_contain_no_raw_phone(client, store):
    client.post("/at/sms", data={"from": "+254711000111", "text": "Nyoka aja"})
    client.post("/at/sms", data={"from": "+254711000111", "text": "4"})
    blob = "".join(p.read_text() for p in store.upload_dir.rglob("*.jsonl"))
    assert "254711000111" not in blob
    assert "ph_" in blob


def test_store_replays_shards_after_restart(client, store, settings):
    from api.store import Store
    _verify(client, "Nyoka aja", "Choka icha")
    again = Store(settings.local_dir)
    assert len(again.translations) == len(store.translations)
    assert len(again.feedback) == 2 and len(again.verified) == 1
    # new process writes to its own shard
    assert again.run_id != store.run_id
    json.dumps(again.progress())


def test_buffer_flushed_on_shutdown(settings, translator, at, tmp_path):
    import threading
    from concurrent.futures import Future
    from fastapi.testclient import TestClient
    from api.server import create_app
    from api.store import Store

    class FakeScheduler:
        def __init__(self, folder):
            self.lock, self.triggered, self.stopped = threading.Lock(), 0, False

        def trigger(self):
            self.triggered += 1
            f = Future()
            f.set_result(None)
            return f

        def stop(self):
            self.stopped = True

    s = Store(tmp_path / "d", scheduler_factory=FakeScheduler)
    assert s.persistence == "hf_dataset"
    with TestClient(create_app(settings, s, translator, at_client=at)) as c:
        c.post("/api/translate", json={"text": "Nyoka aja"})
        assert s.scheduler.triggered == 0
    assert s.scheduler.triggered == 1 and s.scheduler.stopped


def test_history_load_failure_keeps_serving(tmp_path):
    from api.store import Store

    def broken():
        raise ConnectionError("hub down")
    s = Store(tmp_path / "d", history_loader=broken)
    assert s.history_error.startswith("ConnectionError")

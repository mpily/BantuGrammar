"""Shared fixtures: an API app wired to a FAKE pipeline (no model, no
network), a temp store, and a recording Africa's Talking client."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.africastalking import ATClient  # noqa: E402
from api.server import create_app  # noqa: E402
from api.settings import Settings  # noqa: E402
from api.store import Store  # noqa: E402
from api.translator import Translator  # noqa: E402

SALT = "test-salt"


def fake_slot(token_idx, slot_type, surface, target_idx, form, confidence="hypothesis"):
    return SimpleNamespace(token_idx=token_idx, slot_type=slot_type, swahili_surface=surface,
                           target_idx=target_idx, needs_rewrite=True, predicted_form=form,
                           confidence=confidence)


def fake_pipeline(text):
    """Mimics the shape of translate_sentence_with_probes()'s report."""
    if text.startswith("Nyoka aja"):
        d = SimpleNamespace(tokens=["Nyoka", "aja"], subject_class="ANIMATE-SG", passthrough=[],
                            secondary_nouns=[], flags=[],
                            slots=[fake_slot(1, "subject_sm", "aja", 0, "ya", "statistical")])
        return {"kitaita_sentence": "Choka yacha.", "decomposition": d, "flags": [], "error": None}
    if text.startswith("Nyoka"):  # single word
        d = SimpleNamespace(tokens=["Nyoka"], subject_class="ANIMATE-SG", passthrough=[],
                            secondary_nouns=[], flags=[], slots=[])
        return {"kitaita_sentence": "Choka.", "decomposition": d, "flags": [], "error": None}
    if text.startswith("BOOM"):
        raise RuntimeError("pipeline exploded")
    toks = text.rstrip(".").split()
    d = SimpleNamespace(tokens=toks, subject_class=None, passthrough=[(i, t) for i, t in enumerate(toks)],
                        secondary_nouns=[], slots=[],
                        flags=[f"No Kitaita translation/mechanism for: {', '.join(toks)} -- kept"])
    return {"kitaita_sentence": text if text.endswith(".") else text + ".", "decomposition": d,
            "flags": d.flags, "error": None}


class RecordingAT(ATClient):
    def __init__(self):
        super().__init__("sandbox", "fake-key", "sandbox")
        self.sent = []

    def send_sms(self, to, message):
        self.sent.append((to, message))
        return True


@pytest.fixture
def settings(tmp_path):
    return Settings(local_dir=str(tmp_path / "api_data"), phone_hash_salt=SALT,
                    trusted_reviewer_keys=frozenset({"rev-key"}), default_mode="probes")


@pytest.fixture
def calls():
    return []


@pytest.fixture
def translator(calls):
    def probes(text):
        calls.append(text)
        return fake_pipeline(text)
    return Translator({"probes": probes, "heuristic": fake_pipeline}, cache_size=8,
                      pipeline_version="test-sha")


@pytest.fixture
def store(settings):
    return Store(settings.local_dir, min_distinct_speakers=2, low_n_threshold=5)


@pytest.fixture
def at():
    return RecordingAT()


@pytest.fixture
def client(settings, store, translator, at):
    from fastapi.testclient import TestClient
    app = create_app(settings, store, translator, at_client=at)
    with TestClient(app) as c:
        yield c

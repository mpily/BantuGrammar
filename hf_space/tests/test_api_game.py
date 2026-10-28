"""Multiplayer word race: rack logic, scoring, votes, player-accepted words,
and the WebSocket protocol end to end (real kitaita_oracle when installed)."""
import random
import time

import pytest

oracle_mod = pytest.importorskip("kitaita_oracle")

from api.game import GameConfig, GameManager, fits_rack, letters_of  # noqa: E402
from api.server import create_app  # noqa: E402
from api.store import Store  # noqa: E402
from kitaita_oracle.tables import load_yaml  # noqa: E402


def test_rack_helpers():
    assert letters_of("Ng’ombe") == list("ngombe")
    assert fits_rack("mundu", list("mundux")) == (True, [])
    assert fits_rack("mundu", list("mund?")) == (True, [4])
    assert fits_rack("mundu", list("mund"))[0] is False


@pytest.fixture
def game(tmp_path):
    store = Store(tmp_path / "d", min_rooms_to_verify=2)
    cfg = GameConfig(min_round_seconds=1, reveal_seconds=1, vote_seconds=2)
    return GameManager(oracle_mod.Oracle(), load_yaml("kitaita_tiles.json"), store, cfg, rng=random.Random(0))


@pytest.fixture
def gclient(settings, translator, at, game):
    from fastapi.testclient import TestClient
    app = create_app(settings, game.store, translator, at_client=at, game=game)
    with TestClient(app) as c:
        yield c


def test_draw_rack_has_vowels(game):
    for _ in range(50):
        rack = game.draw_rack()
        assert len(rack) == 10 and sum(l in "aeiou" for l in rack) >= 4


def recv_until(ws, kind, limit=30):
    for _ in range(limit):
        m = ws.receive_json()
        if m["type"] == kind:
            return m
    raise AssertionError(f"no {kind}")


def test_room_lifecycle_word_and_vote(gclient, game):
    code = gclient.post("/api/game/rooms").json()["code"]
    assert gclient.get(f"/api/game/rooms/{code}").json()["exists"] is True
    assert gclient.get("/api/game/rooms/ZZZZZ").json()["exists"] is False
    with gclient.websocket_connect(f"/ws/game/{code}") as a, gclient.websocket_connect(f"/ws/game/{code}") as b:
        a.send_json({"type": "join", "name": "Amina"})
        wa = recv_until(a, "welcome")
        b.send_json({"type": "join", "name": "Amina"})
        wb = recv_until(b, "welcome")
        assert wb["name"] != wa["name"]  # names made unique
        b.send_json({"type": "start"})
        assert "host" in recv_until(b, "error")["message"]  # only the host starts
        a.send_json({"type": "start", "rounds": 1, "seconds": 30})
        st = recv_until(a, "state")
        while st["phase"] != "round":
            st = recv_until(a, "state")
        room = game.get(code)
        room.rack = list("wawemanyaz")
        # valid word from the rack
        a.send_json({"type": "word", "word": "wawemanya"})
        r = recv_until(a, "word_result")
        assert r["valid"] and r["points"] > 0 and r["unverified"] in (True, False)
        # not in rack
        a.send_json({"type": "word", "word": "mundu"})
        assert recv_until(a, "word_result")["reason"] == "not_in_rack"
        # rejected by the oracle -> vote
        a.send_json({"type": "word", "word": "zawa"})
        r = recv_until(a, "word_result")
        assert r["valid"] is False and r["votable"] is True
        a.send_json({"type": "challenge", "word": "zawa"})
        req = recv_until(b, "vote_request")
        assert req["word"] == "zawa" and req["by"] == wa["name"]
        b.send_json({"type": "vote", "vote_id": req["vote_id"], "accept": True})
        res = recv_until(a, "vote_result")
        assert res["accepted"] and res["yes"] == 1 and res["points"] > 0
    # stored without nicknames, and the oracle now knows the word
    row = game.store.word_votes[-1]
    assert row["word"] == "zawa" and row["accepted"] and "Amina" not in str(row)
    pw = gclient.get("/api/player_words").json()["words"]
    assert pw == [{"word": "zawa", "status": "player_accepted", "n_rooms": 1, "n_rooms_rejected": 0,
                   "first_accepted_at": pw[0]["first_accepted_at"], "verified_at": None,
                   "translation": None, "word_type": None, "noun_class": None}]
    a2 = game.oracle.analyse("zawa")
    assert a2["valid"] and a2["parses"][-1]["word_type"] == "player_accepted"


def test_vote_rejects_on_tie_or_silence(game):
    import asyncio
    room = game.create_room()

    class WS:
        async def send_json(self, m):
            pass

    a, b = room.add_player("a", WS()), room.add_player("b", WS())

    async def run():
        room.phase, room.rack = "round", list("zawaqqqqqq")
        await room.challenge(a.id, "zawa")
        v = next(iter(room.votes.values()))
        await room.close_vote(v)  # nobody voted
        assert game.store.word_votes[-1]["accepted"] is False
        await room.challenge(a.id, "zawa")
        v2 = [x for x in room.votes.values() if not x.done][0]
        await room.cast_vote(b.id, v2.id, False)
        assert v2.done and game.store.word_votes[-1]["accepted"] is False
    asyncio.run(run())
    assert game.store.player_words() == []


def test_promotion_two_rooms_and_reviewer(game, gclient, settings):
    s = game.store
    s.add_word_vote({"word": "kuvava", "room_id": "r1", "accepted": True, "yes": 2, "no": 0, "kind": "room_vote"})
    assert s.player_words()[0]["status"] == "player_accepted"
    s.add_word_vote({"word": "kuvava", "room_id": "r1", "accepted": True, "yes": 2, "no": 0, "kind": "room_vote"})
    assert s.player_words()[0]["status"] == "player_accepted"  # same room twice doesn't count
    s.add_word_vote({"word": "KuVava", "room_id": "r2", "accepted": True, "yes": 1, "no": 0, "kind": "room_vote"})
    assert s.player_words()[0]["status"] == "verified"
    # reviewer
    body = {"word": "kuvava", "approve": False}
    assert gclient.post("/api/player_words/review", json=body).status_code == 403
    r = gclient.post("/api/player_words/review", json=body, headers={"X-Reviewer-Key": "rev-key"})
    assert r.status_code == 200 and r.json()["words"] == []
    assert gclient.get("/api/player_words").json()["words"] == []


def test_round_ends_and_reveals_words(gclient, game):
    code = gclient.post("/api/game/rooms").json()["code"]
    with gclient.websocket_connect(f"/ws/game/{code}") as a:
        a.send_json({"type": "join", "name": "Solo"})
        recv_until(a, "welcome")
        a.send_json({"type": "start", "rounds": 1, "seconds": 1})
        st = recv_until(a, "state")
        while st["phase"] != "final":
            st = recv_until(a, "state", limit=50)
        assert "words" in st["players"][0]


def test_websocket_rejects_foreign_origin(gclient):
    from starlette.websockets import WebSocketDisconnect
    code = gclient.post("/api/game/rooms").json()["code"]
    with pytest.raises(WebSocketDisconnect):
        with gclient.websocket_connect(f"/ws/game/{code}", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()


def test_game_disabled_without_oracle(client):
    assert client.post("/api/game/rooms").status_code == 503


def test_review_queue_and_review_with_translation_and_class(game, gclient):
    s = game.store
    s.add_word_vote({"word": "kifumbi", "room_id": "r1", "accepted": True, "yes": 2, "no": 1, "kind": "room_vote"})
    s.add_word_vote({"word": "zawa", "room_id": "r1", "accepted": False, "yes": 0, "no": 2, "kind": "room_vote"})
    key = {"X-Reviewer-Key": "rev-key"}
    assert gclient.get("/api/player_words/review_queue").status_code == 403
    q = gclient.get("/api/player_words/review_queue", headers=key).json()
    assert {w["word"]: w["status"] for w in q["words"]} == {"kifumbi": "player_accepted", "zawa": "rejected_by_rooms"}
    assert "chi" in q["noun_classes"]
    kif = next(w for w in q["words"] if w["word"] == "kifumbi")
    assert (kif["votes_yes"], kif["votes_no"], kif["review"]) == (2, 1, None)
    # rejected-by-rooms words are not in the public feed
    assert [w["word"] for w in gclient.get("/api/player_words").json()["words"]] == ["kifumbi"]

    # approve with translation + noun class
    body = {"word": "kifumbi", "approve": True, "translation": "kiti", "word_type": "noun", "noun_class": "chi"}
    r = gclient.post("/api/player_words/review", json=body, headers=key)
    assert r.status_code == 200
    w = r.json()["words"][0]
    assert (w["status"], w["translation"], w["word_type"], w["noun_class"]) == ("verified", "kiti", "noun", "chi")
    # a reviewer can rescue a word the room voted down
    r = gclient.post("/api/player_words/review", json={"word": "zawa", "approve": True, "word_type": "verb"}, headers=key)
    assert r.json()["words"][0]["status"] == "verified"
    assert game.oracle.analyse("zawa")["valid"]
    # the queue now shows the reviews, unreviewed first
    q = gclient.get("/api/player_words/review_queue", headers=key).json()["words"]
    assert all(w["review"] for w in q) and q[0]["review"]["approved"]
    # validation
    bad = [{"word": "x", "approve": True, "noun_class": "chi", "word_type": "verb"},
           {"word": "x", "approve": True, "word_type": "noun", "noun_class": "NOPE!"},
           {"word": "x", "approve": True, "word_type": "pronoun"}]
    for b in bad:
        assert gclient.post("/api/player_words/review", json=b, headers=key).status_code == 422
    r = gclient.post("/api/player_words/review", json={"word": "x", "approve": True, "word_type": "noun",
                                                      "noun_class": "zz"}, headers=key)
    assert r.status_code == 422 and "unknown noun class" in r.json()["error"]
    # declining removes it from the feed and the checker
    gclient.post("/api/player_words/review", json={"word": "zawa", "approve": False}, headers=key)
    assert "zawa" not in [w["word"] for w in gclient.get("/api/player_words").json()["words"]]
    assert game.oracle.analyse("zawa")["valid"] is False

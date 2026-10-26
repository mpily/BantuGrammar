"""Multiplayer Kitaita word race (skribbl-style rooms, no accounts).

A host creates a room and shares its link; players join with a nickname.
Each round everyone gets the same letter tiles and a time limit to make
Kitaita words. Words are checked by the marker-oracle (kitaita_oracle,
installed from mpily/bantugrammar-app) and scored with its tile values.
A word the oracle rejects can be put to the room: if most of the other
players vote to accept it, it scores, and it is recorded as a
player-accepted word (unverified) that the oracle then accepts for
everyone. Rooms live in memory only (like skribbl.io); votes are stored
durably through the feedback Store. Nicknames are never stored.

Protocol (JSON over WebSocket /ws/game/{code}): see api/CONTRACT.md.
"""
from __future__ import annotations

import asyncio
import logging
import random
import secrets
import string
import time
import unicodedata
from dataclasses import dataclass, field

log = logging.getLogger("api.game")

VOWELS = set("aeiou")
APOSTROPHES = "'’‘ʼ`"
CODE_ALPHABET = "".join(c for c in string.ascii_uppercase if c not in "IO")


def letters_of(word: str) -> list[str]:
    w = unicodedata.normalize("NFC", word or "").lower()
    return [c for c in w if c not in APOSTROPHES and not c.isspace()]


def fits_rack(word: str, rack: list[str]) -> tuple[bool, list[int]]:
    """Can `word` be spelled from the rack (each tile once, '?' = blank)?
    Returns (ok, indexes of letters played with a blank)."""
    pool = list(rack)
    blanks = []
    for i, ch in enumerate(letters_of(word)):
        if ch in pool:
            pool.remove(ch)
        elif "?" in pool:
            pool.remove("?")
            blanks.append(i)
        else:
            return False, []
    return True, blanks


@dataclass
class GameConfig:
    max_players: int = 10
    max_rooms: int = 200
    default_rounds: int = 3
    default_round_seconds: int = 90
    vote_seconds: int = 20
    rack_size: int = 10
    min_vowels: int = 4
    min_word_letters: int = 2
    long_word_letters: int = 7
    long_word_bonus: int = 10
    reveal_seconds: int = 8
    min_round_seconds: int = 20
    empty_room_ttl_seconds: int = 600


@dataclass
class Player:
    id: str
    name: str
    ws: object = None
    score: int = 0
    connected: bool = True


@dataclass
class Vote:
    id: str
    word: str
    by: str
    eligible: set
    ends_at: float
    yes: set = field(default_factory=set)
    no: set = field(default_factory=set)
    done: bool = False
    task: object = None


class Room:
    def __init__(self, code: str, manager: "GameManager"):
        self.code = code
        self.id = "room_" + secrets.token_hex(8)  # stored with votes; codes get reused
        self.m = manager
        self.players: dict[str, Player] = {}
        self.host: str | None = None
        self.phase = "lobby"  # lobby | round | reveal | final
        self.rounds = manager.cfg.default_rounds
        self.seconds = manager.cfg.default_round_seconds
        self.round_no = 0
        self.rack: list[str] = []
        self.ends_at = 0.0
        self.round_words: dict[str, list[dict]] = {}
        self.votes: dict[str, Vote] = {}
        self.timer = None
        self.empty_since: float | None = None

    # ------------------------------------------------------------------ state
    def public_state(self) -> dict:
        reveal = self.phase in ("reveal", "final")
        return {
            "type": "state", "room": self.code, "phase": self.phase, "host": self.host,
            "round": self.round_no, "rounds": self.rounds, "seconds": self.seconds,
            "rack": [{"letter": l, "score": self.m.tile_score(l)} for l in self.rack],
            "ends_at": self.ends_at,
            "players": [{"id": p.id, "name": p.name, "score": p.score, "connected": p.connected,
                         "n_words": len(self.round_words.get(p.id, [])),
                         **({"words": self.round_words.get(p.id, [])} if reveal else {})}
                        for p in self.players.values()],
        }

    async def send(self, player: Player, msg: dict) -> None:
        if player.ws is None or not player.connected:
            return
        try:
            await player.ws.send_json(msg)
        except Exception:
            player.connected = False

    async def broadcast(self, msg: dict) -> None:
        for p in list(self.players.values()):
            await self.send(p, msg)

    async def broadcast_state(self) -> None:
        await self.broadcast(self.public_state())

    # ------------------------------------------------------------------ players
    def add_player(self, name: str, ws) -> Player:
        name = " ".join((name or "").split())[:16] or "Player"
        taken = {p.name for p in self.players.values()}
        base, n = name, 2
        while name in taken:
            name = f"{base[:13]} {n}"
            n += 1
        p = Player(id="p_" + secrets.token_hex(4), name=name, ws=ws)
        self.players[p.id] = p
        if self.host is None or not self.players[self.host].connected:
            self.host = p.id
        self.empty_since = None
        return p

    async def drop_player(self, pid: str) -> None:
        p = self.players.get(pid)
        if p is None:
            return
        p.connected = False
        p.ws = None
        if self.phase == "lobby":
            del self.players[pid]
        connected = [q for q in self.players.values() if q.connected]
        if self.host == pid:
            self.host = connected[0].id if connected else None
        if not connected:
            self.empty_since = time.time()
        for v in list(self.votes.values()):
            v.eligible.discard(pid)
            await self.maybe_close_vote(v)
        await self.broadcast_state()

    # ------------------------------------------------------------------ rounds
    async def start(self, pid: str, rounds=None, seconds=None) -> None:
        if pid != self.host:
            raise GameError("only the host can start")
        if self.phase not in ("lobby", "final"):
            raise GameError("a game is already running")
        self.rounds = max(1, min(int(rounds or self.rounds), 10))
        self.seconds = max(self.m.cfg.min_round_seconds, min(int(seconds or self.seconds), 300))
        for p in self.players.values():
            p.score = 0
        self.round_no = 0
        await self.next_round()

    async def next_round(self) -> None:
        self.round_no += 1
        self.rack = self.m.draw_rack()
        self.round_words = {pid: [] for pid in self.players}
        self.phase = "round"
        self.ends_at = time.time() + self.seconds
        await self.broadcast_state()
        self.timer = asyncio.ensure_future(self._end_round_after(self.seconds))

    async def _end_round_after(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        self.phase = "reveal" if self.round_no < self.rounds else "final"
        self.ends_at = time.time() + (self.m.cfg.reveal_seconds if self.phase == "reveal" else 0)
        await self.broadcast_state()
        if self.phase == "reveal":
            await asyncio.sleep(self.m.cfg.reveal_seconds)
            if self.phase == "reveal":
                await self.next_round()

    # ------------------------------------------------------------------ words
    def _score(self, word: str, blanks: list[int]) -> int:
        ls = letters_of(word)
        pts = sum(0 if i in blanks else self.m.tile_score(c) for i, c in enumerate(ls))
        if len(ls) >= self.m.cfg.long_word_letters:
            pts += self.m.cfg.long_word_bonus
        return pts

    async def submit_word(self, pid: str, word: str) -> dict:
        if self.phase != "round":
            raise GameError("no round is running")
        w = " ".join((word or "").split()).lower()
        ls = letters_of(w)
        if len(ls) < self.m.cfg.min_word_letters or " " in w:
            return {"type": "word_result", "word": w, "valid": False, "points": 0, "reason": "too_short_or_spaces",
                    "votable": False}
        if any(x["word"] == w for x in self.round_words.get(pid, [])):
            return {"type": "word_result", "word": w, "valid": False, "points": 0, "reason": "already_played",
                    "votable": False}
        ok, blanks = fits_rack(w, self.rack)
        if not ok:
            return {"type": "word_result", "word": w, "valid": False, "points": 0, "reason": "not_in_rack",
                    "votable": False}
        result = self.m.oracle.analyse(w)
        if result["valid"]:
            pts = self._score(w, blanks)
            unverified = all(p.get("uses_unverified_rules") for p in result["parses"])
            self._award(pid, w, pts, "oracle", unverified)
            await self.broadcast_state()
            return {"type": "word_result", "word": w, "valid": True, "points": pts, "reason": None,
                    "unverified": unverified, "word_types": sorted({p["word_type"] for p in result["parses"]}),
                    "votable": False}
        return {"type": "word_result", "word": w, "valid": False, "points": 0, "reason": result["reason"],
                "votable": True}

    def _award(self, pid: str, word: str, pts: int, how: str, unverified: bool) -> None:
        self.round_words.setdefault(pid, []).append({"word": word, "points": pts, "how": how,
                                                     "unverified": unverified})
        self.players[pid].score += pts

    # ------------------------------------------------------------------ votes
    async def challenge(self, pid: str, word: str) -> None:
        if self.phase != "round":
            raise GameError("votes can only be called during a round")
        w = " ".join((word or "").split()).lower()
        ok, _ = fits_rack(w, self.rack)
        if not ok or self.m.oracle.analyse(w)["valid"]:
            raise GameError("only a word made from the rack that the checker rejected can be put to a vote")
        if any(v.word == w and v.by == pid and not v.done for v in self.votes.values()):
            raise GameError("this word is already being voted on")
        if any(x["word"] == w for x in self.round_words.get(pid, [])):
            raise GameError("you already scored this word")
        eligible = {q.id for q in self.players.values() if q.connected and q.id != pid}
        if not eligible:
            raise GameError("a vote needs at least one other player")
        v = Vote(id="v_" + secrets.token_hex(4), word=w, by=pid, eligible=eligible,
                 ends_at=time.time() + self.m.cfg.vote_seconds)
        self.votes[v.id] = v
        msg = {"type": "vote_request", "vote_id": v.id, "word": w, "by": self.players[pid].name,
               "ends_at": v.ends_at}
        for q in self.players.values():
            if q.id in eligible:
                await self.send(q, msg)
        await self.send(self.players[pid], {"type": "vote_pending", "vote_id": v.id, "word": w,
                                            "n_voters": len(eligible), "ends_at": v.ends_at})
        v.task = asyncio.ensure_future(self._close_vote_after(v))

    async def cast_vote(self, pid: str, vote_id: str, accept: bool) -> None:
        v = self.votes.get(vote_id)
        if v is None or v.done or pid not in v.eligible:
            raise GameError("you can't vote on this")
        (v.yes if accept else v.no).add(pid)
        (v.no if accept else v.yes).discard(pid)
        await self.maybe_close_vote(v)

    async def maybe_close_vote(self, v: Vote) -> None:
        if not v.done and v.eligible and v.eligible <= (v.yes | v.no):
            await self.close_vote(v)
        elif not v.done and not v.eligible:
            await self.close_vote(v)

    async def _close_vote_after(self, v: Vote) -> None:
        await asyncio.sleep(max(0.0, v.ends_at - time.time()))
        await self.close_vote(v)

    async def close_vote(self, v: Vote) -> None:
        if v.done:
            return
        v.done = True
        yes, no = len(v.yes), len(v.no)
        accepted = yes > no and yes >= 1  # majority of the votes cast; ties and silence reject
        pts = 0
        if accepted and v.by in self.players:
            ok, blanks = fits_rack(v.word, self.rack)
            pts = self._score(v.word, blanks if ok else [])
            self._award(v.by, v.word, pts, "vote", True)
        self.m.record_vote(self, v.word, accepted, yes, no, len(v.eligible))
        await self.broadcast({"type": "vote_result", "vote_id": v.id, "word": v.word, "accepted": accepted,
                              "yes": yes, "no": no, "by": self.players[v.by].name if v.by in self.players else None,
                              "points": pts})
        await self.broadcast_state()


class GameError(Exception):
    pass


class GameManager:
    def __init__(self, oracle, tiles: dict, store=None, cfg: GameConfig | None = None,
                 rng: random.Random | None = None):
        self.oracle = oracle
        self.tiles = tiles
        self.store = store
        self.cfg = cfg or GameConfig()
        self.rng = rng or random.Random()
        self.rooms: dict[str, Room] = {}
        self._scores = {t["grapheme"]: t["score"] for t in tiles["tiles"] if t.get("in_bag")}
        self.bag = [t["grapheme"] for t in tiles["tiles"] for _ in range(t["bag"])] + ["?"] * tiles["blank"]["bag"]
        self.refresh_player_words()

    def tile_score(self, letter: str) -> int:
        return 0 if letter == "?" else int(self._scores.get(letter, 0) or 0)

    def draw_rack(self) -> list[str]:
        for _ in range(200):
            rack = self.rng.sample(self.bag, self.cfg.rack_size)
            if sum(1 for l in rack if l in VOWELS) >= self.cfg.min_vowels:
                return rack
        return rack

    def refresh_player_words(self) -> None:
        if self.store is not None and hasattr(self.oracle, "set_player_words"):
            self.oracle.set_player_words(self.store.player_words())

    def record_vote(self, room: Room, word: str, accepted: bool, yes: int, no: int, n_eligible: int) -> None:
        if self.store is None:
            return
        try:
            self.store.add_word_vote({"word": word, "room_id": room.id, "accepted": accepted, "yes": yes, "no": no,
                                      "n_eligible": n_eligible, "kind": "room_vote"})
            if accepted:
                self.refresh_player_words()
        except Exception as e:
            log.error("could not record vote: %s", type(e).__name__)

    def create_room(self) -> Room:
        self.gc()
        if len(self.rooms) >= self.cfg.max_rooms:
            raise GameError("too many rooms right now, try again later")
        for _ in range(50):
            code = "".join(self.rng.choice(CODE_ALPHABET) for _ in range(5))
            if code not in self.rooms:
                room = Room(code, self)
                room.empty_since = time.time()
                self.rooms[code] = room
                return room
        raise GameError("could not allocate a room code")

    def get(self, code: str) -> Room | None:
        return self.rooms.get((code or "").upper())

    def gc(self) -> None:
        now = time.time()
        for code, r in list(self.rooms.items()):
            if r.empty_since is not None and now - r.empty_since > self.cfg.empty_room_ttl_seconds:
                if r.timer:
                    r.timer.cancel()
                del self.rooms[code]

    async def handle(self, room: Room, ws) -> None:
        """One WebSocket connection. First message must be {"type":"join","name":...}."""
        player = None
        try:
            first = await ws.receive_json()
            if first.get("type") != "join":
                await ws.send_json({"type": "error", "message": "send join first"})
                return
            if sum(1 for p in room.players.values() if p.connected) >= self.cfg.max_players:
                await ws.send_json({"type": "error", "message": "room is full"})
                return
            player = room.add_player(first.get("name", ""), ws)
            await ws.send_json({"type": "welcome", "player_id": player.id, "name": player.name, "room": room.code})
            await room.broadcast_state()
            while True:
                msg = await ws.receive_json()
                try:
                    t = msg.get("type")
                    if t == "start":
                        await room.start(player.id, msg.get("rounds"), msg.get("seconds"))
                    elif t == "word":
                        await room.send(player, await room.submit_word(player.id, msg.get("word", "")))
                    elif t == "challenge":
                        await room.challenge(player.id, msg.get("word", ""))
                    elif t == "vote":
                        await room.cast_vote(player.id, msg.get("vote_id", ""), bool(msg.get("accept")))
                    elif t == "ping":
                        await room.send(player, {"type": "pong"})
                    else:
                        raise GameError(f"unknown message type {t!r}")
                except GameError as e:
                    await room.send(player, {"type": "error", "message": str(e)})
        except Exception as e:  # disconnects end up here
            if type(e).__name__ not in ("WebSocketDisconnect", "ClientDisconnected"):
                log.info("game connection closed: %s", type(e).__name__)
        finally:
            if player is not None:
                await room.drop_player(player.id)

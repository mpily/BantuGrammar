"""FastAPI app factory. app.py mounts the existing Gradio UI onto this app,
so the UI and the REST API share one process and one loaded model."""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import re

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from . import API_VERSION
from .africastalking import ATClient, Conversations, sms_reply_for_translation
from .game import GameConfig, GameError, GameManager
from .schemas import (MACHINE_LABEL, FeedbackRequest, FeedbackResponse, HealthResponse, PlayerWordReview,
                      PlayerWordsResponse, ProgressResponse, ReviewQueueResponse, RoomCreated, RoomInfo,
                      TranslateRequest, TranslateResponse, VerifiedPairsResponse)
from .settings import Settings
from .store import Store
from .translator import Translator
from .util import hash_identifier, normalize_phone, parse_iso, utcnow_iso

log = logging.getLogger("api")
SPACE_DIR = Path(__file__).resolve().parent.parent
LOCALHOST_ORIGIN_RE = r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
PIPELINE_FILES = ("kitaita_e2e_pipeline.py", "kitaita_sentence_generator.py", "nllb_probe_identification.py",
                  "kitaita_verified_corrections.py", "extract_swahili_grammar.py",
                  "nllb_concord_rewrite_pipeline.py", "page1_vocabulary.json", "page2_noun_classes.json",
                  "page3_verb_rules.json", "config.yaml")


def file_hashes(base: Path = SPACE_DIR) -> dict[str, str]:
    out = {}
    for name in PIPELINE_FILES:
        p = base / name
        if p.exists():
            out[name] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def compute_pipeline_version(base: Path = SPACE_DIR) -> str:
    """git SHA when available (env PIPELINE_VERSION, then `git rev-parse`),
    else a hash of the pipeline files so the version still changes whenever
    the code does."""
    v = os.environ.get("PIPELINE_VERSION")
    if v:
        return v
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=base, capture_output=True,
                             text=True, timeout=5).stdout.strip()
        if sha:
            return f"git-{sha}"
    except Exception:
        pass
    blob = "".join(f"{k}:{v}" for k, v in sorted(file_hashes(base).items()))
    return "files-" + hashlib.sha256(blob.encode()).hexdigest()[:12]


def create_app(settings: Settings, store: Store, translator: Translator, at_client: ATClient | None = None,
               game: GameManager | None = None) -> FastAPI:
    at_client = at_client or ATClient(settings.at_username, settings.at_api_key, settings.at_environment,
                                      settings.at_sender_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        # Space shutdown/restart: push whatever is still buffered locally
        ok = store.flush()
        log.info("final feedback flush %s", "ok" if ok else "skipped/failed")
        store.stop()

    app = FastAPI(title="Kitaita translator API", version=API_VERSION, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=settings.allow_origins,
                       allow_origin_regex=LOCALHOST_ORIGIN_RE, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-Reviewer-Key"])
    app.state.settings, app.state.store, app.state.translator = settings, store, translator

    @app.exception_handler(HTTPException)
    async def _http_exc(request: Request, exc: HTTPException):
        return JSONResponse({"api_version": API_VERSION, "error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_exc(request: Request, exc: RequestValidationError):
        detail = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse({"api_version": API_VERSION, "error": "invalid request", "detail": detail},
                            status_code=422)

    # ------------------------------------------------------------------ helpers
    def hash_phone(phone: str) -> str:
        return hash_identifier(normalize_phone(phone), settings.phone_hash_salt or "", "ph")

    def hash_speaker(speaker_id: str) -> str:
        if speaker_id.startswith("ph_"):
            return speaker_id  # already a salted phone hash (internal channels)
        if settings.phone_hash_salt:
            return hash_identifier(speaker_id, settings.phone_hash_salt, "spk")
        return "spk_u_" + hashlib.sha256(speaker_id.encode()).hexdigest()[:24]

    def run_translation(text: str, mode: str, channel: str) -> dict:
        result, hit, latency = translator.translate(text, mode)
        row = store.add_translation({
            "source": text, "swahili_mt": text, "mode": mode, "channel": channel,
            "kitaita": result["kitaita"], "kitaita_marked": result["kitaita_marked"],
            "unknown_words": result["unknown_words"], "vocabulary_gaps": result["vocabulary_gaps"],
            "slots": result["slots"], "flags": result["flags"], "noun_class": result["noun_class"],
            "pipeline_version": translator.pipeline_version, "config_hash": settings.config_hash(),
            "latency_ms": latency, "cached": hit,
        })
        return row

    def record_feedback(translation_id: str, speaker_hash: str, *, stars, native_output, channel,
                        comment=None, approve=False, trusted=False, speaker_name=None) -> tuple[dict, bool]:
        return store.add_feedback({
            "translation_id": translation_id, "speaker_id": speaker_hash, "stars": stars,
            "native_output": native_output, "comment": comment, "channel": channel,
            "approve": approve, "trusted_reviewer": trusted, "speaker_name": speaker_name,
        })

    conversations = Conversations(
        store,
        translate_fn=lambda text, channel: run_translation(text, settings.default_mode, channel),
        cached_fn=lambda text: translator.cached(text, settings.default_mode),
        feedback_fn=lambda tid, ph, stars, native_output, channel: record_feedback(
            tid, ph, stars=stars, native_output=native_output, channel=channel),
        hash_phone=hash_phone, at=at_client, ttl_minutes=settings.sms_state_ttl_minutes)
    app.state.conversations = conversations

    def check_at_request(token: str | None):
        if not settings.phone_hash_salt:
            raise HTTPException(503, "PHONE_HASH_SALT secret not configured; refusing to handle phone numbers")
        if settings.at_webhook_token and token != settings.at_webhook_token:
            raise HTTPException(403, "bad webhook token")

    # ------------------------------------------------------------------ routes
    @app.get("/api/health", response_model=HealthResponse)
    def health():
        return HealthResponse(ok=True, pipeline_version=translator.pipeline_version,
                              pipeline_loaded=translator.loaded, persistence=store.persistence,
                              at_configured=at_client.configured)

    @app.post("/api/translate", response_model=TranslateResponse)
    def translate(req: TranslateRequest):
        if len(req.text) > settings.max_chars:
            raise HTTPException(422, f"text longer than {settings.max_chars} characters")
        mode = req.mode or settings.default_mode
        row = run_translation(req.text.strip(), mode, "web")
        verified = store.verified_for_source(row["swahili_mt"])
        return TranslateResponse(
            id=row["id"], source=row["source"], swahili_mt=row["swahili_mt"],
            kitaita=row["kitaita"], kitaita_marked=row["kitaita_marked"],
            status="machine" if row["kitaita"] else "unknown", label=MACHINE_LABEL,
            unknown_words=row["unknown_words"], vocabulary_gaps=row["vocabulary_gaps"],
            slots=row["slots"], flags=row["flags"], mode=mode,
            verified_alternative=({"kitaita": verified["kitaita"], "verified_at": verified["verified_at"]}
                                  if verified else None),
            pipeline_version=row["pipeline_version"], latency_ms=row["latency_ms"], cached=row["cached"])

    @app.post("/api/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest, x_reviewer_key: str | None = Header(default=None)):
        if store.get_translation(req.translation_id) is None:
            raise HTTPException(404, f"translation_id {req.translation_id!r} not found")
        trusted = bool(x_reviewer_key) and x_reviewer_key in settings.trusted_reviewer_keys
        if req.approve and not trusted:
            raise HTTPException(403, "approve requires a valid X-Reviewer-Key")
        row, promoted = record_feedback(
            req.translation_id, hash_speaker(req.speaker_id), stars=req.stars, native_output=req.native_output,
            channel=req.channel, comment=req.comment, approve=req.approve, trusted=trusted,
            speaker_name=req.speaker_name if req.name_consent else None)
        return FeedbackResponse(id=row["id"], translation_id=req.translation_id,
                                translation_status=store.translation_status(req.translation_id),
                                promoted=promoted)

    @app.get("/api/progress", response_model=ProgressResponse)
    def progress():
        return ProgressResponse(**store.progress(), generated_at=utcnow_iso())

    @app.get("/api/verified_pairs", response_model=VerifiedPairsResponse)
    def verified_pairs(since: str | None = Query(default=None),
                       kind: str | None = Query(default=None, pattern="^(word|sentence)$")):
        since_dt: datetime | None = None
        if since:
            try:
                since_dt = parse_iso(since)
            except ValueError:
                raise HTTPException(422, "since must be an ISO-8601 timestamp")
        rows = store.verified_pairs(since_dt, kind)
        # explicit allow-list: no speaker ids, feedback text, or method details
        return VerifiedPairsResponse(pairs=[{k: r[k] for k in ("id", "swahili", "kitaita", "kind", "noun_class",
                                                               "level", "verified_at")} for r in rows])

    @app.post("/at/sms")
    async def at_sms(request: Request, background: BackgroundTasks, token: str | None = Query(default=None)):
        check_at_request(token)
        form = await request.form()
        phone, text = str(form.get("from", "")), str(form.get("text", ""))
        if not phone:
            raise HTTPException(422, "missing 'from'")

        def work():
            try:
                reply = conversations.handle_sms(phone, text)
            except Exception as e:
                log.error("sms handling failed: %s", type(e).__name__)
                reply = "Sorry, something went wrong. Please try again."
            at_client.send_sms(phone, reply)

        background.add_task(work)  # AT ignores the webhook body; reply goes out via the SMS API
        return PlainTextResponse("OK")

    @app.post("/at/ussd")
    async def at_ussd(request: Request, background: BackgroundTasks, token: str | None = Query(default=None)):
        check_at_request(token)
        form = await request.form()
        phone, text = str(form.get("phoneNumber", "")), str(form.get("text", ""))
        if not phone:
            raise HTTPException(422, "missing 'phoneNumber'")
        try:
            reply = conversations.handle_ussd(phone, text, background.add_task)
        except Exception as e:
            log.error("ussd handling failed: %s", type(e).__name__)
            reply = "END Sorry, something went wrong."
        return PlainTextResponse(reply)

    # ------------------------------------------------------------------ game (multiplayer word race)
    app.state.game = game

    def require_game() -> GameManager:
        if game is None:
            raise HTTPException(503, "the word game is not available on this server")
        return game

    @app.post("/api/game/rooms", response_model=RoomCreated)
    def create_room():
        g = require_game()
        try:
            room = g.create_room()
        except GameError as e:
            raise HTTPException(503, str(e))
        return RoomCreated(code=room.code, ws_path=f"/ws/game/{room.code}")

    @app.get("/api/game/rooms/{code}", response_model=RoomInfo)
    def room_info(code: str):
        room = require_game().get(code)
        if room is None:
            return RoomInfo(code=code.upper(), exists=False)
        return RoomInfo(code=room.code, exists=True, phase=room.phase,
                        n_players=sum(1 for p in room.players.values() if p.connected))

    @app.get("/api/player_words", response_model=PlayerWordsResponse)
    def player_words(status: str | None = Query(default=None, pattern="^(player_accepted|verified)$")):
        words = store.player_words()
        if status:
            words = [w for w in words if w["status"] == status]
        return PlayerWordsResponse(words=words)

    def require_reviewer(key: str | None) -> None:
        if not key or key not in settings.trusted_reviewer_keys:
            raise HTTPException(403, "requires a valid X-Reviewer-Key")

    def known_noun_classes() -> list[str]:
        oracle = getattr(game, "oracle", None)
        return sorted(getattr(oracle, "concord", {}) or {})

    @app.get("/api/player_words/review_queue", response_model=ReviewQueueResponse)
    def review_queue(x_reviewer_key: str | None = Header(default=None)):
        require_reviewer(x_reviewer_key)
        return ReviewQueueResponse(words=store.review_queue(), noun_classes=known_noun_classes())

    @app.post("/api/player_words/review", response_model=PlayerWordsResponse)
    def review_player_word(req: PlayerWordReview, x_reviewer_key: str | None = Header(default=None)):
        require_reviewer(x_reviewer_key)
        classes = known_noun_classes()
        if req.noun_class and classes and req.noun_class not in classes:
            raise HTTPException(422, f"unknown noun class {req.noun_class!r}; known: {', '.join(classes)}")
        store.add_word_vote({"word": req.word, "room_id": None, "accepted": req.approve, "yes": 0, "no": 0,
                             "n_eligible": 0, "kind": "reviewer", "translation": req.translation,
                             "word_type": req.word_type, "noun_class": req.noun_class})
        if game is not None:
            game.refresh_player_words()
        return PlayerWordsResponse(words=[w for w in store.player_words() if w["word"] == req.word.strip().lower()])

    origin_re = re.compile(LOCALHOST_ORIGIN_RE)

    @app.websocket("/ws/game/{code}")
    async def game_socket(ws: WebSocket, code: str):
        origin = ws.headers.get("origin")
        if origin and origin not in settings.allow_origins and not origin_re.fullmatch(origin):
            await ws.close(code=1008)
            return
        g = game
        room = g.get(code) if g is not None else None
        await ws.accept()
        if room is None:
            await ws.send_json({"type": "error", "message": "room not found", "fatal": True})
            await ws.close()
            return
        await g.handle(room, ws)

    return app


def build_game(store: Store, settings: Settings) -> GameManager | None:
    """The game needs the marker-oracle (pip package kitaita-oracle from
    mpily/bantugrammar-app). Without it the game endpoints answer 503."""
    try:
        from kitaita_oracle import Oracle
        from kitaita_oracle.tables import load_yaml
    except ImportError:
        log.warning("kitaita_oracle not installed: word game disabled")
        return None
    known = set(GameConfig.__dataclass_fields__)
    cfg = GameConfig(**{k: v for k, v in settings.game.items() if k in known})
    return GameManager(Oracle(), load_yaml("kitaita_tiles.json"), store, cfg)


def build_store(settings: Settings) -> Store:
    """Real store: replays the private dataset and starts a CommitScheduler
    when HF_TOKEN is set; local-only otherwise."""
    history_loader = scheduler_factory = None
    if settings.hf_token:
        def history_loader():
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import RepositoryNotFoundError
            try:
                return snapshot_download(settings.dataset_repo, repo_type="dataset", token=settings.hf_token,
                                         allow_patterns=["data/**"],
                                         local_dir=Path(settings.local_dir) / "history")
            except RepositoryNotFoundError:
                return None  # first run: CommitScheduler creates the repo

        def scheduler_factory(folder: Path):
            from huggingface_hub import CommitScheduler
            return CommitScheduler(repo_id=settings.dataset_repo, repo_type="dataset", folder_path=folder,
                                   every=settings.commit_every_minutes, private=True,
                                   token=settings.hf_token, allow_patterns=["data/**"])
    return Store(settings.local_dir, min_distinct_speakers=settings.min_distinct_speakers,
                 min_rooms_to_verify=int(settings.game.get("min_rooms_to_verify", 2)),
                 trusted_reviewer_approval=settings.trusted_reviewer_approval,
                 low_n_threshold=settings.low_n_threshold, history_loader=history_loader,
                 scheduler_factory=scheduler_factory)


def build_app(tables, settings: Settings | None = None) -> FastAPI:
    """Production wiring used by app.py (tables = the already-loaded lookup
    tables; the probes/NLLB are loaded once by app.py)."""
    from .settings import load_settings
    from .translator import pipeline_translator
    settings = settings or load_settings()
    version = compute_pipeline_version()
    store = build_store(settings)
    store.write_manifest({"pipeline_version": version, "config": settings.public_dict(),
                          "config_hash": settings.config_hash(), "file_hashes": file_hashes(),
                          "persistence": store.persistence, "history_error": store.history_error})
    translator = pipeline_translator(tables, settings.lru_cache_size, version)
    return create_app(settings, store, translator, game=build_game(store, settings))


__all__ = ["create_app", "build_app", "build_store", "sms_reply_for_translation"]

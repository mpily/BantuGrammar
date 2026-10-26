# Addendum for the research repo CLAUDE.md — Space API (hackathon, 29 Oct 2026)

Append this section to the existing `CLAUDE.md` in `github.com/mpily/BantuGrammar`.
It adds a public API to the existing Docker Space `mpily/BantuGrammar`. All
existing research rules in this file still apply; this section never
overrides them.

## Scope

The research repo owns the Space and everything that needs the model or the
feedback data:

- `/api/translate` — the pipeline (NLLB → probes → concord rewrite → dict)
- `/api/feedback` — native-speaker star ratings + corrected output
- `/api/progress` — aggregates for the Progress tab
- `/api/verified_pairs` — read-only feed of speaker-verified pairs, consumed
  by the app repo's learner drills
- `/at/sms`, `/at/ussd` — Africa's Talking webhooks (they need a server, and
  the Space is the only compute)

Scrabble, the marker-oracle and learner drills live in the separate app repo
`mpily/bantugrammar-app` (static page, no server). Don't build them here.

## Rules

- **The API wraps the pipeline; it never changes pipeline logic** to make the
  demo look better. Pipeline changes follow the normal research process
  (one change per experiment, logged in `status.md`).
- Put the API in its own package (`api/`), separate from the research code,
  so research modules don't import FastAPI/Gradio.
- **Never present unverified Kitaita as correct.** Every Kitaita string
  carries a `status`: `machine` | `speaker_corrected` | `verified`.
  Translator output is labelled "machine translation — awaiting
  native-speaker review".
- **Don't invent Kitaita.** Unknown forms come back as `unknown`.
- **The Space disk is not persistent.** Feedback goes to a private HF Dataset
  repo (`mpily/bantugrammar-feedback`) via `huggingface_hub.CommitScheduler`
  from a local JSONL buffer, flushed on shutdown. Fallback: Supabase free tier
  — ask before switching.
- **Privacy**: never store raw phone numbers; store a salted hash (salt in
  Space secrets). Speaker names only with consent.
- **Secrets** (HF token, AT API key, salt) in Space secrets / `.env`, never in
  git.
- The API is versioned (`api_version` in every response) and its contract is
  documented in `api/CONTRACT.md`. The app repo depends on it — breaking
  changes need a version bump and a note to the user.
- Free tiers only. If something needs payment, stop and ask.

## Architecture

```
          ┌──────────── HF Space (Docker, CPU Basic, free) ───────────┐
App page ▶│ FastAPI                                                    │
(GitHub   │  /api/translate        /api/feedback     /api/progress     │
 Pages)   │  /api/verified_pairs   /at/sms           /at/ussd          │
AT ──────▶│  /  existing Gradio UI (gr.mount_gradio_app)               │
          └─────────────┬─────────────────────────────────────────────┘
                        │ CommitScheduler (~5 min)
                        ▼
          private HF Dataset mpily/bantugrammar-feedback
          (translations.jsonl, feedback.jsonl, verified_pairs.jsonl)
```

- Mount Gradio onto FastAPI so the UI and REST share one process and one
  loaded model. Keep the existing Gradio tabs working.
- Load NLLB once at startup. Prefer a CTranslate2 int8 conversion for CPU
  speed; measure p50/p95 latency on the Space and log it in `status.md`.
  Target < 5 s per sentence. LRU-cache translations by normalized input.
- CORS: allow only the app's origin (GitHub Pages URL) and localhost for dev.

## API contracts

Pydantic models for every request/response.

`POST /api/translate`
```json
// request
{"text": "Nyoka aja", "src": "swh", "tgt": "dav"}
// response
{"api_version":"1", "id":"tr_01H...", "source":"Nyoka aja",
 "swahili_mt":"...", "kitaita":"Choka yacha", "status":"machine",
 "slots":[{"slot":"a-","controller":"nyoka","relation":"nsubj",
           "rewritten_to":"ya-","verified_form":false}],
 "pipeline_version":"git-sha", "latency_ms":1830}
```
`slots` exposes the probe/rewrite decisions so speakers and judges can see why
a form was produced.

`POST /api/feedback`
```json
{"translation_id":"tr_01H...", "stars":4, "native_output":"Choka yacha",
 "speaker_id":"spk_hash", "comment":"optional", "channel":"web|sms|ussd"}
```
Validate: stars 1–5; `native_output` optional; `translation_id` must exist.

**Promotion rule** (in `config.yaml`): a Kitaita sentence becomes `verified`
when ≥2 distinct speakers submit the identical normalized `native_output`, or
a designated trusted reviewer approves it. Otherwise `speaker_corrected`.

`GET /api/progress` — counts (translations, feedback, unique speakers,
verified pairs); mean stars with n and 95% CI; ≥4-star rate with Wilson CI;
star histogram; exact-match rate and chrF of machine vs speaker output; trend
by day. **Always show n.** Flag "low n" when n < 5.

`GET /api/verified_pairs?since=<iso>&kind=word|sentence`
→ `{"api_version":"1", "pairs":[{"id","swahili","kitaita","kind",
"noun_class","level","verified_at"}]}`. Only `status == verified`. Public
read; no speaker IDs or feedback text in this feed.

## Africa's Talking

- Use the AT **sandbox** for development and the demo.
- `/at/sms`: inbound Swahili → reply with Kitaita + "Reply 1–5 to rate, or
  send the correct Kitaita". Conversation state per hashed number, stored in
  the feedback layer, not only in memory.
- `/at/ussd`: 1 Translate, 2 Daily word (from verified pairs), 3 Rate last
  translation. USSD times out fast: respond < 2 s — return a cached result or
  "result will arrive by SMS" and send it asynchronously.
- Keep SMS within 160 GSM-7 chars where possible (watch ’ vs ').

## Testing

- `pytest`: schemas, promotion rule, progress stats (incl. low-n flag),
  verified_pairs never leaks unverified rows or speaker data, phone hashing.
- `scripts/smoke.sh` hits every endpoint on the live Space after each deploy.
- Log in `status.md`: latency p50/p95, cold-start time, deploy SHA.
- Free CPU Spaces sleep after 48 h idle — wake the Space before the demo.

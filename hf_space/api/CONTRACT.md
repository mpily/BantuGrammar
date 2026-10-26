# Kitaita Space API — contract (api_version "1")

Base URL: `https://mpily-bantugrammar.hf.space`. Interactive schema:
`/api/docs` (OpenAPI at `/api/openapi.json`). The Gradio UI stays at `/`.

The app repo (`mpily/bantugrammar-app`) depends on this contract. **Breaking
changes need an `api_version` bump and a note to the user.** Adding optional
response fields is not breaking.

## Conventions

- Every JSON response, including errors, has `"api_version": "1"`.
- Errors: `{"api_version":"1","error":"...","detail":...}` with HTTP 403 / 404
  / 422 / 503.
- Every Kitaita string has a `status`:
  `machine` | `speaker_corrected` | `verified` | `unknown`.
  `unknown` means no Kitaita form is known — the API never invents one.
- Machine output is labelled `"machine translation — awaiting native-speaker
  review"` (`label` field). Show this label next to it.
- CORS: `https://mpily.github.io` and `http://localhost:<any port>` only.
- Translations (`tr_…`), feedback (`fb_…`) and verified pairs (`vp_…`) use
  sortable ULID-style ids.

## `GET /api/health`

```json
{"api_version":"1","ok":true,"pipeline_version":"git-ad0940dde663",
 "pipeline_loaded":true,"persistence":"hf_dataset","at_configured":true}
```
`persistence` is `local_only` when `HF_TOKEN` is missing (data is lost on
restart). Use this endpoint to wake the Space.

## `POST /api/translate`

Request:
```json
{"text":"Nyoka aja","src":"swh","tgt":"dav","mode":"probes"}
```
- `text`: 1–1000 chars. `src` must be `swh` (Swahili input; the pipeline
  starts from Swahili). `tgt` must be `dav`.
- `mode` (optional): `probes` (default, full pipeline: NLLB encoder +
  trained probes) or `heuristic` (adjacency decomposer, comparison only).

Response:
```json
{"api_version":"1","id":"tr_01K…","source":"Nyoka aja","swahili_mt":"Nyoka aja",
 "kitaita":"Choka yacha.","kitaita_marked":"Choka yacha.","status":"machine",
 "label":"machine translation — awaiting native-speaker review",
 "unknown_words":[],"vocabulary_gaps":0,
 "slots":[{"slot":"aja","controller":"Nyoka","relation":"nsubj","rewritten_to":"ya",
           "verified_form":false,"slot_type":"subject_sm","confidence":"statistical",
           "token_index":1,"sentence_index":0}],
 "flags":["…pipeline caveats…"],"mode":"probes",
 "verified_alternative":null,
 "pipeline_version":"git-…","latency_ms":1830,"cached":false}
```
- `swahili_mt`: the Swahili the pipeline analysed. With `src=swh` it equals
  the input; the field is kept so an English-input mode can be added later
  without breaking the shape.
- `kitaita`: pipeline output verbatim, or `null` (then `status` is
  `unknown`). The pipeline keeps words it has no Kitaita for **as the
  original Swahili word**; those are listed in `unknown_words` and wrapped as
  `[?word]` in `kitaita_marked`. **Display `kitaita_marked`, not `kitaita`.**
- `vocabulary_gaps`: the pipeline's own count of "no Kitaita for this" flags.
- `slots`: the concord decisions behind the output, from the pipeline's slot
  ledger. `relation`: `nsubj` (verb subject prefix), `amod` (adjective),
  `nmod:poss` (possessive), `nmod` (associative `-a`), `det`
  (demonstrative, `-ote`). `confidence` is the pipeline's own tier
  (`verified` / `statistical` / `hypothesis` / `guessed` / `none`);
  `verified_form` is `confidence == "verified"`. Heuristic mode records fewer
  slots than probes mode.
- `verified_alternative`: if speakers have verified a Kitaita form for the
  same input, it's given here (`status: verified`). The machine output is
  still labelled `machine`.
- `cached`: answered from the LRU cache (key = NFC-normalized input with
  collapsed whitespace; **case is kept** because the pipeline uses
  capitalization). Every call still gets its own `id` so it can be rated.

## `POST /api/feedback`

```json
{"translation_id":"tr_01K…","stars":4,"native_output":"Choka yacha",
 "speaker_id":"spk_hash","comment":"optional","channel":"web",
 "speaker_name":"optional","name_consent":false,"approve":false}
```
- `translation_id` must exist (404 otherwise).
- `stars`: 1–5 or omitted. At least one of `stars` / `native_output` is
  required. (Stars are optional so an SMS user can send only a correction.)
- `channel`: `web` | `sms` | `ussd`.
- `speaker_id`: any stable pseudonymous id from the client. The server stores
  only a salted hash of it.
- `speaker_name` is stored **only** when `name_consent` is `true`.
- `approve: true` = trusted reviewer approval; needs header
  `X-Reviewer-Key: <key>` (403 otherwise). With no `native_output` it approves
  the machine output as is — not allowed if that output has unknown words.

Response:
```json
{"api_version":"1","id":"fb_01K…","translation_id":"tr_01K…",
 "translation_status":"speaker_corrected","promoted":false}
```

**Promotion rule** (`config.yaml` → `promotion`): a translation becomes
`verified` when ≥ `min_distinct_speakers` (2) distinct speakers submit the
identical normalized `native_output`, or a trusted reviewer approves it.
Otherwise any correction makes it `speaker_corrected`. Normalization: NFC,
curly → straight apostrophes, lowercase, collapsed whitespace, no trailing
`.!?;:,`. The verified form is the first submitted surface form.

Caveat: `speaker_id` is client-supplied, so "distinct speakers" can be
faked by a single client. Reviewer approval is the trustworthy path.

## `GET /api/progress`

```json
{"api_version":"1",
 "counts":{"translations":120,"feedback":31,"unique_speakers":6,"verified_pairs":4},
 "mean_stars":{"mean":3.4,"ci95":[2.9,3.9],"n":28,"low_n":false},
 "four_plus_rate":{"rate":0.46,"ci95":[0.29,0.64],"k":13,"n":28,"low_n":false},
 "star_histogram":{"1":3,"2":5,"3":7,"4":8,"5":5},
 "exact_match":{"rate":0.2,"ci95":[0.07,0.45],"k":3,"n":15,"low_n":false},
 "chrf":{"mean":61.2,"ci95":[52.0,70.4],"n":15,"low_n":false},
 "trend":[{"date":"2026-10-29","translations":80,"feedback":20,"rated":18,
           "mean_stars":3.5,"low_n":false}],
 "low_n_threshold":5,"generated_at":"2026-10-29T12:00:00+00:00"}
```
- Mean stars: t-based 95% CI (`ci95` null when n < 2). Rates: Wilson 95% CI.
- `exact_match` / `chrf`: machine `kitaita` vs speaker `native_output`
  (normalized as above), over feedback rows that include a correction.
  chrF uses sacrebleu defaults (char order 6, β = 2), 0–100.
- **Always show n.** `low_n` is true when n < `low_n_threshold` (5).

## `GET /api/verified_pairs?since=<iso>&kind=word|sentence`

Public, read-only. Only `status == verified`. No speaker ids, names, comments
or feedback text.
```json
{"api_version":"1","pairs":[{"id":"vp_01K…","swahili":"Nyoka","kitaita":"Choka",
  "kind":"word","noun_class":"ANIMATE-SG","level":1,
  "verified_at":"2026-10-29T12:00:00+00:00","status":"verified"}]}
```
- `since`: ISO-8601; returns pairs verified strictly after it (422 if
  unparseable). `kind`: `word` (one token) or `sentence`.
- `noun_class`: the pipeline's agreement-class label for single-word pairs
  (e.g. `7`, `ANIMATE-SG`); `null` for sentences. Agreement class ≠
  morphological class for animates.
- `level`: learner difficulty heuristic — 1 = word, 2 = ≤ 5 words, 3 = longer.

## Africa's Talking webhooks (sandbox)

Set the callback URLs in the AT sandbox dashboard to `…/at/sms` and
`…/at/ussd` (append `?token=<AT_WEBHOOK_TOKEN>` if that secret is set). Both
return 503 if `PHONE_HASH_SALT` is not configured. Phone numbers are stored
only as salted hashes (`ph_…`).

`POST /at/sms` (form: `from`, `text`, …) — returns `OK` at once; the reply is
sent through the AT SMS API:
- Swahili text → `<kitaita_marked> (machine, unverified). Reply 1-5 to rate,
  or send the correct Kitaita` (fit to 160 GSM-7 chars where possible).
- Then `1`–`5` → rating; any other text → saved as a correction. Resending the
  same Swahili, or prefixing `T ` / `TAFSIRI `, starts a new translation.
- State per hashed number lives in the feedback store (`sessions` table) and
  expires after `sms_state_ttl_minutes` (60).

`POST /at/ussd` (form: `sessionId`, `phoneNumber`, `text`, …) — plain-text
`CON …` / `END …`:
- `""` → menu: `1 Translate`, `2 Daily word`, `3 Rate last translation`.
- `1*<swahili>` → cached result inline; otherwise `END … will arrive by SMS`
  and the translation runs after the response (USSD needs < 2 s).
- `2` → word of the day from verified word pairs (rotates daily).
- `3` → shows the last translation; `3*<1-5>` saves the rating.

## Storage

Private HF Dataset `mpily/bantugrammar-feedback`, written by
`huggingface_hub.CommitScheduler` every 5 min and once on shutdown. Each
process writes its own shard: `data/<table>/<run_id>.jsonl` for tables
`translations`, `feedback`, `verified_pairs`, `sessions`, `runs` (run
manifest: resolved config with secrets redacted + pipeline file hashes). On
startup all shards are downloaded and replayed. Per-process shards (instead
of single `translations.jsonl` files) mean a restart can never overwrite
earlier data.

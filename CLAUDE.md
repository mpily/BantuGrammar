# CLAUDE.md — Kitaita concord project

Read this file and `status.md` (results log, newest entries last) before
proposing any work. This version (2026-10-05) records the pivot from
internal steering to **probe-guided rule-based concord rewriting**.

> Reconstruction note: rebuilt from the project instructions + status.md
> because the previous CLAUDE.md was not available when this was written.
> Hard-rule numbers #2, #4, #5, #9, #10 are preserved because status.md
> cites them; #1, #3, #6, #7, #8 are reconstructed — check against the
> old file before deleting it.

---

## 1. Goal and deployment pipeline (revised)

Help preserve Kitaita (Kidaw'ida, ISO `dav`; Bantu, Kilimanjaro-Taita,
Guthrie E74) by reusing a Swahili-capable MT model instead of training
on scarce Kitaita data.

1. **Generate Swahili** with NLLB (unsteered).
2. **Analyse**: encode the Swahili output with the NLLB encoder; a
   structural (distance) probe + a relation probe (DepProbe-style)
   recover a labelled dependency structure over **subword/morpheme
   positions**.
3. **Intervene**: the Swahili→Kitaita dictionary supplies each noun's
   Kitaita form and Kitaita agreement class.
4. **Find agreement targets**: slots linked to the noun by an
   agreement-bearing relation (nsubj → subject prefix, obj → object
   marker, amod/det/nmod:poss → modifier prefix, acl:relcl → relative
   marker), filtered by "currently carries the noun's concord".
5. **Reinflect** those slots via the concord-series correspondence
   table (Swahili class → Kitaita concord, hand-built with a speaker);
   morphological rewrite where no Swahili series matches Kitaita.
6. **Substitute vocabulary** from the dictionary.

Example: *Nyoka aja* → target *a-* (nsubj subject prefix of *nyoka*) →
*Nyoka yaja* → dictionary → *Choka yacha* (Kitaita forms to verify with
a speaker). Note agreement class ≠ morphological class for animates.

Methodological template: Zmigrod et al. (ACL 2019) — analyse → intervene
→ infer new tags → reinflect. Internal steering is now a **research /
fallback track**, not the primary mechanism (see §5 for why).

## 2. Why the pivot (one paragraph)

Across French/mBART, French/NLLB and Swahili/NLLB, steering produced
real, control-validated causal effects but never reliable known-target
agreement: best French same-noun strict 0.36 (n=11); Swahili per-noun
gradient attribution 25/57 (43.9%) on the toy matrix, with recurring
failure modes (lexical injection, referent substitution, escape to
unmarked forms, garbage collapse at high alpha). A text-level edit
outperformed steering on the same nouns (2026-09-24), and a
deterministic adjacency rewrite solved the hardest real-text case
(`kitunguu chekundu → nyekundu`) on the first try (2026-10-05). The
remaining research question is **target finding**, not steering.

## 3. Staged plan (current)

| Stage | What | Gate to move on |
|---|---|---|
| A | **Gold set**: 100–200 real NLLB Swahili sentences, native-speaker annotated at the **morpheme-slot** level (slot → controlling noun, relation). Stratify: relative clauses, object markers, coordination, two nouns sharing a concord series (same class, or 3/11 `u-`, 4/6/9 `i-`/`yangu`), long sentences. Plus a **tokenizer–morpheme alignment audit** (how often NLLB subwords align with Swahili morpheme boundaries, e.g. `a-li-ki-nunua`). | Annotation guide frozen; inter-annotator check on ≥20 sentences if a second speaker is available. |
| B | **Baseline**: existing adjacency + concord-prefix rewrite (`nllb_swahili_rule_based_rewrite_prototype.py`) scored on gold (slot-level P/R/F1, per stratum). | Numbers logged. If F1 ≥ ~0.95 on all strata, skip C–D. |
| C | **Structural probe on NLLB encoder**. Training data, in order of preference: Steimel & Kübler SWH_UDT (silver; check release); zero-shot from other UD languages (Chi, Hewitt & Manning 2020; DepProbe-style source-language selection). Layer sweep. **Context-stability check**: distance Spearman binned by sentence length (EWT first, existing pipeline), and same sentence encoded alone vs inside a paragraph. | Beats B on the hard strata by a pre-registered margin. |
| D | **Relation probe** (start from DepProbe code) → relation→slot mapping, hand-specified for Swahili concord. Decide morpheme-level training targets (word-level UD gives the whole verb one node). | Slot-level F1 on gold ≥ B + margin. |
| E | **Reinflection**: verified Swahili concord tables (extend `rewrite` dictionary; words not in it are silently skipped — log coverage), then Kitaita correspondence table with a speaker. Fallback segmenter for badly-tokenized verbs (rule-based / SALAMA-style analyzer). | Coverage report; speaker sign-off on table. |
| F | **Evaluation** with native Kitaita speakers (prefer naive annotators; forced choice, fluency + adequacy). Baselines: naive dictionary substitution; adjacency rewrite (B); optional counterfactual re-translation (swap noun for a target-class proxy in the English source, diff outputs); steering. Test sets stress long-distance and ambiguous agreement. | — |

Sentence boundaries: probe distances exist only within a sentence.
Cross-sentence object markers/pronouns are a coreference problem — out
of scope for C–D, track separately.

## 4. Models and data

- **NLLB-200** (`facebook/nllb-200-distilled-1.3B` for recent work; 3.3B
  used for earlier class-7 work — never mix in one comparison). Swahili
  competence verified directly (EN↔SW, SW→SW) on 3.3B; spot-check
  distilled models the same way before relying on them.
- mBART-50 (base and mmt) has **no usable Swahili** — confirmed by native
  speaker. French/Romance stage only.
- Lexicons: `swahili_noun_lexicon_agreement_v2.py` (319 nouns, 11
  agreement classes, labels derived from NLLB concord behaviour, author-
  knowledge glosses), `fr_it_gender_dict.py`, `french_noun_lexicon.py`
  (all unverified against a corpus — flagged).
- Kitaita data: Mbogho et al. (2025) methodology; Lacuna Fund 2021
  Swahili↔Kitaita/Kalenjin/Dholuo corpus (check release status);
  KenCorpus Kitaita archive. eBible corpus loader exists but data was
  never fetched.
- Downloads manifest: `docs/reproduction_downloads.md`.

## 5. Durable findings (do not re-derive)

- Grammatical gender / noun class is **decodable** (embedding and
  stream) with real selectivity over random-relabel controls; whitened
  mean-centred class centroid ≈ probe direction (cos 0.935).
  Decodability ≠ causal steerability (repeatedly).
- Swahili agreement class follows **animacy** over shape class (*kipepeo,
  paka, nyoka* take class-1 concord); kinship possessives irregular
  (*wajomba zangu*).
- Raw-embedding between-class separation ≈ random control at n=144/159/
  319 — a real null, not underpowered.
- First-subword pooling collapsed 39% of Swahili nouns onto shared
  vectors — **always pool whole words**.
- Causal decoders cannot reach pre-nominal positions from a noun patch;
  last decoder layer cannot propagate across positions (gap test).
- Steering "successes" were often artifacts: lexical injection
  (embedding-diff), referent substitution, predicate-noun agreement,
  escape to unmarked forms (*huitwa*, *ni*), repetition collapse.
- Steering success is predicted by **baseline competitiveness** (unsteered
  rank of the target), not class membership (logistic reanalysis, n=57).
- Gradient-selected units are mostly **per (source, target) word pair**,
  not per noun.
- Natural FR/IT text exercised steering on 1 of ~20 nouns; many real
  occurrences have **no observable agreement marker** — stratify eval.

## 6. Hard rules

1. *(reconstructed)* Never claim a model/corpus covers a language or
   phenomenon without checking directly (native-speaker check where
   possible).
2. Verify the pipeline actually propagates/measures what it claims
   before trusting a run (gap tests for position effects; tooling
   self-checks — e.g. substring matches like `fleur`⊂`fleuriste`).
3. *(reconstructed)* Always run the strict metric + random + orthogonal
   controls (+ reversed sign where meaningful). For rewrite/probe work:
   compare against the adjacency baseline and a random-target baseline.
4. Log the full resolved config (and file hashes) in every result JSON.
5. Report n and confidence intervals; no selection or claims on n < 8.
6. *(reconstructed)* Retract explicitly in status.md when a result fails;
   never silently overwrite.
7. *(reconstructed)* Log every session's results in status.md.
8. *(reconstructed)* Null hypotheses ("relatedness helps", "more data
   helps", "the tree helps") must be beaten, not assumed.
9. One change per experiment.
10. Extend canonical pipelines; don't fork them.

Additional rules learned:
- Score agreement only on words **controlled by the target noun**; a
  gendered/class-marked word belonging to another noun is not a success.
- Select hyperparameters on validation data disjoint from test; use
  intention-to-treat cohorts (no post-treatment exclusion).
- Logit margin is not a proxy for generation quality; always check
  decoded text.
- Checkpoint any run > 10 min (per-item resume).
- Alphas/scales never transfer across models or layers — measure the
  activation norm first.

## 7. Working conventions

- `status.md` = results log; this file = goals, stages, rules.
- External critiques: self-contained `*_prompt.md`; responses saved as
  `*_claude.md` / `*_chat.md` / `*_gemini.md`.
- One change per experiment; strict metric + controls every time.

## 8. Open decisions

- Probe training data: SWH_UDT (silver) vs zero-shot transfer vs both.
- Probe granularity: subword-position targets for verb slots — how to
  derive them (rules from relation labels vs hand annotation).
- Encode per sentence vs per paragraph — **resolved for attention**
  (checked directly, 2026-10-05: no effect); still open for the
  structural probe.
- Whether counterfactual re-translation is a baseline or a component.
- Chaga (E60) as a comparison donor later.
- **(New, 2026-10-05) Morpheme-level agreement-target identification**:
  encoder attention gives one validated, zero-shot, training-free signal
  — adjective concord, noun→piece attention direction, heads
  L2H12/L1H11/L1H2 strongest. Confirmed on real gold text, position-
  confound-controlled, and confirmed by direct head/column inspection to
  be genuinely selective (suppresses the wrong noun to ~zero), not an
  artifact. Subject and object verb agreement do NOT have a validated
  attention signal: the subject direction is a generic previous-token-
  head artifact (attends to whatever sits one position back regardless
  of identity — only looked like subject-tracking in short hand-built
  sentences); the object direction is statistically real at the
  population level but dominated by trivial same-word BPE
  reconstruction, with the real signal living in a thin, often-zero
  residual — not reliable per-sentence. Two probe-based routes also
  failed to generalize: a word-level probe applied to raw subword
  tokens (mostly a word-order artifact), and a probe hand-trained on a
  10-sentence mini-benchmark (100% in-sample, chance-level held-out —
  clean overfit). Full chain of checks in status.md, 2026-10-05
  cont. 1–8.
- **(New, 2026-10-05) Coordination and semantic/animacy agreement
  override have no settled rule — this blocks treating rule-based verb
  templates as an automatically safe fallback.** Real example: "Chama
  tawala cha PPP ... na kile cha PTI ... walilaumu tukio hili" (gold
  corpus) uses human-plural wa- on a coordinated pair of class-7 "chama"
  nouns. Three sources gave three different answers for the "correct"
  agreement: gold text (wa-), a naive coordination rule (vi-, plural of
  the shared class), NLLB's own EN→SW generation of an equivalent
  sentence (singular ki-, possibly closest-conjunct agreement). Needs a
  native speaker, not inference from any tool tried so far (status.md
  2026-10-05 cont. 9). Any rule-based verb-template segmenter must
  explicitly FLAG coordinated subjects and known animacy-override nouns
  (kipepeo/paka/nyoka/chama-type, §5) rather than silently guessing.
- **(New, 2026-10-05) SW→EN translation is not a morphology-correctness
  check.** Confirmed directly: NLLB translates several, only-one-
  grammatical subject-prefix variants of the same sentence identically.
  Every SW→EN translation verification in this project (this session
  and presumably prior ones) should be read as "the gloss/meaning is
  right," never as confirmation that a specific agreement morpheme is
  correct Swahili.

## 9. Key references

Hewitt & Manning 2019 (structural probe); Chi, Hewitt & Manning 2020
(cross-lingual); Müller-Eberstein et al. 2022 (DepProbe, labelled
directed trees, code: personads/depprobe); Zmigrod et al. 2019
(dependency-based reinflection for gender swapping); Steimel & Kübler
2023 (Swahili UD treebank, git.steimel.info/ksteimel/SWH_UDT);
Hurskainen SALAMA (rule-based Swahili MT with class tags); Hewitt &
Liang 2019 (control tasks); relexification precedent (Fadaee,
Pourdamghani); "Dictionaries to the Rescue" (2025); URIEL/lang2vec;
Maseno/Princeton 2026 Kenyan ASR (auxiliary language helps only at ~zero
target data — tested fine-tuning, not this pipeline).

---

## 10. Space API (hackathon, 29 Oct 2026)

Adds a public API to the existing Space `mpily/BantuGrammar`. All
research rules above still apply; this section never overrides them.
### Scope

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

### Rules

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

### Architecture

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

### API contracts

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

### Africa's Talking

- Use the AT **sandbox** for development and the demo.
- `/at/sms`: inbound Swahili → reply with Kitaita + "Reply 1–5 to rate, or
  send the correct Kitaita". Conversation state per hashed number, stored in
  the feedback layer, not only in memory.
- `/at/ussd`: 1 Translate, 2 Daily word (from verified pairs), 3 Rate last
  translation. USSD times out fast: respond < 2 s — return a cached result or
  "result will arrive by SMS" and send it asynchronously.
- Keep SMS within 160 GSM-7 chars where possible (watch ’ vs ').

### Testing

- `pytest`: schemas, promotion rule, progress stats (incl. low-n flag),
  verified_pairs never leaks unverified rows or speaker data, phone hashing.
- `scripts/smoke.sh` hits every endpoint on the live Space after each deploy.
- Log in `status.md`: latency p50/p95, cold-start time, deploy SHA.
- Free CPU Spaces sleep after 48 h idle — wake the Space before the demo.

### Implementation notes (2026-10-26)

Where the code differs from the spec above, and why:

- **Location**: the API is `hf_space/api/` (the `hf_space/` folder is what
  deploys to the Space). Tests: `hf_space/tests/`. Contract:
  `hf_space/api/CONTRACT.md`. Promotion rule: `hf_space/config.yaml`.
  Smoke test: `scripts/smoke.sh`.
- **Space SDK**: the Space's README says `sdk: gradio`, not Docker.
  `app.py` now mounts Gradio onto FastAPI and runs uvicorn itself, which
  works under the Gradio SDK. A `Dockerfile` is included for switching to
  `sdk: docker` (+ `app_port: 7860`) if needed.
- **Storage layout**: per-process shards `data/<table>/<run_id>.jsonl`
  instead of single `translations.jsonl` / `feedback.jsonl` /
  `verified_pairs.jsonl` files, so a restart (or a failed startup download)
  can never overwrite earlier data.
- **Two feedback datasets**: the existing Gradio "Submit feedback" form still
  writes to `mpily/kitaita-translation-feedback` and the Progress tab still
  reads it (left as is so the existing tabs keep working). The API writes to
  `mpily/bantugrammar-feedback`. Merging them is an open decision.
- **CTranslate2 not used**: the probes need NLLB encoder hidden states from
  the PyTorch model; swapping in CT2 would change the pipeline, which the
  API must not do. Any CT2 work goes through the normal research process.
- **`src` is `swh` only**: the pipeline starts from Swahili text. EN→SW via
  `english_to_swahili()` would load a second NLLB copy (memory on CPU
  Basic), so it is not exposed yet.
- **`stars` optional** in `/api/feedback` (at least one of `stars` /
  `native_output` required), so an SMS user can send only a correction.
- **Unknown words**: the pipeline keeps words it has no Kitaita for as the
  original Swahili word. The API lists them in `unknown_words` and wraps
  them as `[?word]` in `kitaita_marked`; clients should display that field.
- **Trusted reviewers** authenticate with `X-Reviewer-Key` (secret
  `TRUSTED_REVIEWER_KEYS`). `speaker_id` is client-supplied, so the
  2-speaker consensus path can be gamed by one client — flagged in the
  contract.
- **Multiplayer word race (user decision, 2026-10-26).** The user asked for a
  free, skribbl-style game for native speakers and chose: rooms hosted on this
  Space (`api/game.py`, WebSocket `/ws/game/{code}`), a round-based word race
  rather than a full Scrabble board, and "majority accepts, logged unverified"
  for words the oracle rejects. Votes go to the `word_votes` table; a word is
  `verified` after acceptance in 2 different rooms or by a trusted reviewer.
  The Space imports the oracle from the app repo (`kitaita-oracle`, pinned
  commit in `hf_space/requirements.txt`): bump the pin when the app's data
  changes. Contract: `api/CONTRACT.md` → "Multiplayer word race".

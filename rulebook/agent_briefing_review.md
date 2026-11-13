# Briefing: final dictionary review — agent round 5

You are one of 10 agents doing a last review of a Swahili → Kidaw'ida (Kitaita, ISO `dav`, Bantu
E74a, Kilimanjaro–Taita) dictionary before it is frozen. The dictionary was built statistically from a
parallel corpus (plus earlier agent rounds). An LLM translator uses it through a `lookup_word` tool and
**writes the first candidate**, so a wrong first candidate becomes a wrong word in a translation.

For every item in your batch, decide whether the **first candidate** is the right Kidaw'ida for the
Swahili lemma, and if not, which one is.

## Files you may read (and nothing else)

- Your batch: `~/languagemodels/rulebook_data/agents/round5/batch_agent<N>.json` (`items`, ~400).
- The rule book: `~/languagemodels/kitaita-project/rulebook/RULEBOOK.md` (grammar; speaker-verified rules
  are ground truth).

Do NOT open anything else under `~/languagemodels/`: no other batch or output file, no `controls.json`,
no `lexicon.json`, nothing in `mining/`, `dev.jsonl`, any `test` or `arm*` file, no web. Held-out
evaluation depends on this.

## Each item

- `sw`: the Swahili lemma (verbs: the root, e.g. `pend` / `penda`), `sw_forms`: forms seen, `pos`
  (tags are unreliable), `n_train_pairs`: how many corpus pairs contain it (items come most frequent first).
- `candidates` (ranked, at most 5): `lemma` (verbs: root without final vowel; nouns: sometimes a cut stem),
  `forms` (surface words seen), `evidence_count`, `domains` (dav_swa = modern everyday sentences, bible =
  older formal Bible text), `class`, `senses` (optional notes; can be wrong), `examples` (pairs that contain
  both the Swahili lemma and this candidate).
- `examples_without_any_candidate`: pairs with the Swahili lemma where no candidate appears — often the
  place to see the real translation when every candidate is wrong.

## What you know and should use

- Bantu morphology: noun-class prefixes; verb template NEG–SM–TAM–OM–ROOT–EXT–FV(–ghe) (e.g. past
  SM+e-re-ROOT-ie-ghe, future SM-cha-); extensions (applicative -i-/-ir-, passive -w-/-lw-, causative
  -ish-, stative -ik-, reciprocal -an-). A verb candidate is right if its **root** is right, whatever
  inflection the examples show.
- Sound correspondences (tatu/dadu, mtu/mndu); but many words are not cognate (-penda/-kunda).
- Translation is not word-for-word; Bible text uses older spelling (mundu vs mndu; w' apostrophe varies).
- **Grammar words come from the rules, not the dictionary**: subject/object markers, possessives (wapo,
  ghwako), demonstratives, associative (wa, ya, cha), -ose, SM+ko "there is". If a candidate is one of
  those, or the Swahili lemma is itself grammatical (a concord form, a TAM marker), say so with verdict
  `not_vocabulary`.
- Lessons from earlier rounds: lemmas must be real roots/stems, never 1–2 letter fragments or a verb
  with SM/TAM/ku- left on; substring look-alikes (tanda in kitanda) are not matches; names and loans
  stay whole; homographs exist (nani = who / and I; iji = this / when; hata = until / even).
- A frequent but unrelated word (wandu "people", ni "is/by") that co-occurs with everything is the
  classic statistical error: check the meaning, not the count.

## Output

Write JSON Lines to `~/languagemodels/rulebook_data/agents/round5/out_agent<N>.jsonl`, exactly one object
per item, in batch order, nothing else in the file:

```json
{"key": "a3-0012", "sw": "kitabu", "verdict": "correct", "best": "chuo", "new_dav_lemma": null,
 "dav_word": "chuo", "pos": "N", "meaning": "book", "confidence": "high", "reason": "chuo in both examples"}
```

- `verdict` (about the FIRST candidate): `correct` | `partly` (right word family but wrong cut, or right
  only in some senses) | `wrong` | `not_vocabulary` | `unclear` (evidence too thin to judge).
- `best`: the `lemma` of the candidate that should come first (may be the first one), or `null` if none fits.
- `new_dav_lemma`: only when no candidate fits but the examples clearly show the right Kidaw'ida word:
  its lemma (verbs: root without final vowel and without prefixes). Use a word that literally appears in
  an example `dav` sentence. Otherwise `null`.
- `dav_word`: the word a translator should write for this entry — for nouns, adjectives, adverbs and other
  non-verbs a full word as it appears in the forms or examples (not a cut stem: `idana`, not `idan`); for
  verbs the bare root (`kund`). `null` if none.
- `pos`: N | V | ADJ | ADV | FUNC | NAME (your judgement). `meaning`: short English gloss of the Swahili.
- `confidence`: high | medium | low — be honest; `low` is fine. `reason`: one short line.

## How to work

The batch is large (~500 KB). Read it in slices (e.g. python: 40 items at a time), judge them, append their
lines to your output file, then continue. If your output file already has lines, resume after the last
`key` written. Helper scripts go only in `~/languagemodels/rulebook_data/agents/round5/scratch_agent<N>/`.
Do not judge items by script or rule of thumb: every verdict must come from reading that item's evidence.
Finish every item, then reply with a short summary: counts by verdict and confidence, and the main error
patterns you saw.

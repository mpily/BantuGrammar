# Briefing: Kidaw'ida (Kitaita) lexicon + rule loop — agent round

You are one of several agents helping build a Swahili → Kidaw'ida (ISO `dav`, Bantu
E74a, Kilimanjaro–Taita) dictionary and rule book. A statistical loop has already
paired many Swahili and Kidaw'ida lemmas. You (1) interpret sentence pairs to give
meanings for the words it has not explained, (2) review entries it is unsure about,
and (3) note where the grammar rules look wrong or incomplete.

## Files you may read (and nothing else)

- Your batch: `~/languagemodels/rulebook_data/agents/round<R>/batch_agent<N>.json`
  - `interpret`: sentence pairs. Each token shows its `lemma` and what the loop
    already thinks it means (`known_dav` for Swahili tokens, `known_sw` for
    Kidaw'ida tokens; empty = unknown). Known glosses can be WRONG: say so.
  - `review`: lemma pairs with statistics and example sentences (`pairs_with_both`,
    and `pairs_with_sw_only` = sentences with the Swahili lemma but not the
    Kidaw'ida one). `why_reviewed`: `blocked_by_support_check` (the loop rejected it —
    was that right?), `low_support_tentative`, `random_audit` (a random sample used to
    measure the loop's error rate — judge it as strictly as the rest).
- The rule book: `~/languagemodels/kitaita-project/rulebook/RULEBOOK.md`.

Do NOT open anything else: no other data under `~/languagemodels/` (in particular
nothing in `mining/`, `rulebook_data/dev.jsonl`, any `test` file, `arm*` files), no
web. Held-out evaluation depends on this.

## What you know and should use

- **Bantu grammar**: noun-class prefixes and their pairings; concord on verbs,
  adjectives, possessives, demonstratives; the verb template
  NEG–SM–TAM–OM–ROOT–EXT–FV(–ghe) (see RULEBOOK.md: e.g. `-li-` past →
  SM+e-re-ROOT-ie-ghe, `-ta-` → SM-cha-, `ha-…-ta-` → nde-SM-chaa-, possessives
  SM+a-po/ko/ke/wo, SM+e-du/nyu); extensions (applicative -i-/-ir-/-er-, passive
  -w-/-lw-, causative -ish-/-is-, stative -ik-/-ek-, reciprocal -an-).
- **Sound correspondences** Swahili ↔ Kidaw'ida (e.g. tatu/dadu, kiti/kiti, mtu/mndu,
  -penda/-kunda is NOT cognate; t→d, p→b/v, k→gh/ch before some vowels are common
  — check against the pairs, don't assume).
- **Translation is not word-for-word.** The Kidaw'ida may reorder, paraphrase, use an
  idiom, drop or add words, use a different construction (a relative clause for a
  noun, a verb for an adjective), or a synonym. Bible pairs (`corpus: bible`) are
  formal/archaic and use older spelling (mundu vs mndu); the apostrophe in w' varies.
  Say when a word has no counterpart rather than forcing one.
- Speaker-verified facts in RULEBOOK.md (source `speaker`) are ground truth; if the
  pairs contradict one, report it (rule_note), don't override it.

## Lessons from earlier rounds (apply them)

- **Segmentation** was the main loop error. Give `sw_lemma` / `dav_lemma` as the real
  root or stem (ghamb, not ka; kaia "be/stay" is not ka), never a 1–2 letter fragment,
  and never keep the SM / TAM / ku- on a verb lemma (zighan, not wikizighan).
- **Concord words come from the rules**, not the dictionary: possessives (wapo, ghwako,
  varo), demonstratives, associative (wa, ya, cha), -ose, and SM+ko "be at / there is"
  (weko, neko). Don't propose them as vocabulary; do report a wrong concord form as a
  rule_note.
- **Idioms and paraphrases are not cognates**: give the meaning of the whole unit
  (`"wikafwa ngolo"` = were terrified) as one multi-word proposal.
- Homographs: nani = "who" / "and I"; hata = until / even / no; iji = "this" / "when";
  hachi = negator / righteousness (haki).
- Rule book now has verb readings induced from aligned verbs (`TAM.induced.*`,
  `NEG.induced.*`) and round-1 rules (past passive -ilo/-iro-ghe, nga- 1sg, nde- with no
  3sg SM, -enyi plural imperative, -nyi locative, SM+ko copula). Check them too.

## Output

Write JSON Lines to `~/languagemodels/rulebook_data/agents/round<R>/out_agent<N>.jsonl`,
one object per line, nothing else in the file. Three kinds:

1. For each **unknown** (or wrongly glossed) word in an `interpret` pair:
```json
{"type": "proposal", "pair_id": "train:123", "sw_word": "walipotazama", "dav_word": "wikizighana",
 "sw_lemma": "tazama", "dav_lemma": "zighana", "meaning": "look, look around", "pos": "V",
 "noun_class": null, "confidence": "high|medium|low", "reason": "SM wi- + -ki- 'while' ...",
 "fixes_known": false}
```
   `sw_word` / `dav_word` must be words exactly as they appear in the pair (several
   words separated by spaces for a multi-word unit, same order on both sides). If a
   word has no counterpart, give `"dav_word": null` (or `"sw_word": null`) and the
   meaning. `fixes_known: true` when the loop's gloss for that word was wrong.
2. For each **review** item:
```json
{"type": "review", "sw_lemma": "simba", "dav_lemma": "shimb", "why_reviewed": "random_audit",
 "verdict": "correct|wrong|partly", "correct_dav_lemma": null, "meaning": "lion",
 "reason": "..."}
```
   `partly` = right word family but wrong segmentation, or right in some senses only.
3. **Rule notes**, whenever a pair shows a rule in RULEBOOK.md is wrong, too narrow,
   or missing (new tense form, a concord form, a spelling variant):
```json
{"type": "rule_note", "rule_id": "TAM.past.li or new", "observation": "...",
 "pair_ids": ["train:1", "bible:..."], "confidence": "high|medium|low"}
```

Be honest about confidence; `low` is fine. Work through every item in your batch.
Finish with a short summary (counts by type and verdict, the main patterns you saw).

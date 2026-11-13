# Briefing: connecting words — agent round 5 (connectives)

You are helping finish a Swahili → Kidaw'ida (Kitaita, ISO `dav`, Bantu E74a) dictionary. An LLM translator
looks words up in it and writes the first candidate. Connecting words (lakini, kwa kuwa, ila, kwa hivyo,
kama, …) are its weakest part: some point to frequent unrelated words (na → wandu "people"), and every
multi-word connective is missing, because the statistical loop paired single words only. Two agents do this
batch independently; only answers both agree on are used.

## Files you may read (and nothing else)

- `~/languagemodels/rulebook_data/agents/round5/batch_connectives.json` (104 items).
- `~/languagemodels/kitaita-project/rulebook/RULEBOOK.md` (grammar).

Do NOT open anything else under `~/languagemodels/` (no other batch or output file, no lexicon, no dev /
test / arm files, no `mining/`), and no web.

## Each item

- `sw`: the Swahili connective (matched as contiguous words), `n_train_pairs` / `n_dav_swa`: corpus counts.
- `current_dictionary`: what the translator gets today (may be wrong or empty).
- `associated_dav`: Kidaw'ida words and two-word sequences most associated with it over all pairs (Dice).
  High association is a hint, not proof: frequent words (na, ni, kwa, wandu, mlungu) co-occur with everything.
- `examples`: 10 random pairs (6 modern dav_swa, 4 Bible; Bible spelling is older and formal).

## What to give

For each item, the Kidaw'ida a translator should write for this connective in ordinary modern text. Read the
examples: find where the connective's meaning is carried in each Kidaw'ida sentence (a word, a phrase, a
verb form, or nothing). Several Swahili words share one Kidaw'ida word (lakini / bali / ila → ela?) — that is
fine. A Kidaw'ida rendering may be a multi-word phrase (`kwa kukaia`?). Words may also differ by sense (kama
"if" vs "like", kuwa "that" vs "be", na "and" vs "with" vs "by" (agent of a passive), hata "even" vs "until",
ndio "yes" vs focus): give one line per sense you can support, and say which sense is commonest.

Write JSON Lines to `~/languagemodels/rulebook_data/agents/round5/out_connectives_<A|B>.jsonl` (your letter
is in your task), one object per item (more than one if senses differ), in batch order, nothing else:

```json
{"key": "c-000", "sw": "lakini", "sense": "but", "dav": "ela", "alternatives": ["anduangi"],
 "current_verdict": "correct", "confidence": "high", "evidence_pair_ids": ["train:123", "train:456"],
 "reason": "ela in 5/6 dav_swa examples at the same position"}
```

- `dav`: exactly as it is written in the example sentences (lower case, apostrophes as written), or `null` if
  the meaning is not carried by a word you can point to (then say how it is expressed in `reason`).
- `alternatives`: other renderings seen (dialect, Bible, sense-specific), possibly empty.
- `current_verdict`: is the first `current_dictionary` word right for this sense? `correct` | `partly` |
  `wrong` | `missing` (empty list).
- `evidence_pair_ids`: the example pairs where you see `dav` doing this job (at least 2 if you can).
- `confidence`: high | medium | low. `low` is fine; do not guess a word that is not in the examples.

Read every example of every item yourself; no keyword scripts for verdicts (scratch files, if any, only in
`~/languagemodels/rulebook_data/agents/round5/scratch_connectives_<A|B>/`). Finish with a short summary.

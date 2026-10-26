---
title: BantuGrammar
emoji: 🏆
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.30.0"
python_version: "3.13"
app_file: app.py
pinned: false
license: mit
short_description: Testing transferrability between different Bantu languages
---

# Kitaita Translator

A probe-guided, rule-based Swahili→Kitaita (Kidaw'ida, ISO `dav`; Bantu,
Kilimanjaro-Taita, Guthrie E74) translation pipeline, built around NLLB
rather than fine-tuned on it.

Kitaita has very little digitized text and no existing MT system. Swahili
is closely related and well-resourced. The null hypothesis — "relatedness
helps" — has to be beaten, not assumed, and every claim behind this demo is
checked against real corpus data, native-speaker judgment, or both before
it's trusted. Full writeup and code: [GitHub](https://github.com/mpily/BantuGrammar).

## What this Space runs

1. **Identify structure**: encode the Swahili sentence with NLLB's encoder;
   trained relation probes recover, per sentence, which noun is each verb's
   subject/object, and which noun each adjective/possessive agrees with.
2. **Resolve agreement classes**: a hand-built Kitaita dictionary supplies
   each noun's Kitaita form and Kitaita noun-class (concord) marker — which
   does **not** always line up with the noun's Swahili class.
3. **Substitute and reinflect**: the verb's subject-marker/TAM and the
   adjective's concord prefix are rewritten for the resolved Kitaita class;
   nouns are substituted via dictionary lookup; everything else is
   reassembled in its original sentence position.

This mirrors Zmigrod et al. (2019)'s analyse → intervene → reinflect
template for morphological rewriting, applied across a language pair
instead of within one.

**Probes mode** runs the real pipeline (NLLB encoder + 5 trained relation/
POS probes). **Heuristic mode** is a plain word-adjacency decomposer kept
only for comparison — it's faster but structurally blind to anything past
simple local adjacency (coordinated subjects, longer-distance agreement).

## Test it, and help find what's missing

**🎲 Give me a sentence to test** hands you a real Swahili sentence (sampled
from this project's own parallel Bible corpus, not synthetic) instead of
you having to think one up. If you speak Kitaita, rate the translation
(1–5) and, if it's wrong, tell us what it should have been — that's logged
for review, not applied automatically. This is the project's main way of
finding gaps it doesn't know about yet, so a low rating is exactly as
useful as a high one.

## Recent additions (2026-10-20)

- **Object markers**: Kitaita person object markers (me/us/them, +2sg
  still unresolved) found by cross-referencing the parallel corpus —
  corpus-inferred hypotheses, not yet speaker-confirmed, not yet wired
  into live output.
- **Default class agreement for headless subjects**: "Kile
  kinachonisikitisha" (that which saddens me) — no concrete noun at all —
  now gets a statistically-grounded default Kitaita subject marker from
  its relative-clause marker's class, instead of refusing to translate.
- **Periphrastic verbs**: "tawala" (rule) has no single-word Kitaita root,
  but *is* a confirmed two-word phrase, "bonya nguma" (lit. "do
  authority") — verified compositionally (fanya→bonya, mamlaka→nguma are
  each independently confirmed) before this mechanism existed at all.
- Several new verb roots and noun translations confirmed directly against
  the parallel corpus (not guessed) — see the project's `docs/status.md`
  for the full evidence trail on every addition, including the ones
  tried and rejected along the way.

## Known gaps (not hidden)

- Dictionary coverage is incomplete — plenty of common nouns, verb roots,
  and adjective-class forms have no Kitaita translation yet; a sentence
  using one will visibly flag it rather than guess.
- Coordination and semantic/animacy agreement override have no settled
  rule yet (flagged, not silently resolved, when detected).
- `-ki-` (participial) verb subjects rely on 4 hand-built syntactic rules,
  not the trained probe — see the GitHub README for why.

First call after the Space wakes up takes longer (NLLB + probes loading);
after that, each translation is just inference.

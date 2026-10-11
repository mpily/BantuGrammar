# Kitaita subspace-transfer project

Testing whether reusing Swahili grammatical structure — rather than naive
dictionary substitution or fine-tuning on scarce data — can produce
usable Kitaita (Kidaw'ida, ISO `dav`; Bantu, Kilimanjaro-Taita, Guthrie
E74) translations from a Swahili-capable MT model (NLLB).

Kitaita has very little digitized text and no existing MT system. Swahili
is closely related and well-resourced. The null hypothesis — "relatedness
helps" — has to be beaten, not assumed, and every claim here is checked
against real corpus data, native-speaker judgment, or both before it's
trusted.

## Pipeline

1. **Generate Swahili** with NLLB (unsteered) from the source text, or
   take real Swahili text directly.
2. **Identify structure**: encode the Swahili with NLLB's encoder; trained
   relation probes recover, per sentence, which noun is each verb's
   subject/object, which noun each adjective/possessive agrees with.
3. **Resolve agreement classes**: the Kitaita dictionary supplies each
   noun's Kitaita form and Kitaita noun-class (concord) marker — which
   does **not** always line up with the noun's Swahili class.
4. **Substitute and reinflect**: the verb's subject-marker/TAM and the
   adjective's concord prefix are rewritten for the resolved Kitaita
   class; nouns are substituted via dictionary lookup; everything else
   is reassembled in its original position in the sentence.

This mirrors Zmigrod et al. (2019)'s analyse → intervene → reinflect
template for morphological rewriting, applied across a language pair
instead of within one.

## Code in this push

- **`kitaita_e2e_pipeline.py`** — the original, full free-text pipeline:
  candidate detection (nouns/verbs/adjectives/possessives/function words),
  probe-based relation identification, hand-built syntactic override
  rules for `-ki-` (participial) verb subjects, object-marker and
  relative-clause rewriting, and position-preserving reassembly. Serves
  `kitaita_local_webapp.py`, a small local tool for testing translations
  against the current pipeline state.
- **`kitaita_relations.py`** — a from-scratch, step-by-step rebuild of
  the identification → concord/TAM substitution → reassembly core, after
  the original pipeline grew hard to reason about. Four composable
  functions, each independently testable:
  - `find_relations()` — subject/object per verb from the trained
    probes, plus the `-ki-`-verb override rules (see below).
  - `align_concord_and_tam()` — for each verb/adjective, finds its
    agreement driver, translates it, resolves its Kitaita class, and
    substitutes concord + TAM (reuses `kitaita_sentence_generator.
    translate()`, the validated substitution engine, rather than
    re-deriving its tables).
  - `dictionary_substitute_remaining()` — plain dictionary lookup for
    every word concord didn't need to reason about.
  - `reassemble_sentence()` — places every piece back at its original
    token position.
- **`nllb_probe_identification.py`** — loads the trained probes and
  exposes the relation-identification primitives (`identify_relations`,
  `find_ki_governor`, `tag_pos`, the win-condition helpers) both
  pipelines above are built on.
- **`saved_identification_probes/`** — trained weights: `nsubj_obj_probe`
  (labelled relation probe, 21-way, covers nsubj/obj/amod/poss/acl/etc.),
  `amod_probe`, `case_probe`, `poss_probe`, `acl_probe` (binary relation
  probes), `pos_probe` (NOUN/VERB/ADJ/PRON/OTHER), `governor_attachment_
  probe` (which token a `-ki-` verb's real syntactic head is),
  `silver_nsubj_ki_probe[_multilayer]` and `xsubj_controls_probe_en`
  (negative results, kept for the record — see below).
- Supporting modules these import: Swahili verb/noun segmentation
  (`extract_swahili_grammar.py`, `bantu_grammar.py`), the Kitaita-side
  concord/TAM/root tables (`kitaita_verified_corrections.py`,
  `kitaita_sentence_generator.py`), the gold-corpus rewrite pipeline used
  to validate concord rules (`nllb_concord_rewrite_pipeline.py`), probe
  architecture and training-data loading (`relation_probe.py`,
  `pos_probe.py`, `order_probe.py`, `structural_probe.py`,
  `conllu_utils.py`), and the 319-word hand-built noun lexicon
  (`swahili_noun_lexicon_agreement_v2.py`). `probe_pipeline_regression.py`
  is a 12-case hand-built regression battery run after every change to
  either pipeline.
- `page1_vocabulary.json` / `page2_noun_classes.json` /
  `page3_verb_rules.json` — the mined Swahili↔Kitaita candidate tables
  (noun/verb-root/adjective-concord correspondences, Dice-aligned across
  3 Bible corpus versions) both pipelines read at startup.

Not included in this push: the training scripts for the probes above
(they need the UD/SUD/AfriSUD treebank data and NLLB itself, and the
probes are already trained and checked in as weights), and the large
body of exploratory/steering research scripts this project also
contains (embedding-space steering, cross-lingual transfer experiments,
etc.) — those live alongside this code but aren't part of the
translation pipeline itself.

## What's been checked, and what hasn't

**Durable, checked findings** (not re-derived; see `docs/status.md` for
the full record if you have it):

- Grammatical noun class is reliably decodable from NLLB's
  representations, with real selectivity over random-relabel controls —
  but decodability is not the same as being able to *steer* generation
  with it; internal activation steering was explored first and
  repeatedly produced real-but-unreliable effects (lexical injection,
  referent substitution, escape to unmarked forms), which is why the
  project pivoted to rule-based rewriting as the primary mechanism.
- Raw attention (both encoder self-attention and encoder-decoder
  cross-attention) was tested twice, independently, as a training-free
  mechanism for verb subject/object tracking. Both looked strong on
  short hand-built sentences and both failed an adversarial,
  position-confound-controlled check on real corpus data — proven to be
  linear-position artifacts, not grammatical tracking. The only
  attention-based signal that survived the same kind of check is noun→
  adjective concord targeting (a different relation).
- The *trained* relation probe (not raw attention) does track subject/
  object correctly even when the subject isn't adjacent to the verb —
  0.90/0.88 accuracy on held-out Swahili, and the same architecture
  transfers zero-shot to English (0.90/0.88 again) via NLLB's shared
  encoder space.
- That same probe is specifically unreliable for one narrow category:
  subject identification for `-ki-` (participial) verbs, because almost
  no training examples exist where that relation is even annotated (5/
  226 in the available treebank) — not underpowered, genuinely absent
  from how dependency treebanks mark non-finite control constructions.
  A dedicated probe trained on an English control-relation annotation
  (`nsubj:xsubj`, 3,020 real examples) and tested zero-shot on Swahili
  came back at chance (7/14) — a plausible but unconfirmed account is
  that English marks this relation through non-finite verb *shape*,
  which Kitaita's `-ki-` form doesn't share. `find_relations()` instead
  uses 4 hand-built syntactic rules for this one category (reported-
  speech verbs, periphrastic "kuwa", perception-verb object control,
  temporal-conjunction defaults), each validated against real corpus
  examples before being trusted.

**Known open gaps**, not hidden:

- Dictionary coverage is incomplete — plenty of common nouns, verb
  roots, and adjective-class forms have no Kitaita translation yet. A
  speaker-facing tool for filling these in is in progress (not part of
  this push).
- The self-loss fallback (NLLB's own generative preference, used when
  none of the 4 `-ki-` rules fire) exists in the original pipeline but
  hasn't been ported into the `kitaita_relations.py` rebuild yet.
- Coordination and semantic/animacy agreement override (e.g. a
  coordinated pair of inanimate nouns taking human-plural agreement in
  real text) have no settled rule.
- Possessive-pronoun concord data only covers one noun class; every
  other class is explicitly flagged as unsupported rather than guessed.

## Running it

```
python3 probe_pipeline_regression.py      # 12-case regression battery
python3 kitaita_local_webapp.py 8765      # local browser UI, both pipelines
```

`kitaita_relations.py` can also be used directly, one step at a time:

```python
from kitaita_sentence_generator import load_pages, build_lookup_tables
from kitaita_relations import find_relations, align_concord_and_tam, \
    collect_pieces, dictionary_substitute_remaining, reassemble_sentence

tables = build_lookup_tables(*load_pages())
tokens = ["Mwanafunzi", "mrefu", "alimwona", "mzee", "akitembea"]
relations = find_relations(tokens, noun_idxs=[0, 3], verb_idxs=[2, 4])
concord = align_concord_and_tam(tokens, [0, 3], [2, 4], [1], tables)
pieces = collect_pieces(concord)
pieces.update(dictionary_substitute_remaining(tokens, set(pieces), tables[0]))
print(reassemble_sentence(tokens, pieces))
```

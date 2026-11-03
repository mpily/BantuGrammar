"""
rulebook_correct.py

2026-11-02. Version B of the user's two-system comparison: deterministic
corrections applied to the fine-tuned NLLB's draft (system A). Every step can
be switched off on its own, so each is evaluated as a separate change.

  rerank    -- among the model's n-best candidates, take the one with the fewest
               noun-phrase agreement violations (a noun directly followed by a
               concord word, or a demonstrative directly before a noun, of a
               class the noun never takes in the training text; data:
               kitaita_agreement_data.json). Ties keep the model's
               order, so this only ever picks text the model itself produced.
  fill      -- a word the flagger marks as copied Swahili gets the round-1 mined
               Kidaw'ida form if one exists (kitaita_mined_lexicon; class-
               dependent words are never filled). Otherwise it stays Swahili.
  unknown   -- a word not in the training vocabulary is replaced by the Swahili
               source word, but only when the alignment is unambiguous: exactly
               one unknown word and exactly one source word that nothing in the
               output accounts for. Otherwise the model's word is kept.
  repeat    -- 3rd+ consecutive copies of a token are dropped.
  subject   -- the pipeline's own analysis of the Swahili source (probe path:
               each verb's subject and the Swahili subject marker) and the
               subject-class resolver (subject_class_resolver.py: class pair from
               the Kidaw'ida noun, number from the Swahili verb) fix the subject
               marker on the draft's verbs (Nguku orekoghe -> erekoghe: nguku is
               an i- noun). Only the marker changes; root and tense stay the
               model's. Surface by class and tense from kitaita_subject_markers.json
               (corpus counts over verified class nouns; used only where one form
               has >= 5 hits and >= 75%). Draft and source verbs are paired in
               order and only when their counts match. User request 2026-11-02.
  verified  -- if the source is a native-speaker-verified sentence
               (kitaita_speaker_verified.json), output it exactly; nothing else runs.
  applicative -- the Swahili source has an applicative verb (walipigania 'fought
               for', base pigan) but the draft's verb on that base's Kidaw'ida
               root has none (w'edelwa 'fought'): add it (w'edeluya), using the
               attested applicative root if round 1 lists one, else the corpus rule
               (root-final ny/nj/s/sh/z/ch/j -> -ir-/-er-, otherwise -i-; lw -> luy).
               Fires only when exactly one draft word matches. User report 2026-11-02.
               One-consonant roots (ku-l-a 'eat' -> -j-) need an object marker
               before ku (a-li-m-ku-l-i-a); user request 2026-11-02.
               Perfective drafts (-ie/-ire/-ere, -ghe) are changed only for a
               speaker-verified stem (lw -> lu: wi-lu-ie-ghe) or a malformed -id-
               before the perfective; after a palatal root, -ere/-ire is also
               applicative + subjunctive (di-bony-er-e), so those are left alone.

Flagged words are never shown as [?] (user decision, 2026-11-02: brackets would
bias the speakers judging A vs B); the default is the Swahili word.
Nothing here is speaker-verified.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from kitaita_mined_lexicon import mined_lookup
from subject_class_resolver import resolve as resolve_subject_class
from rulebook_flagger import flag, norm

_DATA = Path(__file__).parent / "kitaita_agreement_data.json"
_ROOTS = Path(__file__).parent / "kitaita_verb_roots.json"
_VERIFIED = Path(__file__).parent / "kitaita_speaker_verified.json"
_SUBJECT_MARKERS = Path(__file__).parent / "kitaita_subject_markers.json"
ALL_STEPS = ("verified", "rerank", "fill", "unknown", "repeat", "applicative", "subject", "subject_probe",
             "subject_checked", "tense", "extension")
# 2026-11-03: the subject step is OFF by default. On 500 novel test sentences it
# changed 27, 4 better / 21 worse, B-A -0.49 [-0.81, -0.22]: the pipeline's
# subject class is often a guess (u- fallback for chungu/kondoo/barabara/jeshi)
# and nouns like charonyi were taken for verbs. docs/status.md 2026-11-03.
DEFAULT_STEPS = tuple(x for x in ALL_STEPS if x not in ("subject", "subject_probe", "subject_checked",
                                                        "tense", "extension"))
# extension forms tried, in order; one is used only if the corpus attests it
EXT_FORMS = {"passive": ("w", "lw", "ilw", "elw", "iw", "ew"), "causative": ("isy", "ish", "esh", "is", "es", "ithy"),
             "stative": ("ik", "ek"), "reciprocal": ("an",),
             # attested forms only (Pikia -> Dekia, huletea -> radaredia)
             "applicative": ("i", "ir", "er")}
MIN_ATTESTED = 2
# Swahili roots whose applicative is a fixed form with no Kidaw'ida applicative:
# ku-p-e-a 'give (to)' = Kidaw'ida ku-nek-a (development check 2026-11-03)
LEXICAL_APPLICATIVES = {"p"}
SM_MIN_COUNT = 5
SM_MIN_SHARE = 0.75
_SM_TAMS = ("cha", "re", "de", "ka", "ki", "we")  # "" = zero-marked present

# Applicative shape by Kidaw'ida root (docs/status.md 2026-11-02 cont. 8): after a
# root-final ny/nj/s/sh/z/ch/j the extension is -ir- (-er- if the root vowel is
# e/o); after any other consonant it is -i-. Attested special case: lw -> luy.
APPLICATIVE_SPECIAL = {"lw": "luy"}
_PALATAL = ("ny", "nj", "sh", "ch", "s", "z", "j")
_FINALS = ("agha", "aa", "a", "e")


@lru_cache(maxsize=1)
def _agreement():
    d = json.loads(_DATA.read_text(encoding="utf-8"))
    return ({f: set(c) for f, c in d["concord_forms"].items()},
            {w: set(v["classes"]) for w, v in d["noun_classes"].items()},
            set(d.get("demonstratives", [])))


@lru_cache(maxsize=1)
def _speaker_verified():
    try:
        d = json.loads(_VERIFIED.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, {}
    sentences = {_sentence_key(x["swahili"]): x for x in d.get("sentences", [])}
    stems = {r: v["stem"] for r, v in d.get("morphology", {}).get("applicative_perfective_stem", {}).items()}
    return sentences, stems


def _sentence_key(text: str) -> str:
    return " ".join(norm(t) for t in text.split() if norm(t))


def applicative_perfective(root: str) -> str:
    """Root + applicative, in front of the perfective -ie. Speaker-verified stems
    first (lw -> lu: wi-lu-ie-ghe); after root-final ny/nj/s/sh/z/ch/j the corpus
    keeps -ir-/-er- (bony-er-ie, rum-ir-ie); elsewhere the applicative -i- merges
    into -ie, so the root is unchanged (vik-ie, nek-ie)."""
    stems = _speaker_verified()[1]
    if root in stems:
        return stems[root]
    if root.endswith(_PALATAL):
        vowels = re.findall(r"[aeiou]", root)
        return root + ("er" if vowels and vowels[-1] in "eo" else "ir")
    return root


def applicative_stem(root: str) -> str:
    """Kidaw'ida root + applicative extension, by the corpus rule."""
    if root in APPLICATIVE_SPECIAL:
        return APPLICATIVE_SPECIAL[root]
    if root.endswith(_PALATAL):
        vowels = re.findall(r"[aeiou]", root)
        return root + ("er" if vowels and vowels[-1] in "eo" else "ir")
    return root + "i"


@lru_cache(maxsize=1)
def _verb_roots():
    return {sw: v["kitaita_roots"] for sw, v in json.loads(_ROOTS.read_text(encoding="utf-8"))["roots"].items()}


_SW_OBJECT_MARKERS = ("ni", "ku", "tu", "wa", "m", "mw", "ki", "vi", "li", "ya", "zi", "i", "u")


def _om_pseudo_root(base: str, roots: dict) -> bool:
    """Round-1 parse noise: 'roots' like kup / nip / wap are an object marker plus
    a 1-2 letter root (-p- 'give'). Skip them."""
    return any(base.startswith(om) and 0 < len(base) - len(om) <= 2 for om in _SW_OBJECT_MARKERS)


# One-consonant Swahili roots (ku-l-a 'eat'). Round 1's root table lists j 'come'
# and p 'give' but not l; l -> j is from the mined lexicon (kula -> kuja, high;
# alikula -> odeja). Not speaker-verified.
# Their applicative is plain -i- even after j/ch: the only two training sentences
# with this shape have zi-li-tu-ku-j-i-a -> ra-di-ch-i-a and zi-me-wa-ku-j-i-a ->
# ra-di-ch-e-a 'came to us/them' (not the palatal -ir-).
SHORT_ROOTS_ADDED = {"l": ["j"]}
_SW_SM = ("a", "u", "wa", "ni", "tu", "m", "mw", "ki", "vi", "i", "li", "ya", "zi", "ku", "pa")
_SW_TAM = ("li", "na", "me", "ta", "ka", "nge", "ngali")


def _kitaita_roots(base: str) -> list[str]:
    return _verb_roots().get(base) or SHORT_ROOTS_ADDED[base]


@lru_cache(maxsize=1)
def _short_applicative_re():
    """Subject + tense + object marker + ku + one-consonant root + i/e + a:
    a-li-m-ku-l-i-a. Monosyllabic verbs keep ku- after the object marker, which
    rules out the look-alikes a-li-l-i-a 'cried' and a-me-ku-li-a 'grew up'
    (-li- 'cry' / -ku- 'grow' + applicative). Without an object marker it does
    not fire."""
    bases = sorted({b for b in _verb_roots() if len(b) == 1 and b not in "aeiou"} | set(SHORT_ROOTS_ADDED))
    alt = lambda xs: "|".join(sorted(xs, key=len, reverse=True))
    return re.compile(rf"(?:{alt(_SW_SM)})(?:{alt(_SW_TAM)})(?:{alt(_SW_OBJECT_MARKERS)})ku({alt(bases)})(?:i|e)a")


def swahili_applicatives(src: str) -> list[tuple[str, str | None]]:
    """(Swahili base root, attested Kidaw'ida applicative root or None) for each
    source word built on base + applicative -i-/-e- (walipigania -> pigan).
    If round 1 lists base+i itself and its Kidaw'ida root is built on the base's
    root (fanyi -> bonyer), that attested root is used; if it is a different
    word (tumi 'use' -> tumi vs tum 'send' -> dum; pati -> nek), the source word
    is treated as lexicalised and skipped."""
    roots = _verb_roots()
    out = []
    for tok in (norm(t) for t in src.split()):
        short = _short_applicative_re().fullmatch(tok)
        if short:
            out.append((short.group(1), None))
            continue
        for base in sorted(roots, key=len, reverse=True):
            if len(base) < 3 or _om_pseudo_root(base, roots):
                continue
            # final -a/-e only: base + i/e + wa is usually the passive (katal-iw-a), not an applicative
            m = re.search(rf"{base}(i|e|li|le)(a|e)$", tok)
            if not m:
                continue
            listed = roots.get(base + m.group(1))
            attested = None
            if listed:
                built_on_base = [r for r in listed if any(r.startswith(br) for br in roots[base])]
                if not built_on_base:
                    break  # lexicalised: a different Kidaw'ida verb
                attested = next((r for r in built_on_base if r not in roots[base]), None)
            out.append((base, attested))
            break
    return out


@lru_cache(maxsize=1)
def _subject_markers():
    """(dominant surface by (class, tense), surfaces with >= 2 hits by class)."""
    counts = json.loads(_SUBJECT_MARKERS.read_text(encoding="utf-8"))["counts"]
    dominant, attested = {}, {}
    for base, by_tam in counts.items():
        for tam, c in by_tam.items():
            surface, n = max(c.items(), key=lambda x: x[1])
            if n >= SM_MIN_COUNT and n / sum(c.values()) >= SM_MIN_SHARE:
                dominant[(base, tam)] = surface
            attested.setdefault(base, set()).update((x, tam) for x, k in c.items() if k >= 2)
    return dominant, attested


@lru_cache(maxsize=1)
def _noun_forms() -> set[str]:
    """Kidaw'ida forms known to be nouns: agreement data, round-1 nouns, voted
    subject classes, and the mined lexicon's noun entries."""
    here = Path(__file__).parent
    forms = set(_agreement()[1])
    for e in json.loads((here / "kitaita_noun_classes_round1.json").read_text(encoding="utf-8"))["nouns"].values():
        forms.update(e["kitaita"])
    forms.update(json.loads((here / "kitaita_subject_classes.json").read_text(encoding="utf-8"))["nouns"])
    mined = json.loads((here / "kitaita_mined_lexicon_round1.json").read_text(encoding="utf-8"))["entries"]
    forms.update(norm(e["kitaita"]) for e in mined.values() if e.get("area") == "nouns" and e.get("kitaita"))
    return forms


def verb_prefix(token: str, base: str):
    """(surface, tense) if token starts with this class's dominant subject marker."""
    dominant = _subject_markers()[0]
    for tam in _SM_TAMS:
        s = dominant.get((base, tam))
        if s and token.startswith(s + tam) and len(token) > len(s + tam) + 1:
            return s, tam
    s = dominant.get((base, ""))
    if s and token.startswith(s) and len(token) - len(s) >= 3 and token[len(s)] not in "aeiou":
        return s, ""
    return None


def draft_candidates(words: list[str]) -> tuple[list[int], dict]:
    """Noun positions and verb positions (-> {class: (surface, tense)}) in a draft.
    Known nouns, -nyi locatives and ku- forms (infinitives: kukaia) are not verbs."""
    nouns = _noun_forms()
    bases = sorted(_subject_markers()[1])
    noun_idxs, verbs = [], {}
    for i, w in enumerate(words):
        t = norm(w)
        if t in nouns:
            noun_idxs.append(i)
        elif len(t) >= 4 and not t.endswith("nyi") and not t.startswith("ku"):
            m = {b: hit for b in bases if (hit := verb_prefix(t, b))}
            if m:
                verbs[i] = m
    return noun_idxs, verbs


def draft_words(src: str, hyp: str) -> list[str]:
    """The rule book's own tokenisation of a draft (same as correct())."""
    return [t for t, _ in flag(src, hyp)]


def _class_of_draft_noun(form: str, clauses: list[dict] | None) -> tuple[str | None, str]:
    """Subject marker of a noun as written in the draft. The form decides when
    both numbers give the same marker; otherwise the number comes from a Swahili
    clause whose subject is this noun."""
    sg, why = resolve_subject_class(None, form, "ki")
    pl, _ = resolve_subject_class(None, form, "vi")
    if sg and sg == pl:
        return sg, why
    from subject_class_resolver import kitaita_forms
    for c in clauses or []:
        if form in kitaita_forms(c.get("subject"), c.get("subject_kitaita")) and c.get("swahili_sm"):
            m, why = resolve_subject_class(c.get("subject"), form, c.get("swahili_sm"))
            return m, f"{why} (number from Swahili {c.get('verb')})"
    return None, f"{form}: number needed, no Swahili clause for it"


# Associative particles (round-1 concord table, ASSOC) and agentive ni: a noun
# right after one is a possessor or an agent (nguo ra mwana, ni lufu), not a subject.
_NOT_SUBJECT_AFTER = {"cha", "gha", "ghwa", "ja", "jha", "kwa", "lwa", "ra", "va", "wa", "ya", "ni"}


@lru_cache(maxsize=1)
def _vocab() -> set[str]:
    d = json.loads((Path(__file__).parent / "kitaita_training_vocab.json").read_text(encoding="utf-8"))
    return {norm(w) for w in d["vocab"]}


@lru_cache(maxsize=1)
def _all_surfaces() -> tuple[str, ...]:
    return tuple(sorted({sf for pairs in _subject_markers()[1].values() for sf, _ in pairs}, key=len, reverse=True))


def inflects(token: str, surface: str, tam: str) -> bool:
    """True if the training text has the same stem with another subject marker
    (ra-dima / wa-dima): evidence the word is a verb (not rahisi, champali)."""
    stem = token[len(surface) + len(tam):]
    vocab = _vocab()
    return any(s2 != surface and (s2 + tam + stem) in vocab for s2 in _all_surfaces())


def _swahili_subject_forms(c: dict) -> set[str]:
    from subject_class_resolver import kitaita_forms
    forms = set(kitaita_forms(c.get("subject"), c.get("subject_kitaita")))
    e = mined_lookup(c.get("subject") or "", "derived")
    if e:
        forms.add(norm(e["kitaita"]))
        forms.update(norm(x) for x in e.get("alternatives", []))
    return forms


def _has_ext(tail: str, ext: str) -> bool:
    if ext == "passive":
        return "w" in tail[:-1]
    if ext == "causative":
        return bool(re.match(r"(i|e)(sy|sh|s|thy)", tail))
    if ext == "stative":
        return bool(re.match(r"(i|e)k", tail))
    if ext == "applicative":
        return bool(re.match(r"(i|ir|er)", tail))
    return tail.startswith("an")


def _find_draft_verb(words, kroots, bare_at, nouns):
    """The one draft word that contains a Kidaw'ida root with a verb prefix
    before it: (word index, root position, root, prefix readings) or None.
    bare_at: for a Swahili imperative, the word position where a bare
    (prefix-less) verb is accepted, else None."""
    from kitaita_verb import parse_prefix
    allow_bare = bare_at is not None
    hits = []
    for i, w in enumerate(words):
        t = norm(w)
        if (t in nouns and i != bare_at) or t.endswith("nyi"):  # nouns and -nyi locatives (irikonyi)
            continue
        # a vowel-initial root may surface with gh- (enda -> -ghend-)
        for kr in [k for k0 in kroots for k in ((k0, "gh" + k0) if k0[:1] in "aeiou" else (k0,))]:
            if len(kr) < 2:
                continue
            start = 0
            while (pos := t.find(kr, start)) >= 0:
                start = pos + 1
                if pos == 0:
                    if allow_bare and i == bare_at:
                        hits.append((i, 0, kr, [{"neg": None, "row": None, "variant": None, "sm": "", "tam": None,
                                                 "om": None}]))
                    continue
                readings = parse_prefix(t[:pos])
                if not readings and kr[0] in "aeiou":
                    # the tense vowel and the root's first vowel merge (we + enda -> wenda)
                    readings = parse_prefix(t[:pos] + kr[0])
                if readings:
                    hits.append((i, pos, kr, readings))
    if len({h[0] for h in hits}) != 1:
        return None
    return max(hits, key=lambda h: len(h[2]))  # the longest root inside that word


def _pick_reading(readings, sw_sm):
    """Reading whose row the Swahili subject marker allows, else the draft's own
    row if all readings agree on it."""
    from kitaita_verb import SW_SM_ROWS
    fit = [r for r in readings if r["row"] in SW_SM_ROWS.get(sw_sm or "", ())]
    pool = fit or readings
    rows = {r["row"] for r in pool}
    if len(rows) != 1:
        return None
    # prefer a reading with a tense marker over one that reads it as an object marker
    return sorted(pool, key=lambda r: (r["tam"] is None, r["om"] is not None))[0]


def _rebuild(word: str, new_core: str) -> str:
    core = re.sub(r"\W+$", "", word)
    if word[:1].isupper():
        new_core = new_core[:1].upper() + new_core[1:]
    return new_core + word[len(core):]


def _tense_target(sw: dict, r: dict, tail: str, kr: str):
    """New (prefix, tail) when the draft's tense clearly contradicts the Swahili
    verb, else None. Modern (davswa) forms, verb_morphology.md rewrite rules."""
    from kitaita_verb import attested_ending, build_prefix
    row, om = r["row"], r["om"]
    if row is None:
        return None
    mood, tam = sw["mood"], sw["tam"]
    if mood == "affirmative" and sw["rel"] == "po" and tam == "li" and r["neg"] is None and r["tam"] != "nda" \
            and (r["tam"] in ("re", "de") or r["variant"] == "e"):
        # -lipo- 'when' -> SM+e-nda-R-a (verb_morphology.md TAM table)
        new_tail = tail[:-5] + "a" if tail.endswith("ieghe") else tail
        return build_prefix(None, row, "e", "nda", om), new_tail, "-lipo- 'when' -> SM+e-nda-"
    if mood == "affirmative" and sw["rel"] and sw["rel"] != "po" and tam == "na" and r["neg"] is None \
            and r["tam"] is None and r["variant"] == "a" and tail.endswith("agha"):
        # present relative -> bare SM R-agha (wanaopigana -> w'ilwagha)
        return build_prefix(None, row, "bare", None, om), tail, "present relative -> bare SM ...-agha"
    if mood == "habitual" and r["neg"] is None and r["tam"] in ("we", "wia", "cha"):
        return build_prefix(None, row, "a", "da", om), tail, "hu- habitual -> SM+a-da-"
    if mood == "affirmative" and not sw["rel"] and r["neg"] is None:
        if tam in ("me", "sha") and (r["tam"] in ("we", "cha", "wia")
                                     or (r["tam"] is None and r["variant"] == "bare")):
            return build_prefix(None, row, "a", None, om), tail, f"-{tam}- perfect -> SM+a"
        if tam == "li" and r["tam"] in ("we", "cha"):
            return build_prefix(None, row, "e", None, om), tail, "-li- past -> SM+e"
        if tam == "ta" and (r["tam"] in ("we", "de", "re") or (r["tam"] is None and r["variant"] == "e")):
            return build_prefix(None, row, "bare", "cha", om), tail, "-ta- future -> SM-cha-"
        # no -na- rule: stative-like verbs use the past-looking form for the present
        # (wanataka -> w'ekunda, nilitaka -> nakunde), development check 2026-11-03
    if mood == "prohibitive" and r["neg"] is None and r["tam"] not in ("se", "seke", "sake") and tail[-1:] in ("a", "e"):
        return build_prefix(None, row, "bare", "se", om), tail[:-1] + "e", "prohibitive -si- -> SM-se-...-e"
    # past negative: Swahili ha-SM-ku-...; the final vowel cannot tell it for loan roots (hakufahamu)
    past_neg = tam == "ku" or (tam is None and sw["fv"] == "a")
    if mood == "negative" and past_neg and r["neg"] == "nde":
        for neg_present in ("agha", "aa"):
            if tail.endswith(neg_present):
                base = tail[:-len(neg_present)]
                cands = [base + f for f in ("ie", "ire", "ere")]
                best = max(cands, key=lambda c: attested_ending(kr + c))
                if attested_ending(kr + best) >= MIN_ATTESTED:
                    return build_prefix(r["neg"], row, r["variant"], r["tam"], om), best, \
                        "past negative -> nde-SM-...-ie"
    return None


def _replace_prefix(word: str, n_letters: int, new: str) -> str:
    """Replace the first n_letters letters of word (apostrophes skipped) with new."""
    k = j = 0
    while j < n_letters:
        if word[k] != "'":
            j += 1
        k += 1
    if word[:1].isupper():
        new = new[:1].upper() + new[1:]
    return new + word[k:]


def violations(hyp: str) -> list[tuple[str, str]]:
    """(noun, concord word) pairs whose classes do not overlap: a noun directly
    followed by a concord word, or a demonstrative directly followed by a noun."""
    concord, nouns, dems = _agreement()
    toks = [norm(t) for t in hyp.split()]
    out = []
    for a, b in zip(toks, toks[1:]):
        if a in nouns and b in concord and not (nouns[a] & concord[b]):
            out.append((a, b))
        elif a in dems and b in nouns and b not in concord and not (nouns[b] & concord[a]):
            out.append((b, a))
    return out


def _swap(raw: str, new: str) -> str:
    """Replace the word inside a raw token, keeping its punctuation and initial capital."""
    m = re.match(r"^(\W*)(.*?)(\W*)$", raw)
    lead, core, trail = m.groups() if m else ("", raw, "")
    if core[:1].isupper():
        new = new[:1].upper() + new[1:]
    return f"{lead}{new}{trail}"


def _covered_source_words(src: str, hyp_tokens: list[str]) -> set[str]:
    """Source words the output accounts for: copied as is, or present as one of
    their mined Kidaw'ida forms."""
    hyp = set(hyp_tokens)
    covered = set()
    for s in {norm(t) for t in src.split()} - {""}:
        e = mined_lookup(s, "derived")
        forms = {s} | ({norm(e["kitaita"])} | {norm(f) for f in e["alternatives"]} if e else set())
        if forms & hyp:
            covered.add(s)
    return covered


def correct(src: str, nbest: list[str], steps=DEFAULT_STEPS, clauses: list[dict] | None = None,
            draft_subjects: dict | None = None) -> dict:
    """nbest: the model's candidates, best first. clauses: the rule-based
    pipeline's per-verb analysis of src (translate_sentence_with_probes()'s
    report["clauses"]), needed by the subject step. Returns the corrected text
    and a list of what changed."""
    changes = []
    notes = []
    edited = set()
    chosen = 0
    if "verified" in steps:
        hit = _speaker_verified()[0].get(_sentence_key(src))
        if hit:
            return {"kitaita": hit["kitaita"], "draft": nbest[0], "chosen_candidate": 0,
                    "changes": [("verified", f"native-speaker-verified sentence ({hit['verified']})")], "notes": []}
    if "rerank" in steps and len(nbest) > 1:
        counts = [len(violations(h)) for h in nbest]
        chosen = min(range(len(nbest)), key=lambda i: (counts[i], i))
        if chosen != 0:
            changes.append(("rerank", f"candidate {chosen + 1} (agreement violations {counts[0]} -> {counts[chosen]})"))
    hyp = nbest[chosen]
    tokens = flag(src, hyp)
    words = [t for t, _ in tokens]

    if "fill" in steps:
        for i, (t, c) in enumerate(tokens):
            if c == "swahili":
                e = mined_lookup(norm(t), "derived")
                if e:
                    words[i] = _swap(t, e["kitaita"])
                    changes.append(("fill", f"{norm(t)} -> {e['kitaita']}"))

    if "subject" in steps and clauses:
        bases = sorted(_subject_markers()[1])
        nouns = _noun_forms()
        draft_verbs = []
        for i, w in enumerate(words):
            t = norm(w)
            # known nouns and -nyi locatives (charonyi 'on the journey') are not verbs
            if len(t) >= 4 and t not in nouns and not t.endswith("nyi"):
                m = {b: hit for b in bases if (hit := verb_prefix(t, b))}
                if m:
                    draft_verbs.append((i, m))
        if len(draft_verbs) != len(clauses):
            notes.append(f"subject: not run ({len(draft_verbs)} verb(s) in the draft, {len(clauses)} in the source)")
        else:
            for (i, m), c in zip(draft_verbs, clauses):
                target_class, why = resolve_subject_class(c.get("subject"), c.get("subject_kitaita"), c.get("swahili_sm"))
                if not target_class:
                    notes.append(f"subject: {norm(words[i])} left as is ({c.get('subject')}: {why})")
                    continue
                if target_class in m or target_class not in _subject_markers()[1]:
                    continue  # the draft already agrees, or no surface data for that class
                surface, tam = next(iter(m.values()))
                if (surface, tam) in _subject_markers()[1][target_class]:
                    continue  # this surface is also used by the subject's class: leave it
                target = _subject_markers()[0].get((target_class, tam))
                if not target:
                    notes.append(f"subject: no attested {target_class}- form before -{tam or 'present'}- "
                                 f"for {norm(words[i])}; left as is")
                    continue
                core = re.sub(r"\W+$", "", words[i])
                new = _replace_prefix(core, len(surface), target) + words[i][len(core):]
                changes.append(("subject", f"{norm(words[i])} -> {norm(new)} (subject {c.get('subject')}: "
                                           f"class {target_class}- from {why}, verb {c.get('verb')})"))
                words[i] = new
                edited.add(i)

    if "subject_probe" in steps and draft_subjects is not None and chosen == 0:
        # subjects found by the relation probes on the Kidaw'ida draft itself
        # (draft_subjects: verb word index -> subject word index, computed on
        # nbest[0] by the caller): no pairing of draft and source verbs
        _, verbs = draft_candidates(words)
        for i, m in sorted(verbs.items()):
            s_idx = draft_subjects.get(i)
            if s_idx is None:
                notes.append(f"subject_probe: {norm(words[i])}: the probes found no subject")
                continue
            target_class, why = _class_of_draft_noun(norm(words[s_idx]), clauses)
            if not target_class:
                notes.append(f"subject_probe: {norm(words[i])} left as is ({why})")
                continue
            if target_class in m or target_class not in _subject_markers()[1]:
                continue
            surface, tam = next(iter(m.values()))
            if (surface, tam) in _subject_markers()[1][target_class]:
                continue
            target = _subject_markers()[0].get((target_class, tam))
            if not target:
                notes.append(f"subject_probe: no attested {target_class}- form before -{tam or 'present'}- "
                             f"for {norm(words[i])}; left as is")
                continue
            core = re.sub(r"\W+$", "", words[i])
            new = _replace_prefix(core, len(surface), target) + words[i][len(core):]
            changes.append(("subject_probe", f"{norm(words[i])} -> {norm(new)} (probe subject "
                                             f"{norm(words[s_idx])}: class {target_class}- from {why})"))
            words[i] = new
            edited.add(i)

    if "subject_checked" in steps and clauses and chosen == 0:
        # 2026-11-03: subject found in the draft (probe on the draft, else the nearest
        # noun before the verb) AND confirmed as the subject of a Swahili verb in the
        # pipeline's analysis; class from the draft's own noun, number from that
        # Swahili clause; class 9/10 nouns accept i- or ri-; verbs must inflect.
        noun_idxs, verbs = draft_candidates(words)
        for i, m in sorted(verbs.items()):
            surface, tam = next(iter(m.values()))
            t = norm(words[i])
            if not inflects(t, surface, tam):
                notes.append(f"subject_checked: {t}: no other subject marker on this stem in the corpus")
                continue
            s_idx = (draft_subjects or {}).get(i)
            if s_idx is None:
                before = [n for n in noun_idxs if n < i]
                s_idx = before[-1] if before else None
            if s_idx is None:
                continue
            if s_idx > 0 and norm(words[s_idx - 1]) in _NOT_SUBJECT_AFTER:
                notes.append(f"subject_checked: {t}: {norm(words[s_idx])} follows "
                             f"{norm(words[s_idx - 1])} (possessor/agent)")
                continue
            f = norm(words[s_idx])
            c = next((c for c in clauses if f in _swahili_subject_forms(c)), None)
            if c is None:
                notes.append(f"subject_checked: {t}: {f} is not a Swahili subject in the pipeline's analysis")
                continue
            target_class, why = resolve_subject_class(c.get("subject"), f, c.get("swahili_sm"))
            if not target_class:
                notes.append(f"subject_checked: {t} left as is ({why})")
                continue
            allowed = {target_class}
            if target_class in ("i", "ri") and not f.startswith("mi"):
                allowed = {"i", "ri"}  # class 9/10: one form for both numbers
            if allowed & set(m) or target_class not in _subject_markers()[1]:
                continue
            if (surface, tam) in _subject_markers()[1][target_class]:
                continue
            target = _subject_markers()[0].get((target_class, tam))
            if not target:
                notes.append(f"subject_checked: no attested {target_class}- form before -{tam or 'present'}- "
                             f"for {t}; left as is")
                continue
            core = re.sub(r"\W+$", "", words[i])
            new = _replace_prefix(core, len(surface), target) + words[i][len(core):]
            changes.append(("subject_checked", f"{t} -> {norm(new)} (subject {f} = Swahili {c.get('subject')}: "
                                               f"class {target_class}- from {why})"))
            words[i] = new
            edited.add(i)

    if ("tense" in steps or "extension" in steps) and chosen == 0:
        # 2026-11-03: the draft verb that translates a Swahili verb is found by its
        # root (kitaita_root_alignment.json), not by position.
        from kitaita_verb import attested_ending
        from swahili_verb import kitaita_roots, parse as parse_swahili_verb
        nouns = _noun_forms()
        for sw_i, sw_tok in enumerate(src.split()):
            sw = parse_swahili_verb(sw_tok)
            if not sw or sw["stem"] in ("w", "kuw", "pokuw"):
                continue
            decs = sw["decompositions"]
            # a whole stem is a separate (lexicalised) verb only if its Kidaw'ida root
            # differs from its peeled form's (piki -> dek is just pik + applicative)
            peeled = {k for r, e in decs if e for k in kitaita_roots(r)}
            decs = [(r, e) for r, e in decs if e or not (set(kitaita_roots(r)) & peeled)]
            for root, exts in decs:
                hit = _find_draft_verb(words, kitaita_roots(root),
                                       sw_i if sw["mood"] == "imperative" else None, nouns)
                if not hit:
                    continue
                i, pos, kr, readings = hit
                if i in edited:
                    break
                t = norm(words[i])
                tail = t[pos + len(kr):]
                if "extension" in steps:
                    for ext in exts:
                        if ext not in EXT_FORMS or _has_ext(tail, ext):
                            continue
                        if ext == "passive" and kr.endswith("w"):
                            continue  # -iw- 'steal': rawiwa is already passive
                        if ext == "applicative" and root in LEXICAL_APPLICATIVES:
                            continue
                        cands = [(attested_ending(kr + f + tail), f) for f in EXT_FORMS[ext]]
                        n, f = max(cands)
                        if n >= MIN_ATTESTED:
                            new = t[:pos] + kr + f + tail
                            changes.append(("extension", f"{t} -> {new} (Swahili {norm(sw_tok)}: {ext} "
                                                         f"-{f}-, {n} corpus tokens end in -{kr + f + tail})"))
                            words[i] = _rebuild(words[i], new)
                            edited.add(i)
                            t, tail = new, f + tail
                            break
                if "tense" in steps and i not in edited and sw["mood"] != "imperative":
                    r = _pick_reading(readings, sw["sm"])
                    tgt = _tense_target(sw, r, tail, kr) if r else None
                    if tgt:
                        new_prefix, new_tail, why = tgt
                        new = new_prefix + kr + new_tail
                        if new != t:
                            changes.append(("tense", f"{t} -> {new} (Swahili {norm(sw_tok)}: {why})"))
                            words[i] = _rebuild(words[i], new)
                            edited.add(i)
                break  # first decomposition found in the draft decides

    if "applicative" in steps:
        for base, attested in swahili_applicatives(src):
            hits = []
            for i, w in enumerate(words):
                if i in edited:
                    continue  # already changed by the extension step
                t = norm(w)
                for kr in _kitaita_roots(base):
                    if kr[-1] in "aeiou" and not attested:
                        continue  # the rule covers consonant-final roots only (churi- 'add' already ends in i)
                    if t in _agreement()[1]:
                        continue  # a known noun, not a verb
                    # a finite verb: subject/tense prefix + root + final, no applicative yet
                    m = re.fullmatch(rf"([a-z]+){re.escape(kr)}({'|'.join(_FINALS)})", t)
                    if m and m.group(1) != "ku":
                        hits.append((i, kr, m.group(2), "plain"))
                        break
                    # perfective (-ie/-ire/-ere, optional -ghe), also the malformed -id- seen in drafts
                    m = re.fullmatch(rf"([a-z]+){re.escape(kr)}(id)?(ie|ire|ere)(ghe)?", t)
                    # Only verified stems (lw -> lu) or a malformed -id- are changed: after a
                    # palatal root, -ere/-ire is also applicative + subjunctive (di-bony-er-e
                    # 'do for us'), so it may already carry the applicative.
                    if m and (kr in _speaker_verified()[1] or m.group(2)):
                        hits.append((i, kr, "ie" + (m.group(4) or ""), "perfective"))
                        break
            if len(hits) == 1:
                i, kr, final, kind = hits[0]
                core = re.sub(r"\W+$", "", words[i])
                trail = words[i][len(core):]
                if kind == "perfective":
                    stem = kr if len(base) == 1 else applicative_perfective(kr)
                    m = re.fullmatch(rf"(.*?){re.escape(kr)}(?:id)?(?:ie|ire|ere)(?:ghe)?", norm(core))
                    new_norm = m.group(1) + stem + final
                    if new_norm != norm(core):
                        # rebuild from the original prefix so its apostrophes survive
                        k, j = 0, len(m.group(1))
                        while j > 0:
                            if core[k] != "'":
                                j -= 1
                            k += 1
                        new = core[:k] + stem + final + trail
                        changes.append(("applicative", f"{norm(words[i])} -> {norm(new)} (Swahili -{base}- + applicative, "
                                        "perfective)"))
                        words[i] = new
                    continue
                stem = attested or (kr + "i" if len(base) == 1 else applicative_stem(kr))
                tail = kr + final
                if norm(core).endswith(tail) and core.lower().replace("'", "").endswith(tail):
                    # edit only the end of the original token, so apostrophes in the prefix survive
                    k = len(core)
                    j = len(tail)
                    while j > 0:  # walk back over len(tail) letters, skipping apostrophes
                        k -= 1
                        if core[k] != "'":
                            j -= 1
                    new = core[:k] + stem + final + trail
                    changes.append(("applicative", f"{norm(words[i])} -> {norm(new)} (Swahili -{base}- + applicative"
                                    + (", attested root)" if attested else ", rule)")))
                    words[i] = new

    if "unknown" in steps:
        # re-flag the current words: a verb the applicative step repaired is no longer unknown
        reflag = flag(src, " ".join(words))
        if len(reflag) == len(words):
            tokens = reflag
        # words the rule book itself edited are never sent back to Swahili
        unknown = [i for i, (_, c) in enumerate(tokens) if c == "unknown" and i not in edited]
        if len(unknown) == 1:
            others = [norm(w) for j, w in enumerate(words) if j != unknown[0]]
            src_words = [norm(t) for t in src.split() if norm(t)]
            free = [s for s in dict.fromkeys(src_words) if s not in _covered_source_words(src, others)]
            if len(free) == 1:
                i = unknown[0]
                changes.append(("unknown", f"{norm(words[i])} -> {free[0]} (Swahili)"))
                words[i] = _swap(words[i], free[0])

    if "repeat" in steps:
        kept = [w for w, (_, c) in zip(words, tokens) if c != "repeat"]
        if len(kept) < len(words):
            changes.append(("repeat", f"dropped {len(words) - len(kept)} repeated token(s)"))
        words = kept

    edited = any(k != "rerank" for k, _ in changes)
    # Re-joining tokens can alter spacing (e.g. "a,b" -> "a, b"); when no word was
    # edited, return the chosen candidate exactly as the model wrote it.
    text = " ".join(words) if edited else hyp
    return {"kitaita": text, "draft": nbest[0], "chosen_candidate": chosen, "changes": changes, "notes": notes}

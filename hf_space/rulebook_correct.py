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
  applicative -- the Swahili source has an applicative verb (walipigania 'fought
               for', base pigan) but the draft's verb on that base's Kidaw'ida
               root has none (w'edelwa 'fought'): add it (w'edeluya), using the
               attested applicative root if round 1 lists one, else the corpus rule
               (root-final ny/nj/s/sh/z/ch/j -> -ir-/-er-, otherwise -i-; lw -> luy).
               Fires only when exactly one draft word matches. User report 2026-11-02.

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
from rulebook_flagger import flag, norm

_DATA = Path(__file__).parent / "kitaita_agreement_data.json"
_ROOTS = Path(__file__).parent / "kitaita_verb_roots.json"
ALL_STEPS = ("rerank", "fill", "unknown", "repeat", "applicative")

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


def correct(src: str, nbest: list[str], steps=ALL_STEPS) -> dict:
    """nbest: the model's candidates, best first. Returns the corrected text and
    a list of what changed."""
    changes = []
    chosen = 0
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

    if "unknown" in steps:
        unknown = [i for i, (_, c) in enumerate(tokens) if c == "unknown"]
        if len(unknown) == 1:
            others = [norm(w) for j, w in enumerate(words) if j != unknown[0]]
            src_words = [norm(t) for t in src.split() if norm(t)]
            free = [s for s in dict.fromkeys(src_words) if s not in _covered_source_words(src, others)]
            if len(free) == 1:
                i = unknown[0]
                changes.append(("unknown", f"{norm(words[i])} -> {free[0]} (Swahili)"))
                words[i] = _swap(words[i], free[0])

    if "applicative" in steps:
        for base, attested in swahili_applicatives(src):
            hits = []
            for i, w in enumerate(words):
                t = norm(w)
                for kr in _verb_roots()[base]:
                    if kr[-1] in "aeiou" and not attested:
                        continue  # the rule covers consonant-final roots only (churi- 'add' already ends in i)
                    # a finite verb: subject/tense prefix + root + final, no applicative yet
                    m = re.fullmatch(rf"([a-z]+){re.escape(kr)}({'|'.join(_FINALS)})", t)
                    if m and m.group(1) != "ku" and t not in _agreement()[1]:  # skip known nouns
                        hits.append((i, kr, m.group(2)))
                        break
            if len(hits) == 1:
                i, kr, final = hits[0]
                stem = attested or applicative_stem(kr)
                core = re.sub(r"\W+$", "", words[i])
                trail = words[i][len(core):]
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

    if "repeat" in steps:
        kept = [w for w, (_, c) in zip(words, tokens) if c != "repeat"]
        if len(kept) < len(words):
            changes.append(("repeat", f"dropped {len(words) - len(kept)} repeated token(s)"))
        words = kept

    edited = any(k != "rerank" for k, _ in changes)
    # Re-joining tokens can alter spacing (e.g. "a,b" -> "a, b"); when no word was
    # edited, return the chosen candidate exactly as the model wrote it.
    text = " ".join(words) if edited else hyp
    return {"kitaita": text, "draft": nbest[0], "chosen_candidate": chosen, "changes": changes}

"""
bootstrap_lexicon.py

2026-11-11. User request: lower the acceptance bar, apply the rules recursively and
move from known to unknown words (not Dice alone), with agents interpreting
sentences from the known lemmas, until every word in both corpora has a meaning
hypothesis. This script is the statistical part; agent proposals come back in
through --agent-files and are re-checked on the corpus here.

Clean train only (rulebook_data/train.jsonl; dev and excluded splits never read).

1. Lemmas. Each word is reduced to a stem with the rule book's affixes
   (Kidaw'ida: NEG / SM series / TAM / OM prefixes, finals -agha/-ie/-ire/-ere/-a/-e
   + -ghe, noun prefixes; Swahili: swahili_verb.parse, else the same kind of
   stripping). Among a word's possible stems the one shared by most word types
   wins. Words of <= 3 letters are their own lemma (function words).
2. Seeds (priority order): speaker-verified nouns and roots; concord / function
   words from rules.json (possessives, demonstratives, associative); agent
   proposals (--agent-files); strings identical on both sides in >= 2 pairs
   (names, loans).
3. Iterate: in every pair, a Swahili lemma is "explained" if a known translation
   of it is in the Kidaw'ida sentence (both tokens marked). The unexplained rest
   votes: 1 unknown on each side = elimination (weight 1); up to MAX_RESIDUAL on
   each side = weight 1/(|U_sw| * |U_dav|). Orthographic similarity adds a small
   bonus. A candidate is
     accepted   elimination >= 2, or votes >= 2 and mutual best
     tentative  elimination >= 1 or votes >= 1, and best for its Swahili lemma
   Accepted and mutual-best tentative pairs become known for the next
   iteration (known -> unknown). Stop when an iteration adds nothing.
Outputs: rulebook/bootstrap_lexicon.json (word pairs, tiers, evidence counts,
iteration learned, pair IDs; no sentences) and coverage per iteration.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "hf_space"))
import attest as A  # noqa: E402
import kitaita_verified_corrections as V  # noqa: E402
from kitaita_verb import SERIES, EXTRA_FORMS, OMS  # noqa: E402
from rulebook_flagger import norm  # noqa: E402

OUT = HERE / "bootstrap_lexicon.json"
MAX_RESIDUAL = 4
ORTHO_BONUS = 0.5  # added to votes when difflib ratio >= 0.75 (loans, names, shared Bantu roots)
MIN_STEM = 1        # 3 stalled the loop (no function words -> residuals too large); Dice floors guard instead
TOP_FUNCTION, FUNCTION_MIN_DICE = 150, 0.2
AGENT_MIN_COOC = 3
MIN_DICE = {"accepted": 0.05, "tentative": 0.02}  # whole-sentence Dice floor (not the only criterion)

DAV_NEG = ("nde", "si")
DAV_SM = sorted({f for r in SERIES for v in range(3) for f in (SERIES[r][v], *EXTRA_FORMS.get((r, v), ()))},
                key=len, reverse=True)
DAV_TAM = ("chaa", "cha", "sake", "seke", "nda", "re", "de", "ka", "ki", "we", "da", "se", "wia")
DAV_OM = tuple(sorted(OMS, key=len, reverse=True))
# passive perfective -ilo/-elo/-iro/-ero(-ghe), locative / plural imperative -(e)nyi: agent round 1
DAV_FINAL = ("iloghe", "eloghe", "iroghe", "eroghe", "ieghe", "ireghe", "ereghe", "eghe", "agha", "enyi", "nyi",
             "ilo", "elo", "iro", "ero", "ire", "ere", "ie", "a", "e")
DAV_NOUN = ("mw", "mu", "wa", "wu", "ki", "ch", "vi", "vy", "ma", "mi", "lu", "ku", "m", "i", "w")
SW_NOUN = ("mw", "wa", "ki", "ch", "vi", "vy", "ma", "mi", "ji", "ku", "m", "u")
SW_FINAL = ("a", "e", "i")


def _strip_prefixes(w: str, chains) -> set[str]:
    """All results of stripping an optional prefix from each slot in order."""
    out = {w}
    for slot in chains:
        nxt = set()
        for x in out:
            nxt.add(x)
            for p in slot:
                if x.startswith(p) and len(x) - len(p) >= 2:
                    nxt.add(x[len(p):])
        out = nxt
    return out


def _strip_final(w: str, finals) -> set[str]:
    out = {w}
    for f in finals:
        if w.endswith(f) and len(w) - len(f) >= 2:
            out.add(w[: -len(f)])
    return out


@lru_cache(maxsize=None)
def dav_stems(w: str) -> frozenset:
    if len(w) <= 3:
        return frozenset({w})
    out = set()
    for x in _strip_prefixes(w, (DAV_NEG, DAV_SM, DAV_TAM, DAV_OM)) | _strip_prefixes(w, (DAV_NOUN,)):
        out |= _strip_final(x, DAV_FINAL)
    # undo m + w/b -> mmb after an object or subject m- (ummbanga = u-m-wang-a; agent round 3)
    for x in list(out):
        if x.startswith("mb") and len(x) >= 4:
            out |= {"w" + x[2:], "b" + x[2:]}
    # stems of 1-3 letters collected glosses of unrelated words (ka, da, ndo: agent round 1)
    # a stem that is only an ending (eghe, agha, ire: werelueghe -> eghe, agent round 2) is not a lemma
    return frozenset(s for s in out if (len(s) >= MIN_DAV_STEM and s not in DAV_FINAL) or s == w)


@lru_cache(maxsize=None)
def sw_stems(w: str) -> frozenset:
    """Swahili: the verb parser's root, else the word itself. Noun-prefix stripping was
    dropped after agent round 1: it merged unrelated words (chumba / umba, majani / kijana)."""
    if len(w) <= 3:
        return frozenset({w})
    p = A.sw_parse(w)
    if p and len(p["root"]) >= 3:
        return frozenset({p["root"], w})
    return frozenset({w})


MIN_DAV_STEM = 4  # a stripped Kidaw'ida stem must keep >= 4 letters (else the word stays whole)
FREQ_KEEP = 300   # frequent short words (<= 5 letters) keep their own form (function words: mara, huku, ndee)


def choose_lemmas(words, stems_fn, freq: collections.Counter | None = None, min_len: int = 4,
                  boost: collections.Counter | None = None) -> dict[str, str]:
    """Each word -> the candidate stem shared by most word types (ties: longer stem).
    Frequent words and words shorter than min_len are their own lemma."""
    support = collections.Counter()
    for w in words:
        support.update(stems_fn(w))
    for l in (boost or {}):
        if l in support:
            support[l] += AGENT_BOOST
    out = {}
    for w in words:
        if (freq and freq[w] >= FREQ_KEEP and len(w) <= 5) or len(w) < min_len:
            out[w] = w
            continue
        c = stems_fn(w)
        best = max(c, key=lambda s: (support[s] if s != w else 1, len(s)))
        out[w] = best if support[best] >= 2 else w
    return out


@lru_cache(maxsize=1)
def names() -> set[str]:
    """Words written capitalised in mid-sentence on both sides of some pair: proper names,
    kept whole (ramothi was stripped to mothi: agent round 1)."""
    import re as _re
    out = set()
    for r in A.pairs():
        caps = lambda t: {norm(w) for w in t.split()[1:] if _re.match(r"[A-Z]", w)}  # noqa: E731
        out |= caps(r["swa"]) & caps(r["dav"])
    return out


AGENT_DIR = Path.home() / "languagemodels" / "rulebook_data" / "agents"
AGENT_BOOST = 10   # agent-given roots count as this many supporting word types


def _cut(lemma: str) -> str:
    """Agent lemmas carry the final vowel (kunda); the stripper removes it (kund)."""
    lemma = norm(lemma)
    return lemma[:-1] if len(lemma) >= 4 and lemma[-1] in "ae" else lemma


@lru_cache(maxsize=1)
def agent_lemmas() -> tuple[dict, dict]:
    """word -> root from agent proposals (high/medium confidence), when the root is inside the word.
    Majority per word. Agent round 2: verb lemmas kept SM/TAM (washoma), nouns were over-cut (kate)."""
    votes = {"sw": collections.defaultdict(collections.Counter), "dav": collections.defaultdict(collections.Counter)}
    for f in sorted(AGENT_DIR.glob("round*/out_agent*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") != "proposal" or o.get("confidence") not in ("high", "medium"):
                continue
            for side, wk, lk in (("sw", "sw_word", "sw_lemma"), ("dav", "dav_word", "dav_lemma")):
                w, l = o.get(wk), o.get(lk)
                if not w or not l or " " in str(w).strip() or " " in str(l).strip():
                    continue
                w, l = norm(w), _cut(l)
                if len(l) >= 2 and l in w:
                    votes[side][w][l] += 1
    return ({w: c.most_common(1)[0][0] for w, c in votes["sw"].items()},
            {w: c.most_common(1)[0][0] for w, c in votes["dav"].items()})


@lru_cache(maxsize=1)
def lemmatizers():
    ps = A.pairs()
    fs = collections.Counter(w for r in ps for w in r["sw"])
    fd = collections.Counter(w for r in ps for w in r["dv"])
    asw, adv = agent_lemmas()
    sw = choose_lemmas(set(fs), sw_stems, fs, min_len=5, boost=collections.Counter(asw.values()))
    dv = choose_lemmas(set(fd), dav_stems, fd, min_len=4, boost=collections.Counter(adv.values()))
    sw.update({w: l for w, l in asw.items() if w in sw})
    dv.update({w: l for w, l in adv.items() if w in dv})
    for n in names():
        sw[n] = n
        dv[n] = n
    return sw, dv


def seeds(sw_lem, dv_lem) -> dict[tuple[str, str], dict]:
    s = {}

    def put(sw, dv, source):
        sw, dv = norm(sw), norm(dv)
        if not sw or not dv:
            return
        ls, ld = sw_lem.get(sw, sw), dv_lem.get(dv, dv)
        s.setdefault((ls, ld), {"source": source, "sw_word": sw, "dav_word": dv})

    for sw, dv in V.VERIFIED_NOUN_TRANSLATIONS.items():
        if isinstance(dv, str):
            put(sw, dv, "speaker_verified")
    for sw, dv in V.VERIFIED_ROOTS.items():
        s.setdefault((sw, dv), {"source": "speaker_verified", "sw_word": sw, "dav_word": dv})
    rules = json.loads((HERE / "rules.json").read_text(encoding="utf-8"))["rules"]
    sw_poss = {"1sg": "angu", "2sg": "ako", "3sg": "ake", "1pl": "etu", "2pl": "enu", "3pl": "ao"}
    sw_agr = {"u": "w", "wi": "w", "ghu": "w", "i": "y", "i_nasal": "y", "ji": "l", "ghi": "y", "chi": "ch",
              "va": "vy", "ri": "z", "lu": "w", "ku": "kw"}
    for r in rules:
        if r["id"].startswith("POSS.") and r["status"] == "active" and "forms" in r["then"]:
            lab = r["id"].split(".")[1]
            for (p, cue), f in zip(sw_poss.items(), r["then"]["forms"].values()):
                if lab in sw_agr and r["evidence"]["by_form"][f]["attested"]:
                    put(sw_agr[lab] + cue, f, "rules.json:" + r["id"])
    return s


def read_agent_files(agent_files) -> tuple[list[dict], set[tuple[str, str]]]:
    """Agent proposals (sw_word, dav_word, meaning) and 'wrong' verdicts (sw_lemma, dav_lemma)."""
    props, wrong = [], set()
    for f in agent_files:
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") == "review" and o.get("sw_lemma") and o.get("dav_lemma"):
                if o.get("verdict") == "wrong":
                    wrong.add((o["sw_lemma"], o["dav_lemma"]))
                    if o.get("correct_dav_lemma"):
                        props.append({"sw_word": o["sw_lemma"], "dav_word": o["correct_dav_lemma"],
                                      "meaning": o.get("meaning"), "confidence": "medium",
                                      "agent_file": Path(f).name})
                elif o.get("verdict") in ("correct", "partly"):
                    props.append({"sw_word": o["sw_lemma"], "dav_word": o["dav_lemma"], "meaning": o.get("meaning"),
                                  "confidence": "medium" if o["verdict"] == "partly" else "high",
                                  "agent_file": Path(f).name})
            elif o.get("type") == "proposal" and o.get("sw_word") and o.get("dav_word"):
                props.append({**o, "agent_file": Path(f).name})
    return props, wrong


SW_AGR = ("w", "y", "l", "ch", "vy", "z", "kw", "p", "mw")
SW_DEM = ("huyu hawa huu hii hili haya hiki hivi hizi huku hapa humu yule wale ule ile lile yale kile vile "
          "zile kule pale mle huyo hao huo hiyo hilo hayo hicho hivyo hizo huko hapo humo").split()


@lru_cache(maxsize=1)
def concord_words() -> tuple[frozenset, frozenset]:
    """Concord words (possessives, -ote, demonstratives, associative): their form comes from the
    rule book's class tables, so the dictionary loop does not pair them as vocabulary (agent round 1:
    zako = wako, vile = seji). They count as explained and are left out of the residuals."""
    sw = set(SW_DEM)
    for a in SW_AGR:
        sw |= {a + x for x in ("angu", "ako", "ake", "etu", "enu", "ao", "ote", "a")}
    import build_rules as R  # rules' own concord forms (POSS / ASSOC / -ose / DEM tables)
    dv = set(R.STOP)
    for row in R.SERIES:
        for f in R.forms(row, 0) + R.forms(row, 2):
            dv.add(f + "ko")          # existential / copula SM+ko (weko, neko, gheko): agent round 1
        dv.add(R.forms(row, 1)[0] + "ro")  # non-human 'its' SM+a-ro (varo, charo): agent round 1
    return frozenset(sw), frozenset(dv)


def run(agent_files: list[Path], max_iter: int = 60, hide: set[str] = frozenset(), quiet: bool = False,
        blocked: set = frozenset()):
    ps = A.pairs()
    sw_words = {w for r in ps for w in r["sw"]}
    dv_words = {w for r in ps for w in r["dv"]}
    sw_lem, dv_lem = lemmatizers()
    S = [[sw_lem[w] for w in r["sw"]] for r in ps]
    D = [[dv_lem[w] for w in r["dv"]] for r in ps]
    known: dict[str, dict[str, dict]] = collections.defaultdict(dict)   # sw lemma -> dav lemma -> info
    for (ls, ld), info in seeds(sw_lem, dv_lem).items():
        if info["source"] == "speaker_verified" and ls in hide:
            continue
        known[ls][ld] = {**info, "tier": "seed", "iteration": 0}
    props, wrong = read_agent_files(agent_files)
    blocked = set(blocked) | wrong
    pending = []
    for o in props:
        for sw_w, dv_w in zip(str(o["sw_word"]).split(), str(o["dav_word"]).split()) \
                if len(str(o["sw_word"]).split()) == len(str(o["dav_word"]).split()) else [(o["sw_word"], o["dav_word"])]:
            sw, dv = norm(sw_w), norm(dv_w)
            if not sw or not dv:
                continue
            ls, ld = sw_lem.get(sw, sw), dv_lem.get(dv, dv)
            if (ls, ld) in blocked or ls in hide:   # hidden seeds stay hidden from agent proposals too
                continue
            pending.append((ls, ld, {"source": "agent_interpretation", "sw_word": sw, "dav_word": dv,
                                     "meaning": o.get("meaning"), "confidence": o.get("confidence"),
                                     "tier": "agent", "iteration": 0, "agent_file": o["agent_file"]}))
    # identical strings on both sides (names, loans)
    same = collections.Counter()
    for s, d in zip(S, D):
        same.update(set(s) & set(d))
    for l, c in same.items():
        if c >= 2 and len(l) >= 3:
            known[l].setdefault(l, {"source": "identical_string", "tier": "seed", "iteration": 0, "pairs": c})
    n_sw_pairs = collections.Counter(l for s in S for l in set(s))
    n_dv_pairs = collections.Counter(l for d in D for l in set(d))
    top_sw = {l for l, _ in n_sw_pairs.most_common(TOP_FUNCTION)}
    top_dv = {l for l, _ in n_dv_pairs.most_common(TOP_FUNCTION)}
    idx_sw, idx_dv = collections.defaultdict(set), collections.defaultdict(set)
    for i, (s, d) in enumerate(zip(S, D)):
        for l in s:
            idx_sw[l].add(i)
        for l in d:
            idx_dv[l].add(i)
    # agent proposals enter only with corpus support (round 2 audit: 8/24 agent entries wrong):
    # not low confidence, and the two lemmas co-occur in >= 2 pairs (>= 1 if high confidence).
    # The rest are kept in agent_unverified, not used to explain sentences.
    csw, cdv = concord_words()
    unverified = []
    for ls, ld, info in pending:
        c = len(idx_sw[ls] & idx_dv[ld])
        conf = info.get("confidence")
        if ls in csw or ld in cdv:
            continue
        # round 3: gated agent entries were still wrong 8/39 -> medium needs >= 3 shared pairs
        if (conf == "high" and c >= 1) or (conf == "medium" and c >= AGENT_MIN_COOC):
            known[ls].setdefault(ld, {**info, "agent_cooc": c})
        else:
            unverified.append({"sw_lemma": ls, "dav_lemma": ld, **info, "agent_cooc": c})
    run.unverified = unverified
    history = []
    for it in range(1, max_iter + 1):
        votes = collections.defaultdict(float)
        elim = collections.Counter()
        ids = collections.defaultdict(list)
        explained_sw = explained_dv = 0
        csw, cdv = concord_words()
        for i, (s, d) in enumerate(zip(S, D)):
            dset = {x for x in d if x not in cdv}
            us, ud_used = set(), set()
            for l in {x for x in s if x not in csw}:
                hit = [m for m in known.get(l, {}) if m in dset]
                if hit:
                    ud_used.update(hit)
                else:
                    us.add(l)
            ud = dset - ud_used
            explained_sw += len(set(s)) - len(us)
            explained_dv += len(ud_used)
            if us and ud and len(us) <= MAX_RESIDUAL and len(ud) <= MAX_RESIDUAL:
                w = 1 / (len(us) * len(ud))
                for a in us:
                    for b in ud:
                        votes[(a, b)] += w
                        if len(ids[(a, b)]) < 3:
                            ids[(a, b)].append(ps[i]["id"])
                if len(us) == 1 and len(ud) == 1:
                    elim[(next(iter(us)), next(iter(ud)))] += 1
        for (a, b) in list(votes):
            if difflib.SequenceMatcher(None, a, b).ratio() >= 0.75:
                votes[(a, b)] += ORTHO_BONUS
        best_sw, best_dv = {}, {}
        for (a, b), v in votes.items():
            if v > best_sw.get(a, (None, 0))[1]:
                best_sw[a] = (b, v)
            if v > best_dv.get(b, (None, 0))[1]:
                best_dv[b] = (a, v)
        added = collections.Counter()
        for (a, b), v in votes.items():
            if a in known and b in known[a]:
                continue
            if len(a) < MIN_STEM or len(b) < MIN_STEM or (a, b) in blocked:
                continue
            if len(a) >= 4 and len(b) <= 2:   # content word -> 2-letter fragment (fik = ka, pit = da)
                continue
            e = elim[(a, b)]
            is_best = best_sw[a][0] == b
            mutual = is_best and best_dv[b][0] == a
            # whole-sentence Dice: a pairing with a word found in thousands of sentences
            # (tunasakata = na) cannot pass on a few leftover votes
            dice = 2 * len(idx_sw[a] & idx_dv[b]) / (n_sw_pairs[a] + n_dv_pairs[b])
            tier = None
            # function-word guard (agent round 2: na = maumivu, kwa = jicho): pairs touching the most
            # frequent lemmas need strong whole-sentence association
            if (a in top_sw or b in top_dv) and dice < FUNCTION_MIN_DICE:
                continue
            if (e >= 2 or (v >= 2 and mutual)) and dice >= MIN_DICE["accepted"]:
                tier = "accepted"
            elif (e >= 1 or v >= 1) and is_best and dice >= MIN_DICE["tentative"]:
                tier = "tentative"
            if tier and (tier == "accepted" or mutual):
                known[a][b] = {"source": "bootstrap", "tier": tier, "iteration": it, "elimination": e,
                               "votes": round(v, 2), "dice": round(dice, 3), "pair_ids": ids[(a, b)]}
                added[tier] += 1
        cov = coverage(S, D, known)
        history.append({"iteration": it, "added": dict(added), "explained_sw_lemma_tokens": explained_sw,
                        "explained_dav_lemma_tokens": explained_dv, **cov})
        if not quiet:
            print(json.dumps(history[-1]), flush=True)
        if not added:
            break
    # support of every non-seed entry: share of the Swahili lemma's sentences that contain
    # the Kidaw'ida lemma, and the other way round
    for a, cands in known.items():
        for b, info in cands.items():
            if info["tier"] in ("accepted", "tentative", "agent"):
                c = len(idx_sw[a] & idx_dv[b])
                info["cooc"] = c
                info["support_sw"] = round(c / n_sw_pairs[a], 3) if n_sw_pairs[a] else 0.0
                info["support_dav"] = round(c / n_dv_pairs[b], 3) if n_dv_pairs[b] else 0.0
                info["n_sw"], info["n_dav"] = n_sw_pairs[a], n_dv_pairs[b]
    return sw_lem, dv_lem, known, history, S, D


ERR_MIN_N, ERR_MAX_SUPPORT = 4, 0.1  # 0.2 over-blocked: agents judged 42/50 blocks right word family (round 1)


def suspicious(known) -> set[tuple[str, str]]:
    """Bootstrap entries whose words rarely occur together: the Swahili lemma occurs in >= ERR_MIN_N
    sentences and fewer than ERR_MAX_SUPPORT of them contain the Kidaw'ida lemma, and vice versa."""
    out = set()
    for a, cands in known.items():
        for b, i in cands.items():
            if (i.get("source") == "bootstrap" or (i.get("source") == "agent_interpretation"
                                                    and i.get("confidence") != "high")) and i["n_sw"] >= ERR_MIN_N and \
                    i["support_sw"] < ERR_MAX_SUPPORT and i["support_dav"] < ERR_MAX_SUPPORT:
                out.add((a, b))
    return out


def run_corrected(agent_files, hide=frozenset(), quiet=False, max_rounds=8):
    """Run to convergence, block entries that fail the support check, re-run; repeat until no new errors."""
    blocked, rounds = set(), []
    for rnd in range(1, max_rounds + 1):
        out = run(agent_files, hide=hide, quiet=True, blocked=blocked)
        bad = suspicious(out[2]) - blocked
        rounds.append({"round": rnd, "new_errors_blocked": len(bad), **out[3][-1]})
        if not quiet:
            print("correction round", json.dumps(rounds[-1]), flush=True)
        if not bad:
            return (*out, blocked, rounds)
        blocked |= bad
    return (*out, blocked, rounds)


def coverage(S, D, known) -> dict:
    """Tokens covered by the lexicon or by the rules' concord words (possessives, demonstratives, SM+ko)."""
    csw, cdv = concord_words()
    dv_known = {m for v in known.values() for m in v} | set(cdv)
    known = {**known, **{w: {} for w in csw}}
    sw_types = collections.Counter(l for s in S for l in s)
    dv_types = collections.Counter(l for d in D for l in d)
    sw_cov = sum(c for l, c in sw_types.items() if l in known)
    dv_cov = sum(c for l, c in dv_types.items() if l in dv_known)
    return {"sw_lemma_types_known": sum(1 for l in sw_types if l in known), "sw_lemma_types": len(sw_types),
            "sw_token_coverage": round(sw_cov / sum(sw_types.values()), 4),
            "dav_lemma_types_known": sum(1 for l in dv_types if l in dv_known), "dav_lemma_types": len(dv_types),
            "dav_token_coverage": round(dv_cov / sum(dv_types.values()), 4)}


def same_lemma(a: str, b: str) -> bool:
    a, b = a.replace("'", ""), b.replace("'", "")
    # stems are cut differently on the two runs (merea / meri, se / sea): count as the same
    # word if one contains the other or the strings are >= 0.6 similar
    return a == b or (min(len(a), len(b)) >= 2 and (a in b or b in a)) or \
        difflib.SequenceMatcher(None, a, b).ratio() >= 0.6


def heldout_check(agent_files) -> dict:
    """Precision/recall on speaker-verified seeds hidden from the run (2 folds)."""
    ps = A.pairs()
    sw_lem, dv_lem = lemmatizers()
    gold = collections.defaultdict(set)
    for (ls, ld), info in seeds(sw_lem, dv_lem).items():
        if info["source"] == "speaker_verified":
            gold[ls].add(ld)
    keys = sorted(gold)
    res = {"accepted": [0, 0], "tentative": [0, 0], "n_hidden": len(keys), "found": 0, "wrong_examples": []}
    for fold in (0, 1):
        hide = set(keys[fold::2])
        known = run_corrected(agent_files, hide=hide, quiet=True)[2]
        for ls in hide:
            got = [(ld, i) for ld, i in known.get(ls, {}).items() if i.get("source") == "bootstrap"]
            if not got:
                continue
            res["found"] += 1
            ld, info = max(got, key=lambda x: x[1]["votes"])
            ok = any(same_lemma(ld, g) for g in gold[ls])
            res[info["tier"]][0] += ok
            res[info["tier"]][1] += 1
            if not ok and len(res["wrong_examples"]) < 15:
                res["wrong_examples"].append(f"{ls}: got {ld}, verified {sorted(gold[ls])}")
    from api.stats import rate_ci  # noqa: E402
    for t in ("accepted", "tentative"):
        k, n = res[t]
        res[t] = {"correct": k, "n": n, "precision_ci": rate_ci(k, n) if n else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-files", nargs="*", type=Path, default=[])
    ap.add_argument("--no-heldout", action="store_true")
    a = ap.parse_args()
    sw_lem, dv_lem, known, history, S, D, blocked, rounds = run_corrected(a.agent_files)
    held = None if a.no_heldout else heldout_check(a.agent_files)
    print("heldout:", json.dumps(held, ensure_ascii=False))
    entries = []
    for ls, cands in known.items():
        for ld, info in cands.items():
            entries.append({"sw_lemma": ls, "dav_lemma": ld, **info})
    OUT.write_text(json.dumps({
        "_meta": {"built_by": "rulebook/bootstrap_lexicon.py", "train_sha16": A.sha16(A.TRAIN),
                  "script_sha16": A.sha16(Path(__file__)), "rules_sha16": A.sha16(HERE / "rules.json"),
                  "agent_files": {str(f): A.sha16(f) for f in a.agent_files},
                  "config": {"max_residual": MAX_RESIDUAL, "ortho_bonus": ORTHO_BONUS,
                             "accepted": "elimination >= 2, or votes >= 2 and mutual best",
                             "tentative": "elimination >= 1 or votes >= 1, best for its Swahili lemma "
                                          "(only mutual-best tentative pairs are fed back)"},
                  "history": history, "correction_rounds": rounds, "heldout_seed_check": held,
                  "blocked_as_errors": sorted(f"{x}={y}" for x, y in blocked)},
        "lemmas": {"sw": {w: l for w, l in sw_lem.items() if w != l},
                   "dav": {w: l for w, l in dv_lem.items() if w != l}},
        "entries": entries, "agent_unverified": getattr(run, "unverified", [])}, ensure_ascii=False, indent=0))
    print(f"{len(entries)} entries -> {OUT}")


if __name__ == "__main__":
    main()

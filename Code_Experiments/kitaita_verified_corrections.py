# -*- coding: utf-8 -*-
"""
kitaita_verified_corrections.py

2026-10-07. Native-speaker ground truth from the first full round of
the Kitaita Sentence Judge (22/22 sentences judged; see docs/
status.md for the full per-sentence analysis). These overrides take
PRIORITY over the statistical candidates everywhere kitaita_sentence_
generator.py looks one up -- this is direct human correction, not a
corpus co-occurrence guess, so it is applied even at n=1 (unlike the
corpus-statistics side of this project, which needs n>=8) -- except
where two corrections genuinely conflicted, which is flagged instead
of merged (see CONFLICTS below).

Every entry's confirmation count is in its comment so a future round
can tell a well-supported override from a single, not-yet-corroborated
one.
"""
from __future__ import annotations

import json as _json
from pathlib import Path as _Path

# ----------------------------------------------------------------- verb roots
# swahili_root -> kitaita_root, overriding root_kitaita/root_kitaita_fallback.
VERIFIED_ROOTS = {
    "on": "won",      # "see" -- root-level alignment had NO candidate at all
                       # (statistical dilution by 'kuw'); confirmed 3x
                       # independently (mungu/mtu/samaki, s1/s5/s14): "...kawona"/"...kawona".
    "simam": "simam",  # "stand" -- root-level alignment's top candidate ('imus',
                        # Dice 0.42, 360 shared verses) was WRONG; speaker confirms
                        # a direct cognate/borrowing of the Swahili root itself (s7:
                        # "Mudi ghukasimama", not "...ghukaimusa"). 1x confirmed,
                        # but unambiguous (no plausible alternative reading).
    "fik": "vik",      # "arrive" -- root-level alignment's candidate ('uw', Dice 0.25,
                        # 68 shared verses) was WRONG; speaker confirms root 'vik' (s86:
                        # "Mwalimu ukavika mbaa", not "...kukauwa..."), an f/v alternation
                        # cognate of the Swahili root itself (cf. 'simam' above). 1x
                        # confirmed; same correction also fixed mwalimu's subject marker
                        # (see VERIFIED_SUBJECT_MARKERS) in the same judgment.
}

# 2026-10-12: 12 verb-root gaps filled via the Kitaita Word Bank Artifact
# (https://claude.ai/artifact/6eozxX4HtKZhce3FTfWnY2, collection
# "contributions") -- these are COMMON Bible-corpus roots (>=3 occurrences)
# that previously had NO Kitaita candidate at all (not low-confidence,
# zero). Same trust tier as the hand-corrected entries above (direct
# submission, not a corpus statistic), not yet cross-checked by a 2nd
# independent round. NOTE: "ish" (live) submitted the SAME as the
# Swahili root itself -- plausible shared-cognate/loan, not flagged as
# an error (unlike the dogo/'ri' noun-table case) since nothing here
# contradicts it.
VERIFIED_ROOTS.update({
    "ahid": "laghir",       # promise
    "eleke": "gheenda",     # head towards / approach
    "fiki": "viki",         # arrive (at) -- cf. "fik"->"vik" above, same f/v cognate pattern
    "fuati": "nughia",      # follow
    "ish": "ish",           # live
    "maliz": "merea",       # finish
    "ongoz": "longoza",     # lead
    "reje": "wuya",         # return
    "taraji": "kund",       # expect/hope
    "tarajiw": "kunde",     # be expected (passive)
    "wasil": "vik",         # arrive -- SAME Kitaita root as "fiki"->"viki" (minus final i);
                              # both Swahili roots gloss "arrive", consistent rather than conflicting.
    "zungumzi": "dedanya",  # converse/talk -- cf. the existing "kudedanya" (converse) note
                              # elsewhere in this project re: re/de segmentation collisions.
})

# 2026-10-20: 4 more root gaps, submitted alongside the descriptive-associative
# examples below (KITAITA_ASSOCIATIVE_DESCRIPTOR_VERIFIED) -- same n=1,
# direct-submission trust tier as every other entry in this table, and
# apply project-wide (any finite verb using these roots too), not just
# inside that construction.
VERIFIED_ROOTS.update({
    "chom": "kor",     # burn/roast ("mti wa kuchoma" -> "mudi ghokora", firewood)
    "onge": "ghor",    # talk/converse ("ulimi wa kuongea" -> "lumi loghora")
    "lind": "lindi",   # guard/protect ("mbwa wa kulinda" -> "koshi elindia") -- NOTE:
                        # decompose_verb("kulinda") itself mis-segments this as
                        # tam='li'(past)+root='nd' (a real, separate bug: 'li' is
                        # both the past-TAM marker AND a legitimate class-5 object
                        # marker elsewhere, so the decomposer's general-purpose
                        # ambiguity-resolution doesn't flag it here as spurious --
                        # same root-vs-prefix collision class as 'ka'/'kabili-').
                        # This entry is keyed by the CORRECT bare root 'lind'
                        # (ku+lind+a); any caller must derive 'lind', not trust
                        # decompose_verb()'s own root_guess for this specific word.
    "chek": "sek",     # laugh ("mtu wa kucheka" -> "mundu oseka") -- ch/s correspondence,
                        # consistent with proto-Bantu *-sek- "laugh" (Swahili's own
                        # "cheka" is independently a reflex of the same proto-root).
})

# 2026-10-20 (cont.): reverse-inference pilot (docs/status.md) -- found by
# stripping concord on BOTH sides of the full parallel Bible corpus
# (31,102 verse pairs) and reading the aligned verses directly, same
# confidence tier as every hand-corrected entry above (direct confirmation,
# trusted at n=1 for a SINGLE clean example -- but both entries below have
# far more than that):
VERIFIED_ROOTS.update({
    "saidi": "tesi",  # help -- 6 independent, near-identical "help me" prayers agree
                        # EXACTLY on both sides (kunisaidia/unisaidie <-> kunitesie/
                        # kunitesia: PSA.22.19, 35.2, 38.22, 40.13, 70.1, 71.12), plus
                        # the NOUN "msaidizi" (helper) <-> "mtesia" independently
                        # confirming the same root in a DIFFERENT word class
                        # (PSA.22.11, 30.10, 72.12). 82 total "saidi"-containing
                        # Swahili tokens found project-wide.
    "fany": "bony",    # do/make -- even stronger: 116 "fanya"-containing Swahili
                        # tokens found, every one of the first 8 read directly
                        # (GEN.7.5, GEN.18.17, EXO.7.6, EXO.16.17, EXO.18.17,
                        # EXO.18.24, EXO.25.13, EXO.28.13) shows a clean "fanya"/
                        # "bonya" match regardless of subject/tense (akafanya<->
                        # ukabonya, wakafanya<->wikabonya, fanya[imperative]<->
                        # kubonye/kubonya). Found as a SIDE EFFECT of explaining
                        # "tawala" (rule) below, not one of the 3 originally
                        # reported candidates -- flagging that explicitly since it
                        # wasn't part of what was directly confirmed before adding.
})

# 2026-10-20 (cont. 3): full-corpus reverse-inference run (all 31,102 verse
# pairs, not a hand-picked few) -- extended decompose_kitaita_verb()
# (kitaita_tam_allomorphy.py) to strip the 3 person object markers found
# earlier today before computing root_guess (fixes the undercounting bug
# that same tool's own sanity check caught), then ran root-level Dice
# alignment (same method as align_roots(), same MIN_FREQ=3 convention)
# restricted to this project's own CURRENT verb-root gap list. 35 of 115
# listed gaps got some candidate; MANUALLY read the verses for every
# candidate with cooccur>=3 before adding anything (same discipline as
# the 'kusudi' retraction earlier today) -- most did NOT survive that
# check (see docs/status.md for the full table and the rejected ones:
# 'hesab'->'tal' was noun/verb-decomposer contamination -- 'tal' is really
# the NOUN 'mtalo' misread as a verb by the SM-stripping prefix match;
# 'agiz'->'laghir' and 'tegeme'->'kusaghik' did not hold up on direct
# reading at all; 'shuhudi' aligned strongly with the DIFFERENT word
# 'memu', but only as a noun sense, not this verb-root slot). Only 2
# survived:
VERIFIED_ROOTS.update({
    "oko": "kir",   # save/rescue -- confirmed 3/3 directly (PSA.80.2 "utuokoe"->
                      # "kudikire"; PSA.118.25 "utuokoe"->"kudikire"; MAT.14.30
                      # "niokoe"->"Nikire"), PLUS this independently reconfirms
                      # BOTH person object markers found earlier today in the
                      # SAME verses (di=us, ni=me -- see KITAITA_PERSON_OBJECT_
                      # MARKER_HYPOTHESIS below, n counts updated). 'kir' is NOT
                      # a generic/overloaded root either -- checked directly: only
                      # ONE other Swahili root ('fun') ever ranks it as the #1
                      # candidate project-wide, out of 2,646 roots checked.
    "ondo": "inj",  # remove (transitive sense ONLY -- "ondoka"/arise is a
                      # DIFFERENT root string, 'ondok' not 'ondo', already kept
                      # separate by the decomposer; confirmed this is genuinely
                      # two different Kitaita answers, not a missed split, same
                      # class of polysemy as 'kuwa'[existential vs copula]
                      # already documented elsewhere in this file). Confirmed
                      # directly: PRO.25.4 "Ondoa takataka" (remove the dross)
                      # -> "Kukainja wuchafu" (ku+ka+inj+a).
})

# 2026-10-20 (cont. 9): direct user bug report ("hata uharibifu uipate
# ngome" -> decompose_verb() returned None outright for 'uipate' -- a
# SEPARATE grammar-gap fix, see extract_swahili_grammar.decompose_verb()'s
# new subjunctive/suffixed-relative branches -- but once that's fixed, the
# ROOT 'pat' itself still had no translation at all). Checked 'pata'
# broadly first (not just this one sentence's sense) -- it turns out to be
# a DIRECT, repeatedly-confirmed SHARED COGNATE, retained VERBATIM in
# Kitaita in at least 6 independent verses, not translated to a different
# word at all: EXO.13.9 "ipate kuwa"->"kupate kuiwadia", LEV.14.21
# "kupata"->"kupata" and "atatwaa...awe"->"upate kubonya", EXO.31.6 "wapate
# kufanya"->"upate/wipate kubonya", JAS.4.13 "kupata faida"->"kupata
# fwaida", PRO.19.25 "atapata kutambua"->"uchapata kumanya". Most of these
# are the GRAMMATICALIZED "SM-pate-VERB" purpose-clause auxiliary ("so
# that... may...", not the literal "get/obtain" sense) -- the literal/
# physical sense in the user's own example ("uharibifu uipate ngome",
# destruction OVERTAKES the fortress) was NOT separately verse-confirmed
# this round, but since the Kitaita word is identical either way, this is
# a safe, low-risk addition either way (worst case: the SAME real Kitaita
# word, not demonstrably wrong for the untested sense).
VERIFIED_ROOTS.update({
    "pat": "pat",
})

# NOT added, found by the same run, explicitly rejected after reading the
# actual verses (kept for the record, same "log a retraction, don't
# silently drop it" discipline as everywhere else in this file):
#   'hesab'  (count/calculate) -> automated top pick 'tal', Dice 0.265,
#       cooccur 18 -- but 'tal' turned out to be the NOUN "mtalo" (count/
#       record), not a verb root at all: decompose_kitaita_verb() matched
#       its "m-" as a subject-marker prefix (a real, separate contamination
#       risk this run exposed -- the SAME SM-prefix list doesn't distinguish
#       "this word is a noun" from "this word is a verb"). The genuine
#       correspondence appears to be sense-dependent (isabu/mtalo, both
#       nouns) -- not resolved to a single verb root.
#   'agiz' (order/command) -> top pick 'laghir', Dice 0.176, cooccur 23 --
#       read both short-verse examples directly, NEITHER shows 'laghir' at
#       all. Likely contamination from 'ahid' (promise), which already,
#       correctly, uses 'laghir' -- the two concepts co-occur often in
#       covenant/legal passages without being the same word.
#   'tegeme' (depend/rely on) -> top pick 'kusaghik', Dice 0.196, cooccur 5
#       -- the 2 examples read directly give 2 DIFFERENT, inconsistent
#       Kitaita renderings ("lindia"/guard, "saghi"), not a stable match.
#   'shuhudi' (witness/testify) -> top pick 'shuhudi' (same string,
#       Dice 0.272) did not actually appear in any of the 4 verses read;
#       the REAL, highly consistent pattern in those same 4 verses is
#       "shuhuda"(testimonies) <-> "memu", 4/4 clean -- but that is the
#       NOUN sense (the content of what is witnessed), not this verb-root
#       gap's VERB sense ("to testify/bear witness" as an act) -- not
#       assumed to be the same without separate verification.
#   'himiz' (urge/press) -> no example verses found at all this round
#       (0 hits even before filtering) -- the automated candidate 'shashir'
#       is UNCHECKED, not merely rejected; flagged as literally untested.
# One promising lead, NOT added (single example, needs more before trusting
# it even at this project's own "flag, don't guess" bar):
#   'peleke' (send) -> 'dumi' (imperative "duma" <-> "duma", ACT.10.5,
#       "peleka watu Yafa" -> "duma wandu Jopa") -- exactly 1 clean example
#       found; the automated score (Dice 0.146, cooccur 6) suggests more
#       support exists in longer verses this pass's short-verse filter
#       skipped, worth a second look, not relied on yet.

# 2026-10-20 (cont. 4): verb root from the SAME run as the noun pass above
# (VERIFIED_NOUN_TRANSLATIONS['kusikia']) -- same underlying root, kept in
# the verb table too since finite forms ("anasikia"/"alisikia") go through
# a completely separate lookup than the noun one.
VERIFIED_ROOTS.update({
    "siki": "sikir",  # hear -- confirmed via the recurring "kusikia habari" (hear
                        # news) collocation, 3/3 clean (see the matching noun entry
                        # above for the citations).
})

# 2026-10-20: "kupa" (to give) -- direct user request ("pea (to give)
# verb ... Kila kikundi kitapewa pesa nyingi"). Root "p" (Swahili
# monosyllabic irregular verb) read directly in all LITERAL-giving
# verses found (not the idiomatic "give honor/glory/justice" sense,
# which this Bible consistently renders with a DIFFERENT, sense-
# specific verb instead -- "kumpa Kaisari kodi"->"kumfunyira" (render/
# pay), "kumpa Mungu utukufu"->"kumkasa" (glorify), "kupewa haki"->
# "kuwoneka"/"kupata wubaa" (be shown/grant justice) -- same "idiom
# dissolves, not a counterexample" pattern as VERIFIED_IDIOMS elsewhere
# in this file, just not yet promoted to that table since no single
# fixed object/periphrasis is obviously shared across all of them):
#   LUK.22.5 "kumpa fedha"(give him money) -> "kumneka feza"
#   REV.2.28 "nitampa ile nyota"(I will give him that star) -> "nicha-
#     mneka ija nyerinyeri"
#   JHN.4.7 "Nipe maji"(give me water) -> "Nineke machi"
#   MAT.5.42 "mpe"(give him!) -> "mneke"
#   LUK.6.30 "Mpe kila akuombaye"(give everyone who asks you) -> "Uo ose
#     ukulombagha kilambo chingi kumneke"
#   PRO.31.31 "Mpe mapato"(give him the profits -- object paraphrased as
#     "ishima"/honor, but the VERB itself still matches) -> "Mneke ishima"
# 6 independent verses, zero counterexamples for the literal sense.
VERIFIED_ROOTS.update({
    "p": "nek",
})
# "pew" (fused root+passive "-ew-", decompose_verb() never separates
# passive morphology -- see its own docstring) is kept as its OWN entry
# rather than assumed to regularly derive from "p"+passive: the only 2
# corpus examples of "kupewa" itself (ISA.43.26, LUK.23.52) were BOTH
# idiomatic ("be shown justice", "asked to be given" via "kuomba", not
# a literal passive at all), so "nekwa" is NOT directly verse-confirmed
# the way "p"->"nek" is -- it is the PRODUCTIVE Bantu passive suffix
# "-w-" applied to the confirmed active root as a single best-effort
# guess, same confidence tier as this file's "single corpus-mined data
# point, generalized" adjective entries, explicitly weaker than "p"
# itself. (The existing root_kitaita_fallback statistical table
# independently proposed the SAME "nekwa" at Dice 0.18/weak before this
# was added, which is modest independent support, not confirmation.)
VERIFIED_ROOTS.update({
    "pew": "nekw",
})

# 2026-10-20: "piga" (to hit/strike), direct user request ("piga -> kaba,
# verify with reverse inference"). Read every short akapiga/wakapiga/
# alipiga/mpige/piga-piga verse found. "piga" is extremely idiomatic in
# Swahili (piga vita=wage war, piga mishale=shoot arrows, piga kelele=
# shout, piga magoti=kneel, piga kambi=pitch camp, piga mbio=run, piga
# moyo konde=take courage) -- EVERY one of those dissolves into a
# DIFFERENT, sense-specific Kitaita verb (bonya+wuda, fumya, kema/ghora
# kwa lwaka, dunga ngokoro, bonya kambi, ghenda na iseghe, kaia na
# ngolo respectively), same "idiom dissolves, not a counterexample"
# pattern as VERIFIED_IDIOMS elsewhere. The user's "kaba" IS confirmed,
# but only for the LITERAL sense (physically strike/hit something/
# someone), 4 independent verses, zero counterexamples within that
# sense: MRK.15.19 "Wakampiga mwanzi wa kichwa"(struck him on the head
# with a reed)->"Wikamkaba na mrangi chongonyi kwake", 2SA.22.14 "alipiga
# radi"(struck/sent lightning)->"orekabieghe lukuwa", MAT.27.30
# "wakampiga-piga kichwani"(struck him repeatedly on the head)->
# "kumkaba...chongonyi", PSA.18.38 "Nitawapiga-piga"(I will strike them
# down)->"Nerewikabieghe".
VERIFIED_ROOTS.update({
    "pig": "kab",
})

# 2026-10-20: "lia" (to cry/weep), direct user request ("lia -> lila,
# verify with reverse inference"). LUK.22.62 "akalia kwa majonzi"(he
# wept bitterly)->"ukalila nandighi" -- 1 direct verb-form confirmation,
# but independently corroborated by this project's OWN existing "kulila"
# (the infinitive used nominally, "the act of crying" -- see
# DEVERBAL_SM_HYPOTHESIS/VERIFIED_SUBJECT_MARKERS above) already using
# the SAME "lil" stem in a totally different grammatical construction --
# 2 independent sources, not just 1 verse. JHN.11.35 "Akalia machozi"
# (lit. "wept tears", an intensified/specific phrasing, "Jesus wept")->
# "ukadikwa ni mbori" uses a DIFFERENT verb+construction ("was overcome
# by tears"), consistent with "akalia machozi" being its own idiom
# distinct from plain "akalia", same "idiom dissolves" pattern as
# "piga" above -- not treated as a counterexample to "lia"->"lil".
VERIFIED_ROOTS.update({
    "li": "lil",
})

# 2026-10-20: "ambia" (to tell, root "ambi") -> Kitaita "ghore". Found
# while investigating the PASSIVE-extension correspondence (see the
# "verb EXTENSIONS" section below): "akaambiwa"(was told) -> "ukaghorelwa"
# in all 5 short verses read (GEN.31.22, 1SA.19.19, 1SA.27.4, 2SA.19.1,
# 2SA.21.11), zero exceptions -- the PASSIVE form is directly confirmed,
# the ACTIVE base "ghore" is inferred from it (passive = base + "-lwa"),
# not independently verse-confirmed on its own.
VERIFIED_ROOTS.update({
    "ambi": "ghore",
})

# 2026-10-20: "penda" (to love, root "pend") -> Kitaita "kund". Also
# found while investigating the passive-extension correspondence:
# "mpendwa"(the-beloved-one, class-1 m- + PASSIVE) -> "Mkundwa" (m- +
# kund + wa) in all 6 short verses read (SNG.1.13, 1.14, 1.16, 2.8,
# 2.10, 3.1), zero exceptions -- same shape as "ambi"->"ghore" above,
# the ACTIVE base is inferred from the passive, not independently
# verse-confirmed alone. Bonus, not separately added: "mpenzi"
# (darling/lover, the "-zi" AGENTIVE nominalization of the SAME root,
# a DIFFERENT derivation than the "-wa" passive participle) ALSO maps
# to "Mkundwa" in all 10 short verses checked (PSA.22.20, PSA.88.18,
# SNG.1.9, 2.2, 2.10, 4.7) -- Kitaita apparently does NOT distinguish
# these two Swahili nominalizations, both just render as "the one who
# is loved" -- a reminder (same as the stative note above) that not
# every Swahili morphological distinction has its own separate Kitaita
# form.
VERIFIED_ROOTS.update({
    "pend": "kund",
})

# ----------------------------------------------------------------- Word Bank negation-tab predicates (one per noun class, NOT "kunda")
# 2026-10-21, direct user report: native speakers struggle with the
# PERSONIFICATION of using "kunda" (love) as the demo verb for EVERY
# Word Bank negation_class row, including inanimate nouns ("the cup
# loves" is exactly the kind of sentence that distracts a speaker from
# the real grammatical question being asked). Replaced with a
# DIFFERENT, semantically-natural, non-personifying verb per noun --
# each checked directly against the full-Bible corpus (SUV/NEN/BHN)
# before adding, same discipline as every other entry in this file.
# Confidence varies a lot by entry (noted individually) -- this is a
# first pass for a Word Bank ELICITATION tab, not a claim that any of
# these are native-speaker-verified; the whole point of the tab is to
# let a speaker correct them.
#
# "chomoza" (rise, of sun/moon/stars) -> root "buk". 4 independent
# verses, including a NEGATED one (confirms the SAME root under "si-"
# too, not just a coincidence of one construction): PSA.104.22 "Jua
# huchomoza"/"lachomoza" -> "Iruwa jikabuka" (3 versions, identical);
# JOB.9.7 "halichomozi" -> "jisebuke" (root "buk" negated). A SEPARATE,
# less-attested root "jok" also appears in 2 verses (GEN.19.23,
# MAT.13.6, both "limechomoza" -> "jajoka") -- NOT used here, kept as a
# flagged alternate; "buk" has more independent support and matches the
# tense (present/habitual) this row actually needs.
VERIFIED_ROOTS.update({"chomoz": "buk"})
# "ongezeka" (increase/multiply) -> root "churik". Extremely clean, 6+
# independent verses, zero exceptions, across all 3 versions: GEN.9.1/
# 9.7 "mkaongezeke" -> "mchurikie" (imperative), EXO.23.30
# "utakapoongezeka" -> "mwachurikia", PSA.107.38 "wakaongezeka" ->
# "wichurikie". Used here for "shamba"/farm (a farm's produce
# increasing) -- a real semantic stretch from "people multiplying" but
# not personifying, and far better-evidenced than the only literal
# "zaa"(yield)-specific match found (LUK.12.16, n=1, weak: "limezaa
# sana" -> "erevaeghe loli-loli", root "va" -- NOT used, too thin).
VERIFIED_ROOTS.update({"ongezek": "churik"})
# "pita" (pass, of time) -> root "si". 2 independent verses: GEN.41.53
# "miaka...ikapita" -> "miaka...ikasia"; SNG.2.11 "kaskazi imepita" ->
# "Kwari yasia". Used for "siku"/day passing.
VERIFIED_ROOTS.update({"pit": "si"})
# "kufa" (die) -> root "fw". Very clean: EXO.7.18/7.21 "watakufa"/
# "wakafa" (of fish in the Nile) -> "richafwa"/"rikafwa", identical
# across all 3 versions, directly about SAMAKI (fish) -- the single
# best-evidenced and most on-topic entry in this whole batch.
VERIFIED_ROOTS.update({"kuf": "fw"})
# "vunjika" (break, of a vessel/fragile object) -> root "barik". Only
# ONE verse gives a literal VESSEL breaking (not a body part/weapon,
# which this project already knows are often separate idioms --
# PSA.37.15/37.17 "nyuta zao zitavunjika"/"itavunjika" give a DIFFERENT
# root, "chikany"/unclear, not used here): PSA.31.12 "chombo
# kilichovunjika" (a broken VESSEL, directly on-topic for "kikombe"/cup
# and, by the same physical-fragility logic, "mayai"/eggs) -> "chombo
# chibarikie". n=1 -- flagged as the weakest-evidenced entry in this
# batch, kept because it is directly on-topic and the alternative was
# no data at all.
VERIFIED_ROOTS.update({"vunjik": "barik"})
# "chosha" (tire/exhaust) -> root "sirish". ECC.10.15 "huwachosha" ->
# "radamsirisha ndighi" (lit. "takes-their strength"), identical across
# SUV/NEN (same underlying verse, so really ONE independent data point,
# not two) -- the weakest-evidenced entry in this batch, used for
# "kukimbia"/running (running tires you) since no better-attested
# alternative was found.
VERIFIED_ROOTS.update({"chosh": "sirish"})
# "tosha" (suffice/be enough) -> root "kat". EXO.36.7 "vilitosha" (of
# building MATERIALS being enough) -> "vakata" -- directly reusable for
# "chakula"/food (food being enough), SAME Kitaita class ('va') as the
# attested example, so the exact surface form transfers unchanged. n=1.
VERIFIED_ROOTS.update({"tosh": "kat"})

# "upendo...utafifia" (love growing cold/fading) -> MAT.24.12, the ONE
# verse found: "lukundo lwa wengi luchahora modo" = root "hor" + the
# FIXED word "modo" (cold) -- a periphrasis, same shape as VERIFIED_
# IDIOMS' other entries (root + an obligatory extra word); the actual
# entry is added to VERIFIED_IDIOMS further below (that dict doesn't
# exist yet at this point in the file). Used for "upendo"/love
# specifically (not a general "poa"/"fifia" rule -- n=1, directly
# on-topic only).

# 2026-10-21: "angusha" (cause to fall/drop) -- direct user correction,
# ONE specific sense only: "alipoangusha kikombe" (when he dropped the
# cup, a HELD OBJECT) -> Kitaita root "to" ("...wendapotoa kikombe").
# NOT a general override for every "angusha" occurrence -- checked
# directly against the corpus before adding this (full-Bible SUV/NEN/
# BHN): the corpus's own "angusha" examples are near-uniformly a
# DIFFERENT concrete sense, toppling a STRUCTURE/CITY or casting a
# PERSON down (GEN.19.25 "Akaangusha miji hiyo" -> "ukaighusha mizi
# yose", root "ighush"; EXO.15.7/PSA.35.5/PSA.44.5/PSA.89.44 all use yet
# other idioms -- "kumba...noko", "winga...noko", "kab"+"ghwa", "ng'arisha"
# -- for "cast down" in a defeat/humiliation sense), never the "drop a
# held object" sense this correction is for. Same "idiom dissolves by
# concrete sense" pattern already on file for "piga"/"kasirisha" (see
# VERIFIED_IDIOMS below) -- this entry covers ONLY the held-object sense;
# treating it as a blanket "angusha"->"to" rule would be wrong for the
# structure/person senses, which keep their own existing (or still-
# unresolved) Kitaita verbs.
VERIFIED_ROOTS.update({
    "angush": "to",
})

# NOT added from the noun pass, explicitly rejected or left unresolved
# after reading the verses (same "log it, don't silently drop it"
# discipline as the verb-roots pass):
#   'kilele'/'vilele' (peak/summit) -- 0 verses under 12 tokens contain it
#       at all; the automated top candidate 'pisga' is almost certainly
#       the place name "Pisgah" (kilele cha mlima Pisga, a real Biblical
#       mountain), not a translation of "peak" -- a false positive the
#       short-verse filter correctly avoided rather than one this session
#       caught by reading a long verse.
#   'pomboo' (dolphin) -- 0 short verses; automated top candidate
#       ('wingire') unchecked, and the HIGHER-cooccurrence runner-up
#       ('mrongo', meaning "tens"/a number) is clearly spurious -- dolphin/
#       badger-skin offering lists in Exodus are long enumerations, likely
#       coincidentally rich in number words.
#   'kina' (depth), 'waokozi' (saviors) -- no example verse found at all
#       (0 hits); flagged as untested, not rejected.
# PARTIALLY resolved, NOT added as a noun translation (a genuine part-of-
# speech mismatch, not a confident finding to force into this table):
#   'ukame' (drought/dryness, abstract NOUN in Swahili) -- the one example
#       found (EZK.19.13, "nchi kavu, ya ukame" / dry land, OF drought) ->
#       "isanga jiomie jisena machi" (land THAT-IS-DRY, without water) --
#       "jiomie" there is a STATIVE/ADJECTIVAL description of the land, not
#       a free-standing abstract noun Kitaita speakers would use the way
#       Swahili uses "ukame" as a subject/object in its own right. Possibly
#       a genuine typological difference (Kitaita may simply lack this
#       abstraction and use a relative clause instead), not resolved to a
#       drop-in noun -- adding "jiomie" to VERIFIED_NOUN_TRANSLATIONS would
#       risk translate() inserting an adjective wherever this noun is
#       expected. Flagged for a speaker, not guessed into the table.

# 2026-10-20 (cont.): 2 of the original 3 reverse-inference candidates are
# explicitly NOT added above, after closer re-reading of the SAME verses --
# a RETRACTION of this session's own first-pass read, same discipline as
# every other correction in this file (never silently drop a wrong claim):
#
# "tawal" (rule) -- NOT a single Kitaita root. 3 independent verses (EXO.
# 15.18, PSA.22.28, PSA.106.41) consistently render it as the Kitaita VERB
# PHRASE "bonya nguma" (lit. "do/exercise power" -- root 'fany'/'bony' above
# + the separate noun 'nguma', NOT fused into one word), including
# correctly re-inflected for voice (PSA.106.41 Swahili "wakawatawala"
# active -> Kitaita "wikabonyerwa nguma" passive, matching the Swahili
# passive context exactly). This project's `translate()` architecture had
# no slot for "root + an obligatory extra word" at the time -- adding
# "tawal"->"bonyanguma" as a literal VERIFIED_ROOTS string would have
# silently produced a wrong inflection point (the final vowel belongs
# after 'bony', not after 'nguma'). RESOLVED 2026-10-20 (cont. 5): see
# VERIFIED_IDIOMS below -- a real periphrastic-construction mechanism was
# added, and "tawala" turns out to be a genuine COMPOSITIONAL paraphrase,
# not an opaque idiom: "tawala" = "fanya mamlaka" (do authority), and BOTH
# halves are independently, strongly confirmed elsewhere in this very file
# ('fany'->'bony' above; 'mamlaka'->'nguma', VERIFIED_NOUN_TRANSLATIONS
# above, 40+ clean matches) -- the 3-verse "bonya nguma" finding wasn't a
# coincidence, it's what that compositional paraphrase predicts exactly.
#
# "kusudi" (intend/purpose) -- RETRACTED. The first-pass read (docs/
# status.md, "narrowed to njama or mipango") does not survive a closer
# re-read of the same 4 verses: in every one, "kusudi"/"makusudi" is
# dissolved into a restructured clause ALONGSIDE "njama"/"mipango", not
# individually equivalent to either -- e.g. PRO.16.30 "kusudi lake ni
# kuwaza yaliyopotoka" (his PURPOSE is to PLAN perverse things, 2 separate
# Swahili concepts) collapses to "wadabonya mipango elaghasha" (lit. "does
# crooked plans", 1 Kitaita clause) -- the Swahili source's OWN purpose/
# plan distinction isn't preserved, so there is no single Kitaita word
# this evidence can pin "kusudi" to. Same idiomatic-dissolution pattern as
# "tawal" above, not a 2-candidate narrowing as first reported.

# ----------------------------------------------------------------- periphrastic verb roots (VERB + a FIXED object noun)
# 2026-10-20 (cont. 5). Direct user request: before falling back to
# kitaita_sentence_generator.translate()'s "Taita-fy" tier (keep the
# Swahili root, Kitaita-inflect around it -- 2026-10-19) for a root with
# NO single-word Kitaita answer, first check whether it's actually a
# CONFIRMED [do/make-type root] + [fixed object noun] periphrasis, like
# "tawala" (rule) = root 'bony' (do/make, VERIFIED_ROOTS['fany']) + fixed
# object 'nguma' (authority, VERIFIED_NOUN_TRANSLATIONS['mamlaka']) --
# i.e. "tawala" really does decompose as "fanya mamlaka" (do authority),
# a genuine Swahili paraphrase, not a made-up gloss; see the 'tawal'
# comment above for the full verse citations and the compositional cross-
# check. Each entry is {"root": Kitaita verb root, "object": the fixed
# Kitaita noun that always follows it, unchanged, never re-inflected,
# "note": why}. Consulted by translate() as a tier BELOW VERIFIED_ROOTS/
# root-level Dice/derived (those, when they exist, are a real single-word
# answer, strictly better evidence) but ABOVE both the noisy per-surface-
# form statistical fallback AND the Taita-fy tier -- a confirmed phrase
# beats an unconfirmed single-word guess either way it could go wrong.
# Scope note: only matches the PLAIN root string ('tawal', from kutawala/
# anatawala/atatawala etc.) -- the applicative ('tawalia') and passive
# ('tawaliwa') extensions decompose to a DIFFERENT root string entirely in
# this project's simple one-final-vowel model (e.g. 'tawaliw', not
# 'tawal') and are NOT covered by this entry; flagged as a known gap, not
# silently assumed to also work.
VERIFIED_IDIOMS = {
    "tawal": {"root": "bony", "object": "nguma",
              "note": "rule = do(fany/bony) + authority(mamlaka/nguma); confirmed compositionally, "
                      "not just as a fixed phrase -- see the 'tawal' comment above"},
    # 2026-10-21: "kasirisha" (to anger/make angry, root "kasirish") --
    # direct user correction, RESOLVED after an initial misreading this
    # session (see below). Real correspondence, confirmed by the user
    # reverse-translating their own target sentence word-by-word:
    # "Mtoto alimkasirisha mzazi" ~ "Mtoto alipandisha mzazi hasira"
    # (lit. "the child raised the parent's anger") -> "Mwana odejosha
    # muvi machu" -- i.e. root "josh" (raise/cause) + the PARENT noun
    # (object 1, "mzazi"->"muvi", see VERIFIED_NOUN_TRANSLATIONS below)
    # + the FIXED anger-object "machu" (object 2). The idiom's own fixed
    # object is "machu", NOT "muvi" -- "muvi" is just mzazi's ordinary
    # noun translation, substituted in before the idiom's fixed object
    # the same way any other noun would be.
    #
    # Earlier pass this session (now corrected) had these swapped --
    # "mzazi"->"machu", idiom object "muvi" -- because the two
    # correspondences were given in a way that read as "kasirisha ->
    # odejosha muvi" / "mzazi -> machu" taken independently. That
    # version directly CONTRADICTED the corpus (machu is robustly,
    # exclusively "anger" there, 150+ occurrences; "wazazi"/parents is
    # consistently "wavi", never "machu") and was flagged rather than
    # applied. This corrected version instead MATCHES the corpus
    # exactly: "ukajokwa ni machu" (lit. "was angered BY anger" = became
    # angry, PASSIVE of a root alternating "jok"/"jos" -- e.g.
    # "Weremjosereghe BWANA machu" = "they angered BWANA [with] anger",
    # SUV/NEN/BHN, 30-65 occurrences of "kasirisha"/"kasirika" each,
    # zero exceptions) -- "jos" there is the SAME root as this entry's
    # "josh", just the ACTIVE-causative vs. the user's PASSIVE example;
    # no register conflict after all, one unified finding. "muvi" (the
    # parent/"mzazi" noun) is NOT otherwise attested in this project's
    # data outside this correction -- flagged, not independently
    # corroborated, but no longer contradicted either.
    "kasirish": {"root": "josh", "object": "machu",
                 "note": "anger (cause) = raise(josh) + [noun] + anger(machu); direct user correction, "
                         "2026-10-21, matches the corpus's own independent 'jos'+'machu' finding"},
    # 2026-10-21: Word Bank negation-tab predicate for "upendo"/love -- see
    # the "chomoz"/"ongezek"/etc. block above (VERIFIED_ROOTS) for the full
    # citation (MAT.24.12, n=1). Keyed by "fifi" -- decompose_verb()'s
    # root_guess for the Swahili verb "fifia" (fade/grow dim/cold), NOT by
    # the Kitaita root "hor" itself (VERIFIED_IDIOMS is always keyed by
    # the SWAHILI root string, same convention as "kasirish"/"tawal" above).
    "fifi": {"root": "hor", "object": "modo",
             "note": "love grows cold = fade(hor) + cold(modo); corpus-inferred, 2026-10-21, n=1 "
                     "(MAT.24.12), Word Bank predicate for 'upendo' only"},
}

# 2026-10-22: "kasirika" (get angry, INTRANSITIVE) -- direct user report
# ("kukasirika I see we are only using udejokwa instead of the full
# phrase"). decompose_verb() strips this to root "kasirik", which had no
# VERIFIED_ROOTS/VERIFIED_IDIOMS entry at all, so it fell to a WEAK
# statistical candidate ("jokwa", Dice 0.17, flagged UNRELIABLE) that
# happened to accidentally match the real root but with no indication
# this verb is a PASSIVE+fixed-agent idiom, same shape as 'furahi'
# (VERIFIED_ROOTS['furahi']) -- the object/agent phrase was simply never
# attached. Reverse-inference search of the full parallel Bible corpus
# for "alikasirika"/"kukasirika": n=8 clean verses (2SA.13.21, DAN.6.14,
# LUK.13.14, ECC.7.9, JON.4.4, JON.4.9, 2CO.7.11, JAS.1.19), zero
# exceptions -- "alikasirika sana" -> "ukajokwa ni machu loli-loli"
# (2SA.13.21), "kukasirika" -> "kujokwa ni machu" (ECC.7.9 etc.). This is
# the PASSIVE counterpart of the causative "kasirisha"/"kasirish" entry
# just above (root alternation "jok"[passive]/"jos"[causative-active],
# same "machu" fixed anger-object both times) -- one unified finding,
# not two unrelated ones. Stored as the BASE (pre-passive) root "jok"
# here, same convention as VERIFIED_ROOTS['furahi']='boi' -- translate()
# applies kitaita_passive_form('jok') = 'jokw' (consonant-final rule) and
# appends the FIXED "ni machu" phrase via `idiom_object` (this verb is
# intransitive in Swahili too -- no real object ever competes for that
# slot), not VERIFIED_IDIOMS (which has no present-tense-override hook;
# see translate()'s own 'kasirik' branch for the thin-but-consistent,
# n=2, present-tense '-da-' evidence: "nadajokwa"/"wadajokwa").
VERIFIED_ROOTS["kasirik"] = "jok"

# ----------------------------------------------------------------- verb EXTENSIONS (passive/stative/causative/etc)
# 2026-10-20, direct user request: "get the various verb forms ... so we
# don't have to duplicate effort for every single [root]. See how the
# different endings map between Swahili and Taita" -- i.e. a GENERAL,
# reusable Swahili-extension -> Kitaita-extension correspondence, so a
# single VERIFIED_ROOTS entry's PASSIVE/STATIVE/etc. forms don't each
# need their own independent reverse-inference pass.
#
# Checked DIRECTLY against the corpus for 3 extensions, with genuinely
# different outcomes -- NOT assumed to generalize uniformly:
#
# PASSIVE ("-wa"/"-ewa" after SWAHILI root, "-w-"/"-iw-" allomorphy) --
# CONFIRMED, GENERALIZES, wired into translate() below as an automatic
# rule (see swahili_passive_base_candidates()/kitaita_passive_form()):
#   - "penda"(love, root "pend") -> Kitaita "kund"; PASSIVE "mpendwa"
#     (the-one-who-is-loved) -> "Mkundwa" (kund+wa), 6 independent
#     verses (SNG.1.13/1.14/1.16/2.8/2.10, 3.1), zero exceptions.
#   - "ambia"(tell, root "ambi") -> Kitaita "ghore" (NEW, added below);
#     PASSIVE "akaambiwa"(was told) -> "ukaghorelwa" (ghore+lwa, "-lw-"
#     allomorph since "ghore" itself ends in a vowel -- mirrors
#     Swahili's OWN consonant/vowel-conditioned "-wa"/"-liwa" split,
#     just with Kitaita's own "l" as the epenthetic consonant instead of
#     Swahili's "i"), 5 independent verses (GEN.31.22, 1SA.19.19,
#     1SA.27.4, 2SA.19.1, 2SA.21.11), zero exceptions.
#   - "p"->"nek" (give, see VERIFIED_ROOTS above) is a 3rd, WEAKER data
#     point in the same direction (root_kitaita_fallback independently
#     proposed "nekwa" at Dice 0.18 before this rule existed).
#   Rule: the KITAITA base root's OWN final sound decides the allomorph
#   (consonant-final -> "-w-", vowel-final -> "-lw-"), NOT the Swahili
#   root's shape -- "ambi" (Swahili, vowel-final) still gets Kitaita
#   "ghore" (also vowel-final, coincidentally) -> "-lw-", but this is
#   KITAITA's own root deciding, not inherited from Swahili.
#
# STATIVE/NEUTER ("-eka"/"-ekana") -- ONE confirmed data point, NOT
# wired into an automatic rule (too little data to trust an automatic
# rule yet, unlike passive above) -- kept here as a recorded finding for
# a future round, not silently assumed:
#   - "ona"(see, root "on") -> Kitaita "won"; STATIVE "onekana"/
#     "ilionekana"(appear/become visible) -> "woneka" (won+eka) in
#     REV.12.1, 1SA.10.16, REV.18.24 (3 verses) -- Swahili's compound
#     "-ekan-" (stative+extra) corresponds to Kitaita's SIMPLE "-eka"
#     (the "-an-" portion has NO separate Kitaita exponent here, it
#     just doesn't carry over) -- a reminder that NOT every Swahili
#     morphological distinction gets its own Kitaita form.
#
# CAUSATIVE ("-isha"/"-esha") -- CHECKED, DOES NOT GENERALIZE, explicitly
# NOT wired into any automatic rule:
#   - "ona"(see, root "on"->"won") -> CAUSATIVE "onyesha"(show) does NOT
#     map to any "won"-derived form at all -- LUK.4.5 "akamwonyesha"->
#     "ukammbonyera" (applicative of "bonya"=do/make, an idiom "do FOR"),
#     JOB.10.2 "Nionyeshe"->"nighorie" (imperative of "ghor"=tell/say,
#     see passive section above) -- TWO DIFFERENT verbs, NEITHER
#     derived from "won", for the SAME Swahili causative. Same "idiom
#     dissolves, root+affix does NOT regularly compose" pattern already
#     seen for "piga"/"sema" -- common/basic verbs' causatives are
#     independently lexicalized here, not transparent affixation.
#
# RECIPROCAL ("-ana") and APPLICATIVE ("-ia"/"-ea") alone (not fused
# with another extension) -- NOT YET CHECKED this round (every short-
# verse candidate tried, e.g. "pendana"/"ambiana"/"fuatana", had 0-1
# corpus occurrences, too little to read) -- flagged as untested, not
# assumed either way.

VOWELS = "aeiou"


def swahili_passive_base_candidates(root_guess: str) -> list[str]:
    """Candidate BASE (pre-passive) Swahili roots for a root_guess that
    might end in a passive marker -- enumerated, not blindly stripped
    (same discipline as find_object_marker_candidates), since Swahili's
    passive allomorph depends on the base root's own shape: most roots
    just add "-w-" (pend -> pendw), but MONOSYLLABIC roots (p, j, l, ...)
    need an epenthetic vowel first, "-ew-"/"-iw-" (p -> pew, not "pw") --
    a well-known, independent Swahili morphophonological rule, not
    something specific to this project. Tries the plain "-w-" strip
    first (shorter, far more common), then the monosyllabic "-ew-"/
    "-iw-" strip. Returns [] if root_guess doesn't end in "w" at all."""
    if not root_guess.endswith("w"):
        return []
    candidates = []
    if len(root_guess) > 2:
        candidates.append(root_guess[:-1])
    if len(root_guess) > 3 and root_guess[-3:-1] in ("ew", "iw"):
        candidates.append(root_guess[:-3])
    return candidates


def kitaita_passive_form(kitaita_base_root: str) -> str:
    """The Kitaita PASSIVE infix for an already-resolved Kitaita base
    root (see this section's own comment for the evidence): "-w-" if
    the root ends in a consonant (kund -> kundw, nek -> nekw), "-lw-" if
    it ends in a vowel (ghore -> ghorelw) -- mirrors Swahili's OWN
    consonant/vowel-conditioned passive split, just with Kitaita's own
    "l" as the epenthetic consonant. The caller appends the verb's
    ORIGINAL final vowel afterward, same as any other root (this
    returns the CONSONANTAL part only, not a complete word)."""
    if kitaita_base_root and kitaita_base_root[-1] in VOWELS:
        return f"{kitaita_base_root}lw"
    return f"{kitaita_base_root}w"

# "kuwa" (be) splits by whether a predicate complement (adjective)
# follows -- a real distinction Swahili's own "kuwa" does not mark but
# Kitaita apparently does. The 'kai' root-level-alignment candidate
# (Dice 0.19, 1790 shared verses -- previously this project's
# strongest-looking root-level match) is WRONG for both; confirmed
# consistently by 8 of 9 relevant judgments:
#   existential (no complement): mungu/mti/siku (s2,s6,s9: "...rekoghe")
#     and nchi-future (s11: "...chakoghe") all confirm root 'kogh'.
#     EXCEPTION, flagged not merged: nyumba (s17) gave "redeka" (the
#     'dek' root below) with NO adjective in that sentence -- either
#     nyumba's class genuinely takes the copula root even existentially,
#     or this one answer doesn't fit the pattern; needs asking directly.
#   copula (+ adjective): mungu/mti/nchi/mtu ALL confirm root 'dek'
#     (s18 "kodeka", s19 "ghodeka", s20 "jedekaa", s21 "odeka") -- 4/4
#     clean, no exceptions.
VERIFIED_KUWA_ROOT_EXISTENTIAL = "kogh"
VERIFIED_KUWA_ROOT_COPULA = "dek"

# ----------------------------------------------------------------- noun translation overrides
# 2026-10-08 cont. 12 (batch 10, s136/s137): FIRST ATTEMPT at this, RETRACTED cont. 16.
# 'mto' (river)'s statistical candidate 'moda' was judged wrong and corrected to 'modo'
# ("Modo ghukaghamba"/"Modo ghwawemanya") -- but direct correction (cont. 16): "mto was
# read as moto and translated to modo instead of moda." Checked directly against
# page1_vocabulary.json: 'modo' is moto (fire)'s OWN statistical candidate (Dice 0.69),
# 'moda' is mto (river)'s (Dice 0.73) -- the original "wrong" judgment was answering
# about a DIFFERENT Swahili word (moto, confused with mto) entirely, not a real
# correction to mto's translation. Override removed; mto falls back to the statistical
# 'moda' again, which was never actually shown wrong for mto itself. See
# VERIFIED_SUBJECT_MARKERS and VERIFIED_SENTENCES below for the matching retractions.
# Keyed by the SWAHILI word (same convention as VERIFIED_SENTENCES), consulted before
# the statistical noun_kitaita lookup.
VERIFIED_NOUN_TRANSLATIONS = {
    # 2026-10-08 cont. 13: direct user request to test "verb-nouns like kuseka" --
    # Swahili class-15 infinitives used nominally ("kucheka" = "the act of laughing"),
    # a category this project's own from-scratch induction had already pulled OUT of
    # every noun class as a 243-member cross-cutting "deverbal" group (page2_noun_
    # classes.json's kitaita_corrected_classes.deverbal) but never actually tested as
    # a sentence SUBJECT. No statistical alignment exists for 'kucheka'/'kuseka' at
    # all (checked directly -- zero entries, whole-word or root-level); this is a
    # HYPOTHESIS, not even corpus-induced: Bantu infinitives are often directly
    # retained/cognate across closely related languages, and the thin root-level
    # alignment for 'chek'->'wasek' (Dice 0.38, n=3, likely a wa-prefixed inflected
    # form rather than a bare root) is at least consistent with a Kitaita root 'sek'
    # existing. Deliberately flagged as UNCONFIRMED in the trace -- see
    # DEVERBAL_SM_HYPOTHESIS below for the subject-marker side of this same test.
    "kuseka": "kuseka",
}

# 2026-10-12: 41 noun-translation gaps filled via the Kitaita Word Bank
# Artifact (https://claude.ai/artifact/6eozxX4HtKZhce3FTfWnY2, collection
# "contributions") -- these are all words from the project's own 319-word
# hand-built Swahili lexicon that had ZERO Kitaita match on file at all
# (not low-confidence, missing entirely). Same trust tier as every other
# direct-submission table in this file. 8 of the 41 were submitted
# IDENTICAL to the Swahili word (dirisha, machungwa, masanduku, mchango,
# mfuko, mwavuli, walimu, kuchelewa) -- plausible shared loanwords (all
# are either Arabic/Swahili loans into Swahili itself, or modern/
# administrative vocabulary, both classes where Bantu-internal borrowing
# into Kitaita is independently plausible), kept as submitted since
# nothing here contradicts them, but flagged here as not yet cross-
# checked the way a flagged contradiction (see DOGO_FORM_BY_SM) would be.
VERIFIED_NOUN_TRANSLATIONS.update({
    "choo": "choronyi", "dirisha": "dirisha", "goti": "kitemberu",
    "huduma": "wutesia", "jambo": "ilagho", "jicho": "iriso",
    "kuchelewa": "kuchelewa", "kufikiri": "kughesha", "kumaliza": "kumerea",
    "kuogelea": "kuogelia", "kupika": "kudeka", "kupumzika": "kughoresha",
    "kushuka": "kusea", "machungwa": "machungwa", "mahindi": "mabemba",
    "masanduku": "masanduku", "mashamba": "mbuwa", "mchango": "mchango",
    "mchezo": "msarigho", "mfuko": "mfuko", "mioyo": "ngolo",
    "mwavuli": "mwavuli", "mzee": "mghosi",
    "mzizi": "muri", "paka": "boi", "pesa": "maghome",
    "sababu": "wundu", "tembo": "chovu", "udogo": "wutini",
    "ukweli": "loli", "uongozi": "wulongozi", "usalama": "wusalama",
    "ustawi": "wubinana", "utoto": "wanake", "viazi": "makaji",
    "visu": "ndagaa", "vyoo": "vyoronyi", "walimu": "walimu",
    "wanaume": "womi", "yai": "ighi",
    # 2026-10-13 (cont. 4): "mwamba" (rock) RETRACTED and replaced -- the
    # user resubmitted it via the Word Bank with a different translation
    # ("mbanga", not "igho") and a class ("ghu"). NOT applying the class
    # side of this update: "mbanga" already exists in this project's own
    # Class Sorter data (noun_sort_data.json) as the Kitaita word for
    # "pango" (cave), classed "i" -- a genuine conflict with the new
    # "ghu" tag, not resolved here (see VERIFIED_SUBJECT_MARKERS note
    # below). The translation itself is applied; "mwamba"'s class is left
    # unset pending that conflict's resolution.
    "mwamba": "mbanga",
    # 2026-10-13 (cont. 4): 17 more nouns added via the Kitaita Word Bank
    # (same artifact/collection as the 2026-10-12 round above).
    "bega": "ionge", "fuvu": "ngoma", "misitu": "misidu", "mkopo": "mkopo",
    "mkristo": "mkristo", "mkurugenzi": "waziri", "mpishi": "mpishi",
    "msichana": "mwai", "mtoto": "mwana", "mvulana": "mndwawomi",
    "mwandishi": "mwandiki", "waandishi": "waandishi", "vitambaa": "vitambaa",
    "nyani": "fuwe", "sungura": "kwaru", "ushirikiano": "mapatano", "uzi": "uzi",
})

# 2026-10-20 (cont. 4): same full-corpus reverse-inference run as the verb-
# roots pass above, now on the Word Bank's NOUN gap list (whole-word Dice
# alignment -- no Kitaita noun-root stripper exists, same convention the
# statistical noun_kitaita table itself already uses). FIRST finding,
# before any new alignment: 58 of 136 listed noun gaps were ALREADY
# resolved (VERIFIED_NOUN_TRANSLATIONS or the statistical noun_kitaita
# table), e.g. "mtoto"->"mwana" -- a bigger staleness gap than the verb-
# root list had (43% vs 12%), never cleaned up at all for nouns. Of the
# 78 genuinely-unresolved gaps, 54 do not appear in this corpus even once
# -- cannot be resolved by this method at all, regardless of effort.
# Of the 24 that DO appear, same "read the verses before trusting the
# score" discipline as the verb-root pass -- 5 survived (not the ~1-in-4
# rate from verbs; nouns' reference/loanword nature made several of these
# easier to confirm):
VERIFIED_NOUN_TRANSLATIONS.update({
    "habari": "malagho",   # news/report -- confirmed 3/4 direct short-verse matches
                             # (PSA.18.44, PSA.22.30, PSA.68.11 "habari"<->"malagho";
                             # GEN.37.5 restructured differently, not contradicting).
    "kipande": "kibelele",  # piece -- confirmed via LUK.24.42 "kipande cha samaki"
                             # ->"kibelele cha samaki"; reinforced by "kibelele"/
                             # "vibelele" independently meaning "piece/unit" elsewhere
                             # in the corpus (LEV.27.4, shekel units). The other 2
                             # short verses read used a different, non-contradicting
                             # construction (a diminutive "ka-mkate", and "madumbu").
    "vyakula": "vindo",     # foods (plural of "chakula", already known -> "vindo") --
                             # confirmed directly as its OWN key too (3/4 clean:
                             # PSA.132.15, MAT.14.16, LUK.10.8) rather than assumed
                             # from the singular; "vindo" is independently documented
                             # elsewhere in this project as inherently plural-shaped,
                             # so this is not a surprising number mismatch.
    "kipimo": "kipimo",     # measure/measurement -- WEAKER evidence than the 3 above:
                             # only 1 of 3 short verses read shows this directly
                             # (JOB.28.25, "kwa kipimo"<->"kipimo chawo", same spelling
                             # both sides -- plausible shared loanword, same tier as
                             # this project's existing "ish"->"ish"/"uzi"->"uzi"
                             # entries), the other 2 verses paraphrase the concept
                             # differently ("kikombe"/cup; dropped entirely) rather
                             # than contradicting it outright.
    "kusikia": "kusikira",  # hearing (class-15 infinitive-as-noun, same category as
                             # "kuchelewa"/"kupumzika" etc. above) -- confirmed via the
                             # specific recurring collocation "kusikia habari" (hear
                             # news) <-> "kusikira malagho"/"wiendasikira malagho",
                             # 3/3 clean (PSA.18.44, 2SA.22.45, + the habari cross-
                             # check above). See VERIFIED_ROOTS['siki'] below for the
                             # matching FINITE-verb root (the same underlying root,
                             # stored separately because this project's noun and verb
                             # lookups are two separate tables, not because the root
                             # differs).
})

# 2026-10-20 (cont. 5): direct user-prompted re-check -- "mamlaka" (authority/
# power/dominion), found while verifying the "tawala" periphrasis (VERIFIED_
# IDIOMS below). OVERWHELMING evidence, not a thin corpus match: 40+ distinct
# clean matches across Daniel, Luke, Romans, Ephesians, Colossians, Revelation
# -- "mwenye mamlaka"<->"uko na nguma"/"wibonyagha nguma", "mamlaka yangu/
# yako/yake"<->"nguma yangu/yako/yake", "mamlaka yote"<->"nguma rose", near-
# 1:1 throughout. NOTE: an earlier single-verse read (DAN.11.3) was
# MISATTRIBUTED to a different word in that same sentence ("ndighi" there
# actually glosses "hodari"/mighty, a SEPARATE Swahili word describing the
# king, not "mamlaka") -- caught and corrected only once the FULL set of
# matches was read, not from that one verse; kept as a reminder that even a
# single plausible-looking match needs the broader check before trusting it.
VERIFIED_NOUN_TRANSLATIONS.update({
    "mamlaka": "nguma",
})

# 2026-10-21: "mzazi" (parent) -> "muvi" -- direct user correction, resolved
# after an initial misreading this session (see VERIFIED_IDIOMS["kasirish"]'s
# own comment for the full story: the user's worked example reverse-
# translates as "Mtoto alipandisha mzazi hasira" = verb(josh) + mzazi(muvi)
# + hasira/anger(machu), word-by-word). NOT independently corpus-checked
# (no isolated singular "mzazi" exists anywhere in the full-Bible corpus to
# check against -- every occurrence is fused with "mama" and collapses to
# plain "mother"/"mao"/"mae"); the corpus's OWN plural "wazazi"->"wavi" is
# at least CONSISTENT with "muvi" being the Kitaita 1/2-class singular
# counterpart of "wavi" (wa-/m- class pairing), though that specific
# singular-plural relationship is a plausible inference, not itself
# directly attested.
VERIFIED_NOUN_TRANSLATIONS.update({
    "mzazi": "muvi",
})

# 2026-10-20: "maisha" (life) -- direct user request, reverse-inferred.
# Checked ~30 short/medium SUV verses containing "maisha" directly:
# "maisha" = Kitaita "irangi" in every single one, overwhelmingly (e.g.
# PSA.23.6 "maisha yangu" -> "irangi japo", GEN.3.14 "maisha yako" ->
# "irangi jako", JOS.1.5/1SA.7.15/1KI.4.21/1KI.11.34/1KI.15.5/1KI.15.6/
# PRO.31.22 "maisha yake" -> "irangi jake", among 20+ more). Its Kitaita
# class is registered below as "ji" (VERIFIED_SUBJECT_MARKERS["irangi"],
# reusing "ituku"/day's already-registered class -- UNCONFIRMED whether
# they're really the same class or this just happens to produce the
# right surface form either way; see OTE_STEM's elision note). "jose"
# ("-ote", all), "japo"/"jako"/"jake" ("-angu"/"-ako"/"-ake", see
# POSS_PERSON_FORM_BY_CLASS below) all showed up repeatedly in those
# exact verses, independently confirming SOME shared class, not just
# the noun itself.
VERIFIED_NOUN_TRANSLATIONS.update({
    "maisha": "irangi",
})

# 2026-10-20: "asubuhi" (morning), direct user request, reverse-inferred.
# Read all 16 short SUV verses containing it: "nakesho" in EVERY single
# one, zero exceptions (GEN.1.13/1.19/1.23, GEN.19.27, GEN.40.6, GEN.44.3,
# EXO.16.19, PSA.73.14, PSA.88.13, PSA.90.5/90.6, PSA.92.2, 1SA.17.16,
# JOB.7.18, EZK.12.8, MAT.21.18) -- one of the cleanest findings in this
# project's whole reverse-inference history. Used adverbially/temporally
# throughout (never takes a class-concord prefix in any example found),
# so no class registration is needed, unlike a regular noun. Bonus,
# not requested but directly visible in the same verses: "jioni"
# (evening) = "nakenyi" with equal consistency (GEN.1.13/1.19/1.23,
# PSA.90.6, 1SA.17.16) -- NOT added here since it wasn't asked for, but
# flagged for a future round.
VERIFIED_NOUN_TRANSLATIONS.update({
    "asubuhi": "nakesho",
})

# 2026-10-22: "jana" (yesterday), direct native-speaker correction --
# "ighuo". Same adverbial/temporal shape as "asubuhi"/"nakesho" above
# (not a class-concording noun), not yet corpus-cross-checked the way
# asubuhi was (16 verses read directly) -- kept at the same trust tier
# as every other direct correction in this file (n=1), not claimed to
# be corpus-confirmed.
VERIFIED_NOUN_TRANSLATIONS.update({
    "jana": "ighuo",
    # 2026-10-22: "leo" (today), direct native-speaker correction --
    # "idime". Same adverbial/temporal shape as "asubuhi"/"jana" above.
    "leo": "idime",
    # 2026-10-22: RETRACTS the 2026-10-20 bonus note above ("jioni" ->
    # "nakenyi", flagged but never added) -- direct native-speaker
    # correction: the real Kitaita for "jioni" (evening) is the FULL,
    # UNSEGMENTED PHRASE "luma lwa kenyi" ("luma" = time, lu-class --
    # the same class as "lukundo"/love, note "lwa" is exactly this
    # project's own already-VERIFIED lu-class associative particle,
    # KITAITA_ASSOCIATIVE_VERIFIED['lu']='lwa' -- a nice independent
    # cross-confirmation of that table, not newly introduced here; +
    # "kenyi", the evening-specific stem). Native speakers confirm
    # "kenyi" ALONE has little to no meaning -- it is NOT a synonym or
    # a shortened form, so this must be stored and used as one fixed
    # phrase, never decomposed or abbreviated to just "kenyi"/"luma".
    # This also directly explains why the "Ambiguous translations" Word
    # Bank round's 2 competing flat candidates ("kenyi" Dice 0.468,
    # "luma" Dice 0.384) were BOTH wrong on their own -- the statistical
    # table has no way to represent "the answer is a fixed 2-word
    # phrase, not a single word" at all, a DIFFERENT failure shape than
    # the singular/plural or sense-distinct cases found in that same
    # round (matoleo/majira/mkwe etc.).
    "jioni": "luma lwa kenyi",
    # 2026-10-22: "wimbo" (song, singular) -> "lumbo", direct native-
    # speaker correction -- same class as "lukundo" (love), 'lu'. And
    # "nyimbo" (songs, plural) -> "chumbo", class 'ri' -- NOT the same
    # root pluralized the way matoleo/majira were earlier this session;
    # a genuinely different Kitaita word for the plural, same shape as
    # ituku/matuku (different root AND different class, singular vs
    # plural, not just a prefix change). See VERIFIED_SUBJECT_MARKERS
    # below for both classes.
    "wimbo": "lumbo",
    "nyimbo": "chumbo",
})

# 2026-10-20: "siku" (day) is Swahili number-invariant (no plural
# marking), but Kitaita uses a DIFFERENT WORD for plural, not just a
# different subject marker on the same root (unlike NUMBER_INVARIANT_
# NOUNS, which is exactly that different, simpler case): singular
# "ituku" (already on file, e.g. JOS.4.14 "Siku ile" -> "Ituku ijo"),
# plural "matuku". Direct user report: translating "siku" with "ituku"
# unconditionally is wrong whenever the context is plural. Scoped fix
# (see lookup_noun_kitaita's plural_context param in kitaita_e2e_
# pipeline.py) covers only the ONE heavily-confirmed trigger directly
# read in the corpus: "siku zote" -- NEVER "ituku zote", ALWAYS "matuku
# ghose" (20+ verses: GEN.3.14, GEN.3.17, PSA.23.6, PSA.27.4, PSA.128.5,
# DEU.4.9, DEU.16.3, JOS.1.5, JOS.4.14, 1SA.1.11, 1SA.7.15, 1KI.4.21,
# 1KI.11.34 ("irangi jake jose", a SEPARATE -ote on "irangi" itself, same
# verse), 1KI.15.5, 1KI.15.6, 2KI.25.29, PRO.31.22, ECC.2.3, ECC.5.18,
# ECC.6.12, ECC.8.15, ECC.9.9x2, ISA.38.20). Plural "siku" WITHOUT "zote"
# (e.g. DEU.6.2's 2nd clause, "siku zako ziongezwe" -> "matuku ghako
# ghikaie mengi") is a real, confirmed counterexample to "only 'zote'
# triggers plural" as a FULLY general rule, but is NOT yet covered --
# this fix is deliberately narrow (one directly-evidenced trigger), not a
# claim of general number detection. "matuku"'s own class is registered
# below as "ghi" (VERIFIED_SUBJECT_MARKERS["matuku"] -- reusing "maji"/
# water's already-established ma-/liquid-noun class, matching the
# associative particle actually observed, "matuku ghose GHA irangi" --
# "gha" is "ghi"'s own tier-1-confirmed associative form, see
# KITAITA_ASSOCIATIVE_VERIFIED). NOTE this is a DIFFERENT Kitaita class
# than "irangi"'s "ji", even though both are Swahili class-6 "ma-" nouns
# -- another instance of this project's repeated finding that Kitaita
# class does not always track Swahili class.
SIKU_PLURAL_TRIGGER = "matuku"

# 2026-10-22: direct user report -- translating English "Pope" goes
# EN->SW "Papa" (the Swahili title for the Pope) -> Kitaita lookup falls
# through to the statistical noun_kitaita candidate for "papa", which is
# this project's own 319-word lexicon's SEPARATE sense of "papa" (shark,
# swahili_noun_lexicon_agreement_v2.py) -> "kukanye"/"kukanya" (Dice 0.33,
# 3 competing partners, page1_vocabulary.json). Checked that candidate
# directly: its one supporting verse (GEN.30.3, "kukanye nao") is Rachel
# telling Jacob to "go in to"/lie with her maid Bilha -- an unrelated verb,
# not a noun for "shark" or "whale" at all, so the statistical candidate
# was never a real translation of "papa" to begin with, let alone of the
# Pope sense. "Papa" (Pope) is a title/loanword in Swahili itself (cf.
# "Papa Fransisko"); treated as a direct pass-through here, same tier as
# this file's other identical-loanword overrides (dirisha, machungwa,
# etc. above) -- NOT a disambiguation of "papa"=shark vs "papa"=Pope in
# general (this table has no sense-marking mechanism), just the override
# needed for the Pope case this was reported against.
VERIFIED_NOUN_TRANSLATIONS.update({
    "papa": "papa",
})

# 2026-10-22: direct native-speaker corrections on the "Ambiguous
# translations" Word Bank round (12 of the 20 rows answered directly in
# chat; the rest -- lugha, mawazo, mapema, kijito, mbuga, viunga, and
# "mkwe"'s exact resolution -- are still pending, NOT assumed resolved
# by omission). Each note below records WHY the statistical table had
# 2-3 competing candidates in the first place -- in every case checked,
# it was not random noise, but a real, nameable distinction (singular/
# plural, a different Swahili word entirely, or a construction this
# project's noun-only statistical table can't represent):
#   "mavuno" (harvest) -> "makuwido": the OTHER candidate, "ekuwida",
#     is not a competing noun at all -- "ngelo ekuwida" = "wakati wa
#     kuvuna" (harvest TIME), the verb "kuwida" (= kuvuna, to harvest)
#     used relatively/associatively, not a 2nd noun reading. The verb
#     root itself is recorded separately, see VERIFIED_ROOTS["vun"]
#     below.
#   "matoleo" (editions/offerings, Swahili PLURAL of "toleo") ->
#     "mafunyo": "ifunyo" is the SINGULAR ("toleo") of the SAME word,
#     not a competing translation -- "matoleo" itself is plural-shaped
#     in Swahili, so the plural form is the right default here (same
#     "which number is the Swahili lemma itself" reasoning as siku/
#     ituku/matuku above, just the opposite direction: the Swahili
#     lemma already IS plural).
#   "majira" (seasons/times, Swahili PLURAL of "jira") -> "vimu": same
#     shape as matoleo -- "kimu" is the singular ("jira"), not a
#     competitor.
#   "kushi" (Kush, a place/kingdom name) -> "kushi": the OTHER
#     candidate, "ethiopia", is a DIFFERENT place name entirely (a
#     plausible corpus co-occurrence, Kush and Ethiopia are mentioned
#     together often) -- "kushi" itself is a direct loanword/proper
#     name, kept unchanged, same tier as "papa" above.
#   "viti" (chairs) -> "vifumbi": the OTHER candidate, "vifumbinyi", is
#     not a competing plural -- it is the LOCATIVE form (for "vitini",
#     "on/at the chairs"), the "-nyi" locative suffix already flagged
#     elsewhere in this project (2026-10-08) acting on the SAME noun.
#     Recorded as its own entry below, not merged into this one.
#   "mitego" (traps) -> "midegho": direct correction AGAINST this
#     session's own new frequency-aware reranking, which had picked
#     "shaghala" (score 0.216) over "midegho" (score 0.197) -- a
#     concrete, honest miss for that heuristic, kept as a reminder it's
#     a re-ranking aid, not a validated answer (see docs/status.md
#     2026-10-22 cont. 2's own "NOT attempted" paragraph).
#   "maskini" (poor/destitute) and "wakawaua" (they then killed [them])
#     are NOT noun-translation ambiguities at all -- see their own
#     notes below/elsewhere; not added here.
VERIFIED_NOUN_TRANSLATIONS.update({
    "mavuno": "makuwido",
    "matoleo": "mafunyo",
    "majira": "vimu",
    "kushi": "kushi",
    "viti": "vifumbi",
    "mitego": "midegho",
    # 2026-10-22: bonus finding, same correction round -- "vitini"
    # (locative of "viti", on/at the chairs) is NOT "viti"'s own
    # ambiguity (see above) but a separate word, confirming the "-nyi"
    # locative suffix this project already flagged in 2026-10-08
    # (e.g. "andenyi"/"ndani") now directly on a 2nd noun.
    "vitini": "vifumbinyi",
    # 2026-10-22: "maskini" (poor/destitute) is Swahili NUMBER-INVARIANT
    # (same word sg/pl), but Kitaita uses a DIFFERENT word for plural --
    # same shape as "siku"/SIKU_PLURAL_TRIGGER above, direct user
    # confirmation: "mkiwa" (singular) / "wakiwa" (plural). Defaulted to
    # the singular here (no plural-detection TRIGGER wired for this word
    # yet, unlike "siku zote" -- flag, don't guess a trigger that hasn't
    # been checked against the corpus).
    "maskini": "mkiwa",
    # 2026-10-22: direct finding, same round, NOT requested but directly
    # given -- "sadaka" (offering) -> "vizongona". Its own Kitaita class
    # is NOT yet confirmed (see KUTEKETEZWA_ROOT_SINGLE_POINT above,
    # which depends on it) -- flagged, not guessed.
    "sadaka": "vizongona",
    # 2026-10-22, separate round (AP News paragraph test) -- direct
    # native-speaker corrections. "swali"/"maswali" classes registered
    # in VERIFIED_SUBJECT_MARKERS (see "ikoto"/"makoto" above).
    "swali": "ikoto",
    "maswali": "makoto",
    # "Wakatoliki" (Catholics) and "kanisa" (church/Church) are direct
    # loanword pass-throughs, same tier as "papa"/"kushi" above.
    "wakatoliki": "wakatoliki",
    "kanisa": "kanisa",
    # 2026-10-22, 2nd follow-up round, same 2 cases genuinely resolved to
    # TWO coexisting valid answers rather than one winner -- "no rule
    # found, alternate" (this project's own standing policy for this
    # exact shape, see [[feedback_methodology_preferences]]), not forced
    # into a single pick:
    #   "lugha" (language): native speaker gives "lumi" AS WELL AS the
    #   already-exported candidate pair "kiteto"(singular)/"viteto"
    #   (plural) -- all 3 apparently usable. CHECKED DIRECTLY: "lumi" is
    #   ALREADY this project's own word for "ulimi" (tongue) --
    #   VERIFIED_SUBJECT_MARKERS["lumi"] = "lu", added independently,
    #   long before this round. NOT treated as a conflict/error --
    #   tongue/language sharing one word is a real, common cross-
    #   linguistic pattern (English "mother tongue") -- but flagged here
    #   explicitly rather than silently reusing a homograph without
    #   noting it. "lumi" used here as the default (shortest, number-
    #   invariant, no sg/pl complication to wire, and ALREADY has a
    #   confirmed class, 'lu', unlike kiteto/viteto below); kiteto/viteto
    #   recorded in LUGHA_ALTERNATE_SG_PL below, NOT discarded.
    #   "mawazo" (thoughts): BOTH exported candidates confirmed valid --
    #   "maririkanyo" used as the default (the higher-scoring of the 2
    #   in this session's own frequency-aware reranking), "maghesho"
    #   recorded as an equally-valid alternate, not a rejected one.
    "lugha": "lumi",
    "mawazo": "maririkanyo",
})
LUGHA_ALTERNATE_SG_PL = {"singular": "kiteto", "plural": "viteto"}  # see "lugha" note above -- NOT
# wired as the default (no established Kitaita class confirmed for kiteto/viteto yet, unlike the
# maskini/matoleo/majira sg/pl pairs above), kept for the record and a future round.
MAWAZO_ALTERNATE = "maghesho"  # see "mawazo" note above -- equally valid, not a rejected candidate.
MASKINI_PLURAL = "wakiwa"  # not wired to any plural-detection trigger yet -- see the note above.

# 2026-10-22: "mavuno"'s own verb root, "kuwida" (= kuvuna, to harvest),
# same direct-correction round -- "kuwida" = ku-wid-a (class-15
# infinitive marker + root + final vowel), so the bare root is "wid".
VERIFIED_ROOTS.update({
    "vun": "wid",
    # 2026-10-22: "furahi" (enjoy/rejoice) -> "boi", direct native-
    # speaker correction ("furahia -> boilwa"), then CROSS-CHECKED
    # against the full corpus by direct reverse-inference request --
    # "boilwa" = root "boi" + this project's OWN already-established
    # regular passive infix (kitaita_passive_form(): vowel-final root ->
    # "-lw-", i.e. "boi"+"lw"+"a" = "boilwa", not a special form). 620
    # corpus tokens, 90 distinct surface forms, extremely clean and
    # consistent with every TAM/person table already on file:
    #   ka-narrative past: "u-ka-boilwa" (EXO.18.9, Jethro/3sg),
    #     "wi-ka-boilwa" (GEN.45.16, 3pl)
    #   cha-future: "ni-cha-boilwa" (PSA.9.2, 1sg), "wi-cha-boilwa"
    #     (PSA.14.7, 3pl), "m-cha-boilwa" (DEU.12.7, 2pl)
    #   da-perfect/stative (used for a GENERAL present/habitual state,
    #     not strictly "has become" -- a real stative-verb aspect
    #     blur, not an error): "wa-da-boilwa" (PSA.1.2/68.3, 3pl,
    #     "wa-" is class 'wi''s own PRESENT_SM_VOWEL_CONFIRMED-shifted
    #     form, consistent with existing tables), "na-da-boilwa"
    #     (PSA.16.9, 1sg)
    #   bare/zero-TAM present-like: "na-boilwa" (GEN.30.13, 1sg, "ni-"
    #     elided before directly-attached root -- the SAME contraction
    #     already wired this session for "na_have"/"nafurahia"-type
    #     verbs, an independent corpus confirmation of that mechanism)
    #   habitual "-agha" suffix: "na-boilwagha" (EZK.18.23, 1sg),
    #     "wi-boilwagha" (PSA.35.26, 3pl)
    #   NEGATIVE: "nde-boilwagha" (PSA.147.10, 3sg, "He is not
    #     pleased by..."), "si-boilwagha" (ISA.1.11, 1sg -- the
    #     already-established irregular 1sg-negative "si-" prefix,
    #     here keeping FV "-a"+habitual rather than the regular
    #     negative-present "-e" shift)
    #   bare infinitive, used NOMINALLY as "joy" itself (like Swahili
    #     "furaha" the noun, not a verb in these examples): "kuboilwa"
    #     (305x, by far the most common form -- GEN.18.12 "nadima
    #     kuboilwa?"<->"nitapata furaha", PSA.4.7 "kuboilwa kubaa"<->
    #     "Umenitia furaha" [great joy])
    #
    # STRUCTURAL FINDING, not yet wired into the pipeline (see docs/
    # status.md for the open question this leaves): this verb has NO
    # plain active transitive form in the corpus at all -- the
    # STIMULUS ("what one rejoices in/is pleased by") is consistently
    # introduced by the preposition "ni" ("by"), e.g. PSA.1.2
    # "wadaboilwa ni Sharia ya BWANA" <-> "sheria ya BWANA
    # ndiyo impendezayo" (the LAW is what PLEASES [them] -- Swahili's
    # own ACTIVE "furahia X" corresponds to Kitaita's PASSIVE-SHAPED
    # "boilwa ni X", exactly matching the user's own direct instinct
    # ("I think it translates directly to 'nafurahishwa na nyimbo
    # zao'"). This "ni" is the PASSIVE AGENT MARKER ("by"), a
    # DIFFERENT word from the invariant copula "ni" ("is/are", wired
    # earlier this session) and from "ni" nowhere stripped as an
    # object marker -- do not conflate the three. NOT generalized into
    # an automatic "object -> 'ni'+object" rewrite rule here: that
    # would change this verb's whole ARGUMENT STRUCTURE (subject
    # stays the same, but what was the Swahili DIRECT OBJECT becomes a
    # PREPOSITIONAL PHRASE in Kitaita), a bigger, sentence-level
    # mechanism than a root substitution -- flagged as an open
    # decision, not silently built.
    "furahi": "boi",
})

# ----------------------------------------------------------------- deverbal/infinitive-noun subject marker: HYPOTHESIS, not corpus-induced
# 2026-10-08 cont. 13. Kitaita class-15-style infinitive-as-noun subjects
# (kuseka/kulia, see VERIFIED_NOUN_TRANSLATIONS above) have NEVER been
# tested as a sentence subject by this pipeline -- there is no statistical
# OR corpus-induced subject-marker candidate for this category at all. The
# value here is borrowed from a DIFFERENT, SIBLING research track in this
# same project (the Swahili-internal concord-rewrite pipeline's own from-
# scratch induction over real Kitaita Bible text, kitaita_agreement_
# evidence.py) which found a candidate class-15 subject marker 'ghu' (see
# [[project-concord-pipeline-state]] memory, "1:u, 2:wi, 3/11:ku, 4/9:i,
# 5:w, 6/10:r, 7:ch, 8:ve, 15:ghu"). This is weaker evidence than even this
# project's own "corpus-induced" tier (VERIFIED_SUBJECT_MARKERS is direct
# human correction, trusted at n=1; the ordinary noun_class_of fallback is
# THIS project's own from-scratch induction) -- kept in its own table,
# NEVER silently treated as verified, so a real judgment here is pure new
# information either way.
DEVERBAL_SM_HYPOTHESIS = {
    "kuseka": "ghu",
    "kulila": "ghu",  # kulia (to cry)'s existing thin statistical candidate (Dice 0.32,
                        # no class data) -- same hypothesis, 2nd independent test
}
# 2026-10-08 cont. 14 (batch 11 judged, s144/s147): RETRACTED. Both tested nouns
# (kuseka, kulila) directly contradict this 'ghu' hypothesis -- the real marker is
# 'ku' (s144 "Kuseka ghukaghamba"->correct "kukaghamba", s147 "Kulila ghukaghamba"
# ->correct "kukaghamba"), 2/2 clean. Kept here as a historical record of a
# sibling-track hypothesis that did NOT transfer; see VERIFIED_SUBJECT_MARKERS
# below for the now-native-speaker-verified value actually used (it takes
# priority in the generator's lookup order, so this retracted table is dead code
# for these two nouns specifically, not deleted).

# ----------------------------------------------------------------- future 'kuwa': which classes need the fuller 'kai' root
# 2026-10-08 cont. 12 (batch 10). The future-'kuwa' bare-root-'k' rule
# ("GENERALIZED, confirmed across 7 more nouns with zero exceptions",
# cont. 16) has its first REAL exceptions -- but they're clean at the
# CLASS level, not per-noun: both tested 'wi'-class nouns (watu s118,
# wazee s130) and both tested 'chi'-class nouns (kikombe s107, kiti
# s125) independently need the FULLER root 'kai' instead of bare 'k'
# ("wichakaia"/"chichakaia", not "wichaka"/"chichaka") -- 2/2 clean on
# each class, no exceptions within either one. This is DIFFERENT from
# the 'i' class, where the SAME future-tense question splits PER NOUN
# (nyumba/shimba take bare 'k'; miti/nyoka need 'kai') -- not
# generalized there, kept as per-noun exact overrides instead (see
# VERIFIED_SENTENCES).
# 2026-10-08 cont. 14, then RE-RETRACTED same day (cont. 15): this set briefly
# excluded "chi" (see the dead paragraph below, kept for the record of what went
# wrong) after a full-regression audit found s107 (kikombe)'s stored correction
# read "Kikombe chichaka" (bare-'k', no '-ia'), contradicting kiti's "chichakaia"
# and cont. 12's own comment. Direct correction from the user (cont. 15): that
# was a MISTYPE in the original judgment, not a real per-noun split -- "the
# class is uniform". s107's stored correction fixed to "Kikombe chichakaia" to
# match; "chi" restored here, and the two exact per-noun overrides this
# round's retraction added are removed again (see VERIFIED_SENTENCES). Net: the
# ORIGINAL cont. 12 reading was right all along; the problem was bad data in the
# judgment database, not a bad generalization.
#   [cont. 14's now-superseded reasoning, kept so a future reader can see what
#    the brief retraction looked like and why: "s107's actual stored correction
#    is 'Kikombe chichaka' (bare-'k', NO '-ia'), not 'chichakaia' as cont. 12's
#    own comment claimed -- a transcription error made while writing that
#    comment, not a re-judgment... the 'chi' class genuinely SPLITS per noun for
#    this exact combination."]
FUTURE_KUWA_FULL_ROOT_MARKERS = {"wi", "chi"}

# ----------------------------------------------------------------- subject markers
# Keyed by the specific KITAITA noun (not generalized to a whole class
# yet -- safer with this little data per class).
VERIFIED_SUBJECT_MARKERS = {
    # 2026-10-23: direct user correction -- "umbali" (distance) -> "wukula"
    # is the SAME class as "wughoma" (mercy). Neither word was independently
    # in this project's data before now, but "wughoma" is itself an
    # abstract "wu-" noun -- and "wukula" sharing the exact "wuk-" 3-letter
    # prefix with 4 ALREADY-confirmed "gh"-class nouns (wukelu/kindness,
    # wukiwa/desolation, wukomboli/redemption, wukongo/sickness -- 4/4,
    # 100%) independently predicts the SAME answer the user just gave
    # directly, by word shape alone (see predict_kitaita_class_from_prefix()
    # in kitaita_e2e_pipeline.py, added the same session this correction
    # was made, for exactly this situation generalized). Two independent
    # signals agreeing, not just one.
    "wukula": "gh",
    "ituku": "ji",    # was "ku" -- confirmed 2x (s8 sem+ka, and isanga's twin case s10)
    "isanga": "ji",   # was "ku" -- confirmed 2x (s10 sem+ka, s11 ta+kuw)
    "irangi": "ji",   # 2026-10-20: "maisha" (life), reverse-inferred -- see VERIFIED_NOUN_
                        # TRANSLATIONS's own comment just above for the full evidence trail.
    "matuku": "ghi",  # 2026-10-20: plural of "ituku"/"siku" (day/days) -- see
                        # SIKU_PLURAL_TRIGGER's own comment for the full evidence trail.
    # 2026-10-13 (cont. 3): direct user correction -- nyumba is "i_nasal"
    # (class 2, same group as "ndagha"/knife, adjective prefix "nd-"), NOT
    # "i_plain" (class 1, "midi"/trees, adjective prefix "mi-") as cont. 2
    # assumed by analogy from Swahili noun shape alone. The SUBJECT MARKER
    # is still plain "i-" (TAM_SM_ALIAS handles that, unchanged) -- only
    # the noun's class IDENTITY changes, which only matters for the
    # adjective tables. baya's native-speaker-verified "izamie" for nyumba
    # (s49) is NOT a counterexample: baya (and zuri) use the bare subject-
    # marker prefix UNIFORMLY, never the mi-/nd- split -- see the comment
    # above BAYA_STEM ("baya... does NOT take the subject-marker prefix at
    # all" was about kubwa; zuri's own comment already says it uses "the
    # SAME strategy as baya", i.e. bare prefix). Only kubwa/refu/dogo/geni/
    # fupi/zito/chafu (the DICT-based "special paradigm" adjectives) use
    # the real class identity and therefore see the split.
    "nyumba": "i_nasal",
    "samaki": "ri",    # no prior marker at all -- new, 1x (s14 "samaki rikawona"). NOTE: samaki
                        # is ANIMATE-SG like mundu/mlungu (which use 'u'), so 'ri' is a genuinely
                        # different marker for a noun this project's Swahili-derived label called
                        # the same class -- another within-Kitaita distinction Swahili doesn't make
                        # (cf. the siku/ituku vs matuku case, and the mtu/samaki 'jua' conflict below).
                        # 2026-10-08 cont. 16: 'ri' here is now understood as the general PLURAL-
                        # reading default for NUMBER-INVARIANT nouns, not a class unique to samaki --
                        # see NUMBER_INVARIANT_NOUNS below for the 10 other nouns sharing this pattern.
    # 2026-10-08 cont. 16: direct correction generalizing the samaki pattern -- these 10
    # nouns are ALSO number-invariant (same Kitaita word for singular and plural, like
    # samaki). Originally defaulted ALL of them to the plural-reading 'ri' (samaki's own
    # majority vote, 7/8 judged sentences) -- 2026-10-08 cont. 18 RETRACTS that default
    # for every one of these 9 except 'ngelo': the user directly classified all 9 into
    # the 'i' (singular-shape) bucket via the Kitaita Class Sorter artifact, a clean,
    # deliberate, independent signal across 9 words at once -- and it actually agrees
    # with the only 2 of these that have a REAL judged sentence (chui/inge, s89,
    # "Inge ikaida mwalimu."; ndege/nyonyi, s87, "Nyonyi iweobua inge." -- both singular
    # 'i-', which this project had been treating as flagged EXCEPTIONS to a 'ri' default
    # that, in hindsight, was over-generalized from samaki alone). 'ngelo' (wakati/time)
    # is the one member the sorter placed in 'ri' instead, matching its ALREADY-correct
    # entry below -- consistent, not a counterexample. 'ri' is now each of these 9
    # nouns' documented PLURAL alternative, not the default; see NUMBER_INVARIANT_SM_
    # PLURAL above. Evidence tier: direct native-speaker CLASSIFICATION (no generated
    # sentence yet, unlike a VERIFIED_SENTENCES entry) -- trusted per this project's
    # standing rule (human input wins over this project's own prior inference), but
    # flagged here as a different kind of evidence than a judged sentence.
    "ngelo": "ri",     # wakati (time) -- sorter-confirmed, matches the samaki-style plural default
    "mbuwa": "i",      # shamba (farm)
    "barua": "i",      # barua (letter) -- a loanword retained as-is in both languages
    "sere": "i",       # amani (peace)
    "inge": "i",       # chui (leopard) -- matches s89's own judged singular answer exactly
    "nguku": "i",      # kuku (chicken)
    "koshi": "i",      # mbwa (dog)
    "supu": "i",       # mchuzi (sauce)
    "ngokoro": "i",    # magoti (knees)
    "nyonyi": "i",     # ndege (bird) -- matches s87's own judged singular answer exactly; the
                        # ("ndege","ogop","na",None) exact override below is now redundant (composes
                        # correctly through this default) but left in place, harmless
    "mwana": "u",      # 2026-10-07 cont. 15: an UNRELATED background rerun of the full induction
                        # pipeline (build_pages.py, kicked off for the separate Judge-page question
                        # this same session) flipped mwana's induced per-noun subject marker from
                        # 'u' to 'j' -- almost certainly induction noise/non-determinism in the
                        # per-noun vote, not a real finding (mwana stayed in the SAME follower-marker
                        # cluster "w", same as mungu/mtu). Overridden back to 'u', which is directly
                        # confirmed by 5 separate correct judgments (s12, s13, s22, s33, s38, s52 all
                        # use 'u-'/'wa-' consistently) -- overwhelming direct human evidence over one
                        # noisy, unsupervised corpus-induction vote. No backup of the pre-rerun JSON
                        # exists (project isn't under version control) -- this override is the fix,
                        # not a revert.
    "indi": "ji",      # 2026-10-07 cont. 19 (batch 5): 'indi' (mfupa/bone) inherited the
                        # RAW induction's bare consonant 'j' (never speaker-confirmed, unlike
                        # isanga/ituku's hand-corrected 'ji') -- s67 ("Indi juchaka"->"Indi
                        # jichaka") and s69 ("Indi judeka..."->"Indi jideka...") both directly
                        # confirm the vowel is 'i', matching isanga/ituku exactly. This single
                        # fix resolves BOTH s67 and s69 automatically via the existing
                        # generalized future-kuwa and KUBWA_IBAA_SM rules.
    # 2026-10-08 cont. 4 (batch 7 judgments, s86/s88/s90/s91/s94/s95):
    # 5 MORE from-scratch-induction class assignments overridden by
    # direct correction, all the SAME pattern as 'indi' above -- the
    # raw per-noun induction vote landed on the wrong class/marker for
    # a noun this project had never speaker-tested before:
    "mwalimu": "u",    # induction gave bare 'k' (flagged, linking-vowel-guessed) -- 3 INDEPENDENT
                        # corrections all agree: s86 ("Mwalimu kukauwa mbaa"->"mwalimu ukavika
                        # mbaa"), s88 ("...kukarwaa..."->"...ukarwaa..."), s96 ("...kukawona...
                        # na kukasimama"->"...ukawona... na ukasimama") -- 'u-' (plain w-class,
                        # same as mtu/mungu/mwana/malaika), not the induced bare 'k'.
    "sanduku": "ji",   # induction gave class 'y'/'i' -- s90 ("Sanduku idengia"->"Sanduku
                        # jidengia") corrects to 'ji-', putting sanduku in the SAME class as
                        # nchi/siku/isanga/ituku/indi, not its own induced 'y' class.
    "mori": "ghu",     # mwezi's Kitaita word; induction gave class 'g'/'i' -- s94 ("Mori
                        # ichaka"->"Mori ghuchaka") corrects to 'ghu-', the SAME class as
                        # mti/msalaba, not its own induced 'g' class.
    # 2026-10-08 cont. 10: RETRACTED. s108/s109 (batch 8's "narrative/
    # future both confirmed correct as generated, 'yu-'") were
    # REVISED by the speaker in batch 9 to "wrong" -- real answers
    # "Shimba ikaghamba"/"Shimba ichaka", marker 'i-', NOT 'y-'/'yu-'
    # at all. There never was a separate 'y' class: 'shimba' is the
    # SAME CLASS AS NYUMBA (bare vowel 'i-'). The original present-
    # tense form "yawemanya" (s95) was never wrong -- it's the SAME
    # 'i'->'ya' present shift (y-GLIDE inserted before the vowel-
    # shifted 'a', the front-vowel mirror of the 'u'->'wa' w-glide)
    # this project had long speculated for nyumba's own class ("By
    # analogy with the w-class's 'w'-glide, a Bantu-typical 'y'-glide
    # is the linguistically likely prediction" -- PRESENT_SM_VOWEL_
    # CONFIRMED below, written before this confirmed it). The 'y' was
    # never the base consonant; it was a glide this project
    # misattributed as the marker itself. Batch 9's existential past
    # (s116, "Shimba erekoghe") independently confirms bare 'i' shifts
    # to 'e' before '-re-', the SAME front-vowel shift 'ji'->'je'
    # already has (see kogh_re_o_shift below) -- now a 3rd confirmed
    # instance of that pattern (ji, i, and the person-marker 'ni'->'ne'
    # from cont. 1).
    # 2026-10-13 (cont. 3): "shimba"/"choka" moved to "i_nasal" alongside
    # "nyumba" -- this project's OWN established precedent (below, and the
    # 2026-10-08 cont.10/cont.6 findings) is that these two share nyumba's
    # SUBJECT-MARKER/TAM behavior exactly ("dissolves entirely into
    # nyumba's class" / "the SAME 'i' class as nyumba/shimba/miti") -- that
    # finding is untouched (TAM_SM_ALIAS keeps it true). What's NEW is the
    # adjective-class identity, which that finding never tested.
    "shimba": "i_nasal",
    "choka": "i_nasal",   # nyoka (snake)'s Kitaita word; had NO class/marker data at all --
                        # s123/s124 (batch 9) confirm the SAME 'i' class as nyumba/shimba/miti
                        # ("Choka ikaghamba"/"Choka yawemanya..."), a 4th independent member.
    # "modo": "ghu" -- REMOVED 2026-10-08 cont. 16: this entry, and the matching
    # VERIFIED_NOUN_TRANSLATIONS["mto"]="modo" override it depended on, were based on a
    # mixup between 'mto' (river) and 'moto' (fire) -- see the retraction note above
    # VERIFIED_NOUN_TRANSLATIONS. The original judgment's own note, "wrong class", reads
    # very differently in hindsight: not "this noun is in a different class" but "this
    # isn't even the right word." mto now falls back to the statistical 'moda' again,
    # untested by any real correction.
    # 2026-10-08 cont. 7 (batch 8 judgments, s100-s107): THREE separate
    # from-scratch-induction guesses (kiti's bare 'k', kitabu's bare
    # 'ma', and kikombe's own 'ch' entry just above -- all three
    # WRONG) turn out to be the SAME real class, base marker 'chi-'
    # (a consonant+vowel marker, NOT a bare consonant needing a
    # linking-vowel guess at all): s100 ("Kifumbi kukaghamba"->
    # "kifumbi chikaghamba"), s102 ("Chuo makaghamba"->"chuo
    # chikaghamba"), s106 ("Kikombe chukaghamba"->"Kikombe
    # chikaghamba") all independently confirm 'chi-' before the
    # consonant-initial narrative '-ka-' -- NOT the generic bare-
    # consonant default ('u') kikombe's old 'ch' entry would have
    # produced. s107 ("Kikombe chuchaka"->"Kikombe chichaka") confirms
    # the SAME 'chi-' before future '-cha-' too. Present tense keeps
    # the already-known 'cha-' (s101/s103 confirm 'chi'+'a'->'cha',
    # the SAME i->a shift 'ji'->'ja' already has, just previously
    # mis-attributed to a nonexistent bare-consonant form). This
    # REPLACES kikombe's old "ch" entry above -- do not reintroduce it.
    "kikombe": "chi",
    "kifumbi": "chi",  # kiti (chair)'s Kitaita word -- induction wrongly gave bare 'k' (the
                        # SAME wrong-induction pattern 'mwalimu' had two rounds ago)
    "chuo": "chi",     # kitabu (book)'s Kitaita word -- induction wrongly gave 'ma'
    "igho": "ji",      # jiwe (stone)'s Kitaita word -- induction gave bare 'j' (no vowel,
                        # never speaker-confirmed); s104 ("Igho jukaghamba"->"Igho jikaghamba")
                        # and s105 ("Igho juwemanya"->"Igho jawemanya") both confirm this is the
                        # SAME already-verified 'ji' class (nchi/siku/mfupa/sanduku), not a
                        # separate bare-'j' class -- the SAME "induction dropped the vowel"
                        # pattern 'indi' had (cont. 19).
    # 2026-10-08 cont. 14 (batch 11, s138/s142): maji/mayai (class-6 "ma-" nouns) --
    # the statistical class guess 'ghu' (same bare-'u'-vowel class as mti/mlima/
    # msalaba/mwezi) is WRONG: s138 ("Machi ghukaghamba"->"Machi ghikaghamba") and
    # s142 ("Maghi ghukaghamba"->"Maghi ghikaghamba") both independently confirm a
    # DIFFERENT front-vowel marker, 'ghi-', 2/2 clean -- same consonant 'gh' as the
    # mti-class, but the OTHER vowel, patterning like 'ji'/'i'/'wi' (unshifted
    # narrative, shifts to 'gha' before present -- already matches s139's
    # independently-judged-correct "ghawemanya" -- and 'ghe' before existential
    # past -- matches s141's "gherekoghe"), not like 'ghu'/'u'. See
    # PRESENT_SM_VOWEL_CONFIRMED and KOGH_RE_FRONT_VOWEL_SHIFTS below for the two
    # TAM-conditioned shifts this adds.
    "machi": "ghi",    # maji (water)'s Kitaita word
    "maghi": "ghi",    # mayai (eggs)'s Kitaita word -- 2nd independent confirmation of 'ghi'
    # 2026-10-08 cont. 14 (batch 11, s144/s147): kuseka/kulila (class-15-style
    # deverbal-noun subjects) -- DEVERBAL_SM_HYPOTHESIS's borrowed 'ghu' guess
    # (above) is WRONG; both tested nouns confirm 'ku-' instead (s144 "Kuseka
    # ghukaghamba"->"Kuseka kukaghamba", s147 "Kulila ghukaghamba"->"Kulila
    # kukaghamba"), 2/2 clean -- now native-speaker-VERIFIED, no longer a bare
    # hypothesis. Notably identical to Swahili's own infinitive prefix 'ku-'
    # (direct retention), and the present-tense shift ('ku'->'kwa', s145) matches
    # this project's OWN already-verified person-marker finding for 2sg ('ku'->
    # 'kwa', PERSON_SUBJECT_MARKERS_BY_TAM above, s83) -- cross-confirms the same
    # glide-insertion pattern independently, from a completely different corner
    # of this project's evidence.
    "kuseka": "ku",
    "kulila": "ku",

    # 2026-10-08 cont. 20: direct user confirmation that the 2 candidate classes
    # flagged cont. 16 ("1 class is missing for vitoi/vadu... Another class missing
    # for lukundo lu...") are real -- "you haven't added the vadu and lukundo classes
    # yet." Deliberately only these 2, NOT 'vitoi' (wezi/thieves): the user left that
    # one specifically unresolved (still "__unsure__" in the Class Sorter, and the
    # project's own notes flag its 'vi-' shape as a genuinely open question, possibly
    # the 'chi' class's plural rather than its own class) -- not assumed settled just
    # because its siblings are. No TAM data of any kind exists yet for either 'lu' or
    # 'va' (not even a judged sentence) -- ROUND_12_SPECS (s148-s155) will show real
    # 'lu-'/'va-' forms once regenerated instead of '?' placeholders, but every
    # TAM-specific cell (vowel shifts, future-root shape, etc.) remains genuinely
    # unconfirmed until those sentences are actually judged.
    "lukundo": "lu",   # upendo (love) -- candidate class 11 'lu-', a 3rd shape in
                        # territory Swahili flattened to 'u-' (alongside this
                        # project's own 'u'/'ghu' split)
    "vadu": "va",       # viatu (shoes, plural of kiatu/chadu) -- candidate plural
                        # counterpart of the 'chi' class (ch/v, the standard Bantu 7/8
                        # pairing)

    # 2026-10-08 cont. 22: "updated. add to the grammar page" -- the user went on to
    # classify the ENTIRE rest of the Class Sorter's pool into 'lu'/'va' (not just
    # upendo/viatu), via the Sorter UI, directly confirming 'vitoi' (wezi/thieves)
    # belongs to 'va' too -- the exact open question cont. 16/20 had deliberately left
    # unresolved. 'va' now reads as the standard Bantu class 7/8 ki-/vi- plural
    # counterpart generally, not just for kiatu/chadu specifically: every 'chi'-class
    # singular this project already has (kikombe/kifumbi/chuo) corresponds to a
    # DIFFERENT, separately-spelled plural Kitaita word here (e.g. kifumbi/chair-sg
    # vs vifumbi/chairs-pl, vuo here vs chuo/kitabu-sg) -- no key collisions, checked
    # directly. 'lu' gained 2 more members beyond upendo, both abstract/body-part-ish
    # nouns (sauti/voice, ulimi/tongue) -- consistent with cont. 20's class-11 framing.
    # Same evidence tier as cont. 18/19 (direct classification, no judged sentence for
    # ANY TAM cell on either class yet) -- checked directly before trusting any of it,
    # every single Kitaita word here already matched this project's own statistical
    # top candidate for its Swahili source, 0 mismatches.
    "lwaka": "lu",     # sauti (voice)
    "lumi": "lu",      # ulimi (tongue)
    "vindo": "va",     # chakula (food)
    "viteto": "va",    # lugha (language)
    "vikomba": "va",   # mikia (tails)
    "vikapu": "va",    # vikapu (baskets)
    "vikombe": "va",   # vikombe (cups)
    "vibelele": "va",  # vipande (pieces)
    "vipimo": "va",    # vipimo (measurements)
    "visiwa": "va",    # visiwa (islands)
    "vuo": "va",       # vitabu (books) -- the PLURAL counterpart of chuo/kitabu-sg
                        # (already 'chi' class), a different word, no collision
    "vifumbi": "va",   # viti (chairs) -- the PLURAL counterpart of kifumbi/kiti-sg
                        # (already 'chi' class), a different word, no collision
    "via": "va",       # viungo (spices/joints)
    "vilambo": "va",   # vyombo (vessels)
    "vumba": "va",     # vyumba (rooms)
    "vitoi": "va",     # wezi (thieves) -- RESOLVES the cont. 16/20 open question:
                        # the 'vi-' shape is the standard class 8 plural, same as every
                        # other noun in this block, not its own separate class

    # ----- 2026-10-08 cont. 18/19: Kitaita Class Sorter batch (120 nouns) -----
    # Direct user request: "update the nouns I grouped previously to the concord table
    # so that we don't have to guess for now." Pulled from the Kitaita Class Sorter
    # artifact's `assignments` database (https://claude.ai/artifact/
    # WTboxS96wq4XE6KpQ3cRjr) -- every entry here is the project's own native speaker
    # DIRECTLY classifying a noun into one of the 9 established classes via the sorter
    # UI. 2026-10-08 cont. 19, direct confirmation: "I did them with native speakers
    # and have high confidence" -- these were NOT a solo guess, so they are trusted at
    # the SAME confidence level as this file's other VERIFIED_* tables, not a lesser
    # tier (see kitaita_class_sorter_classifications.py, included on the Concord
    # Review's noun-class page alongside the sentence-tested members, distinguished
    # there only by HOW each was confirmed -- classification vs. a judged sentence --
    # not by how much either is trusted). The one thing still genuinely open per noun:
    # no TAM has actually been exercised in a real sentence for these yet, so a test
    # sentence is still new information about the SPECIFIC inflected forms, even
    # though the base class itself is high-confidence. Checked directly before
    # trusting any of it: every single Kitaita word below already matched this
    # project's own statistical top candidate for its Swahili source (0 translation
    # mismatches across 131 sorted nouns) -- so no VERIFIED_NOUN_TRANSLATIONS entries
    # were needed, only subject-marker ones.
    # 18 nouns were left "unsure" by the user and are deliberately NOT included here
    # (lukundo/vadu/vitoi among them -- the 2 brand-new candidate classes this batch
    # was partly testing, plus chakula/lugha/mikia/sauti/ulimi/vikapu/vikombe/vipande/
    # vipimo/visiwa/vitabu/viti/viungo/vyombo/vyumba). ONE sorted item was excluded
    # even though it had a class: 'vilima' (hills) was sorted as 'mighondinyi', which
    # is 'mighondi' (hills, plural) + the locative suffix '-nyi' -- see LOCATIVE_SUFFIX
    # below; it is a locative FORM, not a plain class member, so adding it here would
    # misrepresent what was actually classified.
    "chombo": "chi", "chumba": "chi", "chughano": "chi", "chadu": "chi",
    "kifuniko": "chi", "kikapu": "chi", "kilole": "chi", "kisiwa": "chi",
    "kitambaa": "chi", "chia": "chi", "charo": "chi",
    "kitoi": "chi",  # mwizi (thief, singular) -- a clean new confirmed 'chi'-class member,
                      # and (with 'vitoi'/wezi/thieves-plural left deliberately unsure) exactly
                      # the ki-/vi- singular/plural pairing this project predicted 'vadu' might
                      # show for 'chadu'/'viatu' too.
    "machu": "ghi", "maonge": "ghi", "meso": "ghi", "madirisha": "ghi",
    "mavuda": "ghi", "magare": "ghi", "magunia": "ghi", "maza": "ghi",
    "malagho": "ghi", "madu": "ghi", "matunda": "ghi", "magho": "ghi",
    "mariwa": "ghi", "maghegho": "ghi", "maghonyi": "ghi", "maghu": "ghi",
    "maringo": "ghi", "makochi": "ghi",
    "mfwano": "ghu", "mkate": "ghu", "munda": "ghu", "mkonu": "ghu",
    "msidu": "ghu", "mumbi": "ghu", "mfunguo": "ghu",
    "moda": "ghu",  # mto (river) -- DIRECT, independent confirmation of this session's own
                     # mto/moto retraction: the user's own sort puts 'moda' (not 'modo') in the
                     # SAME 'ghu' class as mti/msalaba/mori/mghondi, exactly as expected once
                     # the moto mixup was caught and undone.
    "ighuri": "i", "elimu": "i", "hachi": "i", "kalamu": "i", "kazi": "i",
    "ndagha": "i", "mirumenyi": "i", "misarigho": "i", "meri": "i",
    "mimbi": "i", "mikate": "i", "mikonu": "i", "mighondi": "i", "meda": "i",
    "ngolo": "i", "ng'ombe": "i", "nguwo": "i", "nyama": "i", "mbogho": "i",
    "bara": "i", "mbanga": "i", "saa": "i", "mbeo": "i",
    "igare": "ji", "ighegho": "ji", "ilagho": "ji", "indu": "ji",
    "itunda": "ji", "irika": "ji",
    "kuboilwa": "ku", "kubora": "ku", "kukimbia": "ku", "kuja": "ku",
    "kughua": "ku", "kuwona": "ku", "kurama": "ku", "kudamba": "ku",
    "kuliwa": "ku", "kushoma": "ku", "kusela": "ku", "kuuza": "ku",
    "kughu": "ku", "kuzoya": "ku", "kukanye": "ku", "kudu": "ku",
    "kughoka": "ku",
    "miri": "ri", "soko": "ri", "ndiwa": "ri",
    "wubanana": "u", "wughanga": "u", "msawi": "u", "mzuri": "u",
    "mghenyi": "u", "mkongo": "u", "mlimi": "u", "mlindiri": "u",
    "mdumiki": "u", "mnughi": "u", "wuhuru": "u", "wuwana": "u",
    "wurigha": "u", "wushapa": "u", "wulacha": "u", "wushu": "u",
    "wughadi": "u", "wughosi": "u", "wulemeri": "u",
    "wasawi": "wi", "walisha": "wi", "wazuri": "wi", "washami": "wi",
    "waghenyi": "wi", "wakongo": "wi", "wabori": "wi", "walimi": "wi",
    "walindiri": "wi", "wanughi": "wi", "waka": "wi", "wai": "wi",
    "wana": "wi", "wadumiki": "wi",
}

# 2026-10-13 (cont. 4): per-noun Kitaita class, now collected directly in
# the Kitaita Word Bank (the per-noun class dropdown added cont. 2) --
# covers both the 2026-10-12 noun batch (translation only, no class until
# now) and the 17 new nouns added this round. Checked each against the
# existing Class Sorter data (noun_sort_data.json, folded into the big
# dict above) before adding -- 3 apparent collisions found, 2 resolved,
# 1 left genuinely unresolved:
#   - "mbuwa" (shamba/farm, existing "i") vs the new "mashamba"->"mbuwa"
#     cls "ri": NOT a conflict -- "mbuwa" is already in NUMBER_INVARIANT_
#     NOUNS, and "i" (singular default) / "ri" (plural default) are
#     EXACTLY the existing documented pair for number-invariant nouns.
#     No override added; already correctly handled.
#   - "ngolo" (moyo/heart, existing "i") vs the new "mioyo"->"ngolo" cls
#     "i_nasal": the old "i" entry predates the i/i_nasal split entirely
#     (it could not have chosen "i_nasal" -- the option didn't exist) --
#     treated as a refinement, not a contradiction; overridden below.
#   - "mbanga": existing Class Sorter entry is "pango" (cave) -> "mbanga",
#     class "i". The new submission is "mwamba" (rock) -> "mbanga", class
#     "ghu" -- a REAL, unresolved conflict (same Kitaita surface form,
#     two different Swahili glosses, two different classes). NOT
#     overridden -- "mwamba"'s translation update is applied above
#     (VERIFIED_NOUN_TRANSLATIONS), but no class is set for it here.
#     Needs a native speaker: is "mbanga" one word covering both cave
#     and rock (and if so, which class), or two distinct words that
#     happen to be homophones?
VERIFIED_SUBJECT_MARKERS.update({
    "ngolo": "i_nasal",   # moyo/heart -- UPDATE, see note above
    "choronyi": "chi", "kitemberu": "chi", "wutesia": "ghu", "ilagho": "ji",
    "iriso": "ji", "kuchelewa": "ku", "kughesha": "ku", "kumerea": "ku",
    "kuogelia": "ku", "kudeka": "ku", "kughoresha": "ku", "kusea": "ku",
    "machungwa": "ghi", "mabemba": "ghi", "masanduku": "ghi", "mchango": "ghu",
    "msarigho": "ghu", "mfuko": "ghu", "mghosi": "u", "muri": "ghu",
    "boi": "i", "maghome": "ghi", "wundu": "ghu", "wutini": "ghu",
    "loli": "ghu", "wulongozi": "ghu", "wusalama": "ghu", "wubinana": "ghu",
    "wanake": "ghu", "makaji": "ghi", "ndagaa": "ri", "vyoronyi": "va",
    "walimu": "wi", "womi": "wi", "ighi": "ji",
    # 17 new nouns (2026-10-13 cont. 4):
    "ionge": "ji", "ngoma": "i_nasal", "misidu": "i", "mkopo": "ghu",
    "mkristo": "u", "waziri": "u", "mpishi": "u", "mwai": "u",
    "mndwawomi": "u", "mwandiki": "u", "waandishi": "wi", "vitambaa": "va",
    "fuwe": "i_nasal", "kwaru": "i_nasal", "mapatano": "ghi", "uzi": "ghu",
})

# 2026-10-22: direct native-speaker correction -- "swali" (question) ->
# "ikoto" (singular, class 'ji'), "maswali" (plural) -> "makoto"
# (plural, class 'ghi'). Same ji-singular/ghi-plural SHAPE this project
# already has for other ji-class nouns in general (e.g. "ituku"/
# "matuku" -- day/days), not a claim that "ikoto"/"makoto" share a root
# with any specific existing entry.
VERIFIED_SUBJECT_MARKERS.update({
    "ikoto": "ji",
    "makoto": "ghi",
    # "papa" (Pope, VERIFIED_NOUN_TRANSLATIONS above) never got a class
    # registered when it was added -- found directly while testing the
    # "na_have"/negation fix on "Papa hana uwezo." (kit_sm/kit_class
    # came back None, breaking every concord mechanism for "papa", not
    # just negation). A human-reference title/proper-noun defaults to
    # this project's ANIMATE-SG class, same as "mtu"/"mwalimu"/"mzee" --
    # not independently native-speaker-tested for "papa" itself, but
    # there is no plausible alternative class for a person reference.
    "papa": "u",
    # 2026-10-22, same AP-paragraph round -- direct confirmation, not an
    # inference this time: user pointed out "wote" (all) next to
    # "Wakatoliki" should already resolve to "wose" via this project's
    # existing -ote concord rule (OTE_STEM="ose", wa-/wi-class elides to
    # bare 'w' -- see apply_ote_concord()'s own evidence, EXO.12.47 etc.,
    # "wote" -> "wose" for 'wa'/'wi'-class nouns). That ONLY resolves if
    # "wakatoliki" itself is class 'wi' -- so this confirms the class,
    # not just plausible by analogy to "watu"/"wandu" as flagged before.
    "wakatoliki": "wi",
    # 2026-10-22: "lumbo" (wimbo/song) -> class 'lu' (same as lukundo/
    # love); "chumbo" (nyimbo/songs, a DIFFERENT word, not this same
    # root pluralized) -> class 'ri'. Direct native-speaker correction.
    "lumbo": "lu",
    "chumbo": "ri",
})

# ----------------------------------------------------------------- adjective concord
# NO Kitaita adjective-concord table existed before this round -- every
# entry here is first-pass seed data, not a mature paradigm. Keyed by
# (swahili_adjective_stem, swahili_class).
# Keyed by (stem, swahili SUBJECT NOUN) -- NOT the hand-labelled Swahili class, because
# nchi and nyumba share the same Swahili-class label ("9/10") but take DIFFERENT 'kubwa'
# forms (s20 "ibaa" vs s51 "mbaa") -- the exact same "same label, different real Kitaita
# class" pattern this project keeps finding (siku/matuku, mtu/samaki). A class-keyed
# table would silently collide these two; a noun-keyed one can't.
VERIFIED_ADJECTIVES = {
    ("kubwa", "mungu"): "mbaa",   # s18 -- "m-" prefix matches the standard ADJ_BASE
                                   # class-1 shape (cf. mzuri's "m-" in Swahili) -- plausible, not just coincidence.
    ("kubwa", "mti"): "mbaa",     # s19 -- SAME form as mungu (ADJ_BASE also gives
                                   # class 1 and 3 the same "m-" prefix in Swahili -- consistent).
    ("kubwa", "nchi"): "ibaa",    # s20 -- DIFFERENT prefix ("i-" not "m-").
    ("kubwa", "nyumba"): "mbaa",  # s51 -- shares mungu/mti's "mbaa", NOT nchi's "ibaa",
                                   # despite nyumba and nchi sharing the Swahili "9/10" label.
    ("kubwa", "siku"): "ibaa",    # s62, batch 4 -- matches nchi, NOT nyumba, confirming the
                                   # split tracks the subject-marker CONSONANT ('ji'), not the
                                   # Swahili class label -- see KUBWA_FORM_BY_SM note below.
    ("baya", "mtu"): "uzamie",    # s21 -- does not share a stem with "-baa", so either a
                                   # separate adjective lexeme for "bad" or a suppletive form.
                                   # ('baya' otherwise uses the generalized BAYA_STEM rule now --
                                   # this entry predates that and is kept for mtu's exact record.)
    # 2026-10-11: promoted from VERIFIED_SENTENCES exact-sentence overrides (cont. 10/11,
    # kikombe/kiti/kitabu, all 'chi'-class nouns) into this flat per-noun table -- those
    # entries only applied to the ONE exact (noun, root='kuw', tam='li') combination they
    # were recorded under, so a 'chi'-class noun in a DIFFERENT tense fell through to the
    # generalized rule below, which didn't know about 'chi' at all and would have wrongly
    # defaulted to 'mbaa'. Same underlying data, now usable everywhere.
    ("kubwa", "kikombe"): "kibaa",
    ("kubwa", "kiti"): "kibaa",
    ("kubwa", "kitabu"): "kibaa",
}

# 'kubwa' (big), generalized for any noun NOT already in the flat table above:
# 2026-10-07 cont. 16 found 'ji' (nchi, siku) takes 'ibaa' while 'u'/'ghu'/'ri'/'i'
# (mungu/mti/mwana/samaki/nyumba) take 'mbaa' -- the split tracks the subject-marker
# CONSONANT, not the Swahili class label. 2026-10-08 cont. 10 separately found 'chi'
# (kikombe/kiti/kitabu, 3 independent nouns) takes a THIRD form, 'kibaa' -- confirmed
# data that was never promoted into this generalized fallback until now (2026-10-11),
# so an UNTESTED 'chi'-class noun would have silently gotten the wrong 'mbaa' default.
# Every other marker tried (u/ghu/ri/i) still defaults to 'mbaa'; markers with NO data
# at all (ghi/ku/lu/va/wi) ALSO default to 'mbaa' as the majority-vote guess, but
# callers should treat that case as a real unknown, not a confirmed answer -- see
# kitaita_sentence_generator.py's trace message, which says so explicitly only for
# the ji/chi-confirmed cases vs the bare fallback.
# 2026-10-11 (cont.): class-level subject-marker FALLBACK for a noun
# with NO Kitaita translation at all -- previously a hard stop that
# aborted the WHOLE clause, including the verb, even though the noun's
# SWAHILI class was already known (confirmed case: 'mzee', lexicon
# class ANIMATE-SG, zero Kitaita candidates at all -- "Mwanafunzi mrefu
# alimwona mzee akitembea..." silently failed to translate 'akitembea'
# at all because of this). Scoped deliberately narrow: "ANIMATE-SG"/
# "ANIMATE-PL" cover BOTH humans (confirmed Kitaita 'u'/'wi', 8+ nouns:
# mungu/mtu/mwana/malaika/mwalimu/mwokozi/mchungaji/mwanamke/mzee) AND
# animals (confirmed to take a DIFFERENT class each -- mbwa/nyoka/
# simba/ndege/chui all 'i', samaki its own 'ri' -- animals don't even
# agree with EACH OTHER). Applying "u"/"wi" to an excluded animal lemma
# would be a confident WRONG guess, not a reasonable default -- this
# project's own repeated "label lies" finding, mattering here too.
SWAHILI_CLASS_DEFAULT_SM_GUESS = {"ANIMATE-SG": "u", "ANIMATE-PL": "wi"}
ANIMATE_SG_EXCLUDED_FROM_DEFAULT = {"mbwa", "nyoka", "simba", "ndege", "chui", "samaki"}

# 2026-10-22: direct user instruction -- "personal pronouns should
# automatically use the agreement of mundu." Found while debugging
# "Mimi ni mrefu." ("I am tall") -- "mimi" (I) DOES have a Kitaita
# translation (the statistical table's "ini"), so it never reached the
# ANIMATE-SG class-label fallback above at all (that fallback only
# fires when there is NO Kitaita translation whatsoever); "ini" itself
# simply has no corrected-class subject-marker data of its own, so
# kit_sm/kit_class came back None, breaking BOTH the copula's own
# subject prefix and the adjective concord ("mrefu"/"mlacha") that
# depends on it. A personal pronoun is unambiguously human regardless
# of which Kitaita word it happens to translate to, so this is keyed
# on the SWAHILI LEMMA, not on "ini" or any other specific Kitaita
# surface form -- robust even if a pronoun's Kitaita translation
# changes later.
#
# CORRECTION, same day: the first version of this fix set kit_sm (and
# therefore BOTH the verb's own subject prefix AND the adjective
# concord) to "u" uniformly -- direct user correction: "Ini nideka
# mlacha", NOT "Ini udeka mlacha". The VERB uses PERSON agreement, the
# ADJECTIVE uses mundu's CLASS agreement -- two different things that
# happen to collide for "mimi" only because this project's class 'u'
# and 1st-person both exist, not because they're the same mechanism.
# `PERSONAL_PRONOUN_KITAITA_CLASS` ("u") is for the ADJECTIVE/concord
# side only now; `SWAHILI_PRONOUN_TO_PERSON` below feeds the VERB's own
# subject marker through this project's EXISTING person-marker system
# (PERSON_SM_DEFAULT -- the "bare citation form" tier, e.g. "ni" for
# 1sg, already established 2026-10-20 for a different gap) instead.
# "yeye"/"wao" (3rd person) have NO entry in `SWAHILI_PRONOUN_TO_PERSON`
# -- this project's person-marker scheme only ever covers 1st/2nd
# person (3rd normally comes from a real noun's own class); verb
# agreement for those two pronouns remains unresolved/flagged, not
# guessed, while the adjective-class side above still applies to all 6.
SWAHILI_PERSONAL_PRONOUNS = {"mimi", "wewe", "yeye", "sisi", "ninyi", "wao"}
PERSONAL_PRONOUN_KITAITA_CLASS = "u"  # mundu's class -- for ADJECTIVE/concord purposes
SWAHILI_PRONOUN_TO_PERSON = {"mimi": "1sg", "wewe": "2sg", "sisi": "1pl", "ninyi": "2pl"}

# 2026-10-20: GENERALIZED the fallback above to every Swahili noun class,
# direct user request ("for sentences where the subject/object is unknown
# ... use the mode per class") -- e.g. "Kile kinachonisikitisha" (that
# which saddens me) has NO concrete noun at all, only a class-7 demonstrative/
# relative marker, so even the EXISTING per-lemma fallback above (which
# still needs a specific noun lemma, just an untranslated one) can't help;
# this is the class-only, no-noun-at-all case.
#
# Computed directly from this project's OWN current data (every lexicon
# noun with a known Swahili class AND a known Kitaita translation --
# VERIFIED_NOUN_TRANSLATIONS first, the statistical noun_kitaita table
# otherwise -- grouped by Swahili class, Kitaita class via resolve_class_
# marker()), NOT assumed or carried over from reference grammar. Same
# "check the mode isn't hiding a real exception cluster" discipline the
# ANIMATE-SG entry above already uses (that one's own comment: "animals
# don't even agree with EACH OTHER") -- applied here by requiring the
# mode to cover clearly over half the sample AND not be a near-tie with
# a runner-up, and by EXCLUDING any class that fails that bar rather than
# guessing anyway:
#   15        ku   17/17 (100%) -- matches the already-known class-15 SM.
#   ANIMATE-PL wi  19/20 (95%)  -- matches the existing entry above exactly.
#   8         va   14/17 (82%)
#   6         ghi  18/22 (82%)
#   7         chi  12/15 (80%)
#   ANIMATE-SG u   21/36 (58%, animal exceptions already excluded above)
#   3         ghu  13/21 (62%)
#   4         i     9/15 (60%)
#   5         ji    9/17 (53%) -- weakest INCLUDED tier, just over half,
#             flagged lower confidence than the rest, not equal to them.
# EXCLUDED, checked and rejected rather than silently omitted:
#   11   (u 9, ghu 7, n=23) -- 'u' and 'ghu' are nearly tied, no real mode.
#   9/10 (i 11, ghi 3, chi 3, +6 others, n=28) -- scattered across 7+
#        classes, no usable majority at all.
# This is a WEAKER evidence tier than VERIFIED_SUBJECT_MARKERS (direct
# human correction) -- a statistical default for when NOTHING else is
# known, not a claim about any specific noun. A caller with a real noun
# (even an untranslated one) should always prefer the per-lemma fallback
# above over this class-only one.
SWAHILI_CLASS_DEFAULT_SM_GUESS.update({
    "15": "ku", "7": "chi", "8": "va", "6": "ghi", "3": "ghu", "4": "i", "5": "ji",
})

KUBWA_FORM_BY_SM = {"ji": "ibaa", "chi": "kibaa", "va": "vibaa"}
# 2026-10-11: 'va' (vitabu/plural-chi-class) -> 'vibaa' added from the SAME
# concord_families.py mining pass that found 'chi'->'kibaa' -- "vikubwa"
# ("big" agreeing with plural books/cups) aligns to "vibaa" at Dice 0.44,
# "high" confidence tier, independently consistent with the already-
# confirmed m-/i-/ki- pattern (va is chi's plural counterpart, vi-baa
# matches chi's ki-baa the same way Swahili's own ki-/vi- pair does).
KUBWA_DEFAULT_FORM = "mbaa"

# 2026-10-20: "-ingi" (many/much) -- direct user request. Read all short
# wengi/vingi/nyingi/mengi verses. Same "own irregular/archaic concord
# series, not the regular kit_sm+stem formula" shape as kubwa (flagged
# directly by the project's own "gh"/"ghi"-class counterexample: a
# formulaic 'kit_sm+ingi' would predict "ghingi", not the observed
# "mengi" -- same irregularity Swahili's OWN "-ingi" paradigm has for
# its ma- class, "maingi" is not a real Swahili word either, "mengi" is).
# Multi-verse confirmations, zero counterexamples within each class:
#   wi (human plural): PSA.3.1 "watesi wangu...wengi"/"ni wengi
#     wanaonishambulia" -> "ni wengi", PSA.35.18 "watu wengi" -> "wandu
#     wengi", PRO.31.29 "Binti za watu wengi"/"wengi"(repeated) -> "Waka
#     wengi" (3 independent verses)
#   va (vi-/plural-chi): PSA.61.6 "vizazi vingi" -> "vivalwa vingi",
#     1CO.12.14 "ni vingi" -> "ni via vingi" (viungo=via, class
#     confirmed above, KUBWA_FORM_BY_SM) (2 verses; 1CO.12.20's "via
#     vilazi" for the SAME "viungo ni vingi" idea is a single-verse
#     ALTERNATE, not conflated in -- a real synonym or a freer
#     paraphrase, not yet resolved either way)
#   ghi (ma-/liquid-class, pesa/maji/matuku's own class): PSA.139.18's
#     comparandum ("si nyingi kuliko mchanga"/mawazo, both resolved
#     class "gh" -- see kitaita_e2e_pipeline.py's own check) -> "ghadakaia
#     mengi"; this is also, not incidentally, the SAME class "pesa"
#     (money) is registered under -- directly relevant to "pesa nyingi"
#   i (plain i-class): 2SA.12.2 "ng'ombe...wengi sana"->"ng'ombe
#     NYINGI" (ng'ombe registered class "i") -- SAME SPELLING as the
#     Swahili word, kept as submitted (not flagged as suspicious the way
#     a genuinely coincidental same-spelling match would be -- "-ingi" is
#     a reconstructed Proto-Bantu root, shared/cognate forms across
#     closely related Bantu languages are the EXPECTED outcome here, not
#     a red flag).
INGI_FORM_BY_SM = {"wi": "wengi", "va": "vingi", "ghi": "mengi", "gh": "mengi", "i": "nyingi"}
# 2026-10-20: "-ingi"'s SWAHILI surface forms don't decompose as a simple
# ADJ_BASE prefix + bare root the way kubwa/baya/etc do -- Swahili itself
# applies its OWN vowel coalescence/epenthesis here (wa+ingi->wengi,
# vi+ingi->vingi, n+ingi->nyingi, ma+ingi->mengi), so detection uses this
# closed set of known SURFACE forms directly (mingi/wingi included as
# the standard reference-grammar forms for classes 3/1 respectively,
# though neither was directly corpus-confirmed this round).
INGI_SURFACE_FORMS = {"wengi", "vingi", "nyingi", "mengi", "mingi", "wingi"}
INGI_DEFAULT_FORM = "wengi"  # majority across the 4 confirmed classes' verse counts -- UNCONFIRMED
                              # for any OTHER class, flagged as such in the trace, same discipline
                              # as KUBWA_DEFAULT_FORM.

# 2026-10-12: Kitaita Word Bank submissions (live Artifact, https://
# claude.ai/artifact/6eozxX4HtKZhce3FTfWnY2, collection "contributions")
# filled every remaining gap in kubwa's table in one round. Real finding:
# this is NOT a uniform "m-" default outside ji/chi/va as KUBWA_DEFAULT_FORM
# implied -- ku/lu/wi/ghi each take their OWN prefix (kubaa/lubaa/wabaa/
# mabaa), only u/ghu/ri actually collapse to the "mbaa" default.
# Merged directly into the same dict (not native-speaker-judgment-tier like
# VERIFIED_ADJECTIVES, but a direct Word Bank submission, trusted the same
# way every other n=1 direct-submission table in this file is).
# 2026-10-13 (cont. 3): the submitted "i":"ibaa" entry is PULLED BACK --
# it assumed the "i" class's adjective prefix matches its subject marker
# ("i-"), which the user's direct correction shows is wrong (the real
# class-1/"i" adjective prefix is "mi-", e.g. "milacha"/"mivui" -- see
# LACHA_FORM_BY_SM/FUPI_FORM_BY_SM below). No confirmed "mi-" form for
# kubwa exists yet, so this falls back to KUBWA_DEFAULT_FORM, flagged,
# rather than guessing "mibaa".
KUBWA_FORM_BY_SM.update({
    "ghi": "mabaa", "ghu": "mbaa", "ku": "kubaa",
    "lu": "lubaa", "ri": "mbaa", "u": "mbaa", "wi": "wabaa",
    # 2026-10-13 (cont. 4): "i" and "i_nasal" BOTH explicitly confirmed
    # "mbaa" via the Word Bank (submitted 20s apart, same value both
    # times -- a deliberate statement, not a duplicate-row error like the
    # earlier dogo/'ri' case). Unlike refu/dogo/fupi, kubwa does NOT split
    # between the two i-subclasses -- both just take the ordinary "m-"
    # default, same as u/ghu/ri. Recorded explicitly rather than left to
    # fall through to KUBWA_DEFAULT_FORM, since this is now confirmed data,
    # not an absence of data.
    "i": "mbaa", "i_nasal": "mbaa",
})

# ----------------------------------------------------------------- flagged conflicts
# Reported, not merged: two judgments that should (on this project's own
# class labels) have matched, but gave different answers.
CONFLICTS = [
    {
        "swahili": "X anajua. (know, present)",
        "context": "mtu and samaki are both labelled ANIMATE-SG, but got DIFFERENT "
                    "corrected forms for the identical verb+tense: mtu -> 'wawemanya' "
                    "(s4), samaki -> 'ramanya' (s15). Possible explanations, not "
                    "resolved here: (a) a real animacy-class override specific to "
                    "'jua' (this project has documented animacy-override behaviour "
                    "before, CLAUDE.md section 8); (b) one of the two answers is an "
                    "inconsistency, not a real grammatical difference. Needs asking "
                    "the native speaker directly -- NOT auto-resolved to either value.",
    },
]

# ----------------------------------------------------------------- exact-sentence overrides
# For the copula+adjective cases: rather than recombine SM+root+adjective
# from scratch (several interacting unknowns at once -- e.g. "kodeka" for
# mungu uses SM "k-", not mungu's otherwise-confirmed "u-", for reasons
# not yet understood), the exact verified sentence is used as-is for
# this precise (noun, swahili_root, swahili_tam, adjective_stem)
# combination. Generalizing the SM/TAM-with-adjective pattern needs
# more examples before it is safe to decompose further.
VERIFIED_SENTENCES = {
    ("mungu", "kuw", "li", "kubwa"): "Mlungu kodeka mbaa.",   # s18
    ("mti", "kuw", "li", "kubwa"): "Mudi ghodeka mbaa.",       # s19
    ("nchi", "kuw", "li", "kubwa"): "Isanga jedekaa ibaa.",    # s20
    ("mtu", "kuw", "li", "baya"): "Mundu odeka uzamie.",       # s21

    # Existential 'kuwa' (no complement): the SM+TAM fusion before the
    # 'kogh' root is NOT a clean, class-independent concatenation --
    # each class's result below contracts differently (mungu's 'u' ->
    # 'o-re-', mti's 'ghu' -> 'gho-' with NO visible 're' at all, siku's
    # 'ji' -> 'je-re-', nchi future keeps 'ji-cha-' intact). One
    # example per class is not enough to safely generalize a fusion
    # rule, so these are exact overrides, not decomposed into reusable
    # subject-marker/TAM rules -- VERIFIED_KUWA_ROOT_EXISTENTIAL above
    # still gives the right ROOT for any OTHER noun/class not listed
    # here, just not necessarily the right fused prefix.
    ("mungu", "kuw", "li", None): "Mlungu orekoghe.",  # s2
    ("mti", "kuw", "li", None): "Mudi ghokoghe.",       # s6
    ("siku", "kuw", "li", None): "Ituku jerekoghe.",    # s9
    # ("nchi", "kuw", "ta", None): "Isanga jichakoghe." -- REMOVED 2026-10-07 cont. 16: this
    # (s11) predates the future-kuwa 'bare root k' finding (cont. 15, confirmed only for
    # nyumba: "ichaka"). Deliberately NOT kept as an override so batch 4 re-generates this
    # combination fresh (now predicting "Isanga jichaka") for the speaker to confirm or
    # correct, rather than silently re-showing the old pre-finding answer.
    ("nyumba", "kuw", "li", None): "Nyumba redeka.",    # s17 -- the one exception that
        # doesn't fit the 'kogh' pattern at all (root 'dek' here, like the COPULA
        # cases, even with no adjective present) -- kept exact rather than forced
        # into either generalization; worth asking the speaker about directly.
    ("mwana", "kuw", "li", None): "Mwana urekoghe.",   # s22 -- CONFLICTS WITH CORPUS: the
        # full Bible corpus has 'orekoghe' 1455x and 'urekoghe' ZERO times (checked
        # directly, 2026-10-07 cont. 14). Kept as-is (never silently overwrite a direct
        # human correction) but flagged -- new w-class nouns NOT yet speaker-tested
        # should default to the corpus form 'o', not this one; worth re-asking the
        # speaker about mwana specifically.
    ("samaki", "kuw", "li", None): "Samaki rirekoghe.",  # s23 -- confirms 'ri' stays 'ri'
        # (no vowel shift) before TAM 're', same as before TAM 'ka' -- the present-
        # tense-only vowel shift (see PRESENT_SM_VOWEL_CONFIRMED below) does NOT
        # apply to 're'/'de'/'cha'.

    # "jua" (know, present): mtu and samaki gave DIFFERENT answers for the
    # identical verb+tense (see CONFLICTS above) -- kept as two separate
    # exact overrides rather than merged into one subject-marker rule.
    ("mtu", "ju", "na", None): "Mundu wawemanya.",   # s4
    ("samaki", "ju", "na", None): "Samaki ramanya.",  # s15
    ("mungu", "ju", "na", None): "Mlungu wawemanya.",  # s35 -- SAME surface form as mtu's
        # (both 'w'-class), corroborating the vowel-shift rule below independent of CONFLICTS.
    ("mti", "ju", "na", None): "Mudi ghwawemanya.",   # s36
    ("nchi", "ju", "na", None): "Isanga jawemanya.",  # s37

    # Copula 'kuwa' + adjective, round 2 (2026-10-07 cont. 14) -- extends VERIFIED_ADJECTIVES'
    # (stem, class) entries below, but the SM+TAM fusion for the VERB half still isn't safely
    # generalizable (see CONFLICTS note on mungu's 3 disagreeing forms), so kept exact:
    ("mungu", "kuw", "li", "baya"): "Mlungu udedeka uzamie.",   # s24 -- disagrees with
        # ("mungu","kuw","li","kubwa")'s "kodeka" (s18): same noun, same root/TAM, DIFFERENT
        # adjective -> different SM+TAM form ("u-de-" vs "ko-"). mungu's copula form is
        # internally inconsistent across judgments; not generalized, see CONFLICTS.
    ("mti", "kuw", "li", "baya"): "Mudi ghodeka ghuzamie.",     # s25
    ("nchi", "kuw", "li", "baya"): "Isanga jideka jizamie.",  # s26 -- CORRECTED 2026-10-07
        # cont. 16, direct instruction: "should be jideka instead of jidedeka" -- resolves
        # the s26/s50 contradiction in favour of the zero-TAM "jideka" (matches siku exactly,
        # as it should: both are 'ji'-class). The original "jidedeka" judgment (accepted at
        # the time) was simply wrong, not a different valid form.
    ("mwana", "kuw", "li", "kubwa"): "mwana odeka mbaa.",       # s27 -- matches mtu's "odeka"
        # (s21) exactly (o-dek-a, zero TAM) -- mwana and mtu agree with each other but NOT
        # with mungu (see CONFLICTS); 2 nouns is still thin, not promoted to a general rule.
    ("samaki", "kuw", "li", "kubwa"): "Samaki redeka mbaa.",    # s28 -- PLURAL reading
        # ("fish [generically] were big") -- see the singular/plural note below, direct
        # user instruction 2026-10-07 cont. 15.

    # Batch 3 (cont. 15), present tense -- re-generated with the '-we-' TAM syllable
    # per direct instruction (both zero and '-we-' are grammatical; '-we-' is preferred
    # to visually parallel Swahili's own explicit -na- marking, confirmed across every
    # noun in this batch: s42/44/45/46/52):
    ("nyumba", "ju", "na", None): "Nyumba iwemanya.",   # s42 -- speaker's note: the zero
        # form ("Nyumba imanya") was ALSO explicitly called "still correct", just not
        # the preferred style -- keep this in mind if zero ever resurfaces as "wrong".
    ("siku", "ju", "na", None): "Ituku jamanya.",  # s43 -- judged CORRECT as the zero-marked
        # form (no '-we-'); kept exact so a republish doesn't silently flip an already-
        # accepted answer to the new '-we-'-preferred default just because it wasn't
        # explicitly re-asked.
    ("samaki", "sem", "na", None): "Samaki raweghamba.",  # s44 -- PLURAL reading (see below)
    ("samaki", "on", "na", None): "samaki rawewona.",     # s45 -- PLURAL reading
    ("mti", "simam", "na", None): "Mudi ghawesimama.",    # s46 -- confirms 'gha-' (not
        # 'ghwa-') was the right consonant+vowel; only the TAM syllable was missing.
    ("mwana", "ju", "na", None): "Mwana wawemanya.",      # s52

    # Batch 3, bare existential 'kuwa', mtu (3rd w-class noun tested) -- confirms the
    # corpus-grounded 'o-' override from cont. 14 (KOGH_RE_WCLASS_VOWEL_CORPUS): CORRECT
    # as generated, no correction needed.
    ("mtu", "kuw", "li", None): "Mundu orekoghe.",   # s47

    # Batch 3, copula+adjective, round 3 -- see the singular/plural and BAYA_STEM notes
    # below for why these don't match what the generalized rule predicted:
    ("samaki", "kuw", "li", "baya"): "Samaki edeka izamie.",    # s48 -- SINGULAR reading
        # ("the [one] fish was bad") -- note this uses 'e-', NOT the 'ri-'/'ra-' this
        # project had been treating as samaki's one-and-only subject marker. CONTRADICTS
        # the BAYA_STEM generalized rule's prediction of 'rizamie' -- see SAMAKI_NUMBER
        # note below, direct user correction 2026-10-07 cont. 15.
    ("nyumba", "kuw", "li", "baya"): "Nyumba edeka izamie.",    # s49 -- nyumba's copula
        # SM is 'e-' (shift from its base 'i-'), matching BAYA_STEM's prediction exactly
        # this time (izamie) -- CONSISTENT with s51 below (2/2 agreement for nyumba).
    ("siku", "kuw", "li", "baya"): "ituku jideka jizamie.",     # s50 -- fixes a real bug:
        # the generated "jidedeka" has a spurious extra TAM syllable ('de') that doesn't
        # belong -- corrected to "jideka" (SM 'ji' + root 'dek' + fv 'a', ZERO TAM,
        # matching mtu/mwana/samaki-singular/nyumba's own "[vowel]+dek+a" pattern exactly).
        # UNRESOLVED INCONSISTENCY, flagged not reconciled: nchi's own s26 ("Isanga
        # jidedeka jizamie") uses the SAME "jidedeka" pattern this correction just said is
        # wrong, but was never flagged as wrong for nchi specifically (the speaker only
        # corrected the missing adjective word on s26, not the verb). Both nchi and siku
        # share subject marker 'ji' and the 'dek' root, so this is a direct contradiction,
        # not a different class -- kept as two separate exact overrides, per the project's
        # "flag, don't force a reconciliation" rule.
    ("nyumba", "kuw", "li", "kubwa"): "Nyumba edeka mbaa.",     # s51 -- TWO corrections at
        # once: verb 'idedeka'->'edeka' (same zero-TAM fix as above) AND adjective
        # 'ibaa'->'mbaa' (nyumba takes the 1/2-and-3-class 'kubwa' form, NOT nchi's
        # 'ibaa', despite both being labelled Swahili class 9/10 -- yet more confirmation
        # that nyumba is a genuinely distinct Kitaita class from nchi, cont. 6's running theme).

    # Batch 3, future existential, nyumba (2nd class tested, only nchi confirmed before):
    # REAL, CONFIRMED EXCEPTION, not a typo -- 'kuwa' in the future tense does NOT use
    # the 'kogh' root at all. Direct user correction: "it is -ka" -- the future form is
    # just SM + TAM '-cha-' + a bare root 'k' + fv 'a' ("i-cha-ka" = "ichaka"), not
    # SM+TAM+'kogh'+'e'. nchi's OWN future form ("Isanga jichakoghe", s11) was never
    # re-examined against this -- may need re-asking, flagged not silently changed.
    ("nyumba", "kuw", "ta", None): "Nyumba ichaka.",  # s53
    ("nchi", "kuw", "li", None): "Isanga jirekoghe.",    # s54, batch 4 -- CONFIRMS 'kogh' is
        # the strong default (see KUWA_PAST note below, now 7/8 nouns)
    ("nchi", "kuw", "ta", None): "Isanga jichaka.",      # s55, batch 4 RE-TEST -- confirms the
        # future bare-root-'k' rule generalizes to nchi too (old pre-finding "jichakoghe" was wrong)
    ("mungu", "kuw", "ta", None): "Mlungu uchaka.",      # s56, batch 4
    ("mti", "kuw", "ta", None): "Mudi ghuchaka.",        # s57, batch 4
    ("samaki", "kuw", "ta", None): "Samaki richaka.",    # s58, batch 4 -- plural reading
    ("siku", "kuw", "ta", None): "Ituku jichaka.",       # s59, batch 4
    ("mwana", "kuw", "ta", None): "Mwana uchaka.",       # s60, batch 4
    ("mtu", "kuw", "ta", None): "Mundu uchaka.",         # s61, batch 4

    # Batch 5 (cont. 19): kept exact specifically because the marker-level rule gives
    # the WRONG answer for these and no clean fix was found -- see the 'ji'-vs-'je'
    # note on KUWA_PAST_ALTERNATES below.
    ("mfupa", "kuw", "li", None): "Indi jerekoghe.",     # s68 -- 'je', NOT 'ji'+re (nchi's
        # own "Isanga jirekoghe", s54, keeps 'ji' unchanged before the same TAM) -- a real,
        # unreconciled split between nchi (ji) and siku+indi (je), both nominally marker 'ji'.
        # 2026-10-08 cont. 7 RETRACTED cont. 8: cont. 7 read s110 (sanduku, generated
        # "Sanduku jirekoghe") as a THIRD "ji" vote, a tie. That generated form was never
        # actually re-examined against the corpus pattern -- direct correction (cont. 8):
        # "it is supposed to be sanduku jerekoghe instead of sanduku jirekoghe" overturns it.
        # Real tally: 3 ('je': siku, mfupa, sanduku) vs 1 ('ji': nchi only) -- 'je' is the
        # default, nchi is the flagged exception, not a tie. Generalized into kogh_re_o_shift
        # ('ji'->'je', the front-vowel mirror of the already-generalized u->o shift) so any
        # FUTURE untested 'ji'-marker noun defaults to 'je', not 'ji'.
    ("msalaba", "kuw", "li", None): "Msalaba ghorekoghe.",  # s71 -- confirms the SAME
        # u-before-'re'-TAM-becomes-'o' vowel shift already known for the bare 'u' class
        # (KOGH_RE_WCLASS_VOWEL_CORPUS) ALSO applies to 'ghu' -> 'gho' -- generalized into
        # the generator directly (see KOGH_RE_O_SHIFT_MARKERS) rather than kept as a one-off.

    # Batch 9 (2026-10-08 cont. 10), copula+adjective -- same reason EVERY other
    # copula+adjective pair in this table is an exact override, not generic composition:
    # the copula SM is irregular per class and doesn't always match the class's own base
    # marker (see CONFLICTS/mungu's 3 disagreeing copula forms). kit_sm is shared with the
    # adjective-concord computation below it in the generator, so shifting it just for the
    # verb would wrongly also shift the adjective prefix -- an exact override avoids that
    # entirely, same as every other entry here.
    ("kikombe", "kuw", "li", "baya"): "Kikombe chedeka chizamie.",   # s114 -- copula SM
        # 'che-' (shifted from the class's own 'chi-'), but the ADJECTIVE prefix stays
        # 'chi-' unshifted (chizamie) -- confirms the copula-SM shift is VERB-specific,
        # doesn't propagate to the adjective concord.
    ("kikombe", "kuw", "li", "kubwa"): "Kikombe chedeka kibaa.",     # s115 -- SAME copula
        # SM shift ('che-'), AND a new 'kubwa' form: 'ki-baa', NOT the general 'mbaa' or
        # the 'ji'-class's 'ibaa' -- plausibly the standard Swahili-class-7 'ki-' adjective
        # prefix (cf. Swahili 'kizuri'), since kikombe/kiti/kitabu are all class 7 nouns.
        # 2026-10-08 cont. 12 (batch 10): CONFIRMED to generalize across the whole class --
        # kiti (s127) and kitabu (s128) BOTH independently give the identical "chedeka
        # kibaa" shape, n=3 clean, no exceptions.
    ("kiti", "kuw", "li", "kubwa"): "Kifumbi chedeka kibaa.",   # s127
    ("kitabu", "kuw", "li", "kubwa"): "Chuo chedeka kibaa.",    # s128
    ("simba", "kuw", "li", "baya"): "Shimba edeka izamie.",   # s117 -- copula SM 'e-'
        # (shifted from the class's own bare 'i-', the SAME shift nyumba's class uses) +
        # adjective 'izamie' -- IDENTICAL surface form to nyumba's own copula+baya answer
        # ("Nyumba edeka izamie.", s49) -- direct confirmation shimba and nyumba really are
        # the same class, down to the copula form (see VERIFIED_SUBJECT_MARKERS retraction).

    # Batch 9/10, future 'kuwa' with NO adjective -- the 'wi'/'chi' classes' need for the
    # fuller 'kai' root is now GENERALIZED directly in the generator (see
    # FUTURE_KUWA_FULL_ROOT_MARKERS, kitaita_sentence_generator.py) since both tested nouns
    # in each class confirmed it -- watu's own entry that used to live here is gone, composes
    # correctly through the general mechanism now (verified byte-for-byte unchanged). The 'i'
    # class, by contrast, SPLITS per noun for this exact combination (nyumba/shimba take bare
    # 'k'; miti/nyoka need 'kai') -- NOT generalized, kept as exact overrides:
    ("miti", "kuw", "ta", None): "Midi ichakaia.",     # s133 -- CONTRADICTS nyumba/shimba's
        # own "ichaka" (bare-k) answers for the identical class marker 'i' -- a genuine
        # per-noun split, not forced either way.
    ("nyoka", "kuw", "ta", None): "Choka ichakaia.",   # s134 -- matches miti, not nyumba/shimba
    # ("kikombe","kuw","ta",None) / ("kiti","kuw","ta",None) exact overrides -- ADDED
    # cont. 14, REMOVED again cont. 15: these existed only because s107's stored
    # correction had a mistyped answer ("chichaka", missing "-ia"), briefly read as a
    # real per-noun split alongside kiti. Direct correction: "I had mistyped, the class
    # is uniform" -- kikombe and kiti both take the fuller 'kai' form after all, so this
    # composes correctly through the ordinary FUTURE_KUWA_FULL_ROOT_MARKERS mechanism
    # (now including "chi" again) with no override needed.

    # Batch 10, present tense exceptions -- kept exact because the GENERALIZED table
    # (PRESENT_SM_VOWEL_CONFIRMED) now predicts the class default, and these two nouns
    # specifically do NOT follow it:
    ("watu", "ju", "na", None): "Wandu wiwemanya.",    # s99 -- UNSHIFTED 'wi-', the flagged
        # exception to the 'wi'->'wa' shift wazee (s132) confirms for the rest of the class
    # ("mto","ju","na",None): "Modo ghwawemanya." -- REMOVED 2026-10-08 cont. 16: s137's
    # correction was actually about 'moto' (fire), confused with 'mto' (river) -- see the
    # retraction notes above VERIFIED_NOUN_TRANSLATIONS and VERIFIED_SUBJECT_MARKERS.
    # mto's present tense is untested again.

    # 2026-10-08 cont. 14: a THIRD such present-tense exception, found by this round's
    # full-regression audit (comparing freshly-generated output against every LIVE
    # judgment in the Judge Artifact, not just the previous run's output -- this one had
    # been silently broken since cont. 10 and never caught before now). ndege (bird,
    # Kitaita 'nyonyi') was accepted as correct (s87, "Nyonyi iweobua inge.", UNSHIFTED
    # 'i-') back in an early round -- cont. 10 then generalized the 'i'->'ya' present
    # shift from 3 OTHER nouns (shimba/miti/nyoka) without re-checking whether it still
    # held for every noun already tested in this class. It doesn't: ndege is the
    # flagged exception, the SAME "one noun doesn't fit the class" pattern as watu/mwana/
    # nchi above, kept as its own exact override rather than silently overwritten by the
    # newer rule or silently left broken:
    ("ndege", "ogop", "na", None): "Nyonyi iweobua inge.",   # s87

    # 2026-10-08 cont. 16: a SECOND exact override removed by the same "class is
    # uniform" retraction that fixed the future-tense cell (see FUTURE_KUWA_FULL_ROOT_
    # MARKERS above) -- ("kiti","kuw","li",None): "Kifumbi cherekoghe." (s126) used to
    # live here as one half of a believed kikombe-vs-kiti split; s113 was independently
    # revised live to "Kikombe cherekoghe" too, so both compose correctly now through
    # the shared KOGH_RE_FRONT_VOWEL_SHIFTS["chi"]="che" mechanism, no override needed.

    # Chui/'inge' (leopard) -- a number-invariant noun (see NUMBER_INVARIANT_NOUNS
    # below). 2026-10-08 cont. 16 originally needed this as an exact override because
    # the number-invariant generalization had just made 'ri-' (plural) the default for
    # 'inge', contradicting this already-accepted singular answer. 2026-10-08 cont. 18
    # RETRACTED that default (direct Class Sorter classification put 'inge' back in
    # 'i'): this override is now redundant (composes correctly through the restored
    # 'i' default) but left in place, harmless:
    ("chui", "pit", "ka", None): "Inge ikaida mwalimu.",   # s89

    # Batch 11 (cont. 14), future 'kuwa', maji/mayai: a THIRD instance of the future-
    # 'kuwa' per-noun split this project already has for the 'i' class (miti/nyoka vs
    # nyumba/shimba) and the 'wi'/'chi' classes (generalized the other way) -- maji and
    # mayai, SAME class ('ghi', confirmed by every OTHER TAM cell both nouns were tested
    # on), give genuinely DIFFERENT future-tense forms: maji skips the '-cha-' TAM
    # syllable entirely (sm+root+fv, "ghi"+"kai"+"a" = "ghikaia"), mayai keeps it (sm+TAM+
    # root+fv, "ghi"+"cha"+"kai"+"a" = "ghichakaia", the SAME shape as the FUTURE_KUWA_
    # FULL_ROOT_MARKERS classes). NOT generalized either way (would require guessing
    # which of the two disagreeing nouns is the class default) -- kept as two separate
    # exact overrides, per the project's standing "flag, don't force" rule:
    ("maji", "kuw", "ta", None): "Machi ghikaia.",       # s140
    ("mayai", "kuw", "ta", None): "Maghi ghichakaia.",   # s143
}

# ----------------------------------------------------------------- 'kuwa' (be), past tense: 'kogh' is the strong default, not a coin-flip
# 2026-10-07 cont. 15/16. Originally framed (per direct instruction,
# cont. 15) as "no rule, alternate" after nyumba's bare existential
# broke the project's old "adjective presence decides" rule. Batch 4
# then filled the one remaining untested noun (nchi, s54) and it came
# back 'kogh' -- the tally across all 8 bare-existential-tested nouns
# is now 7 'kogh' (mungu/mti/siku/mwana/samaki/mtu/nchi) to 1 'dek'
# (nyumba only). This is NOT a 50/50 alternation; 'kogh' is a strong,
# well-confirmed default and nyumba is a single flagged exception, not
# evidence of genuine free variation. Kept the alternates list (below)
# for the copula-adjective contexts, where the picture is still
# genuinely mixed (mungu's own 3 disagreeing copula answers, cont. 14/
# 15) -- but for the BARE existential (no adjective), the generator now
# defaults to 'kogh' with real confidence, not an arbitrary pick.
# Each full alternate is (root, tam, final_vowel) -- the TAM and final
# vowel are COUPLED to the root choice, not independent: every real
# 'kogh' answer takes TAM '-re-' + fv 'e' (orekoghe/ghokoghe/jerekoghe/
# urekoghe/rirekoghe), every real 'dek' answer takes ZERO TAM + fv 'a'
# (odeka/edeka/redeka/jideka/ghodeka/kodeka) -- picking a root without
# its matching TAM/fv would produce an unattested combination.
KUWA_PAST_ALTERNATES = [("kogh", "re", "e"), ("dek", "", "a")]  # index 0 = current default (continuity only)

# ----------------------------------------------------------------- number-invariant nouns: 'i' (singular) vs 'ri' (plural)
# 2026-10-07 cont. 15: originally found as a samaki-specific quirk ("for samaki in
# singular form uses e/i and in plural form uses ra/re/ri") -- Swahili 'samaki' (fish)
# is NUMBER-INVARIANT (same word singular and plural, like nchi/nyumba/siku), so every
# test sentence had been silently AMBIGUOUS for number, explaining samaki's apparently
# inconsistent subject marker:
#   PLURAL ('ri'/'ra'/'re', what VERIFIED_SUBJECT_MARKERS["samaki"] already encodes as
#     the default): s14 rikawona, s23 rirekoghe, s28 redeka, s32 rikaendasikira,
#     s40 rikaghamba, s44 raweghamba, s45 rawewona -- 7 of 8 judged samaki sentences.
#   SINGULAR ('e'/'i'): s48 edeka -- 1 of 8, but a direct, explicit correction.
#
# 2026-10-08 cont. 16, GENERALIZED by direct correction: this is NOT samaki-specific --
# "some words use the same word for singular version and plural version like in
# Swahili ... a lot of the words in i class are in ri class when in plural." Direct
# list of further number-invariant nouns, all independently cross-checked against
# page1_vocabulary.json's own statistical candidate for the Swahili source word (every
# one matches cleanly): ngelo (wakati/time), mbuwa (shamba/farm), barua (barua/letter,
# a loanword retained as-is), sere (amani/peace), inge (chui/leopard), nguku
# (kuku/chicken), koshi (mbwa/dog), supu (mchuzi/sauce), ngokoro (magoti/knees), nyonyi
# (ndege/bird) -- PLUS samaki itself. This reframes the project's own 'ri' class (see
# export_grammar_pages.py) as NOT a separate base class at all: it is the generic
# PLURAL-reading reflex of the 'i' class for any noun that doesn't distinguish number,
# the same way Swahili's own number-invariant nouns (often borrowed/loan words) behave.
# Not resolved into a single rule per noun (nothing in the Swahili source disambiguates
# number), so the generator keeps defaulting every number-invariant noun to the plural
# reading ('ri'-shape) and these constants document the singular alternative. Also
# still the reason samaki (and now every noun in NUMBER_INVARIANT_NOUNS) is excluded
# from BAYA_STEM's automatic prefixing below (it predicted the plural 'rizamie' for
# samaki; the real answer, s48, was singular 'izamie').
NUMBER_INVARIANT_NOUNS = {
    "samaki", "ngelo", "mbuwa", "barua", "sere", "inge", "nguku", "koshi", "supu",
    "ngokoro", "nyonyi",
}
NUMBER_INVARIANT_SM_SINGULAR = {"default": "i", "copula": "e"}  # 'i' elsewhere, 'e' before 'dek' (1 data point, samaki s48)
NUMBER_INVARIANT_SM_PLURAL = {"default": "ri", "present": "ra", "copula": "re"}  # the existing default (samaki)

# ----------------------------------------------------------------- subject-marker vowel alternation
# 2026-10-07 cont. 14. Direct user observation: the subject marker's
# CONSONANT stays fixed across tenses, but its VOWEL changes depending
# on the TAM that follows -- first seen in the speaker's own present-
# tense corrections (mungu/mtu 'u'->'wa', mti 'ghu'->'ghwa' or 'gha',
# nchi 'ji'->'ja', samaki 'ri'->'ra'), then DOUBLE-CHECKED DIRECTLY
# against the full Bible corpus (31,103 verses) per direct request,
# rather than trusted on 5 isolated sentences alone:
#
#   waghamba 479, wawona 79, wamanya 22    (w-class, zero-marked present)
#   ughamba 1, umanya 2                    (w-class, same, rare alternative)
#   ghaghamba 8, ghawona 2                 (gh-class, zero-marked present)
#   jaghamba 1, jamanya 1                  (j-class, zero-marked present)
#   wawemanya 0, ghwawemanya 0, jawemanya 0, ramanya 0, rawona 0, raghamba 0
#     (the explicit '-we-' TAM syllable the speaker actually WROTE in 4 of 5
#     corrections is essentially UNATTESTED; 'wemanya' occurs exactly once,
#     source unclear)
#
# CONCLUSION: the "vowel becomes 'a' before present tense" rule itself
# is now corpus-confirmed (580+ tokens, 3 independent classes, zero
# counterexamples). The specific TAM REALIZATION the speaker wrote
# ('-we-') is NOT corpus-supported -- the corpus overwhelmingly prefers
# a ZERO-marked present (bare [consonant]+a+root). Both are real
# (neither invented), so neither is silently discarded: the generator
# below defaults to the corpus-dominant zero form and explicitly flags
# that the speaker's own answer used the rarer '-we-' variant instead.
#
# gh-class correction: the speaker's answer used a glide ('ghwa-'), but
# the corpus has ZERO hits for any 'ghwa'+root combination and 10 hits
# for the glide-less 'gha'+root -- the generator uses 'gha', not the
# speaker's 'ghwa', and flags the disagreement rather than picking
# silently.
PRESENT_SM_VOWEL_CONFIRMED = {
    "u": "wa",     # w-class: glide 'w' inserted (bare-vowel marker). Corpus: waghamba 479,
                    # wawona 79, wamanya 22 vs ughamba/umanya 1/2 -- 'wa' is clearly dominant.
    "gh": "gha",   # gh-class: the noun-class table stores this BARE (no vowel at all -- it only
                    # gets a linking 'u' inserted later for consonant-initial TAMs, which zero-
                    # marked present is not, so present tense must be matched against the bare
                    # form here). NO glide -- corpus (ghaghamba 8, ghawona 2) contradicts the
                    # speaker's 'ghwa' (0 hits for any ghwa-root combination tried).
    "ghu": "gha",  # same class, in case it ever arrives already linking-vowelled -- same target.
    "ji": "ja",    # j-class: corpus jaghamba 1, jamanya 1 -- thin but real, matches speaker's 'ja-'.
    "ri": "ra",    # r-class (samaki): speaker-confirmed only ('ramanya') -- ZERO corpus hits
                    # for ANY ra-+root combination tried (raghamba/rawona/ramanya/raendasikira/
                    # rasimama all 0), though the 'ri-'/'rika-' marker itself IS productive
                    # elsewhere (130 types, 308 tokens) -- a genuine data-sparsity gap for this
                    # SPECIFIC combination, not evidence against the rule.
    "i": "ya",     # nyumba's bare-vowel class: this project originally predicted exactly this
                    # by analogy with the w-class's 'w'-glide ("Bantu-typical 'y'-glide... the
                    # linguistically likely prediction") but left it unconfirmed -- now directly
                    # confirmed, n=3 independent nouns: shimba (s95, "yawemanya"), miti (s121,
                    # "Midi yawemanya"), nyoka (s124, "Choka yawemanya..."). A front-vowel 'y'-
                    # glide mirroring the back-vowel 'w'-glide exactly as predicted. (2026-10-08
                    # cont. 10: this entry was wrongly keyed "y" through cont. 7-9 -- 'shimba' was
                    # thought to be its own 'y'-class rather than nyumba's 'i'-class; see
                    # VERIFIED_SUBJECT_MARKERS above for the retraction.)
    "chi": "cha",  # 2026-10-08 cont. 7 (was wrongly keyed "ch", bare, cont. 4 -- that was a
                    # premature guess at the base form from present-tense data alone; batch 8
                    # (s100/s102/s106/s107) directly confirmed the base marker is 'chi-', not
                    # bare 'ch-' -- vowel i->a before present, the SAME shift 'ji'->'ja' already
                    # has (kikombe/kiti/kitabu, n=3 across narrative+future+present).
    "wi": "wa",    # 2026-10-08 cont. 12: RETRACTS the cont. 10 "wi doesn't shift" reading, which
                    # was based on watu alone (s99) -- wazee (s132, "Waghosi wawemanya") shifts
                    # i->a exactly like every other class. watu itself is now the flagged
                    # EXCEPTION (kept as its own exact VERIFIED_SENTENCES override, "Wandu
                    # wiwemanya", unshifted) -- the SAME "one noun doesn't fit the class" pattern
                    # as mwana (u->o) and nchi (ji->je).
    "ghi": "gha",  # 2026-10-08 cont. 14 (batch 11): maji's class (machi/maghi, see
                    # VERIFIED_SUBJECT_MARKERS) -- confirmed by s139, already judged correct as
                    # generated ("Machi ghawemanya") BEFORE this project knew the base marker was
                    # 'ghi' rather than 'ghu' (the shift target 'gha' happens to be identical
                    # either way, which is why s139 passed without a correction).
    "ku": "kwa",   # 2026-10-08 cont. 14 (batch 11): kuseka/kulila's class -- s145 ("Kuseka
                    # ghawemanya"->"Kuseka kwawemanya") confirms a w-glide insertion before the
                    # shifted vowel (k+u -> k+w+a), the SAME pattern as the 'u'->'wa' w-class
                    # shift, just with a filled consonant slot -- cross-confirmed independently by
                    # this project's OWN person-marker finding for 2sg ('ku'->'kwa', s83).
}
PRESENT_TAM_DEFAULT = "we"  # 2026-10-07 cont. 15, OVERRIDES the cont. 14 corpus-frequency
                              # choice: batch 3 confirmed the zero form is ALSO grammatical
                              # (s42's own note: "still correct") but the project prefers
                              # '-we-' explicitly, to visually parallel Swahili's own -na-
                              # marking, per direct instruction -- a style choice, not a
                              # correctness fix. Confirmed across every class in batch 3
                              # (s42/44/45/46/52 all added '-we-' back in).
PRESENT_TAM_ZERO_VARIANT = ""  # corpus-dominant form (see cont. 14 token counts above) --
                                 # also grammatical, just not the generator's default anymore.

# 2026-10-09 (cont. 13). Direct user observation: the probe-based pipeline
# now correctly IDENTIFIES the subject of a Swahili '-ki-' (conditional/
# participial) verb (this session's cont. 12-13 work) but still fails to
# TRANSLATE it -- kitaita_sentence_generator.translate() had NO branch at
# all for `tam_label == "conditional"` (TAM_SWAHILI_TO_LABEL already
# labels Swahili 'ki' as "conditional", but nothing downstream ever
# checked that label), so it fell through to the generic "no mapping
# rule" flag and literally emitted a "?" placeholder TAM syllable in the
# output (e.g. "u?sela") -- a separate, pre-existing gap from the
# subject-identification work, not something introduced by it.
#
# kitaita_tam_allomorphy.py already has a NATIVE-SPEAKER-VERIFIED Kitaita
# "conditional" TAM category (2026-10-07) with two attested variants,
# KITAITA_TAM_VARIANTS["conditional"] = ["ki", "ka"] -- this was
# apparently never wired into the production translate() path at all.
# The project's own standing hypothesis (docs/status.md 2026-10-07 cont.
# 5, never resolved with a speaker): 'ka' outnumbers 'ki' ~7-8:1 in
# Kitaita corpus text, but both occur after every subject-marker class,
# and the split LOOKS exactly parallel to Swahili's own -ka-(narrative/
# "and then") vs -ki-(conditional/temporal/converbal "while/if") split --
# i.e. two different grammatical categories misfiled under one
# "conditional" label, not free variation. Swahili's OWN narrative -ka-
# already maps to Kitaita -ka- elsewhere in this generator (confirmed,
# attested identical retained form) -- choosing Kitaita 'ki' here, for
# Swahili's SEPARATE conditional/converbal -ki-, completes that same
# parallel-structure hypothesis rather than reusing 'ka' for both.
# EXPLICITLY NOT NATIVE-SPEAKER-CONFIRMED for this specific Swahili-
# source mapping (unlike every other TAM branch in translate(), which is
# either corpus-confirmed with real support or native-speaker-verified)
# -- a hypothesis being tried, not an established fact; flagged plainly
# wherever it is used, never silently presented as equivalent confidence
# to the other TAM branches.
CONDITIONAL_TAM_HYPOTHESIS = "ki"

# Existential 'kuwa' (root 'kogh') + past (-li-/-re-): corpus strongly
# prefers subject-marker vowel 'o' for the bare 'u' marker (orekoghe
# 1455 vs urekoghe 0) -- CONTRADICTS the mwana correction (s22, "Mwana
# urekoghe"). Not silently resolved (mwana's exact override above is
# kept as given); this constant is what the generator uses for any
# OTHER, not-yet-tested noun ending in this vowel in this combination.
# 2026-10-07 cont. 19 (batch 5): GENERALIZED beyond the bare 'u' marker
# -- msalaba (s71, "Msalaba ghurekoghe" -> "Msalaba ghorekoghe")
# confirms the SAME u-before-this-TAM-becomes-o shift also applies to
# 'ghu' -> 'gho' (matching mti's own long-standing exact answer,
# "Mudi ghokoghe", which had this shift baked in from the start but
# was never generalized into a rule until this noun confirmed it
# independently). Applied to any marker ENDING in 'u', not just the
# bare marker, via KOGH_RE_O_SHIFT below.
KOGH_RE_WCLASS_VOWEL_CORPUS = "o"  # corpus: orekoghe 1455 vs urekoghe 0

# 2026-10-08 cont. 8: direct correction -- sanduku's existential past is
# "Sanduku jerekoghe", NOT "jirekoghe" as generated (and as a prior
# judgment had briefly, mistakenly, accepted). This REVERSES the cont. 7
# reading of the ji/je question: it is NOT a 2-2 tie. siku, mfupa, and
# NOW sanduku all independently shift 'ji' -> 'je' before '-re-' -- the
# SAME front-vowel mirror of the back-vowel 'u'->'o' shift this function
# already generalizes, just never connected to it before. nchi is the
# ONE exception (kept as its own exact VERIFIED_SENTENCES override,
# "Isanga jirekoghe", unaffected by this change) -- 3 shifting vs 1
# flagged exception, not an even split. 2026-10-08 cont. 10 extended
# this to 2 MORE front-vowel markers (bare 'i': shimba/miti; 'wi':
# watu/wazee) -- NOT a general "ends in i" rule, each front-vowel-
# ending marker listed explicitly rather than matched by suffix.
#
# 'chi' was deliberately ABSENT from this dict from cont. 12 through cont. 15: kikombe
# (s113, read as "chirekoghe") and kiti (s126, "cherekoghe") looked like a genuine
# per-noun split, the SAME pattern the 'ji' class has (nchi vs siku/mfupa/sanduku).
# 2026-10-08 cont. 16 RETRACTS this, the SAME SESSION as the future-tense retraction
# (see FUTURE_KUWA_FULL_ROOT_MARKERS above): s113 was independently revised live in the
# Judge Artifact ("Kikombe cherekoghe", status flipped to wrong) -- the SAME "class is
# uniform" correction extends to this cell too, not just the future one. 'chi' now
# shifts i->e here exactly like every other class in this table; the ("kiti","kuw",
# "li",None) exact override this split used to need is removed (composes correctly
# through the shared mechanism now).
KOGH_RE_FRONT_VOWEL_SHIFTS = {
    "ji": "je",  # confirmed 2026-10-08 cont. 8, n=3 (siku, mfupa, sanduku); nchi is the ONE
                  # flagged exception, kept as its own exact VERIFIED_SENTENCES override
    "i": "e",    # confirmed 2026-10-08 cont. 10, n=2 (shimba s116, miti s122) -- the SAME
                  # noun-class as nyumba (whose own existential past is a SEPARATE irregularity,
                  # the copula root 'dek', not 'kogh'+'re' at all -- unaffected, exact override)
    "wi": "we",  # confirmed 2026-10-08 cont. 10/12, n=2 (watu s119, wazee s131 -- "Waghosi werekoghe")
    "ghi": "ghe",  # confirmed 2026-10-08 cont. 14 (batch 11), n=1 (maji s141, "Machi gherekoghe")
    "chi": "che",  # confirmed 2026-10-08 cont. 16 (RETRACTS the cont. 12 "split" reading), n=2
                    # (kikombe s113 revised, kiti s126)
}

def kogh_re_o_shift(kit_sm: str) -> str:
    """Subject markers shift their final vowel before existential 'kogh'
    + TAM 're': markers ending in 'u' (bare 'u', 'ghu', or any future
    marker of this shape) shift to 'o' -- confirmed on 2 independent
    markers (bare 'u': mungu/mtu; 'ghu': mti/msalaba), 0 counterexamples
    (mwana is a flagged, kept-as-is exception, handled via its own exact
    VERIFIED_SENTENCES entry, not here). Specific markers ending in a
    front vowel shift it to 'e' -- see KOGH_RE_FRONT_VOWEL_SHIFTS; this
    is NOT a general "ends in i" rule, kept as an explicit lookup rather
    than a suffix check (every front-vowel marker tried so far DOES
    shift here, 'chi' included as of 2026-10-08 cont. 16, but the
    lookup is kept explicit rather than assumed for any future marker
    not yet tested). 'ri' is untouched -- no analogous shift found for
    it (samaki keeps its own vowel unchanged before 're')."""
    if kit_sm and kit_sm.endswith("u"):
        return kit_sm[:-1] + KOGH_RE_WCLASS_VOWEL_CORPUS
    if kit_sm in KOGH_RE_FRONT_VOWEL_SHIFTS:
        return KOGH_RE_FRONT_VOWEL_SHIFTS[kit_sm]
    return kit_sm

# ----------------------------------------------------------------- person (1st/2nd) subject markers
# 2026-10-08. Direct user request: test first/second-person sentences.
# Every prior round only ever generated 3rd-person sentences (the
# generator's `translate()` always required a Swahili SUBJECT NOUN) --
# this is the first look at "I"/"you"/"we"/"you-pl" at all.
#
# UNLIKE everything else in this file, these STARTED as corpus-induced
# hypotheses, not native-speaker-verified (see kitaita_person_marker_
# investigation.py for method and caveats) -- the same "aligned, not
# verified" confidence tier extract_kitaita_grammar.py uses. Kept in a
# SEPARATE table from VERIFIED_SUBJECT_MARKERS for exactly that reason
# -- that table's module-level comment says its entries are trusted
# "even at n=1" because they're direct human correction. As of
# 2026-10-08 cont. 4 (the Judge Artifact's first real person-sentence
# judgments, s73-s84), several cells ARE now native-speaker-verified --
# marked explicitly below, cell by cell, rather than promoting the
# whole table at once (several cells are STILL only the original
# corpus hypothesis and must not be treated as if judged).
#
# 2026-10-08, same session, direct correction: "the personal markers
# also change depending on TAM markers" -- i.e. don't treat the
# future-tense forms (ni/ku/di/m) as THE person marker and present/
# past as a separate incomplete afterthought; each person's surface
# form is itself a function of the following TAM, the SAME kind of
# relationship PRESENT_SM_VOWEL_CONFIRMED and kogh_re_o_shift already
# established for NOUN-class markers (consonant tends to stay fixed,
# vowel/form shifts by TAM) -- now confirmed to apply to person
# markers too, keyed as (person, swahili_tam_label) instead of one
# table per tense discovered ad hoc. Evidence per cell (all thin, n<8
# -- reported as counts, no strong claim, per this project's own n>=8
# confidence rule):
#
#   future: clean, multi-root, zero counterexamples (after excluding
#     the 'fanya'/'bony' root's own irregular bare-'ku-' future as a
#     per-lexeme TAM quirk, same pattern 'kuwa' already has) --
#     1sg 'ni', 2sg 'ku', 1pl 'di', 2pl 'm', all + the ALREADY-
#     confirmed future TAM '-cha-'.
#   present: 1sg 'na' (n=1, "ninajua"->"namanya" -- vowel i->a before
#     present, consistent with the already-confirmed noun-class
#     vowel-shift rule, just never tested on a person marker); 2sg
#     'ku' (n=1, "unajua"->"kumanyagha" -- consonant UNCHANGED from
#     the future form, no vowel shift visible, plus an unexplained
#     '-gha' suffix not seen elsewhere -- flagged as a loose end, not
#     resolved); 1pl/2pl have no present-tense data at all.
#   past/existential ('kuwa' root specifically, SM+TAM '-re-'+kogh):
#     1sg 'ne' (n=2, "nerekoghe ngighamba"/"nereobueghe" -- vowel
#     i->e before '-re-' TAM, the FRONT-VOWEL mirror of kogh_re_o_
#     shift's back-vowel u->o, but NOT the same mechanism: noun-class
#     'ji'/'ri' markers are confirmed to stay UNCHANGED before '-re-'
#     (see kogh_re_o_shift's own docstring) while this person marker
#     shifts -- person markers and noun-class markers do NOT share
#     one vowel-alternation paradigm, consistent with this project's
#     repeated finding that the two are genuinely different systems).
#     Directly corroborated by GEN.18.30's real attested "nerekoghe"
#     (1sg + existential 'kogh' + the SAME '-re-'+'e' shape the noun-
#     class existential already uses) -- the clearest person-marker
#     data point found so far.
#     1pl: CONFLICTING, not forced to one answer (n=1 'dere', matching
#     1sg's i->e pattern, vs n=2 'da', which looks zero/narrative --
#     'dere' used as the default since it matches the cross-person
#     pattern, 'da' kept as a flagged alternate, same style as
#     KUWA_PAST_ALTERNATES/SAMAKI_SM_SINGULAR below).
#     2sg: only 1 data point, and it's for Kitaita's NARRATIVE '-ka-'
#     (not '-li-'/'-re-' -- the Swahili source said "ulisikia" but
#     Kitaita rendered it "ukasikira", another instance of the
#     already-documented "Kitaita text doesn't always mirror the
#     Swahili TAM" finding) -- 'u' (CONSONANT DROPS from the future
#     form's 'ku' -- plausibly haplology, ku+ka -> ukaa, to avoid two
#     syllables both starting with 'k', but reported as the surface
#     form only, no phonological mechanism asserted). 1pl/2pl have no
#     narrative-tense data.
#
# 2026-10-08 cont. 4: the Judge Artifact's first real person-sentence
# judgments (s73-s84) came back and directly corrected 3 of these
# cells -- now VERIFIED, not just corpus-hypothesis:
#   present, 1sg: judge wrote "nawemanya" for "Ninajua." (corrected
#     from the generator's zero-marked "Namanya.", note: "Include TAM
#     marker") -- confirms the MARKER 'na-' the corpus hypothesis
#     already had right, but overturns the ZERO-marked present choice:
#     person present tense takes the SAME explicit '-we-' TAM the
#     noun-class generator already defaults to (PRESENT_TAM_DEFAULT),
#     not a person-specific zero exception. The zero-vs-'-we-' special
#     case in the generator's present-tense branch for `person` is
#     removed; persons now use the ordinary default like nouns do.
#   present, 2sg: judge wrote "Kwawemanya" for "Unajua." (corrected
#     from the flagged "?manya.") -- 'kwa-', NOT the thin corpus guess
#     'ku-' (unshifted). This is the SAME vowel-shift-before-present
#     pattern as the noun-class 'u'->'wa' marker, just applied to a
#     2-character marker: 'ku' (consonant k, vowel u) -> 'kwa' (vowel
#     shifts to 'a', 'w' glide inserted) -- the corpus's lone "unajua"
#     ->"kumanyagha" hit (unshifted, odd '-gha' suffix) is now known to
#     be a weaker/different-register form, not the main pattern.
#   past (existential 'kuwa'), 1sg: judge wrote "Nerekoghe" for
#     "Nilikuwa." (corrected from the flagged "?rekoghe.") -- EXACTLY
#     matches the pre-existing corpus hypothesis 'ne-' + '-re-' +
#     'kogh' + 'e'. First direct confirmation of a person-marker
#     hypothesis that was right on the first try; promoted from
#     hypothesis to verified with zero change needed to its value.
# Still NOT native-speaker-verified (unchanged corpus hypotheses,
# s73-s81 future cells were all independently judged "correct" too,
# so those 4 ARE now verified as well): 1pl/2pl present (no data at
# all, still flagged); 1pl past 'de'/'da' conflict (not asked); 2sg
# narrative 'u' (not asked).
PERSON_SUBJECT_MARKERS_BY_TAM = {
    ("1sg", "future"): "ni",   # VERIFIED (s73/74/80 judged correct), n=7+ corpus too
    ("2sg", "future"): "ku",   # VERIFIED (s75/76/81 judged correct), n>=3 corpus too
    ("1pl", "future"): "di",   # VERIFIED (s77/78 judged correct)
    ("2pl", "future"): "m",    # VERIFIED (s79 judged correct)
    ("1sg", "present"): "na",  # VERIFIED (s82) -- marker confirmed, TAM below also changed to '-we-'
    ("2sg", "present"): "kwa", # VERIFIED (s83) -- corrects the old corpus-guess 'ku' (unshifted)
    ("1sg", "past"): "ne",     # VERIFIED (s84) -- kuwa-existential context; exact match to hypothesis
    ("1pl", "past"): "de",     # still hypothesis only (n=1, conflicts with 'da' n=2 -- see note above)
    ("2sg", "ka"): "u",        # still hypothesis only (n=1 corpus, not asked)
}
PERSON_SUBJECT_MARKERS_1PL_PAST_ALTERNATE = "da"  # n=2, zero/narrative-looking, not forced over 'de'

# 2026-10-20: direct user request -- a default person marker for a TAM
# category this table has NO specific entry for at all (the exact gap
# "Umesongeza kiti." hit: 2sg + perfect, nothing in the table above).
# Derived from the table above itself (not hand-copied, so it can't drift
# out of sync with it) rather than invented separately: the FUTURE-tense
# form is used as each person's base/default, because it is the one TAM
# category with a VERIFIED marker for EVERY person AND the one that both
# 1sg and 2sg's OWN already-confirmed vowel shifts (ni->na, ku->kwa,
# consonant-stays-vowel-becomes-a before present -- the SAME pattern
# PRESENT_SM_VOWEL_CONFIRMED already documents for noun-class markers)
# shift AWAY FROM, i.e. future looks like the unshifted/citation form for
# both persons tested, not an arbitrary pick among several options. This
# is a WEAKER tier than a real (person, TAM)-specific entry above (used
# ONLY when one doesn't exist) and weaker still for 1pl/2pl, where this
# project has NO second TAM category's data to cross-check the pattern
# against at all (only confirmed for 1sg/2sg) -- flagged as such by the
# caller, not presented as equal confidence.
PERSON_SM_DEFAULT = {person: marker for (person, tam), marker in PERSON_SUBJECT_MARKERS_BY_TAM.items()
                      if tam == "future"}

# ----------------------------------------------------------------- adjective concord: 'baya' (bad)
# 2026-10-07 cont. 14. Round-2 corrections for 'baya' kept surfacing
# the noun's OWN subject-marker-class prefix in front of a shared
# stem ("-zamie"): mtu/mwana (u-class) -> 'uzamie', mti (gh-class) ->
# 'ghuzamie', nchi (j-class) -> 'jizamie'. DOUBLE-CHECKED directly
# against the corpus (not just 3 data points): uzamie 12, ghuzamie 13,
# jizamie 36, AND -- never asked of the speaker at all --
# **rizamie 66** (samaki's own 'ri-' prefix) and **izamie 70**
# (nyumba's own 'i-' prefix, DIFFERENT from nchi/siku's 'ji-' despite
# sharing this project's "9/10" label -- consistent with the project's
# repeated finding that this label hides real within-Kitaita splits).
# All five are genuinely attested words, not just plausible guesses.
# RULE (corpus-confirmed, 5/5 classes, zero counterexamples):
#   'baya' (bad) = [noun's own subject-marker-class prefix] + 'zamie'
# This is a DIFFERENT concord strategy than 'kubwa' (big) uses --see
# VERIFIED_ADJECTIVES below, which stays a flat per-class lookup
# because 'kubwa' does NOT take the subject-marker prefix at all
# (checked directly: ghubaa/ribaa/jibaa are all 0 in the corpus, ruling
# that out) -- it instead uses a separate ADJ_BASE-style paradigm
# (mbaa for 1/2 AND 3, ibaa for 9/10), with real but still-unmapped
# kibaa (92)/mibaa (51)/vibaa (5) forms suggesting a 7/4/8-class
# paradign this project hasn't assigned to specific Kitaita nouns yet
# -- flagged as an open lead, not pursued further this round. Two
# different lexical adjectives, two different concord systems; do not
# conflate them.
BAYA_STEM = "zamie"

# ----------------------------------------------------------------- 2026-10-13: 'i' class split (ADJECTIVE CONCORD ONLY)
# Direct user correction: this project's kit_sm "i" class may be
# conflating two historically distinct Bantu classes that happen to
# share ONE subject marker -- Swahili class 4 ("mi-", e.g. miti) and
# class 9/10 (nasal "N-", e.g. ndizi) both take "i-" subject agreement,
# but DIFFERENT adjective concord ("miti mirefu" vs "ndizi ndefu").
# Real signal already in this project's OWN data pointed the same way
# before this was raised: the 2026-10-12 Kitaita Word Bank round gave
# "nd-"-prefixed forms (ndini/ndui) for 'dogo'/'fupi' at the 'i' cell,
# while 6 OTHER roots (kubwa/chafu/zito/refu/geni/zuri) gave plain
# "i-" forms for the SAME cell -- the Word Bank never distinguished
# which noun-type the submitter had in mind, so that split looked like
# noise until the user's correction explained it structurally.
#
# CHECKED DIRECTLY before assigning any noun to the nasal subgroup
# (do not repeat the "label lies" mistake by analogy alone): this
# project's HIGHEST-trust data (native-speaker Judge Artifact, not the
# Word Bank) already tested 'baya' on nyumba (s49) and simba (s117) --
# BOTH independently confirm the PLAIN form "izamie", not a nasal
# "ndzamie". This POSITIVELY RULES OUT nyumba/simba as the nasal
# subgroup. mbwa/nyoka/ndege/chui are grouped with them here by this
# project's own established precedent that every 'i'-class member
# behaves identically elsewhere (2026-10-08 cont. 10: "shimba dissolves
# entirely into nyumba's 'i' class") -- ASSUMED by that precedent, NOT
# independently confirmed for any adjective yet. 'miti' (the other
# historical source of kit_sm "i", Swahili class 4) is plain by
# definition, not by inference.
#
# 2026-10-13 (cont. 3) -- direct user correction, replacing the 'ndizi'
# analogy with real tracked examples: "class 1 should be like midi
# [miti/trees] and class 2 like nyumba. the whole ndizi thing was an
# analogy not to be used for the tables." So: class 1 ("i", plain) =
# miti-type, adjective prefix "mi-" (confirmed: "milacha"/tall,
# "mivui"/short -- NOT bare "i-" as cont. 2 guessed by assuming the
# adjective prefix always matches the subject marker). Class 2
# ("i_nasal") = nyumba-type (also "ndagha"/knife, adjective "ndacha"/
# tall), adjective prefix "nd-" -- unchanged from cont. 2's guess, this
# part happened to be right. nyumba/shimba/choka moved from "i" to
# "i_nasal" above (VERIFIED_SUBJECT_MARKERS) to reflect this.
#
# 2026-10-13 (cont. 2) -- REARCHITECTED per direct user instruction:
# "it would be just easier to come up with a new -i- class that has
# the same subject markers as the existing i- class but that has
# different adjective markers the way in Swahili they are treated as
# different classes." The previous version of this fix (same day,
# cont. 0) treated the split as an invisible lookup-time hack keyed off
# a hardcoded noun set -- it worked, but a noun's "real" class was
# never recorded anywhere a human (or the Word Bank) could see or set
# it. Now "i_nasal" is a GENUINE, independently selectable class, same
# status as "u"/"ghu"/"ji"/etc: a noun's class CAN be stored as
# "i_nasal" directly (VERIFIED_SUBJECT_MARKERS, noun_class_of, the
# Word Bank's noun dropdown) and that is what feeds the adjective
# FORM_BY_SM tables below, directly, with NO per-noun lookup table at
# all -- i.e. exactly the architecture every OTHER class already uses.
# The only thing "i_nasal" needs that no other class does is to tell
# the TAM/verb-subject-marker machinery "treat me as 'i'" (since that
# half of the paradigm genuinely IS shared, per the Swahili class
# 4-vs-9/10 precedent) -- that is TAM_SM_ALIAS below, applied ONCE,
# immediately after a noun's class is first resolved, before any TAM-
# conditioned vowel-shift logic runs.
TAM_SM_ALIAS = {"i_nasal": "i"}
# 2026-10-13 (cont. 3) -- IMPORTANT exception found while fixing cont. 2's
# nyumba/baya mistake: 'baya' and 'zuri' do NOT use the mi-/nd- split at
# all -- both use the BARE subject marker directly (kit_sm, already
# TAM_SM_ALIAS-normalized), same as every other class. This is why
# nyumba's native-speaker-verified baya ("izamie", s49) is correct even
# though nyumba is class 2 ("i_nasal") -- baya was never going to say
# "ndzamie" regardless of which i-subclass nyumba is in. Confirmed by
# this project's OWN existing comments: BAYA_STEM's "kubwa does NOT take
# the subject-marker prefix at all... separate ADJ_BASE-style paradigm"
# (i.e. kubwa is the one that varies independently of kit_sm, not baya),
# and ZURI_ROOT's "the SAME strategy as baya... NOT kubwa's reduced
# series". So ONLY the DICT-based "special paradigm" adjectives (kubwa/
# refu/dogo/geni/fupi/zito/chafu) see the mi-/nd- split -- they use the
# raw, un-normalized class (kit_class in translate(), or kit_sm directly
# in compute_adjective_concord(), which never normalizes at all). baya/
# zuri use the TAM-normalized kit_sm instead. No separate "formula
# prefix" table is needed -- kit_sm already collapses "i_nasal" back to
# "i" for exactly this purpose.

# ----------------------------------------------------------------- adjective concord: beyond kubwa/baya
# 2026-10-11, direct user request: "a per class table that gives the
# adjective's prefix for every adjective and an adjective root table,
# similar to what we have for verbs." Checked first whether the raw
# material already existed -- it did: concord_families.py/build_pages.py
# (2026-10-07) already mined 9 adjective families from the full Bible
# corpus (same Dice-score-across-versions method as root_kitaita for
# verbs), but the results were never processed past the flagged "open
# lead" note just above (kibaa/mibaa/vibaa) and never wired into
# translate() -- which still only special-cased kubwa/baya. Processed
# that existing data (extract_adjective_concord_tables.py), NOT a new
# corpus mining pass.
#
# **Real finding, resolving part of the "open lead" above**: kubwa's
# own prefix series (m-/i-/ki-/vi- for default/ji/chi/va-class nouns)
# is a REDUCED series, consistently shorter than the noun's own full
# subject-marker prefix (kit_sm) -- ji->i (not ji), chi->ki (not chi),
# va->vi (not va). This looks like Kitaita may have its own distinct
# ADJECTIVE-concord series, parallel to how Swahili's own ADJ_BASE
# differs from SM_TABLE. Recorded as a HYPOTHESIS (from one adjective's
# data only), not fact:
KITAITA_ADJ_BASE_HYPOTHESIS = {"default": "m", "ji": "i", "chi": "ki", "va": "vi"}


def kitaita_adj_prefix(kit_sm: str) -> str:
    """The reduced adjective-concord prefix for a given KITAITA class
    marker, per KITAITA_ADJ_BASE_HYPOTHESIS -- ONE function, reused by
    every caller (kitaita_sentence_generator.py's translate() AND
    kitaita_e2e_pipeline.py's compute_adjective_concord()), so this
    lookup can't fork into two copies the way the object-marker/
    adjective-rule logic already has, three times, this session."""
    return KITAITA_ADJ_BASE_HYPOTHESIS.get(kit_sm, KITAITA_ADJ_BASE_HYPOTHESIS["default"])

# **'zuri' (good/beautiful) -- the one OTHER adjective with 2+
# independently-confirmed classes, both "high" confidence**: 'i-class'
# ("mizuri"->"ighokie", Dice 0.40) and 'wi-class' ("wazuri"->
# "wighokie", Dice 0.58) BOTH use the noun's FULL subject-marker prefix
# (kit_sm) + shared root 'ghokie' -- the SAME strategy as 'baya', NOT
# kubwa's reduced series. This directly CONTRADICTS treating kubwa's
# reduced series as "the" adjective default: one other adjective tested
# follows baya's full-kit_sm pattern instead. Genuinely unresolved with
# n=2 adjectives pointing two different ways -- needs either more
# corpus data or direct native-speaker testing, not asserted either way.
ZURI_ROOT = "ghokie"  # corpus statistical candidate (Dice 0.40-0.58), NOT speaker-verified like baya

# 2026-10-20: "safi" (clean/pure), reverse-inferred -- read all 26 short
# SUV/TAITA verses containing "safi" directly (not Dice alone). Same
# strategy/evidence tier as ZURI_ROOT: bare kit_sm + shared stem, corpus-
# mined, NOT native-speaker-verified. Confirmed across 6 independent
# classes with zero counterexamples in direct noun+adjective/relative-
# clause use: i- (MAT.27.59 "nguwo ielie", 1TI.3.9 "nia ielie", PRO.8.10/
# ISA.13.12 "dhahabu ieleshero" -- a narrower fixed collocation, not this
# general stem, see below), ri- (JOB.25.5 "nderielie", DEU.14.11 "rielie"),
# vi- (DEU.14.20 + LUK.11.41 "vielie", 2 independent verses), wi- (PSA.73.1
# + MAT.5.8 "wielie", 2 independent verses), u- (PRO.15.26 "uelie"), gha-
# (JOB.16.17 "ghaelie"). Several verses render "safi" via a clearly
# DIFFERENT, non-adjective-concord mechanism and are NOT counterexamples:
# "kuela" (PSA.51.7, JHN.15.3 -- a verb, "to become/be clean", same root
# different derivation), "wueli" (1PE.3.2 -- abstract noun "purity", no
# final -e), "ndicha" (PRO.31.22 -- fabric-quality "fine", a different
# sense of "safi" than "clean/pure"), "wuchafu" (ISA.28.8 -- litotes via
# the antonym "chafu"+negation), "madedo mecha" (PRO.15.4 -- free paraphrase).
SAFI_STEM = "elie"

# 2026-10-22: "moja" (one/a single), DIRECT NATIVE-SPEAKER CONFIRMATION
# (not corpus-mined like zuri/safi above -- a STRONGER evidence tier),
# given as a correction on the "Ambiguous translations" Word Bank round
# (this project's own statistical table had no way to represent this at
# all: it just offered 2 competing FLAT noun candidates, "jimweri"/
# "imweri", for a word that isn't a flat noun to begin with). Same
# strategy as zuri/safi -- full kit_sm + shared stem -- confirmed on 2
# INDEPENDENT, already-established classes: "ituku jimweri" (siku moja/
# one day, class 'ji') and "chadu chimweri" (kiatu kimoja/one shoe,
# class 'chi', "chadu" already registered as kiatu's Kitaita word --
# see VERIFIED_SUBJECT_MARKERS above). Zero counterexamples; both
# classes agree on the SAME strategy zuri/safi already use, so treated
# as CONFIRMED for those two classes and an EXTRAPOLATED hypothesis
# elsewhere, not asserted for every class.
MOJA_ROOT = "mweri"

# 2026-10-22: "kuteketezwa"/"kuchomeka" (be burnt), SAME direct-correction
# round -- user: "vizongona vekora" = "sadaka iliyochomeka" (the burnt
# offering), i.e. also used ADJECTIVALLY with class concord, not as a
# flat noun ("vizongona"/"vekora" were this word's own 2 flat statistical
# noun candidates -- "vizongona" is NOT a Kitaita word for "kuteketezwa"
# at all, it is the Kitaita word for "sadaka"/offering the adjective is
# modifying -- see VERIFIED_NOUN_TRANSLATIONS["sadaka"] below).
# NOT generalized into a concord branch the way MOJA_ROOT above was:
# only ONE data point, and "vizongona"'s own Kitaita class is not yet
# independently confirmed anywhere else in this project (unlike ituku/
# chadu above), so there is no second, independently-classed example to
# agree with -- recorded as a flagged single point, pending either a
# 2nd class or a direct class confirmation for "vizongona" itself.
KUTEKETEZWA_ROOT_SINGLE_POINT = {"kitaita_class_guess": "va", "stem": "kora",
                                   "example": "vizongona vekora = sadaka iliyochomeka (the burnt offering)",
                                   "confidence": "single data point, governing noun's own class unconfirmed"}

# 2026-10-20: "-ote" (all/whole), reverse-inferred -- direct user request
# ("let's work on ... other groups of words such as prepositions" led
# into this while checking -ote/-angu/-ako/-ake alongside it). Read many
# short SUV verses containing wote/yote/vyote/chote/kote/lote/pote/zote
# directly. Finding: Kitaita renders "-ote" as [GOVERNING NOUN's own
# subject-marker class, its FINAL VOWEL ELIDED if the class ends in a
# vowel] + stem "ose" -- e.g. EXO.12.47/PSA.7.1/PSA.33.8 "Waisraeli/wandu
# wote" -> "wose" (class "wa"/"wi" -> elide -> "w" + "ose"), DEU.14.11
# "nyonyi wote" -> "rose" (class "ri" -> "r" + "ose"), DEU.14.20 + LUK.
# 11.41 "vilambo vyote" -> "vose" (class "va" -> "v" + "ose", 2
# independent verses), GEN.47.9-style "matuku ghose" (class "ghi" ->
# "gh" + "ose"), PSA.23.6-style "irangi jose" (class "ji" -> "j" + "ose"),
# MAT.14.7 "lo lote" -> "chose" (class "chi" -> "ch" + "ose"), MAT.8.19
# "ko kote" -> "kose" (class "ku" -> "k" + "ose"). The elision (a/i/u + o
# -> o, i.e. the class's own vowel is absorbed before "-ose"'s own "o")
# is DIFFERENT from BAYA_STEM/ZURI_ROOT/SAFI_STEM's formula (none of
# those stems start with "o", so none of them ever exercised this) --
# NOT assumed to generalize beyond "-ose" specifically. Same evidence
# tier as ZURI_ROOT/SAFI_STEM otherwise: corpus-mined, NOT native-
# speaker-verified. See kitaita_e2e_pipeline.apply_ote_concord() for the
# elision implementation (kit_sm.rstrip(vowels) or kit_sm, then +"ose").
OTE_STEM = "ose"
OTE_STEM_CONFIRMED_CLASSES = {"wa", "wi", "ri", "va", "ghi", "ji", "chi", "ku"}

# **The other 6 mined stems: single, low-to-medium-confidence data
# points, one Swahili-class each, explicitly NOT generalized to other
# classes** (unlike zuri/baya, there's no second confirming class to
# show the prefix pattern at all -- using these for an untested class
# would be a pure guess). Keyed by (stem) -- callers must check the
# noun's class matches `swahili_class_seen` before using; otherwise
# there is no data at all. 'dogo' in particular is flagged EXTRA
# untrustworthy: this project separately confirmed its best corpus
# match ("mtini") is more likely a collision with the UNRELATED noun
# "mtini" (fig tree) than a real adjective-concord form (the 2026-10-08
# "mdogo swallowed as a noun" bug was exactly this word pair).
# 2026-10-12: 'dogo'/'geni'/'zito'/'chafu'/'fupi' ALL PROMOTED out of this
# table (same way 'refu' was on 2026-10-11) -- the Kitaita Word Bank round
# (see DOGO_FORM_BY_SM etc. below) supplied real per-kit_sm-class data for
# every one of them, superseding these single-point corpus guesses. Kept
# commented, not deleted, per this file's own convention:
#   "dogo":  {"swahili_class_seen": "1", "kitaita_word": "mtini", "dice": 0.60} -- LOW TRUST,
#            likely the mdogo/mtini noun-collision artifact (2026-10-08); the new
#            DOGO_FORM_BY_SM data for 'u'/'ghu' independently also gives "mtini",
#            so this guess happened to be right despite the flagged collision risk.
#   "geni":  {"swahili_class_seen": "1", "kitaita_word": "mghenyi", "dice": 0.60} -- confirmed
#            correct by the new data (u/ghu -> "mghenyi").
#   "zito":  {"swahili_class_seen": "5", "kitaita_word": "jikambie", "dice": 0.44} -- WRONG,
#            shares no root at all with the new data's "lemere"-based forms.
#   "chafu": {"swahili_class_seen": "2", "kitaita_word": "makakala", "dice": 0.58} -- WRONG,
#            shares no root at all with the new data's "lue"-based forms.
#   "fupi":  {"swahili_class_seen": "6", "kitaita_word": "ivui", "dice": 0.67} -- confirmed
#            correct by the new data (ji class -> "ivui" exactly).
ADJ_SINGLE_CLASS_CANDIDATE: dict[str, dict] = {}

# 2026-10-11 (cont. 8): direct user instruction -- "For mrefu-> tall we
# already have a very good concord for it in page one of the Kitaita
# dictionary. mlacha, ndacha, walacha, milacha, let's use those ones for
# now... the concord should be based off of Kitaita concord not swahili
# concord since the classes don't necessarily line up." Re-mined the
# SAME already-extracted page1_vocabulary.json family data the original
# (single-point) 'refu' entry above came from, and found it actually has
# THREE member forms, not one -- the earlier extraction only kept the
# best-Dice SINGLE candidate per stem and discarded the other 2 classes.
#   mrefu  (sg., class 1/3, default/'u' bucket) -> 'mlacha', Dice 0.53, LOW (1 version)
#   warefu (pl., class 2 human, kit_sm 'wi')    -> 'walacha', Dice 0.86, HIGH (3 versions agree)
#   mirefu (pl., class 4, mi-, non-human)        -> 'milacha', Dice 0.57, HIGH (3 versions agree)
# 'ndacha' (Dice 0.27) was the WEAKER of two competing candidates for the
# SAME mrefu slot 'mlacha' (Dice 0.53) already won -- a losing competitor,
# not a second attested form (shares no root with 'lacha' at all) -- kept
# out as noise, same discipline as every other low-Dice-competitor case.
# mlacha landing on the SAME 'm-' default prefix kubwa's 'mbaa' already
# uses is a second, INDEPENDENT root now pointing at that slot -- stronger
# cross-root support for "default"="m" than before, still not proof.
# 'milacha' (class 4, mi-) has no established Kitaita kit_sm cross-
# reference anywhere else in this project yet (no other table maps a
# kit_sm value to class-4 plural non-human nouns) -- kept as its own
# flagged candidate below, NOT folded into the kit_sm-keyed table, so it
# can't be silently misapplied to the wrong class.
LACHA_ROOT = "lacha"
LACHA_DEFAULT_FORM = "mlacha"          # mrefu, class 1/3 default/'u' bucket -- LOW confidence, Dice 0.53, n=1 version
LACHA_FORM_BY_SM = {"wi": "walacha"}   # warefu, class 2 human plural, kit_sm 'wi' -- HIGH confidence, Dice 0.86, n=3 versions

# 2026-10-12: Kitaita Word Bank submissions filled every other class.
# REVERSAL worth flagging explicitly: the 'ri' answer submitted is
# "ndacha" -- the EXACT form this file previously dismissed above as
# "the WEAKER of two competing [statistical] candidates... a losing
# competitor, not a second attested form... kept out as noise". A
# direct submission for exactly that cell independently landing on
# "ndacha" is real evidence it was never noise for the 'ri' class
# specifically, just wrongly excluded by a statistical-only check that
# had no way to see per-class irregularity. Also note a likely
# nasal-assimilation pattern across roots (Bantu class 9/10 N- prefix
# + stop/continuant-initial stem): 'ri'+'lacha' -> 'ndacha' here lines
# up with 'i'/'ri'+'tini'/'vui' also surfacing as 'nd-' elsewhere in
# this same submission round (see DOGO_FORM_BY_SM/FUPI_FORM_BY_SM) --
# a hypothesis from 2-3 roots, not asserted as a general rule.
LACHA_FORM_BY_SM.update({
    "chi": "kilacha", "ghi": "malacha", "ghu": "mlacha",
    "ji": "ilacha", "ku": "kulacha", "lu": "lulacha", "ri": "ndacha",
    "u": "mlacha", "va": "vilacha",
    # 2026-10-13 (cont. 3): direct user correction -- the class-1/"i"
    # ("midi"/trees) adjective prefix is "mi-", not bare "i-" (the
    # original "ilacha" submission assumed the adjective prefix always
    # matches the subject marker, which is wrong for this class).
    # "milacha" is the user's own confirmed example for "tall". Class-2/
    # "i_nasal" ("nyumba"/"ndagha") gets "ndacha" -- the user's OTHER
    # confirmed example, independently matching the 'ri' class's own
    # "ndacha" above (the nasal augment "nd-" is plausibly invariant
    # across classes, not specific to 'ri').
    "i": "milacha",
    "i_nasal": "ndacha",
})
LACHA_CLASS4_CANDIDATE = {
    "kitaita_word": "milacha", "dice": 0.57,
    "note": "mirefu, class 4 (mi-, plural non-human, e.g. 'misida milacha' = tall forests) -- HIGH-dice "
            "statistical candidate but NO confirmed Kitaita kit_sm cross-reference for this class exists "
            "elsewhere in the project yet; kept unmapped rather than guessed at which kit_sm it belongs to.",
}

# ----------------------------------------------------------------- 2026-10-12: Kitaita Word Bank round
# Four more adjective roots filled with near-complete per-kit_sm-class
# data in one round (Kitaita Word Bank Artifact, https://claude.ai/
# artifact/6eozxX4HtKZhce3FTfWnY2, collection "contributions",
# submittedAt 2026-10-12). Same trust tier as every other direct-
# submission table in this file (KUBWA_FORM_BY_SM/LACHA_FORM_BY_SM
# above) -- a real answer, not a corpus statistic, but NOT yet cross-
# checked by a second independent judgment round the way VERIFIED_
# ADJECTIVES' Judge-Artifact entries are. Each DEFAULT_FORM is kept only
# as a defensive fallback for a kit_sm this round didn't cover (should
# rarely fire -- coverage below is near-complete).
DOGO_DEFAULT_FORM = "mtini"
DOGO_FORM_BY_SM = {
    "chi": "kitini", "ghi": "matini", "ghu": "mtini",
    "ji": "itini", "ku": "kutini", "lu": "lutini", "u": "mtini",
    "va": "vitini", "wi": "watini",
    # 2026-10-13: was keyed "i" -- RELABELED "i_nasal" (its own genuine
    # class now, see TAM_SM_ALIAS above) after the user's class-split
    # correction. No noun currently has class "i_nasal", so a plain "i"
    # noun like nyumba now correctly falls to DOGO_DEFAULT_FORM (flagged
    # unconfirmed) instead of silently getting this nasal form it never
    # actually confirmed.
    "i_nasal": "ndini",
}
# 'ri' DELIBERATELY EXCLUDED: the submitted value was "mbaa" -- IDENTICAL
# to kubwa's ('big') own 'ri' answer, submitted 20 SECONDS after it
# (2026-10-12T12:01:09 vs 12:01:29), and sharing no root at all with
# every other dogo ('small') answer's "-tini"/"-dini" shape. Overwhelming
# likelihood this is a stray duplicate/copy-paste from the adjacent
# kubwa row, not a real answer for "small" -- flagged for re-submission
# rather than silently used or silently dropped from the record.
DOGO_RI_SUBMITTED_BUT_REJECTED = "mbaa"  # likely data-entry error, see note above -- NOT in DOGO_FORM_BY_SM

GENI_DEFAULT_FORM = "mghenyi"
GENI_FORM_BY_SM = {
    "chi": "kighenyi", "ghi": "maghenyi", "ghu": "mghenyi",
    "ji": "ighenyi", "ku": "kughenyi", "lu": "lughenyi", "ri": "mghenyi",
    "u": "mghenyi", "va": "vighenyi", "wi": "waghenyi",
    # 2026-10-13 (cont. 3): "i":"ighenyi" PULLED BACK, same reasoning as
    # KUBWA_FORM_BY_SM above -- geni shows the SAME reduced-series pattern
    # kubwa/dogo/fupi/refu do (chi->ki-, wi->wa-, ghi->ma-), so it's in
    # the same "needs the real mi-/nd- split" group, and no "mi-" form is
    # confirmed for it yet. Falls back to GENI_DEFAULT_FORM, flagged.
    # 2026-10-13 (cont. 4): "i_nasal" now CONFIRMED via the Word Bank:
    # "mghenyi" -- identical to geni's own "m-" default (u/ghu/ri), so
    # geni's nasal subgroup does NOT take an "nd-" form, unlike refu/
    # dogo/fupi. The plain "i" cell is still unconfirmed (see above).
    "i_nasal": "mghenyi",
}

FUPI_DEFAULT_FORM = "mvui"
FUPI_FORM_BY_SM = {
    "chi": "kivui", "ghi": "mavui", "ghu": "mvui",
    "ji": "ivui", "ku": "kuvui", "lu": "luvui", "ri": "ndui",
    "u": "mvui", "va": "vivui", "wi": "wavui",
    # 2026-10-13 (cont. 2): what was the original "i"-keyed Word Bank
    # submission ("ndui") is kept here under "i_nasal" -- it was answered
    # for the nasal subgroup all along, just mislabeled before the split
    # existed. 'ri' is left AS-IS (ndui) -- this is a SEPARATE, still-
    # open question (see LACHA_FORM_BY_SM's 'ri' note vs GENI_FORM_BY_SM's
    # 'ri'->"mghenyi"), not folded into this fix.
    "i_nasal": "ndui",
    # 2026-10-13 (cont. 3): direct user correction -- "mivui" is the
    # user's own confirmed example for class-1/"i" ("midi"/trees)
    # "short", previously unconfirmed/no-data.
    "i": "mivui",
}
# 'ri' -> "ndui" (not the "default -m-" or own-prefix pattern every
# other class here follows) -- same "nd-" pattern noted at
# LACHA_FORM_BY_SM's 'ri' entry above, now seen on a 2nd root with a
# v-initial stem; still a hypothesis across 2-3 roots, not a general rule.

# zito ('heavy') and chafu ('dirty') previously had only a single, LOW-
# Dice statistical guess each (ADJ_SINGLE_CLASS_CANDIDATE above, now
# commented out) with roots ("jikambie", "makakala") that share NOTHING
# with the roots submitted here ("lemere", "lue") -- those guesses were
# simply wrong, not approximately right; replaced outright, not merged.
# Both tables use the submitter's LITERAL per-class answer, not a
# formula -- unlike zuri/baya, zito and chafu do NOT reduce ji/chi/va/
# ghi/wi the way dogo/geni/fupi/kubwa do (e.g. chafu 'chi' -> "chilue",
# not "kilue"; chafu 'wi' -> "wilue", not "walue"; zito 'ghi' ->
# "ghilemere", not "malemere") -- a genuinely different adjective
# concord sub-pattern, not an inconsistency to resolve away.
ZITO_DEFAULT_FORM = "ulemere"
ZITO_FORM_BY_SM = {
    "chi": "chilemere", "ghi": "ghilemere", "ghu": "ghulemere",
    "ji": "jilemere", "ku": "kulemere", "lu": "lulemere", "ri": "rilemere",
    "u": "ulemere", "va": "vilemere", "wi": "wilemere",
    # 2026-10-13 (cont. 3): "i":"ilemere" PULLED BACK -- same reasoning as
    # kubwa/geni, pending a confirmed "mi-" form. WORTH FLAGGING: unlike
    # kubwa/dogo/geni/fupi/refu, zito does NOT show the reduced-series
    # pattern at all (chi/wi/ghi here use the noun's OWN full subject-
    # marker string unreduced -- "chilemere" not "kilemere", "wilemere"
    # not "walemere", "ghilemere" not "malemere") -- same "always bare
    # subject marker" shape as baya/zuri. If that holds, zito might NOT
    # need the mi-/nd- split at all, and "ilemere" could simply be
    # correct as the uniform i-class answer -- not acted on without
    # confirmation (would need re-wiring zito to use kit_sm like baya/
    # zuri, not kit_class/adj_class).
    # 2026-10-13 (cont. 4): the flag above is PARTIALLY RESOLVED -- a
    # fresh "i_nasal" submission gives "ndilemere", which is NOT the
    # same as the (still-unconfirmed) plain form, so zito DOES
    # distinguish the two i-subclasses after all for at least this cell.
    # Note the SHAPE is different from refu/dogo/fupi's nd-REPLACES-
    # stem-initial-consonant pattern ("ndacha" from "lacha", not
    # "ndlacha"): here it looks like "nd-" simply PREFIXES onto the
    # already-"i"-shaped form ("nd"+"ilemere"="ndilemere"). Recorded
    # as-is, not forced into the other roots' pattern.
    "i_nasal": "ndilemere",
}

CHAFU_DEFAULT_FORM = "ulue"
CHAFU_FORM_BY_SM = {
    "chi": "chilue", "ghi": "ghilue", "ghu": "ghulue",
    "ji": "jilue", "ku": "kulue", "lu": "lulue", "ri": "rilue",
    "u": "ulue", "va": "vilue", "wi": "wilue",
    # 2026-10-13 (cont. 3): "i":"ilue" PULLED BACK -- same reasoning and
    # same "possibly doesn't need the split at all" flag as ZITO_FORM_
    # BY_SM above (chafu shows the identical unreduced chi/wi/ghi shape:
    # "chilue" not "kilue", "wilue" not "walue", "ghilue" not "malue").
    # 2026-10-13 (cont. 4): same resolution as zito above -- "i_nasal"
    # confirmed "ndilue" ("nd-" prefixed onto the "i"-shaped "ilue", same
    # pattern as zito's "ndilemere", not refu/dogo/fupi's stem-initial
    # replacement). Plain "i" is still unconfirmed.
    "i_nasal": "ndilue",
}

# 2026-10-11 (cont.): direct correction -- route via KITAITA class
# groupings, not the Swahili-label matching above. This project has
# repeatedly found Swahili class labels unreliable ("label lies");
# matching `swahili_class_seen` against a target noun's OWN Swahili
# class (as the ADJ_SINGLE_CLASS_CANDIDATE path above still does) means
# a real candidate can sit unused even when the TARGET noun's Kitaita
# class is one this candidate's own form already demonstrates working
# for (confirmed case: 'dogo'/'mtini' failed to apply to kijana/mdawana
# because kijana's SWAHILI class resolves to '7/8', even though
# mdawana's own KITAITA class ('u', default) is exactly the class
# 'mtini' is itself shaped for).
#
# Checked directly: does each candidate's OWN Kitaita surface form
# decompose against the SAME reduced series KITAITA_ADJ_BASE_HYPOTHESIS
# already uses for kubwa (m-/i-/ki-/vi-)? If so, the root generalizes
# across EVERY Kitaita class via that series, the same mechanism kubwa
# itself uses for its own root 'baa' -- just a single data point per
# root here, not kubwa's 3 independently-confirmed classes, so still a
# WEAKER generalization than kubwa, not equal confidence, and reported
# as such.
_REDUCED_PREFIXES_LONGEST_FIRST = sorted(set(KITAITA_ADJ_BASE_HYPOTHESIS.values()), key=len, reverse=True)


def _decompose_via_reduced_series(word: str) -> tuple[str, str] | tuple[None, None]:
    for p in _REDUCED_PREFIXES_LONGEST_FIRST:
        if word.startswith(p):
            return p, word[len(p):]
    return None, None


# stem -> root; generalizes across ALL Kitaita classes via KITAITA_ADJ_BASE_HYPOTHESIS
# (kit_sm -> reduced prefix -> prefix+root), same substitution KUBWA_FORM_BY_SM uses.
# Confirmed directly (2026-10-11): 'dogo'->'tini', 'geni'->'ghenyi', 'chafu'->'akakala',
# 'fupi'->'vui' all decompose cleanly against m-/i-/ki-/vi-. 'refu'/'walacha' and
# 'zito'/'jikambie' do NOT (neither starts with any of those four strings) -- those two
# stay in ADJ_SINGLE_CLASS_CANDIDATE above as genuinely unmapped leads, not forced.
ADJ_SINGLE_ROOT_GENERALIZED: dict[str, str] = {}
for _stem, _cand in ADJ_SINGLE_CLASS_CANDIDATE.items():
    _prefix, _root = _decompose_via_reduced_series(_cand["kitaita_word"])
    if _prefix is not None:
        ADJ_SINGLE_ROOT_GENERALIZED[_stem] = _root

# ----------------------------------------------------------------- past tense (-li-) re/de by subject marker: direct overrides
# 2026-10-08 cont. 4. The statistical allomorphy table (kitaita_tam_
# allomorphy.py's output, consulted by kitaita_sentence_generator.
# past_variant_for_sm) has NO entry at all for some subject markers --
# not "below n=8", genuinely zero data -- because it was computed
# BEFORE several markers were hand-corrected into existence (e.g.
# nchi/ituku/isanga/indi were only corrected to 'ji' afterward, so the
# allomorphy pass never saw 'ji' as a marker at all). 'sanduku' (s90,
# "Sanduku jidengia") is the first-ever test of 'ji' + a plain (non-
# 'kuwa') past tense, and is simply trusted at n=1 like every other
# VERIFIED_* table in this file -- overrides past_variant_for_sm's
# statistical lookup entirely for the markers listed here.
VERIFIED_PAST_TAM_BY_SM = {
    "ji": "de",  # s90, sanduku: "Sanduku jidengia" (ji + de + ngi + a)
}

# ----------------------------------------------------------------- locative suffix '-nyi': FLAGGED, NOT YET INTEGRATED
# 2026-10-08 cont. 16. Direct user observation: "the suffix nyi added to a noun makes
# it a location similar to swahili. mighondinyi -> milimani (at the hills)." 'mighondi'
# is the plural of 'mghondi' (mountain, this project's own 'ghu'-class member, see
# CLASSES above) -- 'mighondi' + '-nyi' = a LOCATIVE meaning, directly paralleling
# Swahili's own '-ni' locative suffix (mlima -> mlimani). This is a completely new
# piece of Kitaita morphology -- NOT a subject-marker or TAM phenomenon, and
# `kitaita_sentence_generator.translate()` has no mechanism for locative sentences at
# all (no Swahili locative input is ever parsed, no '-nyi' attachment rule exists).
# Recorded here as a flagged, pre-registered finding for future work, deliberately NOT
# wired into the generator yet -- doing so would require (a) parsing Swahili '-ni'
# locatives out of the source sentence, (b) a rule for which noun FORM '-nyi' attaches
# to (this one example attaches to the PLURAL 'mighondi', not the singular 'mghondi' --
# untested whether that is required or incidental to this one example), and (c) at
# least one more independent example before treating the attachment rule as general.
# 2026-10-08 cont. 18: the SAME word (vilima/hills -> 'mighondinyi') came back up
# independently in the Class Sorter batch, placed in the 'ku' bucket -- NOT treated as
# a second confirming data point for a subject-marker class (see the exclusion note on
# VERIFIED_SUBJECT_MARKERS above): a locative-suffixed form isn't a plain class member,
# so classifying it into a bucket doesn't mean anything about its class the way the
# other 120 sorted nouns' placements do. Still just the one real example of the
# suffix itself.
LOCATIVE_SUFFIX = "-nyi"  # mirrors Swahili '-ni'; single example so far: mighondinyi = milimani

# ----------------------------------------------------------------- candidate new classes: FLAGGED, UNTESTED
# 2026-10-08 cont. 16. Direct user report of (at least) 2 missing classes, identified
# by Kitaita word alone (no sentence context given) -- cross-referenced directly
# against page1_vocabulary.json to recover a plausible Swahili source for each, then
# queued as ROUND_12_SPECS (kitaita_sentence_batch.py) rather than guessed into a
# table: no subject-marker or TAM data exists for any of these three yet, so nothing
# below is an override -- this is a cross-reference record only, kept so a future
# reader can see WHY these three specific Swahili words were chosen for the batch.
#   'lukundo' -- candidate source 'upendo' (love, Dice 0.63, Swahili class 11 'u-').
#     If confirmed, Kitaita 'lu-' would be a genuine 3rd shape in territory Swahili
#     flattened into 'u-' (classes 3 and 11 both surface as 'u-' in Swahili; this
#     project's own 'u'/'ghu' split already shows Kitaita keeps MORE distinctions
#     here than Swahili does -- 'lu' would be a third).
#   'vadu' -- candidate source 'viatu' (shoes, Dice 0.87, Swahili class 8 'vi-'),
#     plural of 'kiatu' -> Kitaita 'chadu' (itself never tested as a SUBJECT, only as
#     an object, s88). If confirmed, 'va-' would be the plural counterpart of the
#     already-established 'chi' class (the standard Bantu 7/8 ch/v pairing).
#   'vitoi' -- candidate source 'wezi' (thieves, Dice 0.69, Swahili class 2 'wa-',
#     plural of 'mwizi'). The 'vi-' SHAPE is unexpected for a human-plural referent
#     (Bantu 'vi-' is typically inanimate) -- a genuine open question, not assumed to
#     be either the 'chi' class's plural or a new human-vi class.

# ----------------------------------------------------------------- closed-class function words
# 2026-10-11 (cont. 8): direct user instruction -- "Na should be a
# dictionary look up since it does not have concord attached to it."
# 'na' here is the CONJUNCTION ("and"/"with"), a different word from the
# '1sg present' TAM marker 'na' already in PERSON_SUBJECT_MARKERS_BY_TAM
# above -- unrelated, don't confuse the two. Not a Dice-aligned
# statistical guess: 'na' appears repeatedly as the plain conjunction in
# real Kitaita Bible-corpus text already quoted elsewhere in this file's
# history (e.g. "isanga wulacha na wushapa ghwaro", "meta inya na pointi
# inya karakara na mmbao", "wulacha meta ikumi na iwi, na wushapa meta
# iwi") -- a shared Bantu cognate with Swahili, confirmed directly from
# corpus text, not statistically inferred.
FUNCTION_WORD_TRANSLATIONS = {
    "na": "na",
    # 2026-10-22: direct native-speaker correction -- "kama" (if/like/as)
    # -> "sa", a genuine substitution (NOT a shared cognate like "na"
    # above -- the two words don't even share a consonant).
    "kama": "sa",
    # 2026-10-22: direct native-speaker correction -- "kwetu" ("at our
    # place/home") -> "kwedu". This is Swahili's LOCATIVE class (class
    # 17, "ku-" + possessive suffix -- "kwetu"/"kwangu"/"kwako"/"kwake"/
    # "kwenu"/"kwao" are each a COMPLETE, standalone adverbial word, not
    # a possessive attached to some other noun), a DIFFERENT "ku-" from
    # this project's existing 'ku' class (the class-15 INFINITIVE-as-
    # noun one, e.g. "kukimbia") -- find_possessive_pronoun()'s governor-
    # before-it model doesn't even apply here (there is no governed
    # noun), so this has to be a flat word lookup, not a concord rule.
    # Direct user instruction: "use the ku- class possessives" for the
    # rest -- i.e. the SAME general [class marker]+[shared person stem]
    # formula as every other class's SINGULAR cells (regular "w"-glide,
    # NOT the class-15 'ku' entry's own irregular "kodu"/"konyu" plural
    # -- that irregularity is specific to the infinitive sense, this is
    # a different class that happens to share the same surface letters).
    # Only "kwetu" itself was DIRECTLY given; the other 5 are
    # EXTRAPOLATED from that one instruction + the regular pattern
    # documented in POSS_PERSON_FORM_BY_CLASS's own comment above --
    # NOT independently confirmed, flagged as such here, not silently
    # equated with the "kwetu" cell's own confidence.
    "kwetu": "kwedu",          # CONFIRMED, direct native-speaker correction
    "kwangu": "kwapo",         # EXTRAPOLATED (regular singular pattern)
    "kwako": "kwako",          # EXTRAPOLATED (regular singular pattern; same string as Swahili)
    "kwake": "kwake",          # EXTRAPOLATED (regular singular pattern; same string as Swahili)
    "kwenu": "kwenyu",         # EXTRAPOLATED (regular "e"-plural pattern, NOT 'ku' class-15's "konyu")
    "kwao": "kwawo",           # EXTRAPOLATED (regular plural pattern)
    # 2026-10-22: bug #2 (locative-existential "yuko"/"yupo"/"yumo",
    # African Storybook error report) -- these locative function words
    # are the pieces of the test sentences ("Yuko wapi paka wangu?",
    # "Yuko chini ya kitanda?", etc.) besides the verb itself. Each
    # REVERSE-INFERRED from the full parallel Bible corpus (`bible
    # collection/data/parallel_full_suv/parallel.jsonl`), confidence
    # noted per entry; "ya"/"na" (the associative/comitative particle
    # that follows each of these in the source phrases, e.g. "juu YA
    # nchi") are deliberately left UNTOUCHED here -- every corpus example
    # keeps "ya" literally unchanged regardless of the governed noun's
    # class, which this project's existing "keep an unmechanized word
    # as-is" fallback already reproduces; adding an explicit "ya"->"ya"
    # entry would risk silently overriding the UNRELATED, genuinely
    # class-varying associative-concord mechanism used elsewhere
    # (ASSOC_PARTICLES/associative_particle_for_class, e.g. "mayai ya
    # kuku" -> "maghi GHA nguku").
    "wapi": "hao",       # CONFIRMED, n>=9 independent verses, zero exceptions (e.g. "Uko wapi?"/
                          # "Yuko wapi Mungu wako?"/"Umetoka wapi?" all -> "...hao?")
    "hapa": "aha",        # CONFIRMED, n>=5 independent verses (e.g. "Yupo hapa Elisha" -> "oko aeni aha")
    "juu": "aighu",        # CONFIRMED, n>=8 independent verses ("juu ya nchi" -> "aighu (ya) ...")
    "chini": "aisi",       # CONFIRMED, n=2 independent verses ("chini ya mkono/miguu" -> "aisi ya ...");
                          # ONE other verse ("chini ya uvuli") instead used "andenyi ya" (= "ndani ya"'s
                          # own equivalent below) -- a legitimate translation choice (being "under" shade
                          # read as being "within" it), not a contradiction, but flagged since it means
                          # "aisi" is not the ONLY attested rendering of "chini ya".
    "ndani": "andenyi",    # CONFIRMED, n=2 independent verses ("ndani ya hekalu/humo nyuma ya pazia")
    "nje": "shighadi",      # CONFIRMED, n=3 independent verses ("nje ya mji" x2, "nje ya nyumba")
    "karibu": "kaavui",     # CONFIRMED, n=2 independent verses ("karibu na Yafa/kisima")
    "nyuma": "nyuma",       # CONFIRMED cognate, kept identical in both languages, n=1 verse direct
                          # (other verses add a "na-" prefix, "nanyuma ya" -- not generalized here,
                          # only the bare form actually needed by the test sentences is wired)
}

# ----------------------------------------------------------------- possessive-pronoun concord
# 2026-10-11 (cont. 8): direct user instruction -- "wire in possessive-
# pronoun detection." Raw data: page1_vocabulary.json's
# concord_families["possessive"] (mined the SAME way as the adjective
# families above) has exactly 6 person-stem families, each with ONLY ONE
# member -- EVERY one is Swahili class 1 ('w-', e.g. 'wangu'="my", for a
# class-1 possessor/possessed pairing). There is NO data at all for any
# other class (in particular, none for class 8 'vyako' -- the exact word
# in the sentence that prompted this). Genuinely thin: unlike kubwa/lacha
# above, there is no second confirmed class to show whether this
# generalizes via kit_sm at all -- kept strictly class-1-gated, NOT run
# through kitaita_adj_prefix/the reduced-series mechanism (that hypothesis
# was built for ADJECTIVE concord specifically; nothing here shows
# possessive concord follows the same series).
# 'wako'/'wake' match their Swahili spelling EXACTLY (best_dice 0.55/0.45,
# both flagged 'low' by the mining script) -- the SAME "same spelling"
# weak-match pattern this project has flagged elsewhere (page1 object-
# marker mining) as possibly a coincidental loanword/cognate match rather
# than an independently confirmed Kitaita form. Reported as the weakest
# tier here, not excluded.
POSS_PERSON_FORM_CLASS1 = {
    "angu": "wapo",   # wangu="my", Dice 0.61
    "ako": "wako",    # wako="your(sg)", Dice 0.55, LOW -- SAME spelling as Swahili
    "ake": "wake",    # wake="his/her", Dice 0.45, LOW -- SAME spelling as Swahili
    "etu": "odu",     # wetu="our", Dice 0.67
    "enu": "onyu",    # wenu="your(pl)", Dice 0.55
    "ao": "wawo",     # wao="their", Dice 0.38
}

# 2026-10-20: SECOND confirmed class for the possessive mechanism above
# (direct user request, reverse-inferred) -- "irangi" (life, class "ji",
# see VERIFIED_SUBJECT_MARKERS) takes "japo"/"jako"/"jake" for -angu/
# -ako/-ake (PSA.23.6 "maisha yangu"->"irangi japo", GEN.3.14/3.17/DEU.
# 6.2/JOS.1.5 etc. "maisha yako"->"irangi jako", JOS.1.5/1SA.7.15/1KI.
# 4.21/1KI.11.34/1KI.15.5/1KI.15.6/PRO.31.22 "maisha yake"->"irangi
# jake" -- dozens of independent verses). Only angu/ako/ake tested for
# this class (no etu/enu/ao evidence found). Generalizes POSS_PERSON_
# FORM_CLASS1 from "class-1 only" into a real cross-class mechanism --
# kept as a SEPARATE table (not merged into POSS_PERSON_FORM_CLASS1
# itself) so that table's own name/comments stay accurate; this one is
# the actual lookup kitaita_e2e_pipeline.py now uses, keyed by kit_sm.
POSS_PERSON_FORM_BY_CLASS = {
    "u": POSS_PERSON_FORM_CLASS1,
    "ji": {"angu": "japo", "ako": "jako", "ake": "jake"},
}

# 2026-10-22: THIRD class, direct user suggestion ("run reverse
# inference to find the possessives") -- `kitaita_possessive_reverse_
# inference.py` (new), searching the full parallel corpus for a known-
# class noun co-occurring with its matching Swahili possessive form.
# "wi" (ANIMATE-PL, e.g. waandishi/walimu/wanaume) takes EXACTLY the
# SAME forms as "u" above (wapo/wako/wake) -- read directly, not just
# Dice-scored: MRK.7.5 "wanafunzi wako" -> "wanughi wako" (clean 1:1,
# same word order); MAT.17.10 "Wanafunzi wake" -> "Wanughi wake" (same);
# ACT.2.18 "watumishi wangu" -> "wadumiki wapo" (same, governor
# immediately precedes the possessive in both languages). This exactly
# matches an ALREADY-ESTABLISHED finding elsewhere in this project --
# 'u'/'wi' (the two ANIMATE classes) share ONE unified template, not
# just a coincidence (see NEGATIVE_ANIMATE_CLASS_PREFIX_BY_TENSE's own
# comment, "an explicit ANIMATE noun subject... follows the PERSON
# template").
POSS_PERSON_FORM_BY_CLASS["wi"] = {"angu": "wapo", "ako": "wako", "ake": "wake"}

# 2026-10-22, SAME DAY: COMPLETES the table -- direct native-speaker
# submission via the Word Bank's new "Possessives" tab (60 cells, every
# remaining class x person, submitted in one batch). Merged in rather
# than replacing the dicts above, so the earlier verse-citations for u/
# ji/wi stay attached to their own entries.
#
# The SINGULAR cells (angu/ako/ake) are STRIKINGLY regular across all
# 12 classes, worth recording as a real finding: every class =
# [class marker, final vowel resolved] + a shared stem ("apo"/"ako"/
# "ake" -- 2sg/3sg "ako"/"ake" are the SAME string as Swahili, a shared
# cognate; 1sg "apo" is NOT cognate with Swahili "-angu" at all). The
# vowel-resolution rule is clean and exceptionless for these 3 cells:
#   - class ends in a consonant+"i" (ji/ghi/chi/ri): the "i" DELETES,
#     bare consonant + stem (ji+apo -> japo, ghi+apo -> ghapo, chi+apo
#     -> chapo, ri+apo -> rapo).
#   - class ends in a consonant+"u" (ghu/ku/lu): the "u" GLIDES to "w"
#     rather than deleting (ghu+apo -> ghwapo, ku+apo -> kwapo, lu+apo
#     -> lwapo) -- a real, systematic i-deletes/u-glides split, not
#     noise (matches ordinary Bantu/Swahili vowel-sequence behavior).
#   - class ends in a consonant+"a" (va): the "a" deletes too (va+apo
#     -> vapo), same side as "i".
#   - class is BARE (no consonant at all -- u/wi, i/i_nasal): the
#     vowel glides instead of deleting, to avoid a zero onset (u->w:
#     wapo/wako/wake; i->y: yapo/yako/yake) -- "i" and "i_nasal" take
#     IDENTICAL possessive forms despite having DIFFERENT adjective-
#     concord forms (see the "i" vs "i_nasal" adjective split elsewhere
#     in this project) -- possessive concord apparently doesn't see
#     that split at all.
#
# The PLURAL cells (etu/enu/ao) are MESSIER -- checked directly, not
# forced into the same tidy rule: "ao" follows the singular pattern
# everywhere (consonant/glide + "awo"). "etu"/"enu" do NOT: most
# classes take "e" (jedu/ghwedu/redu/chedu/ghedu/vedu/yedu + their
# "-enyu" counterparts), but bare "u", "ku", AND "lu" -- NOT "ghu",
# despite also ending in "-u" -- take "o" instead (odu/onyu, kodu/
# konyu, lodu/lonyu). This "o" subgroup doesn't reduce to "classes
# ending in -u" (ghu is one and doesn't take it) or any other single
# feature tested -- reported exactly as submitted, not smoothed into a
# false regularity.
POSS_PERSON_FORM_BY_CLASS["ji"].update({"etu": "jedu", "enu": "jenyu", "ao": "jawo"})
POSS_PERSON_FORM_BY_CLASS["wi"].update({"etu": "wedu", "enu": "wenyu", "ao": "wawo"})
POSS_PERSON_FORM_BY_CLASS.update({
    "ghu": {"angu": "ghwapo", "ako": "ghwako", "ake": "ghwake",
            "etu": "ghwedu", "enu": "ghwenyu", "ao": "ghwawo"},
    "ri": {"angu": "rapo", "ako": "rako", "ake": "rake",
           "etu": "redu", "enu": "renyu", "ao": "rawo"},
    "chi": {"angu": "chapo", "ako": "chako", "ake": "chake",
            "etu": "chedu", "enu": "chenyu", "ao": "chawo"},
    "ghi": {"angu": "ghapo", "ako": "ghako", "ake": "ghake",
            "etu": "ghedu", "enu": "ghenyu", "ao": "ghawo"},
    "ku": {"angu": "kwapo", "ako": "kwako", "ake": "kwake",
           "etu": "kodu", "enu": "konyu", "ao": "kwawo"},
    "lu": {"angu": "lwapo", "ako": "lwako", "ake": "lwake",
           "etu": "lodu", "enu": "lonyu", "ao": "lwawo"},
    "va": {"angu": "vapo", "ako": "vako", "ake": "vake",
           "etu": "vedu", "enu": "venyu", "ao": "vawo"},
    "i": {"angu": "yapo", "ako": "yako", "ake": "yake",
          "etu": "yedu", "enu": "yenyu", "ao": "yawo"},
    "i_nasal": {"angu": "yapo", "ako": "yako", "ake": "yake",
                "etu": "yedu", "enu": "yenyu", "ao": "yawo"},
})

# ----------------------------------------------------------------- Kitaita NEGATION (present)
# 2026-10-15. Native-speaker-submitted via the Kitaita Word Bank's
# "Negation" tab (16 real data points, one per class/person, all built
# around the same "kupenda"/"kund" test root for a clean minimal-pair
# comparison -- see docs/status.md 2026-10-15 cont. 2/3/4). Induced
# pattern, zero exceptions except the two flagged below:
#   NEGATIVE = "nde" + [BARE, UNSHIFTED subject-marker] + root + "-e"
#     (final vowel "-e", not "-a" -- mirrors Swahili's own negative-
#     present FV shift -a->-i, just with Kitaita's own vowel)
#   EXCEPT 1sg, which is IRREGULAR: "si" + root + "-e", no "nde" at all
#     (parallels Swahili's own suppletive 1sg "si-", a striking but
#     directly-submitted, not assumed, parallel).
# The "[BARE, UNSHIFTED subject-marker]" slot is the class's own
# VERIFIED_SUBJECT_MARKERS string (u/ghu/ji/ri/i/chi/ghi/ku/lu -- NOT
# the present-tense vowel-shifted form, e.g. class-1 "u-", not "wa-")
# for a noun-governed clause, confirmed directly: "Mwalimu ndeukunde"
# analog (3sg) and "Watu ndewikunde" analog (3pl) both use the BARE
# class marker exactly, not PRESENT_SM_VOWEL_CONFIRMED's shifted form.
# For a PERSON-governed clause (no noun), 2sg/1pl/2pl use the SAME
# marker already verified for FUTURE tense (PERSON_SUBJECT_MARKERS_BY_
# TAM), not present's kwa-/na- -- confirmed directly: 2sg "ku" and 1pl
# "di" are an EXACT match to that table's future-tense entries; negation
# evidently patterns with the future-tense person-marker set, not
# present's. 2pl's submitted "ndemukunde" is just future's bare "m" +
# the SAME default linking-vowel 'u' this generator's existing bare-
# consonant-before-consonant-initial-TAM logic already inserts -- not a
# new rule, confirms the existing mechanism generalizes here too.
NEGATIVE_MARKER_KITAITA = "nde"
NEGATIVE_FINAL_VOWEL = "e"
# 2026-10-15: submitted literally as "ndevikunde" -- marker "vi", NOT the
# expected bare "va" (every other class matched its own VERIFIED_SUBJECT_
# MARKERS string exactly). Kept AS SUBMITTED (trust the speaker over the
# regularized pattern) rather than silently "corrected" to "va" -- flagged
# as unconfirmed whether this is a real va->vi alternation specific to
# negation or a one-off typo; needs a second data point.
NEGATIVE_SM_BY_CLASS_OVERRIDE = {"va": "vi"}
# 2sg/1pl/2pl only -- 1sg is NEGATIVE_PERSON_IRREGULAR below (not this
# table); 3sg/3pl are noun-governed (class 'u'/'wi'), not person at all,
# so they go through NEGATIVE_SM_BY_CLASS_OVERRIDE/VERIFIED_SUBJECT_
# MARKERS like any other noun subject, not through this table.
NEGATIVE_PERSON_SM = {"2sg": "ku", "1pl": "di", "2pl": "m"}
NEGATIVE_PERSON_IRREGULAR = {"1sg": "si"}  # replaces the whole "nde"+SM pattern

# ----------------------------------------------------------------- Kitaita NEGATION, extended to past/future/subjunctive-imperative
# 2026-10-21, direct user request ("update the negation table"). The
# Kitaita Word Bank's negation_class/negation_person rows (added
# 2026-10-21 cont. 1, 64 cells across 4 tenses) came back with 62/64
# real answers -- queried via ArtifactData and analysed here. Trusted at
# n=1 per cell like every VERIFIED_* table in this file. Two real
# findings up front, both consistent with this project's own prior
# architecture rather than contradicting it:
#   (1) PERSON markers and NOUN-CLASS markers behave DIFFERENTLY under
#       negation outside of present tense -- not a new split, this
#       project already keeps PERSON_SUBJECT_MARKERS_BY_TAM and
#       VERIFIED_SUBJECT_MARKERS as separate systems; negation inherits
#       that same separation.
#   (2) NEGATIVE PAST uses a TAM ("-re-...-eghe") that is NOT each
#       class's own AFFIRMATIVE past-TAM allomorph (VERIFIED_PAST_TAM_
#       BY_SM only has "ji"->"de", yet "ji"'s NEGATIVE past submission
#       uses "-re-", not "-de-") -- i.e. the negative paradigm has its
#       OWN dedicated past-TAM, independent of the affirmative one,
#       exactly parallel to Swahili itself (affirmative past "-li-" vs.
#       negative past "-ku-" are totally different morphemes, not a
#       negated version of the same TAM). Not assumed, directly
#       supported: 9/10 classes tested show the identical "-re-...
#       -eghe" shape regardless of what each class's own affirmative
#       past marker actually is.
#
# PERSON: literal prefix string that precedes root+FV (FV is ALWAYS
# "e", zero exceptions, across every person/tense tested including
# future and subjunctive). Each cell trusted as submitted, NOT forced
# into a single phonological rule -- e.g. past's "nde"->"ndo" vowel
# shift fires for 2sg/2pl ("ndoko"/"ndomu") but not 1pl ("ndede") despite
# all three being person markers -- kept exactly as given rather than
# regularized, since a wrong regularization is worse than a literal
# table here. 2026-10-21 correction: this dict originally also had
# "3sg"/"3pl" entries, copied from the Word Bank's "negp:3sg"/"negp:3pl"
# rows -- but `person` is NEVER "3sg"/"3pl" anywhere in this codebase
# (PERSON_LABELS only has 1sg/2sg/1pl/2pl; those 2 rows are actually
# explicit ANIMATE NOUNS, mwalimu/watu, despite the "negp" id prefix).
# Moved to NEGATIVE_ANIMATE_CLASS_PREFIX_BY_TENSE below (that table's
# own comment explains why they need a SEPARATE table, not just a
# differently-keyed entry here -- their past/future/subjunctive shapes
# genuinely differ from the other 1st/2nd-person cells').
NEGATIVE_PERSON_PREFIX_BY_TENSE = {
    ("1sg", "present"): "si", ("2sg", "present"): "ndeku", ("1pl", "present"): "ndedi",
    ("2pl", "present"): "ndemu",
    ("1sg", "past"): "se", ("2sg", "past"): "ndoko", ("1pl", "past"): "ndede",
    ("2pl", "past"): "ndomu",
    # FUTURE: retains the affirmative future TAM "-cha-" inside the
    # negative (unlike past, which drops its affirmative TAM entirely
    # and substitutes a dedicated negative one) -- SM itself unshifted.
    ("1sg", "future"): "sicha", ("2sg", "future"): "ndekucha", ("1pl", "future"): "ndedicha",
    ("2pl", "future"): "ndemcha",
    # SUBJUNCTIVE/IMPERATIVE: a COMPLETELY different template -- no
    # "nde" at all, not even for 1sg (which is suppletive in every OTHER
    # tense here); every person takes its ORDINARY affirmative subject
    # marker + the infix "-se-". Clean, zero exceptions across all 6
    # persons/animate-nouns tested (4 here, 2 more in NEGATIVE_ANIMATE_
    # CLASS_PREFIX_BY_TENSE). This "-se-" infix is ANIMATE-specific -- the
    # level subjunctive below turned out to reuse the PRESENT template
    # instead, not this one; checked directly, not assumed to match.
    ("1sg", "subjunctive_imperative"): "nise", ("2sg", "subjunctive_imperative"): "kuse",
    ("1pl", "subjunctive_imperative"): "dise", ("2pl", "subjunctive_imperative"): "mse",
}
NEGATIVE_PERSON_FINAL_VOWEL = "e"  # confirmed across all 4 tenses, no exceptions

# CLASS (noun-governed clauses): present and subjunctive/imperative turn
# out to be the SAME template (no separate subjunctive marking surfaces
# for a class/noun subject the way it does for a person subject --
# checked directly: "Mori ndeghubuke" has the identical "nde"+SM+root+FV
# shape as the present row, just a different, already-correct root).
# Future keeps FV "-a-", NOT "-e-" the way every other negative form in
# this file does (confirmed across 9/10 classes, zero exceptions; 'ku'/
# kukimbia future was skipped by the speaker, no data). Past uses the
# dedicated "-re-...-eghe" negative-past TAM described above; the
# class's own SM form going into it is stored literally per class
# below (NOT run through this project's existing u->o/i->je shift
# rules for "-re-" elsewhere -- checked directly and they do NOT match:
# only "ghu" shows a shift here (u->o, "ghore"), "ku" [also u-final]
# does not, and "ji" stays "ji" rather than the existing rule's "je" --
# a real finding, not forced to agree with the unrelated affirmative-
# existential-"-re-" shift rule). One exception, flagged not merged:
# "va" (chakula/"tosha") submitted a DIFFERENT shape entirely ("nde"+
# "vi"+"de"+root+"a", the ordinary "de" past TAM + FV "a", no "-re-
# ...-eghe" at all) -- kept separate rather than forced into the 9-class
# pattern; possibly va's own past-TAM is genuinely "de" (this project's
# VERIFIED_PAST_TAM_BY_SM already has other classes using "de"), or
# possibly a less careful answer (submitted hours later, in the same
# batch as future/subjunctive) -- not resolved either way.
NEGATIVE_CLASS_FUTURE_FINAL_VOWEL = "a"
NEGATIVE_CLASS_PAST_SM_FORM = {  # the SM-shaped material that precedes "-re-" (past) / "-de-" (va only)
    "ghu": "ghore", "i": "ire", "i_nasal": "ire", "ji": "jire", "ri": "rire",
    "chi": "chire", "ghi": "ghire", "ku": "kure", "lu": "lure",
}
NEGATIVE_CLASS_PAST_SUFFIX = "eghe"  # root + this, e.g. "buk" -> "bukeghe"
NEGATIVE_CLASS_PAST_VA_OVERRIDE = {"sm_form": "vide", "suffix": "a"}  # see comment above; "nde"+"vide"+root+"a"

# 2026-10-21: direct finding while cross-checking the Word Bank's "negp:
# 3sg"/"negp:3pl" rows (mwalimu/class 'u', watu/class 'wi' -- despite the
# "negp" id prefix, these are explicit ANIMATE NOUNS, not pro-dropped
# person pronouns) against the 10 ordinary-noun negation_class rows
# above: mwalimu/watu do NOT follow the 9-class "-re-...-eghe" past /
# "-a" future / "nde+root+e" subjunctive pattern at all -- they follow
# the SAME template as 1st/2nd-PERSON subjects instead (FV always "e";
# past identical to present, no dedicated past-TAM; subjunctive uses
# the "-se-" infix, not "nde"). Consistent with this project's existing
# ANIMATE-SG/ANIMATE-PL vs. ordinary-noun-class architectural split
# (SWAHILI_CLASS_DEFAULT_SM_GUESS etc.) -- animacy, not literally
# "is there a pro-dropped pronoun", is what decides which negative
# template applies. Checked directly: "Mwalimu ndeuchakunde"(future,
# FV "e") / "Mwalimu usekunde"(subjunctive, "-se-" infix) / "Ndeukunde"
# (past, IDENTICAL to present) -- all 3 match the person template, zero
# match the ordinary-class template. Classes 'u'/'wi' are the only ones
# this project currently treats as animate-noun-governed in this way;
# checked in translate() BEFORE the ordinary NEGATIVE_CLASS_* tables.
NEGATIVE_ANIMATE_CLASS_PREFIX_BY_TENSE = {
    ("u", "present"): "ndeu", ("u", "past"): "ndeu", ("u", "future"): "ndeucha",
    ("u", "subjunctive_imperative"): "use",
    ("wi", "present"): "ndewi", ("wi", "past"): "ndewi", ("wi", "future"): "ndewicha",
    ("wi", "subjunctive_imperative"): "wise",
}
# 2026-10-21: "ghi" (maghi/eggs) was submitted as "nderighekundeghe" --
# every OTHER class matches the "SM+re" letter order exactly (chire,
# jire, rire, kure, lure, ire); "ghi" alone is an EXACT ANAGRAM of the
# expected "ghire" with two letters transposed ("righe" vs "ghire") --
# almost certainly a typo, not a genuine exception, so "ghire" (the
# regularized form) is what is actually used above, not "righe" --
# flagged here rather than silently assumed.

# ----------------------------------------------------------------- Kitaita ASSOCIATIVE ('of') concord
# 2026-10-15. Direct native-speaker corrections, trusted at n=1 each like
# every other VERIFIED_* table in this file:
#   "mayai ya kuku" (chicken's eggs, 'mayai'/'maghi' = class 'ghi') -> Kitaita
#   "maghi gha nguku" -- associative particle "gha".
#   "upendo wa Mungu" (God's love, 'upendo'/'lukundo' = class 'lu') -> Kitaita
#   "lukundo lwa Mlungu" -- associative particle "lwa".
# 'ghi' -> 'gha' happens to be an EXACT match for that class's independently-
# verified PRESENT-TENSE subject-marker shift (PRESENT_SM_VOWEL_CONFIRMED) --
# but 'lu' -> 'lwa' is NOT: a naive "consonant stays, vowel becomes a"
# extrapolation from the bare marker 'lu' would predict "la", not "lwa" --
# the 'w' glide here matches the SAME glide 'ku'->'kwa' already shows in
# that same present-tense table (bare-vowel-final markers sometimes get a
# glide before present tense, sometimes don't -- 'ghu'->'gha' has none --
# and apparently the SAME split applies to associative concord, not
# predictable from the class shape alone). Checked here first (kitaita_
# e2e_pipeline.associative_particle_for_class), before falling back to the
# present-tense-shift hypothesis for any OTHER class -- every class not
# listed here remains an unconfirmed hypothesis, not assumed to follow
# either pattern.
KITAITA_ASSOCIATIVE_VERIFIED = {"ghi": "gha", "lu": "lwa"}

# 2026-10-19. Direct user correction -- a DIFFERENT associative construction
# from the genitive/possessive one above, NOT a special case of it: Swahili
# "[noun] -a ku-VERB" (descriptive/purposive complement -- "water FOR
# DRINKING", not "water OF [someone]") takes a DIFFERENT allomorph of the
# associative particle AND fuses directly with the verb root (the
# infinitive's own "ku-" is dropped, not kept as a separate word):
#   "Maji ya kunywa" (drinking water) -> Kitaita "Machi ghenywa" -- particle
#   "ghe" + root "nywa" (ku-NYWa, "to drink") fused into ONE word, not
#   "Machi ghe nywa" or "Machi ghe kunywa".
#   Ownership/possession with the SAME noun+particle is UNAFFECTED --
#   direct confirmation: "maji ya Bill" (Bill's water) stays "Machi gha
#   Bill", the ordinary KITAITA_ASSOCIATIVE_VERIFIED particle, two separate
#   words, Bill (a proper noun) untranslated. The particle allomorph and
#   the fusion are BOTH conditioned on the complement being an infinitive
#   verb, not on the noun's class alone.
# n=1 (class 'ghi' only) -- NOT assumed to generalize to any other class by
# formula (e.g. 'lu' -> 'lwa' in the genitive table above is NOT a simple
# vowel shift from 'lu', so there is no safe default "change the final
# vowel to 'e'" rule to extrapolate from here either). Coincidentally,
# 'ghi'->'ghe' is the SAME STRING as this class's subject-marker shift
# before the UNRELATED existential 'kogh'+'-re-' past construction
# (KOGH_RE_FRONT_VOWEL_SHIFTS['ghi'] = 'ghe', kitaita_sentence_generator.py)
# -- flagged as a possible instance of one broader "front-vowel-shift
# before certain following material" phonological pattern, but NOT merged
# with that table or assumed to be the same rule: different construction,
# checked separately, kept separate until a native speaker confirms they
# really are the same phenomenon.
KITAITA_ASSOCIATIVE_DESCRIPTOR_VERIFIED = {"ghi": "ghe"}

# 2026-10-20: 4 more classes, direct user examples:
#   "mti wa kuchoma" (firewood) -> "mudi ghokora" -- particle "gho" + root
#     "kor" (VERIFIED_ROOTS["chom"]).
#   "ulimi wa kuongea" -> "lumi loghora" -- particle "lo" + root "ghor"
#     (VERIFIED_ROOTS["onge"]).
#   "mbwa wa kulinda" -> "koshi elindia" -- particle "e" + root "lindi"
#     (VERIFIED_ROOTS["lind"]).
#   "mtu wa kucheka" -> "mundu oseka" -- particle "o" + root "sek"
#     (VERIFIED_ROOTS["chek"]).
#   "miti ya kuchoma" -> "midi ekora" -- SAME particle "e" as "mbwa"/
#     "koshi" above (both resolve_class_marker() to the bare class key
#     "i" -- see below), SAME root "kor" as "mti"/singular above
#     (confirms "chom"->"kor" generalizes across classes, as a root
#     translation should -- only the AGREEMENT prefix differs by class,
#     exactly as the architecture already assumes elsewhere).
#
# KEYS ARE THE ACTUAL RUNTIME VALUE kitaita_e2e_pipeline.resolve_class_
# marker() RETURNS for these specific Kitaita nouns, confirmed by DIRECT
# lookup against this project's tables, NOT the "canonical" named class
# (e.g. 'ghu'/'i_nasal') a human would reach for from memory. This
# matters because several of these nouns have NO VERIFIED_SUBJECT_
# MARKERS entry of their own, so resolve_class_marker() falls through to
# the RAW, un-aliased noun_class_of induction, whose own "subject_marker"
# field is a BARE, shorter string that does not disambiguate every
# canonical class (e.g. 'gh' alone doesn't say whether a noun is 'ghu' or
# 'ghi' -- confirmed directly: "mudi"/mti's raw class is bare 'gh', NOT
# the full 'ghu', even though "mudi" is independently documented
# elsewhere (Word Bank demonstrative data) as THE 'ghu'-class example
# noun). Keying by the canonical name here would have silently never
# matched at runtime -- checked this directly before finalizing these
# keys, not assumed:
#   'gh'  -- "mudi" (mti/tree); canonically class 'ghu', but resolve_
#            class_marker() returns the bare, un-aliased 'gh' for this
#            noun (no VERIFIED_SUBJECT_MARKERS override exists for it).
#   'lu'  -- "lumi" (ulimi/tongue); VERIFIED_SUBJECT_MARKERS gives 'lu'
#            directly, same key the genitive table already uses.
#   'i'   -- "koshi" (mbwa/dog, VERIFIED_SUBJECT_MARKERS override) AND
#            "midi" (miti/trees, the RAW un-aliased fallback) -- BOTH
#            resolve to the SAME bare string 'i' despite koshi and midi
#            being semantically unrelated nouns; whether they are
#            genuinely the same CONCORD class or just a coincidental
#            string collision between two different raw/verified
#            sources is not established here -- flagged, not assumed,
#            same discipline as the project's existing i/i_nasal split
#            for adjectives (a DIFFERENT class pair that happens to
#            share a subject marker too).
#   'u'   -- "mundu" (mtu/person); canonically class 'u' (confirmed
#            directly elsewhere -- the Word Bank's own demonstrative
#            gloss for class 'u' lists "mundu/person" explicitly), and
#            resolve_class_marker()'s raw fallback also returns bare 'u'
#            for it (no ambiguity for this one class specifically).
# Still n=1 per key -- not extrapolated to any OTHER class by formula.
KITAITA_ASSOCIATIVE_DESCRIPTOR_VERIFIED.update({
    "gh": "gho", "lu": "lo", "i": "e", "u": "o",
})

# ----------------------------------------------------------------- Kitaita OBJECT MARKERS (person)
# 2026-10-20. CORPUS-INFERRED HYPOTHESIS, NOT native-speaker-verified --
# a DIFFERENT, weaker confidence tier than every VERIFIED_* table above
# (which is direct human confirmation, trusted even at n=1). This table
# is the "aligned"/"statistical" tier extract_kitaita_grammar.py's own
# docstring already defines: a ranked candidate list FOR a native speaker
# to check, not a fact. Per direct user request, loaded into the Kitaita
# Word Bank Artifact's existing "Object markers" tab (object_person rows,
# `current`/`tier` fields) rather than wired into the live translation
# pipeline -- this project's own footer note already flags object marking
# as a deliberate, total "zero" placeholder because no data existed; now
# there is a HYPOTHESIS, which belongs in front of a speaker before it
# starts getting inserted into every translated sentence, not after.
#
# Method: found by isolating verses where a Swahili verb carries a KNOWN
# person object marker on the ALREADY-CONFIRMED root 'saidi' (help, see
# VERIFIED_ROOTS above) and reading off whatever sits between the Kitaita
# word's own SM/TAM and its root 'tesi' in the matching verse.
#
#   1sg "me" -- Kitaita 'ni', IDENTICAL to Swahili's own 'ni'. n=7, zero
#     exceptions (PSA.22.19/38.22/71.12/119.173, "kunisaidia"<->"kunitesie"/
#     "kunitesia", + 3 more). Bantu 1st/2nd-person subject AND object
#     markers are frequently syncretic (same string both roles) cross-
#     linguistically -- consistent with, not contradicted by, this
#     project's own existing Kitaita 1sg SUBJECT-marker-for-person data
#     using the same string elsewhere.
#   1pl "us" -- Kitaita 'di', DIFFERENT from Swahili's own 'tu'. n=3, zero
#     exceptions (JOS.10.6, 2CH.32.8, MRK.9.22, "kutusaidia"<->"kuditesie"/
#     "uditesie"). A genuine, independently-interesting finding: this
#     project should NOT assume Kitaita object markers always match
#     Swahili's just because 1sg happened to.
#   "them"/class-2 -- Kitaita 'wi'. n=10/13 (3 noisy outliers from a crude
#     automated match that isn't position-aware within the verse, not
#     re-checked by hand individually). Matches this project's OWN
#     already-established class 'wi' (wandu/wazee, "people/elders") --
#     same string already used as that class's SUBJECT marker elsewhere,
#     consistent with the SAME SM/OM syncretism pattern Swahili itself
#     shows for its class 2 (both 'wa'). NOTE: Swahili's object marker
#     'wa' is itself ambiguous between "them" (class 2) and "you-all"
#     (2pl person) -- already flagged in extract_swahili_grammar.py's own
#     PERSON_OBJECT_MARKERS comment -- so this Kitaita finding inherits
#     that SAME ambiguity, not a new one.
#   2sg "you" -- NOT included. The one candidate example found
#     (2SA.10.11, "kuacha kunitesie, ... niacha nikutesie") is genuinely
#     ambiguous on inspection: "ku" there could be the 2sg object marker
#     (mirroring Swahili's own 'ku') OR the bare-infinitive marker (same
#     string, same ambiguity already documented for the Swahili-side
#     decomposer, see kitaita_e2e_pipeline.apply_associative_descriptor_
#     concord's 2026-10-20 comment) -- this ONE example cannot
#     distinguish the two readings, and no 2nd independent example was
#     found this round. Flagged as unresolved rather than guessed.
KITAITA_PERSON_OBJECT_MARKER_HYPOTHESIS = {
    # 2026-10-20 (cont. 3): n for 1sg/1pl bumped from the full-corpus
    # reverse-inference run below -- "oko"/"kir" (save/rescue, a SECOND,
    # independent root from 'tesi'/help) reconfirmed BOTH in its own
    # verses (MAT.14.30 "niokoe"->"Nikire"; PSA.80.2/118.25 "utuokoe"->
    # "kudikire") -- two unrelated verbs agreeing on the same object-
    # marker strings is stronger evidence than one verb alone, though
    # still corpus-inferred, not speaker-verified.
    "1sg": {"marker": "ni", "n": 8, "note": "identical to Swahili's own 'ni'; now confirmed on 2 "
                                             "independent roots ('tesi'/help, 'kir'/save)"},
    "1pl": {"marker": "di", "n": 4, "note": "DIFFERS from Swahili's own 'tu'; now confirmed on 2 "
                                             "independent roots ('tesi'/help, 'kir'/save)"},
    "class2_or_2pl": {"marker": "wi", "n": 10,
                        "note": "matches this project's existing class 'wi' (wandu/wazee) subject "
                                "marker; inherits Swahili's own 2pl/class-2 object-marker ambiguity"},
}

# ----------------------------------------------------------------- Kitaita DEMONSTRATIVE ('this'/'that') concord
# 2026-10-15. Direct native-speaker correction chain (both same-day
# retractions kept brief -- see docs/status.md cont. 11/12/15 for the
# full history, not repeated here):
#
# 1. "hii kazi" (this work, 'kazi' = class 'i') -> first recorded as
#    "iri", WRONG.
# 2. Corrected to "ihi" for class 'i' -- matches this class's own
#    Bible-corpus-mined candidate (concord_families['demonstrative'],
#    Dice 0.27, previously unused).
# 3. "iri" is real too, but NOT a plural variant of class 'i' --
#    direct correction: "iri is for the ri class as it is the class
#    samaki (plural) belongs to. I didn't mean you create a new
#    class." 'ri' is an EXISTING Kitaita class (samaki/fish, the
#    plural-reading counterpart of number-invariant nouns like
#    samaki/ngelo -- see VERIFIED_SUBJECT_MARKERS), not a new sub-
#    split of 'i'. Simple fix: "iri" goes directly on 'ri', same flat
#    one-value-per-class shape every other VERIFIED_* table uses.
#
# Also per direct correction: Swahili demonstratives have THREE
# distance degrees, not two -- proximal ("hii"), a middle/referential
# degree ("hiyo" -- "that [already mentioned/near you]"), and distal
# ("ile"). `nllb_concord_rewrite_pipeline.DEM_REFERENTIAL_TABLE` (new)
# covers the Swahili side; KITAITA_DEM_REFERENTIAL_VERIFIED below is
# the Kitaita side, currently empty -- no data for any class yet.
# don't assume it generalizes.
#
# n=1, trusted like every other VERIFIED_* table in this file. Distal
# ("that") has zero data for any class/number -- not attempted,
# flagged not guessed.
#
# Editable live via kitaita_local_webapp.py's "/demonstratives" page and
# the Kitaita Word Bank Artifact's "Demonstratives" tab, grouped by
# Kitaita class, with the (low-confidence, statistical-only) corpus
# hints shown for classes with no verified data yet. Edits call
# save_demonstrative_concord() below, which writes kitaita_
# demonstrative_concord.json AND mutates the three dicts below IN PLACE
# (never rebound) so every module that already did `from kitaita_
# verified_corrections import KITAITA_DEM_PROXIMAL_VERIFIED` (e.g.
# kitaita_e2e_pipeline) sees an edit immediately, no restart needed.
# 2026-10-16: all 12 classes submitted via the Word Bank's "Demonstratives"
# tab (contributions collection, "dem:<cls>:<degree>" docs) -- closed-set
# whole-word forms, not derived from any prefix rule (consistent with this
# series' own "do NOT decompose productively" design, see module docstring
# above). 'i' and 'i_nasal' submitted IDENTICAL forms for all 3 degrees --
# a real, notable finding (demonstratives apparently do NOT carry the i/
# i_nasal distinction that adjective concord does), not an error, kept as
# given. Single-contributor data (n=1 per cell, same trust tier as every
# other VERIFIED_* table) -- not yet cross-checked against a second
# speaker or real sentence context.
KITAITA_DEM_PROXIMAL_VERIFIED = {"u": "uhu", "ghu": "ughu", "ji": "iji", "ri": "iri", "i": "ihi",
                                   "i_nasal": "ihi", "chi": "ichi", "wi": "awa", "ghi": "agha",
                                   "ku": "uku", "lu": "ulu", "va": "ivi"}
KITAITA_DEM_REFERENTIAL_VERIFIED = {"u": "uo", "ghu": "ugho", "ji": "ijo", "ri": "iro", "i": "io",
                                      "i_nasal": "io", "chi": "icho", "wi": "awo", "ghi": "agho",
                                      "ku": "uko", "lu": "ulo", "va": "ivo"}
KITAITA_DEM_DISTAL_VERIFIED = {"u": "uja", "ghu": "ghuja", "ji": "ija", "ri": "irija", "i": "ija",
                                 "i_nasal": "ija", "chi": "ichija", "wi": "awaja", "ghi": "aghaja",
                                 "ku": "ukuja", "lu": "uluja", "va": "vija"}

_DEM_CONCORD_PATH = _Path(__file__).with_name("kitaita_demonstrative_concord.json")
_DEM_DEGREES = ("proximal", "referential", "distal")


def _load_demonstrative_concord_file() -> None:
    """Overlay kitaita_demonstrative_concord.json's current values onto
    the three dicts above, IN PLACE (clear+update, never rebind) so every
    module that imported them by name keeps seeing the live data."""
    if not _DEM_CONCORD_PATH.exists():
        return
    data = _json.loads(_DEM_CONCORD_PATH.read_text(encoding="utf-8"))
    tables = {"proximal": KITAITA_DEM_PROXIMAL_VERIFIED, "referential": KITAITA_DEM_REFERENTIAL_VERIFIED,
              "distal": KITAITA_DEM_DISTAL_VERIFIED}
    for degree, table in tables.items():
        for cls in data.get("classes", []):
            value = (cls.get(degree) or {}).get("value", "").strip()
            if value:
                table[cls["id"]] = value
            else:
                table.pop(cls["id"], None)


def save_demonstrative_concord(updates: dict) -> list:
    """updates: {class_id: {"proximal": str, "referential": str, "distal": str}}
    -- any subset of the 3 keys per class. Writes kitaita_demonstrative_
    concord.json (every other field -- nearest_swahili_class, sample_members,
    corpus_hints, notes -- is preserved untouched) and reloads the 3 live
    dicts above. Treated as trusted, direct human input, same tier as every
    other VERIFIED_* table in this file (applied even at n=1) -- NOT
    re-derived or guessed. Returns the full updated class list."""
    data = _json.loads(_DEM_CONCORD_PATH.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in data["classes"]}
    for cls_id, vals in updates.items():
        c = by_id.get(cls_id)
        if c is None:
            continue
        for degree in _DEM_DEGREES:
            if degree not in vals:
                continue
            v = vals[degree].strip()
            slot = c[degree]
            slot["value"] = v
            slot["status"] = "workbench-edit" if v else "unverified"
    _DEM_CONCORD_PATH.write_text(_json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    _load_demonstrative_concord_file()
    return data["classes"]


def get_demonstrative_concord_classes() -> list:
    """Current full class list (page content), straight from the JSON
    file -- used by both the GET page render and the POST response."""
    if not _DEM_CONCORD_PATH.exists():
        return []
    return _json.loads(_DEM_CONCORD_PATH.read_text(encoding="utf-8"))["classes"]


_load_demonstrative_concord_file()

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
    "ituku": "ji",    # was "ku" -- confirmed 2x (s8 sem+ka, and isanga's twin case s10)
    "isanga": "ji",   # was "ku" -- confirmed 2x (s10 sem+ka, s11 ta+kuw)
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

KUBWA_FORM_BY_SM = {"ji": "ibaa", "chi": "kibaa", "va": "vibaa"}
# 2026-10-11: 'va' (vitabu/plural-chi-class) -> 'vibaa' added from the SAME
# concord_families.py mining pass that found 'chi'->'kibaa' -- "vikubwa"
# ("big" agreeing with plural books/cups) aligns to "vibaa" at Dice 0.44,
# "high" confidence tier, independently consistent with the already-
# confirmed m-/i-/ki- pattern (va is chi's plural counterpart, vi-baa
# matches chi's ki-baa the same way Swahili's own ki-/vi- pair does).
KUBWA_DEFAULT_FORM = "mbaa"

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
FUNCTION_WORD_TRANSLATIONS = {"na": "na"}

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

"""Version B rule book (rulebook_correct.py): pure Python, runs in CI."""
from rulebook_correct import correct, violations


def test_agreement_violation_detected_both_orders():
    assert violations("kwa ugho isanga ja mbembe")      # cl.3/14 demonstrative before a cl.5 noun
    assert not violations("kwa ijo isanga ja mbembe")    # cl.5 demonstrative: fine
    assert violations("isanga ugho")                     # noun then wrong-class demonstrative


def test_rerank_picks_fewer_violations_and_ties_keep_model_order():
    nbest = ["Uganda yawoka kwa ugho isanga", "Uganda yawoka kwa ijo isanga", "Uganda yawoka kwa icho isanga"]
    r = correct("Uganda imeanza", nbest, steps=("rerank",))
    assert r["chosen_candidate"] == 1 and r["kitaita"] == nbest[1]
    same = ["Mwana walala.", "Mwana wadalala."]
    assert correct("Mtoto analala.", same, steps=("rerank",))["chosen_candidate"] == 0


def test_fill_replaces_copied_swahili_only_when_mined_form_exists():
    r = correct("Kampuni itazidi kuimarisha huduma.", ["Kampuni ichachuria kuimarisha huduma."], steps=("fill",))
    assert "kuimarisha" not in r["kitaita"] and r["kitaita"].endswith(".")
    kept = correct("Kampuni zzqxv.", ["Kampuni zzqxv."], steps=("fill",))
    assert kept["kitaita"] == "Kampuni zzqxv."  # no mined form: stays Swahili


def test_unknown_word_replaced_only_when_alignment_is_unambiguous():
    r = correct("Mtoto analala zzqsw.", ["Mwana walala qqzzx."], steps=("unknown",))
    assert r["kitaita"] == "Mwana walala zzqsw."
    two_free = correct("Mtoto zzqsw zzqsy.", ["Mwana qqzzx."], steps=("unknown",))
    assert two_free["kitaita"] == "Mwana qqzzx."  # two candidate source words: keep the model's word


def test_repeats_removed_and_draft_kept_for_comparison():
    r = correct("Wakina mama", ["W'eke w'eke w'eke w'eke mao"], steps=("repeat",))
    assert r["kitaita"] == "W'eke w'eke mao" and r["draft"] == "W'eke w'eke w'eke w'eke mao"


def test_no_brackets_ever():
    r = correct("Mtoto analala zzqsw.", ["Mwana walala qqzzx."])
    assert "[?" not in r["kitaita"]


def test_applicative_added_when_the_draft_drops_it():
    # user report 2026-11-02: "watoto walipigania chakula" -> draft "w'ana w'edelwa vindo"
    r = correct("Watoto walipigania chakula", ["W'ana w'edelwa vindo"], steps=("applicative",))
    assert r["kitaita"] == "W'ana w'edeluya vindo"
    assert correct("Alimfanyia kazi", ["Wambonya kazi"], steps=("applicative",))["kitaita"] == "Wambonyera kazi"


def test_applicative_left_alone_without_one_in_the_source_or_for_passives():
    assert correct("Watoto walipigana", ["W'ana w'edelwa"], steps=("applicative",))["changes"] == []
    assert correct("Wamekataliwa", ["W'alegha"], steps=("applicative",))["changes"] == []


def test_speaker_verified_sentence_wins():
    r = correct("Hawa wazee ndio walipigania uhuru wetu.", ["Aw'a w'aghosi niw'o w'ilwidieghe w'uhuru ghodu"])
    assert r["kitaita"] == "Awa waghosi niwo wiluieghe wuhuru ghodu"
    assert r["changes"][0][0] == "verified"


def test_verified_applicative_perfective_and_id_repair_before_unknown_step():
    # speaker-verified wi-lu-ie-ghe: lw + applicative + perfective -> lu-ie
    for draft in ("W'aghosi niw'o w'ilwidieghe w'uhuru", "W'aghosi niw'o w'ilwieghe w'uhuru"):
        assert correct("Wazee ndio walipigania uhuru", [draft])["kitaita"] == "W'aghosi niw'o w'iluieghe w'uhuru"


def test_palatal_root_ere_left_alone():
    # di-bony-er-e 'do for us' already carries the applicative
    assert correct("Tutendee mema", ["Dibonyere mecha"], steps=("applicative",))["changes"] == []


def test_one_consonant_root_applicative():
    # ku-l-a 'eat' -> Kidaw'ida -j- (kula -> kuja); a-li-m-ku-l-i-a 'ate for/on him'
    r = correct("Mama alimkulia chakula mtoto", ["Mao odemja mwana vindo"], steps=("applicative",))
    assert r["kitaita"] == "Mao odemjia mwana vindo"  # plain -i-, as attested ra-di-ch-i-a 'came to us'
    assert correct("Mvua zilitukujia", ["Mbua radicha"], steps=("applicative",))["kitaita"] == "Mbua radichia"
    # look-alikes without an object marker before ku: 'cried', 'grew up', 'cried for you'
    for src in ("Mama alilia", "Mtoto amekulia mjini", "Alikulia"):
        assert correct(src, ["Mao odeja"], steps=("applicative",))["changes"] == []


KUKU = "Kuku alipokuwa anarudi nyumbani alikutana na Mama yake Jongoo."
KUKU_CLAUSES = [{"verb": v, "subject": "kuku", "subject_kitaita": "nguku", "swahili_sm": "a"}
                for v in ("alipokuwa", "anarudi", "alikutana")]


def test_subject_marker_matched_to_pipeline_subject_class():
    # user report 2026-11-02: person-class o-/u- carried over from Swahili a-; nguku is an i- noun
    r = correct(KUKU, ["Nguku ingelo orekoghe ukiwuya nyumbenyi ukakwana na mae Jongoo."],
                steps=("subject",), clauses=KUKU_CLAUSES)
    assert r["kitaita"] == "Nguku ingelo erekoghe ukiwuya nyumbenyi ikakwana na mae Jongoo."
    assert any("ukiwuya" in n for n in r["notes"])  # i-ki below the 75% bar: left alone, noted


def test_subject_step_needs_matching_verb_counts_and_a_class():
    two = correct(KUKU, ["Nguku orekoghe nyumbenyi ukakwana"], steps=("subject",), clauses=KUKU_CLAUSES)
    assert two["changes"] == [] and "not run" in two["notes"][0]
    none = [dict(c, subject="zzqx", subject_kitaita=None) for c in KUKU_CLAUSES]  # no class data
    assert correct(KUKU, ["Nguku orekoghe ukiwuya ukakwana"], steps=("subject",), clauses=none)["changes"] == []
    assert correct(KUKU, ["Nguku orekoghe ukiwuya ukakwana"], steps=("subject",))["changes"] == []


def test_subject_step_leaves_agreeing_verbs_alone():
    cl = [{"verb": "alikuja", "subject": "mtu", "subject_kitaita": "mundu", "swahili_sm": "a"}]
    assert correct("Mtu alikuja", ["Mundu ukacha"], steps=("subject",), clauses=cl)["changes"] == []


def test_subject_step_needs_the_swahili_number():
    no_number = [dict(c, swahili_sm=None) for c in KUKU_CLAUSES]
    assert correct(KUKU, ["Nguku orekoghe ukiwuya ukakwana"], steps=("subject",), clauses=no_number)["changes"] == []


def test_nouns_and_locatives_are_not_taken_for_verbs():
    cl = [{"verb": "Amerudi", "subject": None, "subject_kitaita": None, "swahili_sm": "a"}]
    r = correct("Amerudi kutoka safari.", ["W'awurieghe kufuma charonyi."], steps=("subject",), clauses=cl)
    assert r["changes"] == []


def test_tense_rules_follow_the_swahili_tense():
    # verb_morphology.md rewrite rules (davswa register); audit of test_novel2, 2026-11-03
    t = lambda src, d: correct(src, [d], steps=("tense",))["kitaita"]
    assert t("Amefika", "Uvika") == "Wavika"                                   # -me- -> SM+a
    assert t("Nitaenda kesho", "Niwenda kesho") == "Nichaenda kesho"           # -ta- -> SM-cha-
    assert t("Usiende na watu wabaya", "Kughendanye na wandu wabaya") == "Kuseghendanye na wandu wabaya"
    assert t("Hakunifahamu", "Ndeunimanyagha") == "Ndeunimanyire"              # past negative -> -ie/-ire
    assert t("Wanataka kwenda", "W'ekunda kughenda") == "W'ekunda kughenda"    # no -na- rule (stative 'want')


def test_extension_rules_insert_only_attested_forms():
    e = lambda src, d: correct(src, [d], steps=("extension",))["kitaita"]
    assert e("Ameongezewa mshahara", "W'echuria mshahara") == "Wechurilwa mshahara"   # passive -lw-
    assert e("Pikia watoto", "Deka wana") == "Dekia wana"                             # applicative, imperative
    assert e("Watoto walipigana", "W'ana w'edelwa") == "W'ana w'edelwa"               # nothing to add
    assert e("Ng'ombe zimeibiwa", "Ng'ombe rawiwa") == "Ng'ombe rawiwa"               # -iw- is already passive
    assert e("Unaweza kunipea maji", "Kwadima kunineka machi") == "Kwadima kunineka machi"  # -pea is lexical

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

"""Rule-book flagger (rulebook_flagger.py): pure Python, runs in CI."""
import rulebook_flagger as F


def cats(src, hyp):
    return dict(F.flag(src, hyp))


def test_names_are_never_flagged_as_errors():
    c = cats("Baada ya miaka mitatu, Magozwe alitoroka.", "Nyuma ya miaka idadu, Magozwe ukakimbia.")
    assert c["Magozwe"] == "name"


def test_known_kidawida_words_pass():
    c = cats("Mtoto analala.", "Mwana walala.")
    assert set(c.values()) == {"ok"}


def test_copied_swahili_is_flagged_but_shared_words_are_not():
    c = cats("Kampuni itazidi kuimarisha huduma na kazi.", "Kampuni ichuriagha kuimarisha na kazi.")
    assert c["kuimarisha"] == "swahili"
    assert c["na"] == "ok"  # established in davswa Kidaw'ida


def test_invented_word_is_unknown_and_repeats_are_caught():
    c = cats("Watu wengi", "zzqxvbn")
    assert c["zzqxvbn"] == "unknown"
    out = F.flag("Wakina mama", "W'eke w'eke w'eke w'eke")
    assert [k for _, k in out][2:] == ["repeat", "repeat"]


def test_comma_glued_tokens_are_split():
    toks = [t for t, _ in F.flag("Ninakusikiza, sema.", "Nakusikira,ghora.")]
    assert len(toks) == 2

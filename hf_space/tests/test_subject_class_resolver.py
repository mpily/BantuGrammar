"""Subject-class resolver: class pair from the Kidaw'ida noun, number from the Swahili verb."""
from subject_class_resolver import resolve


def test_number_comes_from_the_swahili_verb():
    assert resolve("kuku", "nguku", "a")[0] == "i"      # verified i-, singular
    assert resolve("kuku", "nguku", "wa")[0] == "ri"    # same form, plural -> class 10
    assert resolve("nguo", None, "zi")[0] == "ri"
    assert resolve("watoto", "wana", "wa")[0] == "wi"


def test_unverified_nouns_use_corpus_and_round1_data():
    assert resolve("jeshi", None, "li")[0] == "ji"      # concord: class 5
    assert resolve("wanyama", "nyamandu", "wa")[0] == "ri"
    assert resolve("kondoo", None, "i")[0] == "i"       # user: kondoo -> ngondi; 9/10
    assert resolve("punda", None, "a")[0] == "u"        # class 9 form, wa- agreement (concord)


def test_unresolvable_cases_return_none():
    assert resolve("leso", None, "i")[0] is None        # no class data
    assert resolve("kuku", "nguku", "ku")[0] is None    # number unknown
    assert resolve("uhuru", "wuhuru", "u", tiers=("concord", "round1"))[0] is None  # class 14: split u-/ghu-


def test_bible_decides_variation():
    # user decision 2026-11-03: the Bible (more standardised) decides
    assert resolve("barabara", "chia", "i")[0] == "i"       # sorted chi-; Bible 9/10 (njia)
    assert resolve("barabara", "chia", "zi")[0] == "ri"
    assert resolve("rehema", "wughoma", "u")[0] == "u"      # class 14 per noun: Bible u- 13 / ghu- 1
    assert resolve("dunia", "wurumwengu", "u")[0] == "ghu"  # Bible u- 1 / ghu- 13
    assert resolve("mapacha", "maza", "ya")[0] == "ghi"     # Bible homograph (mambo): no override

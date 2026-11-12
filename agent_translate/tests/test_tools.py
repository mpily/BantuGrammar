"""Unit tests for agent_translate/tools.py (CLAUDE_rulebook.md §6.5): every tool, unknown inputs, flagged
cases (coordination, animacy override, conflicting nouns), JSON-serialisable outputs, no exceptions."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agent_translate import tools as T  # noqa: E402

STATUSES = {"ok", "unknown", "unattested", "flag"}


def _ok(r):
    json.dumps(r)
    assert r["status"] in STATUSES
    assert "rule_ids" in r
    return r


def test_schemas_match_tools():
    names = [t["name"] for t in json.loads((Path(T.__file__).parent / "tool_schemas.json").read_text())]
    assert names == list(T.TOOL_NAMES)
    for t in json.loads((Path(T.__file__).parent / "tool_schemas.json").read_text()):
        assert set(t) == {"name", "description", "input_schema"}


def test_parse_swahili_verb():
    r = _ok(T.parse_swahili_verb("walipigania"))
    assert r["status"] == "ok" and r["parses"][0]["tense_key"] == "past"
    assert _ok(T.parse_swahili_verb("hakufahamu"))["parses"][0]["negative"] is True
    assert _ok(T.parse_swahili_verb("xqzt"))["status"] == "unknown"
    assert _ok(T.parse_swahili_verb(""))["status"] == "unknown"


def test_lookup_word():
    r = _ok(T.lookup_word("kitabu"))
    assert r["candidates"] and r["candidates"][0]["lemma"] == "chuo"
    assert _ok(T.lookup_word("anapenda"))["candidates"][0]["lemma"] == "kund"   # speaker-verified root
    assert _ok(T.lookup_word("xyzzyq"))["status"] == "unknown"
    assert _ok(T.lookup_word("kitabu", pos="V"))["status"] == "unknown"


def test_noun_info_verified_and_unknown():
    r = _ok(T.noun_info("nguku"))
    assert r["class"] == "i" and r["number_invariant"] and r["animate"]
    assert _ok(T.noun_info("qqqq"))["status"] == "unknown"


def test_noun_info_conflict():
    r = _ok(T.noun_info("chia"))
    assert r["conflict"] is True and r["class"] == "chi" and r["readings"]
    assert "NOUN.conflict.chia" in r["rule_ids"]


def test_noun_info_animacy_flag():
    r = _ok(T.noun_info("choka", sw_noun="nyoka"))
    assert r["status"] == "flag" and "CLASS.animate_override" in r["rule_ids"]


@pytest.mark.parametrize("slot", T.CONCORD_SLOTS)
def test_concord_every_slot(slot):
    r = _ok(T.concord("ji", slot))
    assert r["status"] != "unknown" or slot == "SELF"


def test_concord_values():
    assert T.concord("ji", "SM_a")["forms"] == ["ja"]
    assert T.concord("wi", "POSS")["forms"]["2pl"] == "wenyu"
    assert T.concord("7", "SM")["forms"] == ["chi"]           # Bantu number accepted
    assert _ok(T.concord("zz", "SM"))["status"] == "unknown"
    assert _ok(T.concord("ji", "NOPE"))["status"] == "unknown"


@pytest.mark.parametrize("feat,form", [
    (dict(dav_root="lw", subj="wi", tense="rel_past_cleft", exts=["applicative"]), "wiluieghe"),  # speaker-verified
    (dict(dav_root="kogh", subj="1sg", tense="past"), "nerekoghe"),                                # speaker-verified
    (dict(dav_root="kund", subj="wi", tense="present"), "wawekunda"),
    (dict(dav_root="won", subj="1sg", tense="future", obj="3sg"), "nichammbona"),
    (dict(dav_root="dim", subj="wi", tense="present", neg=True), "ndewidimagha"),
    (dict(dav_root="fw", subj="3sg", tense="future", neg=True), "ndechaafwa"),
    (dict(dav_root="fung", subj="ri", tense="perfect", exts=["passive"]), "rafungwa"),
])
def test_generate_verb_forms(feat, form):
    r = _ok(T.generate_verb(feat))
    assert r["form"] == form and r["rule_ids"]


def test_generate_verb_flags_and_unknown():
    assert _ok(T.generate_verb(dict(dav_root="kab", subj="wi", tense="past", subj_coordinated=True)))["status"] == "flag"
    assert _ok(T.generate_verb(dict(dav_root="kab", subj="ji", tense="past", subj_noun_sw="paka")))["status"] == "flag"
    assert _ok(T.generate_verb(dict(dav_root="kab", subj="zz", tense="past")))["status"] == "unknown"
    assert _ok(T.generate_verb(dict(dav_root="kab", subj="wi", tense="nonsense")))["status"] == "unknown"
    assert _ok(T.generate_verb(dict(dav_root="kab", subj="wi", tense="narrative", neg=True)))["status"] == "unknown"
    assert _ok(T.generate_verb({}))["status"] == "unknown"
    assert _ok(T.generate_verb(None))["status"] == "unknown"


def test_generate_verb_present_has_zero_alternative():
    r = T.generate_verb(dict(dav_root="kund", subj="wi", tense="present"))
    assert r["alternatives"][0]["form"] == "wakunda"


def test_generate_modifier():
    assert _ok(T.generate_modifier("ji", "baa", "adj"))["form"] == "ibaa"
    assert T.generate_modifier("ji", "ingi", "num")["form"] == "jingi"
    assert T.generate_modifier("wi", "2pl", "poss")["form"] == "wenyu"
    assert T.generate_modifier("ji", "x", "dem_near")["form"] == "iji"
    assert _ok(T.generate_modifier("ji", "baa", "nope"))["status"] == "unknown"
    assert _ok(T.generate_modifier("zz", "baa", "adj"))["status"] == "unknown"


def test_retrieve_examples():
    r = _ok(T.retrieve_examples("Simba hula nyasi", 3))
    if r["status"] == "ok":
        assert len(r["examples"]) == 3 and all(e["pair_id"].startswith(("train:", "bible:")) for e in r["examples"])
        manifest = json.loads((Path(T.__file__).parents[1] / "rulebook" / "split_manifest.json").read_text())
        bad = set(manifest["excluded_ids"]) | set(manifest["dev_ids"]) | set(manifest["dropped_train_ids"])
        assert not {e["pair_id"] for e in r["examples"]} & bad


def test_check_translation():
    r = _ok(T.check_translation("Nilikuwa nyumbani", "Nerekoghe nyumbenyi"))
    assert r["summary"]["n_all_checks_pass"] == 1
    bad = T.check_translation("Nilikuwa nyumbani", "Nachakogha nyumbenyi")
    assert bad["verbs"][0]["status"] in ("flag", "unknown")
    assert _ok(T.check_translation("", ""))["status"] == "unknown"


def test_call_dispatch():
    assert T.call("concord", {"cls": "ji", "slot": "SM"})["forms"] == ["ji"]
    assert T.call("nope", {})["status"] == "unknown"

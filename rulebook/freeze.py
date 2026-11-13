"""
freeze.py

2026-11-12, CLAUDE_rulebook.md §7. Writes rulebook/MANIFEST.json: sha256 of the frozen rule book,
dictionary, tools, retrieval index source, every input file, the excluded-ID list, agent outputs that
fed the lexicon, and the git commit. After the rulebook-v1 tag: no rule edits until arm C/E/A
results are in; fixes go to rulebook-v2.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path.home() / "languagemodels" / "rulebook_data"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


frozen = ["rulebook/rules.json", "rulebook/RULEBOOK.md", "rulebook/lexicon.json", "rulebook/bootstrap_lexicon.json",
          "rulebook/dav_vocab.json", "rulebook/split_manifest.json", "rulebook/induced_verb_rules.json",
          "rulebook/induced_extension_rules.json", "rulebook/validation_results.json", "rulebook/speaker_additions.json", "rulebook/agent_round5_decisions.json",
          "agent_translate/tools.py", "agent_translate/tool_schemas.json", "agent_translate/tests/test_tools.py"]
code = sorted(str(p.relative_to(ROOT)) for p in HERE.glob("*.py"))
inputs = ["hf_space/kitaita_speaker_verified.json", "hf_space/kitaita_verified_corrections.py", "hf_space/kitaita_verb.py",
          "hf_space/swahili_verb.py", "hf_space/rulebook_correct.py", "hf_space/rulebook_flagger.py",
          "hf_space/kitaita_root_alignment.json", "hf_space/page1_vocabulary.json",
          "hf_space/kitaita_demonstrative_concord.json", "hf_space/kitaita_bible_class_decisions.json"]
not_used = ["hf_space/kitaita_mined_lexicon_round1.json", "hf_space/kitaita_verb_roots.json",
            "hf_space/kitaita_noun_classes_round1.json", "hf_space/kitaita_subject_classes.json",
            "hf_space/kitaita_subject_markers.json", "hf_space/kitaita_agreement_data.json",
            "hf_space/kitaita_training_vocab.json"]
outside = {"train.jsonl (retrieval index source, lexicon/rules evidence)": DATA / "train.jsonl",
           "dev.jsonl (validation only)": DATA / "dev.jsonl"}
agents = sorted(DATA.glob("agents/*/out_agent*.jsonl")) + sorted(DATA.glob("agents/*/batch_agent*.json")) + \
    sorted(DATA.glob("agents/segment*/words.json")) + \
    sorted(DATA.glob("agents/round5/out_connectives_*.jsonl")) + sorted(DATA.glob("agents/round5/batch_connectives.json"))
TAG = sys.argv[1] if len(sys.argv) > 1 else "rulebook-v1"
man = {
    "tag": TAG,
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    "frozen_sha256": {p: sha(ROOT / p) for p in frozen},
    "code_sha256": {p: sha(ROOT / p) for p in code},
    "inputs_sha256": {p: sha(ROOT / p) for p in inputs},
    "inputs_left_out_contaminated": {p: sha(ROOT / p) for p in not_used},
    "inputs_left_out_note": "built from the pool that includes arm-A val and test duplicates (status 2026-11-11 "
                            "step 1); user decision: leave out. kitaita_root_alignment.json is used only by the "
                            "Swahili parser to pick readings.",
    "outside_git_sha256": {k: sha(v) for k, v in outside.items()},
    "agent_files_sha256": {str(p.relative_to(DATA)): sha(p) for p in agents},
    "excluded_ids": {"source": "rulebook/split_manifest.json", "n_excluded": None, "n_dropped_train": None, "n_dev": None},
    "changes_from_v1": ("rules.json: + CLASS.wu_prefix.modifier_ghu; lexicon.json: look-alike classes (gh/ghu) vote as "
                        "one group -> 94 candidates get class ghu (25 plurals follow), new mod_class fields; tools.py: "
                        "noun_info returns mod_class; tool_schemas.json: descriptions say which class to pass")
                       if TAG == "rulebook-v2" else
                       ("v2 + rules.json: FV.perfective.other_eo.ie_by_root (e/o roots -ie, kogh keeps -e), "
                        "FV.perfective.no_vowel.ie (ch -> chie), EXT.passive.w_final.flag; lexicon.json: verified verb "
                        "roots p, li tagged V; tools.py: lookup_word short-root fallback (alimpa -> p -> nek), the two "
                        "perfective rules, the passive flag") if TAG == "rulebook-v3" else
                       ("v3 + lexicon.json: speaker additions (rulebook/speaker_additions.json: sasa -> idana); "
                        "tools.py lookup_word: pos mismatch keeps candidates (pos_note), non-verb candidates carry the "
                        "full `word`, bare one-syllable verbs (pa -> p -> nek); tool_schemas: lookup_word description")
                       if TAG == "rulebook-v4" else
                       ("v4 + lexicon.json: final agent review (round 5; Fable 5.2 agents until the usage limit, then Opus; rulebook/agent_round5_merge.py "
                        "-> agent_round5_decisions.json): first candidates promoted / added with corpus support / removed, "
                        "full words for non-verbs, connectives incl. multi-word (two agents agree, >= 3 pairs). Agent "
                        "judgement, not speaker verification") if TAG == "rulebook-v5" else None,
    "rule": "No rule or lexicon edits after this tag until arm C/E/A results are in; fixes go to rulebook-v2 and are reported as such.",
}
sm = json.loads((HERE / "split_manifest.json").read_text())
man["excluded_ids"].update(n_excluded=len(sm["excluded_ids"]), n_dropped_train=len(sm["dropped_train_ids"]),
                           n_dev=len(sm["dev_ids"]))
(HERE / "MANIFEST.json").write_text(json.dumps(man, indent=1))
print(man["git_commit"], len(man["agent_files_sha256"]), "agent files")

"""
tool_bridge.py -- the plumbing between a model and the existing tools.

The model never runs code. It sends a tool name + JSON arguments; this file runs the
matching function in agent_translate/tools.py and returns the result as JSON text.
At the bottom, YOU play the model: you choose the calls by hand.
"""
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]      # kitaita-project/
sys.path.insert(0, str(PROJECT))

from agent_translate import tools                  # the 8 frozen tools (rulebook-v1)

SCHEMAS = PROJECT / "agent_translate" / "tool_schemas.json"


def load_schemas() -> list[dict]:
    """The tool definitions exactly as the model will receive them."""
    with SCHEMAS.open("r", encoding="utf-8") as schema_file:
        tool_details = json.load(schema_file)
        return tool_details


def show_schemas(schemas: list[dict]) -> None:
    """Print what the model is told about each tool."""
    for s in schemas:
        print(s['name'], s['input_schema']['required'], s["description"][:80])
        


def run_tool(name: str, args: dict) -> str:
    """What the harness does for every tool call the model makes."""
    # TODO: call tools.call(name, args) and return it as a JSON string
    #       hint: json.dumps(..., ensure_ascii=False) keeps Kidaw'ida apostrophes readable
    return json.dumps(tools.call(name,args),ensure_ascii=False)


def step(name: str, args: dict) -> dict:
    """Run one call, print request and result, return the parsed result for the next step."""
    print(f"\n>>> {name}({json.dumps(args, ensure_ascii=False)})")
    text = run_tool(name, args)
    print(text[:600])                    # results can be long; the model sees all of it
    return json.loads(text)


if __name__ == "__main__":
    #print(load_schemas()[0]["input_schema"].keys())
    show_schemas(load_schemas())

    # ---- You are the model now. Sentence:
    sw = "Hawa wazee ndio walipigania uhuru wetu"
    # speaker-verified answer: "Awa waghosi niwo wiluieghe wuhuru ghodu"

    # 1. Analyse the verb.
    parse = step("parse_swahili_verb", {"word": "walipigania"})
    tense = parse["parses"][0]["tense_key"]
    exts = parse["parses"][0]["exts"]
    root = parse["parses"][0]["root"]
    # TODO: from parse["parses"], pick a reading and note its tense_key and exts

    # 2. Find the Kidaw'ida root for the verb.
    kidaw_root = step("lookup_word",{'sw':root})
    # TODO: step("lookup_word", {...}) -- try the word, then the root from step 1.
    #       Compare what comes back with the verified root lw ('fight').

    # 3. Find the subject noun's class (it controls the verb's subject marker).
    kidaw_sub = step("lookup_word",{"sw":"wazee"})["candidates"][1]["lemma"]
    kidaw_noun = step("noun_info",{"dav_noun":kidaw_sub})
    print("kidaw_noun structure", kidaw_noun)
    kidaw_sm = step("concord",{"cls":kidaw_noun["class"],"slot":"SM"})["forms"][0]
    # TODO: step("lookup_word", {"sw": "wazee"}) to get the Kidaw'ida noun,
    #       then step("noun_info", {"dav_noun": ...})

    # 4. Generate the verb from the features you collected.
    kidaw_verb = step("generate_verb",{"features":{"dav_root":kidaw_root,"subj": kidaw_sm, "tense":tense,"exts":exts}})
    # TODO: step("generate_verb", {"features": {
    #           "dav_root": ..., "subj": ..., "tense": ..., "exts": [...]}})
    #       Then try tense="rel_past_cleft" (relative after 'ndio') and compare with wiluieghe.

    # 5. Check your draft sentence.
    step("check_translation",{"sw":"walipigania","dav":kidaw_verb})
    # TODO: step("check_translation", {"sw": sw, "dav": "<your draft>"})


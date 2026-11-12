"""
retrieval_demo.py -- the "R" in RAG, on its own.

Retrieval = find training pairs whose Swahili side looks like the input sentence,
then paste them into the prompt as worked examples. No model here: just look
at what comes back and decide whether it would help a translator.

Two retrievers side by side:
  - the tool (retrieve_examples): character 3-5-gram TF-IDF, so 'walipigania' matches 'anapigania'
  - a naive one written here: count shared whole words
Run:  ~/languagemodels/bert-env/bin/python HaikuHarness/retrieval_demo.py ["your sentence" ...]
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tool_bridge import run_tool                 # file 1
from agent_translate import tools

SENTENCES = [
    "Hawa wazee ndio walipigania uhuru wetu",
    "Mtoto hakula chakula jana",
    "Mwalimu aliwaambia wanafunzi waende nyumbani mapema kwa sababu mvua ilikuwa inanyesha",
]


def retrieve(sw: str, k: int = 5) -> list[dict]:
    """The tool's retriever, called exactly as the model will call it."""
    return json.loads(run_tool("retrieve_examples", {"sw_sentence": sw, "k": k}))["examples"]


def show(examples: list[dict], title: str) -> None:
    print(f"  -- {title}")
    for e in examples:
        print(f"  {e['similarity']:>6}  {e['swahili']}")
        print(f"          {e['kidawida']}")


def as_prompt_block(examples: list[dict]) -> str:
    """Examples as the text a model reads. Scores and IDs are left out: they mean nothing to it."""
    return "\n\n".join(f"Swahili: {e['swahili']}\nKidaw'ida: {e['kidawida']}" for e in examples)


@lru_cache(maxsize=1)
def _train_rows() -> list[dict]:
    return [json.loads(l) for l in open(tools.TRAIN, encoding="utf-8")]   # keys: id, src, swa, dav


def word_overlap_retriever(sw: str, k: int = 5) -> list[dict]:
    """Naive baseline: number of shared whole words (lower-cased, punctuation stripped)."""
    def words(s):
        return {w.strip(".,;:!?\"'()").lower() for w in s.split()} - {""}
    q = words(sw)
    scored = sorted(_train_rows(), key=lambda r: -len(q & words(r["swa"])))[:k]
    return [{"pair_id": r["id"], "corpus": r["src"], "swahili": r["swa"], "kidawida": r["dav"],
             "similarity": len(q & words(r["swa"]))} for r in scored]


if __name__ == "__main__":
    for sw in sys.argv[1:] or SENTENCES:
        print("\n" + "=" * 78 + f"\n{sw}")
        ex = retrieve(sw)
        show(ex, "tool: char n-gram TF-IDF (similarity = cosine)")
        show(word_overlap_retriever(sw), "naive: shared whole words (similarity = count)")
        print("  -- what the model would read:")
        print("  " + as_prompt_block(ex[:2]).replace("\n", "\n  "))
    print("\nEvery tool call above was logged to ~/languagemodels/rulebook_data/retrieval_log.jsonl")

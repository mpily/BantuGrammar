"""
one_call.py -- the simplest possible model call: one request, one answer, no tools.

This is arm E (zero-shot). It is also the base the agent loop builds on: agent_loop.py sends
the same kind of request, plus `tools=`, and repeats it while the model asks for tools.

Run:  ~/languagemodels/bert-env/bin/python HaikuHarness/one_call.py "Hawa wazee ndio walipigania uhuru wetu"
Needs credentials: ANTHROPIC_API_KEY, or an `ant auth login` profile.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import anthropic

import prompts

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 16000
EFFORT = "medium"          # Haiku 5.5's default, set explicitly so it is logged


def text_of(response) -> str:
    """Join the text blocks of a reply (thinking blocks are skipped)."""
    return "\n".join(b.text for b in response.content if b.type == "text")


def usage_of(response) -> dict:
    u = response.usage
    return {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}


def translate_zero_shot(sw: str, client=None, model: str = MODEL) -> dict:
    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        output_config={"effort": EFFORT},
        system=prompts.SYSTEM_E,
        messages=[{"role": "user", "content": prompts.user_message(sw)}],
    )
    reply = text_of(response)
    return {"arm": "E", "sw": sw, "translation": prompts.extract_translation(reply), "reply": reply,
            "stop_reason": response.stop_reason, "n_requests": 1, "usage": usage_of(response),
            "tool_calls": []}


if __name__ == "__main__":
    out = translate_zero_shot(" ".join(sys.argv[1:]) or "Hawa wazee ndio walipigania uhuru wetu")
    print(out["reply"])
    print("\n->", out["translation"], "|", out["usage"])

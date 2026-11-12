"""
agent_loop.py -- arm C: the tool-use loop (the "agent").

    send sentence + tool schemas  ->  model replies
      stop_reason "tool_use":  run every requested tool (tool_bridge.run_tool),
                               send all results back in ONE user message, repeat
      stop_reason "end_turn":  done; read <translation> from the text

The model never runs code: it only names a tool and its JSON arguments. We run it and reply.
Written as a manual loop on purpose (instead of the SDK's tool runner) so every step is visible.

Run:  ~/languagemodels/bert-env/bin/python HaikuHarness/agent_loop.py "Hawa wazee ndio walipigania uhuru wetu"
      ... --dry-run   (no API call: a scripted fake model exercises the plumbing)
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prompts
from one_call import EFFORT, MAX_TOKENS, MODEL, text_of, usage_of
from tool_bridge import load_schemas, run_tool

MAX_REQUESTS = 30          # hard stop; a sentence normally needs 5-15


def _block_to_dict(b) -> dict:
    """SDK content blocks -> plain dicts for the transcript file."""
    if hasattr(b, "model_dump"):
        return b.model_dump(exclude_none=True)
    return dict(vars(b))


def translate_with_tools(sw: str, client=None, model: str = MODEL, verbose: bool = False) -> dict:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    tools = load_schemas()
    messages = [{"role": "user", "content": prompts.user_message(sw)}]
    tool_calls, usage, transcript = [], [], []
    stop = None
    for _ in range(MAX_REQUESTS):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            output_config={"effort": EFFORT},
            cache_control={"type": "ephemeral"},   # tools + system + history repeat every turn: cache them
            system=prompts.SYSTEM_C,
            tools=tools,
            messages=messages,
        )
        usage.append(usage_of(response))
        stop = response.stop_reason
        # Append the whole reply (text, thinking and tool_use blocks), never just its text.
        messages.append({"role": "assistant", "content": response.content})
        transcript.append({"role": "assistant", "content": [_block_to_dict(b) for b in response.content]})

        if stop == "pause_turn":
            continue
        if stop != "tool_use":          # end_turn, max_tokens, refusal: stop here
            break

        results = []
        for b in response.content:
            if b.type != "tool_use":
                continue
            text = run_tool(b.name, b.input)
            status = json.loads(text).get("status")
            tool_calls.append({"name": b.name, "input": b.input, "status": status})
            if verbose:
                print(f"  tool {b.name}({json.dumps(b.input, ensure_ascii=False)[:120]}) -> {status}")
            # Tools never raise; a crash inside one comes back as status "unknown" with an "error" key.
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": text,
                            "is_error": "error" in json.loads(text)})
        messages.append({"role": "user", "content": results})
        transcript.append({"role": "user", "content": results})
    else:
        stop = "max_requests"

    reply = text_of(response)
    tot = {k: sum(u[k] for u in usage) for k in usage[0]}
    return {"arm": "C", "sw": sw, "translation": prompts.extract_translation(reply), "reply": reply,
            "stop_reason": stop, "n_requests": len(usage), "usage": tot, "tool_calls": tool_calls,
            "transcript": transcript}


# ---------------------------------------------------------------- offline fake model (--dry-run)
class ScriptedClient:
    """Stands in for anthropic.Anthropic(): asks for retrieve_examples + parse_swahili_verb on the
    first turn, then answers with the top retrieved example. Tests the loop, not translation."""

    def __init__(self):
        self.messages = self

    def create(self, **kw):
        msgs = kw["messages"]
        u = SimpleNamespace(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        if len(msgs) == 1:
            sw = msgs[0]["content"].split("\n\n", 1)[1]
            calls = [SimpleNamespace(type="tool_use", id="t1", name="retrieve_examples", input={"sw_sentence": sw, "k": 3}),
                     SimpleNamespace(type="tool_use", id="t2", name="parse_swahili_verb", input={"word": sw.split()[-1]})]
            return SimpleNamespace(content=calls, stop_reason="tool_use", usage=u)
        first = json.loads(msgs[-1]["content"][0]["content"])
        top = first["examples"][0]["kidawida"] if first.get("examples") else "?"
        text = SimpleNamespace(type="text", text=f"Dry run.\n<translation>{top}</translation>")
        return SimpleNamespace(content=[text], stop_reason="end_turn", usage=u)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    client = ScriptedClient() if "--dry-run" in sys.argv else None
    out = translate_with_tools(" ".join(args) or "Hawa wazee ndio walipigania uhuru wetu", client=client, verbose=True)
    print(out["reply"])
    print("\n->", out["translation"])
    print(f"   {out['n_requests']} requests, {len(out['tool_calls'])} tool calls, stop={out['stop_reason']}, usage={out['usage']}")

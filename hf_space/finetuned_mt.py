"""
finetuned_mt.py

2026-11-02. System A of the user's two-system comparison: the fine-tuned
NLLB-600M (arm 2: dav_swa train + Bible NEN pairs), plus the rule-book flagger
run on its draft (rulebook_flagger.py). Independent of the rule-based pipeline:
nothing here changes kitaita_e2e_pipeline.py.

The model is loaded once, in a background thread at startup, from the private
repo in config.yaml (finetuned.model_repo) using the HF_TOKEN secret. If it
cannot be loaded, translate() reports why instead of failing the Space.
Output is machine translation, not reviewed by native speakers.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import yaml

_CFG = (yaml.safe_load((Path(__file__).parent / "config.yaml").read_text()) or {}).get("finetuned", {})
_state = {"model": None, "tok": None, "error": None, "loading": False}
_lock = threading.Lock()


def _load():
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        repo, token = _CFG["model_repo"], os.environ.get("HF_TOKEN") or None
        tok = AutoTokenizer.from_pretrained(repo, src_lang="swh_Latn", token=token)
        model = AutoModelForSeq2SeqLM.from_pretrained(repo, token=token).eval()
        from rulebook_flagger import resources
        resources()  # build the flagger's vocabulary/stem tables now, not on the first request
        _state.update(model=model, tok=tok)
        print(f"Fine-tuned model loaded: {repo}", flush=True)
    except Exception as e:  # keep the Space up; report the reason in the UI
        _state["error"] = f"{type(e).__name__}: {e}"
        print(f"Fine-tuned model NOT loaded: {_state['error']}", flush=True)
    finally:
        _state["loading"] = False


def start_loading():
    if _state["model"] is None and not _state["loading"] and _CFG.get("model_repo"):
        _state["loading"] = True
        threading.Thread(target=_load, daemon=True).start()


def status() -> str:
    if _state["model"] is not None:
        return "ready"
    if _state["loading"]:
        return "loading"
    return f"unavailable ({_state['error'] or 'not configured'})"


def translate(text: str) -> dict:
    """{'kitaita': A's top-1, 'flags': [(token, category)] on it,
        'version_b': rule-book correction of A (rulebook_correct.correct), 'status'}"""
    if _state["model"] is None:
        return {"kitaita": None, "flags": [], "version_b": None, "status": status()}
    import torch
    from rulebook_correct import correct
    from rulebook_flagger import flag
    tok, model = _state["tok"], _state["model"]
    with _lock, torch.no_grad():
        enc = tok([text], return_tensors="pt", truncation=True, max_length=256)
        gen = model.generate(**enc, forced_bos_token_id=tok.convert_tokens_to_ids("dav_Latn"),
                             num_beams=int(_CFG.get("num_beams", 4)),
                             num_return_sequences=int(_CFG.get("num_beams", 4)),
                             max_new_tokens=int(_CFG.get("max_new_tokens", 128)))
    nbest = tok.batch_decode(gen, skip_special_tokens=True)
    hyp = nbest[0]
    return {"kitaita": hyp, "flags": [(t, c) for t, c in flag(text, hyp) if c != "ok"],
            "version_b": correct(text, nbest), "status": "ready"}

# -*- coding: utf-8 -*-
"""
kitaita_local_webapp.py

2026-10-09. Direct user request: "the website does not run on the full
pipeline? I would like to test the full pipeline as well." The published
Kitaita Workbench Artifact (https://claude.ai/artifact/21wvpDBL5sRNusGHPoyibP)
is a claude.ai Artifact -- a client-side-only JS sandbox, structurally
unable to execute Python or load NLLB/the trained probes (confirmed
directly, repeatedly, this session) -- so it only ever ran the JS-ported
heuristic decomposer, never the real probe-based pipeline.

This is the real thing: a small local web server, running on THIS
machine, that loads NLLB + every trained probe ONCE at startup and
serves a page backed directly by translate_sentence_with_probes() --
the exact same function tested via Bash throughout this session, not a
reimplementation. Stdlib only (http.server), no new dependencies.

Usage:
    python3 kitaita_local_webapp.py [port]
    then open http://localhost:<port>/ in a browser on this machine.
"""
from __future__ import annotations

import html
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from kitaita_e2e_pipeline import translate_sentence_with_probes, translate_sentence
from kitaita_sentence_generator import load_pages, build_lookup_tables

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Kitaita Workbench (local, full pipeline)</title>
<style>
  body { font-family: -apple-system, system-ui, sans-serif; max-width: 760px; margin: 32px auto; padding: 0 16px;
         background: #f6f3ee; color: #241d15; }
  h1 { font-size: 1.4rem; }
  .note { color: #5c5244; font-size: 0.85rem; margin-bottom: 20px; }
  textarea { width: 100%; font-size: 1.1rem; padding: 10px; border-radius: 8px; border: 1px solid #ddd4c2;
             font-family: Georgia, serif; min-height: 50px; box-sizing: border-box; }
  button { margin-top: 10px; padding: 8px 18px; font-size: 0.95rem; border-radius: 6px; border: 1px solid #a8491f;
           background: #a8491f; color: white; cursor: pointer; }
  button:disabled { opacity: 0.5; }
  .examples button { background: none; color: #a8491f; border: 1px solid #ddd4c2; margin: 3px 4px 0 0;
                      font-size: 0.8rem; padding: 4px 10px; }
  #out { margin-top: 20px; }
  .sentence { font-family: Georgia, serif; font-size: 1.6rem; font-weight: 600; margin-bottom: 8px; }
  .error { color: #9a3b2e; font-family: monospace; font-size: 0.85rem; }
  .flags { margin-top: 10px; }
  .flags li { font-family: monospace; font-size: 0.78rem; background: #f1ece1; border-left: 3px solid #ddd4c2;
              padding: 6px 8px; margin-bottom: 6px; border-radius: 4px; list-style: none; }
  .trace li { font-family: monospace; font-size: 0.75rem; color: #5c5244; margin-bottom: 4px; list-style: none; }
  .toggle { font-size: 0.8rem; color: #5c5244; cursor: pointer; margin-top: 14px; display: inline-block; }
  .badge { display: inline-block; font-size: 0.7rem; background: #e4efe6; color: #3c6b4a; padding: 2px 8px;
           border-radius: 10px; margin-left: 8px; }
</style>
</head>
<body>
<h1>Kitaita Workbench <span class="badge">local, real NLLB + probes</span></h1>
<p class="note">This runs <code>translate_sentence_with_probes()</code> directly -- the real NLLB encoder + the
5 trained relation/POS probes, not the JS heuristic the published Artifact uses (Artifacts can't run Python/ML
at all). Probes were preloaded at server startup, so each translation below is just inference, not training.</p>
<textarea id="input">Mbwa mkubwa aliona Kijana mdogo akitembea.</textarea>
<div><button id="go">Translate (probes)</button> <button id="goHeuristic" style="background:#5c5244;border-color:#5c5244;">Translate (heuristic, for comparison)</button></div>
<div class="examples">
  Try:
  <button data-ex="Mbwa mkubwa aliona Kijana mdogo akitembea.">Mbwa mkubwa aliona Kijana mdogo akitembea.</button>
  <button data-ex="Mwalimu alisoma kitabu kisha akalala.">Mwalimu alisoma kitabu kisha akalala.</button>
  <button data-ex="Kitabu cha mwalimu kilikuwa kikubwa.">Kitabu cha mwalimu kilikuwa kikubwa.</button>
  <button data-ex="Askari alimwona mwizi.">Askari alimwona mwizi.</button>
</div>
<div id="out"></div>
<script>
const out = document.getElementById("out");
const input = document.getElementById("input");
document.querySelectorAll(".examples button").forEach(b => b.onclick = () => { input.value = b.dataset.ex; run("probes"); });
document.getElementById("go").onclick = () => run("probes");
document.getElementById("goHeuristic").onclick = () => run("heuristic");

async function run(mode) {
  out.innerHTML = "<p>Running (first call after server start may take a few seconds to load NLLB)...</p>";
  const res = await fetch("/api/translate", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({text: input.value, mode})
  });
  const data = await res.json();
  let h = "";
  if (data.kitaita_sentence) h += `<div class="sentence">${esc(data.kitaita_sentence)}</div>`;
  if (data.error) h += `<div class="error">${esc(data.error)}</div>`;
  if (data.flags && data.flags.length) {
    h += `<ul class="flags">${data.flags.map(f => `<li>${esc(f)}</li>`).join("")}</ul>`;
  }
  if (data.trace && data.trace.length) {
    h += `<details><summary class="toggle">Evidence trail (${data.trace.length})</summary><ul class="trace">${data.trace.map(t => `<li>${esc(t)}</li>`).join("")}</ul></details>`;
  }
  if (data.server_error) h += `<div class="error">SERVER ERROR: ${esc(data.server_error)}</div>`;
  out.innerHTML = h;
}
function esc(s) { const d = document.createElement("div"); d.textContent = s; return d.innerHTML; }
</script>
</body>
</html>
"""

_tables = None


def _get_tables():
    global _tables
    if _tables is None:
        p1, p2, p3 = load_pages()
        _tables = build_lookup_tables(p1, p2, p3)
    return _tables


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {fmt % args}")

    def do_GET(self):
        if self.path != "/":
            self.send_response(404)
            self.end_headers()
            return
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/translate":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        text, mode = payload.get("text", "").strip(), payload.get("mode", "probes")
        resp = {"kitaita_sentence": None, "error": None, "flags": [], "trace": []}
        try:
            tables = _get_tables()
            fn = translate_sentence_with_probes if mode == "probes" else translate_sentence
            r = fn(text, tables)
            resp["kitaita_sentence"] = r.get("kitaita_sentence")
            resp["error"] = r.get("error")
            resp["trace"] = r.get("trace", [])
            if "flags" in r:
                resp["flags"] = r["flags"]
            elif r.get("decomposition") is not None and hasattr(r["decomposition"], "flags"):
                resp["flags"] = r["decomposition"].flags
            else:
                resp["flags"] = []
        except Exception as e:
            resp["server_error"] = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        body = json.dumps(resp, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print("Loading vocabulary/grammar tables...", flush=True)
    _get_tables()
    print("Pre-warming NLLB + trained probes (this is the ~10-15s model-load cost, once)...", flush=True)
    from nllb_probe_identification import load_probes
    load_probes()
    print(f"Ready. Open http://localhost:{port}/ in a browser on this machine.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Smoke-test every Space API endpoint on the live Space after a deploy, and
# print latency p50/p95 to paste into docs/status.md.
#
#   scripts/smoke.sh [BASE_URL] [N]
#   BASE_URL defaults to https://mpily-bantugrammar.hf.space ; N = translate calls (default 10)
#   AT_WEBHOOK_TOKEN in the environment is appended to /at/* calls if set.
#
# Side effects: N translations are logged (channel "web"). Feedback and SMS
# are only exercised through validation-error paths, so no ratings are
# written and no SMS is sent.
set -uo pipefail
BASE="${1:-https://mpily-bantugrammar.hf.space}"
N="${2:-10}"
TOKEN_QS=""; [ -n "${AT_WEBHOOK_TOKEN:-}" ] && TOKEN_QS="?token=${AT_WEBHOOK_TOKEN}"
fail=0
check() { # name expected actual
  if [ "$2" = "$3" ]; then echo "ok   $1 ($3)"; else echo "FAIL $1: expected $2, got $3"; fail=1; fi
}
code() { curl -s -o /dev/null -w '%{http_code}' --max-time 120 "$@"; }

echo "== waking $BASE (cold start can take minutes)"
t0=$(date +%s)
for i in $(seq 1 60); do
  [ "$(code "$BASE/api/health")" = "200" ] && break
  sleep 10
done
echo "health reachable after $(( $(date +%s) - t0 ))s"
curl -s "$BASE/api/health"; echo

check "GET /api/health" 200 "$(code "$BASE/api/health")"
check "GET / (Gradio UI)" 200 "$(code "$BASE/")"
check "GET /api/progress" 200 "$(code "$BASE/api/progress")"
check "GET /api/verified_pairs" 200 "$(code "$BASE/api/verified_pairs?kind=word")"
check "GET /api/verified_pairs bad since" 422 "$(code "$BASE/api/verified_pairs?since=notadate")"
check "POST /api/translate invalid" 422 "$(code -X POST -H 'Content-Type: application/json' -d '{"text":""}' "$BASE/api/translate")"
check "POST /api/feedback unknown id" 404 "$(code -X POST -H 'Content-Type: application/json' \
  -d '{"translation_id":"tr_smoke_missing","stars":3,"speaker_id":"smoke"}' "$BASE/api/feedback")"
check "POST /api/feedback stars=9" 422 "$(code -X POST -H 'Content-Type: application/json' \
  -d '{"translation_id":"tr_x","stars":9,"speaker_id":"smoke"}' "$BASE/api/feedback")"
check "POST /at/ussd menu" 200 "$(code -X POST -d 'sessionId=smoke&serviceCode=*384#&phoneNumber=%2B254700000000&text=' "$BASE/at/ussd$TOKEN_QS")"
check "POST /at/sms missing from" 422 "$(code -X POST -d 'text=hi' "$BASE/at/sms$TOKEN_QS")"
check "CORS app origin" "https://mpily.github.io" "$(curl -s -o /dev/null -D - -X OPTIONS \
  -H 'Origin: https://mpily.github.io' -H 'Access-Control-Request-Method: POST' "$BASE/api/translate" \
  | tr -d '\r' | awk -F': ' 'tolower($1)=="access-control-allow-origin"{print $2}')"

SENTENCES=("Mbwa mkubwa aliona mwanafunzi." "Mwalimu alisoma kitabu kisha akalala." "Kitabu cha mwalimu kilikuwa kikubwa."
  "Askari alimwona mwizi." "Simba mkubwa alikimbia." "Mtoto mdogo alilala." "Mama alipika chakula."
  "Watoto walicheza nje." "Nyumba yetu ni kubwa." "Mbwa alikula nyama.")
lat_file=$(mktemp)
for i in $(seq 0 $((N - 1))); do
  s="${SENTENCES[$((i % ${#SENTENCES[@]}))]}"
  body=$(curl -s --max-time 120 -X POST -H 'Content-Type: application/json' \
    -d "{\"text\":\"$s\",\"src\":\"swh\",\"tgt\":\"dav\"}" "$BASE/api/translate")
  python3 - "$body" >>"$lat_file" <<'PY' || { echo "FAIL translate: $body"; fail=1; }
import json, sys
b = json.loads(sys.argv[1])
assert b["api_version"] == "1" and b["id"].startswith("tr_") and b["status"] in ("machine", "unknown"), b
print(b["latency_ms"], int(b["cached"]))
PY
done
python3 - "$lat_file" <<'PY'
import sys
rows = [tuple(map(int, l.split())) for l in open(sys.argv[1]) if l.strip()]
def pct(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None
for label, sel in (("uncached", 0), ("cached", 1)):
    xs = [l for l, c in rows if c == sel]
    print(f"translate latency_ms {label}: n={len(xs)} p50={pct(xs, 50)} p95={pct(xs, 95)}")
PY
rm -f "$lat_file"
[ $fail = 0 ] && echo "SMOKE OK" || { echo "SMOKE FAILED"; exit 1; }

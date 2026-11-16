#!/usr/bin/env bash
# End-to-end smoke test for the ensemble profile: router -> ensemble-api ->
# svm-api + embed-api. Checks the response carries a scenario and the
# stacker's probabilities, that svm-api's /issues answers, and that the
# label-shift state endpoint is reachable. Tears everything down after.

set -euo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] || cp .env.example .env

for f in ../docs/metrics/improved_model_2/model.joblib \
         ../docs/metrics/ensemble_stack/stack.joblib \
         ../data/cache/knn_embed/index.npz; do
    [[ -f "$f" ]] || { echo "!! missing artifact: $f"; exit 1; }
done

cleanup() { docker compose --profile ensemble down -v >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo ">> building + starting (profile ensemble, router -> ensemble-api)"
PRIMARY_URL=http://ensemble-api:8000 docker compose --profile ensemble up -d --build

echo ">> waiting for router /healthz"
for i in $(seq 1 60); do
    curl -sf http://localhost:8080/healthz >/dev/null && break
    sleep 3
    [[ $i -eq 60 ]] && { echo "!! not healthy in time"; docker compose --profile ensemble logs; exit 1; }
done

echo ">> POST /predict"
BODY=$(curl -sf -X POST http://localhost:8080/predict \
  -H 'content-type: application/json' \
  -d '{"text":"The debt collector keeps calling my workplace about a debt that is not mine and refuses to send written validation."}')
echo "   response: $BODY"
echo "$BODY" | python -c '
import json, sys
d = json.loads(sys.stdin.read())
assert d["model_name"] == "ensemble_stack_v2", d["model_name"]
assert d["scenario"] in ("auto", "assisted", "contested", "degraded"), d
assert d["scenario"] != "degraded", "embed-api did not contribute: " + str(d)
assert d["stack_team_probs"] and abs(sum(d["stack_team_probs"].values()) - 1) < 1e-4
assert isinstance(d["top_issues"], list)
print("   schema: OK; team", d["team"], "scenario", d["scenario"])
'

echo ">> ensemble-api /prior and svm-api /issues (inside the compose network)"
docker compose --profile ensemble exec -T ensemble-api python -c "
import urllib.request, json
print('   prior:', json.load(urllib.request.urlopen('http://localhost:8000/prior', timeout=5))['active'])
req = urllib.request.Request('http://svm-api:8000/issues', method='POST',
    data=json.dumps({'text': 'My mortgage servicer misapplied my escrow payment and started foreclosure.', 'team': 'T06'}).encode(),
    headers={'content-type': 'application/json'})
out = json.load(urllib.request.urlopen(req, timeout=10))
assert all(r['issue'].startswith('T06_') for r in out['top_issues']), out
print('   issues for T06:', [r['issue'] for r in out['top_issues']])
"
echo ">> ensemble smoke test passed"

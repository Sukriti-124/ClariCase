#!/usr/bin/env bash
# End-to-end smoke test for the Phase 1 slice.
# Brings the compose stack up, hits /predict, validates the response shape,
# then tears down. Fails fast on any step.

set -euo pipefail

cd "$(dirname "$0")/.."

[[ -f .env ]] || cp .env.example .env

cleanup() {
    docker compose down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo ">> building"
docker compose build

echo ">> starting"
docker compose up -d

echo ">> waiting for router /healthz"
for i in $(seq 1 30); do
    if curl -sf http://localhost:8080/healthz >/dev/null; then
        break
    fi
    sleep 2
    if [[ $i -eq 30 ]]; then
        echo "!! router did not become healthy in time"
        docker compose logs
        exit 1
    fi
done

echo ">> POST /predict"
BODY=$(curl -sf -X POST http://localhost:8080/predict \
  -H 'content-type: application/json' \
  -d '{"text":"The bank reported wrong information to the credit bureau and has not corrected it after multiple disputes."}')

echo "   response: $BODY"

echo "$BODY" | python -c '
import json, sys
d = json.loads(sys.stdin.read())
assert d["model_name"] == "improved_model_1", d
assert d["model_version"].startswith("sha256:"), d
assert isinstance(d["team"], str) and d["team"]
assert 0.0 <= d["team_confidence"] <= 1.0
assert isinstance(d["top_issues"], list)
assert isinstance(d["auto_route"], bool)
assert d["request_id"]
assert isinstance(d["latency_ms"], int) and d["latency_ms"] >= 0
print("   schema: OK")
'

echo ">> smoke test passed"

# Product-first issue classification

This backend takes the product/team predicted by an existing model and asks an
LLM to identify the issue from that product's complete issue catalog.

```
Complaint → existing product model → predicted team
          → LLM receives complaint + team's issue names → validated issue decision
```

It reads the repository's original `data/processed/team_mapping.csv` and
`issue_mapping.csv`: 11 teams and 73 product-specific issues. It does not apply
the two historical label merges used by some ML pipelines. Every allowed issue
is supplied with its readable name and product-specific label, not just its ID.
The prompt includes guidance for commonly confused issues. It does not use
training examples, change dataset labels, or require a GPU, encoder, or model
artifact. This is separate from `agentic-api`'s older top-candidate `/choose`
service; existing prediction and explanation endpoints are unchanged.

## Run locally

From the repository root, with Python 3.11 or newer:

```bash
python -m venv .venv-issue
.venv-issue/bin/pip install ./platform/shared './platform/services/issue-api[test]'
ISSUE_BACKEND=mock .venv-issue/bin/uvicorn issue_api.main:app --port 8081
```

Visit [the interactive API docs](http://localhost:8081/docs). Mock mode only
checks wiring: it always returns `insufficient_context`, clearly identified as
mock. It is not an accuracy demonstration.

For real inference, set these variables in your shell or a private environment
file. An OpenAI-compatible chat-completions endpoint is required:

```dotenv
ISSUE_BACKEND=openai_compat
ISSUE_BASE_URL=https://your-provider.example/v1
ISSUE_API_KEY=your-private-key
ISSUE_MODEL=your-provider-model-id
ISSUE_RESPONSE_FORMAT=json_object
ISSUE_PRODUCT_URL=http://localhost:8080
```

Start using `.venv-issue/bin/uvicorn issue_api.main:app --port 8081 --env-file /path/to/private.env`.
Choose an actual model ID offered by your provider. `json_object` requests JSON;
`json_schema` additionally sends a product-specific schema with allowed IDs for
providers supporting strict structured output. `none` omits `response_format`
for other compatible servers. All modes use the same application validation.
Unsupported provider options fail visibly; they do not trigger a silent fallback.
Real requests send the complaint to the configured provider and may incur usage
charges. No live-provider accuracy claim is made by the automated tests.

## API

**Already have a predicted team?** Call `POST /classify-issue`:

```bash
curl http://localhost:8081/classify-issue \
  -H 'Content-Type: application/json' \
  -d '{"text":"I paid my credit card bill on time, but the bank charged me a late fee.","team":"T03","request_id":"demo-1"}'
```

Use the canonical team ID from the model (`T01`–`T11`). `GET /catalog` returns
team IDs, product names, and issues; `GET /catalog?team=T03` filters one product.
An illustrative real-provider decision, **not a measured model result**, is:

```json
{
  "status": "classified",
  "team": "T03",
  "team_name": "Credit Card Team",
  "product": "Credit card",
  "issue_id": "T03_I004",
  "issue_label": "Credit card :: Fees or interest",
  "candidate_issue_ids": [],
  "evidence": ["the bank charged me a late fee"],
  "reason": "The dispute concerns a fee.",
  "review_required": true,
  "model": "your-provider-model-id",
  "backend": "openai_compat",
  "catalog_version": "content-hash",
  "prompt_version": "product-issue-v1",
  "request_id": "demo-1",
  "latency_ms": 1234
}
```

**Want both stages in one request?** Set `ISSUE_PRODUCT_URL` to the existing
model/router service's base URL and call `POST /predict` with `text` and optional
`request_id`. The service calls that URL's `/predict`, validates the response,
then supplies its predicted team to the LLM. The response contains:

- `team_prediction`: the existing model's prediction and probabilities.
- `issue_result`: the validated issue decision above.
- `review_required`: currently always `true` for the complete workflow.

The product service must use the shared `PredictResponse` contract. Request IDs
are propagated to both stages. Set the older ensemble `AGENTIC_URL` to empty if
you want the product stage to remain purely ML. Do not point `ISSUE_PRODUCT_URL`
back to this issue service.

## Decisions and failure handling

| Status | Meaning | Issue assigned? |
|---|---|---|
| `classified` | One allowed issue selected with a supporting quote | Yes |
| `ambiguous` | Two or more allowed issues remain plausible | No; alternatives returned |
| `insufficient_context` | Too little detail, or none of the listed issues fits | No |
| `product_mismatch` | Narrative contradicts the predicted product | No; team is not silently changed |

Historical near-synonyms remain in the catalog. A narrative that cannot
distinguish them should produce ambiguity, not an invented distinction. The
model sees the product as a prediction rather than unquestionable truth.

The application rejects malformed JSON, unknown/cross-product IDs, inconsistent
status fields, missing evidence, nonverbatim quotes, and truncated generations.
It never assigns the first issue on a parsing error. A matching quote verifies
provenance, not the correctness of the model's reasoning. Every result therefore
requires review until performance and an automation policy have been evaluated;
no LLM-generated confidence score is treated as a calibrated probability. In a
combined response, do not use `team_prediction.auto_route` as permission to
automatically accept the issue.

HTTP failures: `422` invalid input/team; `502` invalid/unavailable provider;
`503` busy or unconfigured product service; `504` timeout. The client should
retain its known product prediction and show the issue as unavailable, not
silently use an ML issue as though it came from this LLM.

Complaints are limited to 20,000 characters. Longer input is rejected rather
than cropped. This is not a token limit: configure a provider/model whose context
can accommodate the full request and catalog. The issue call has a total
`ISSUE_TIMEOUT_S` budget (default 25 seconds), including cancellation, and a
per-process concurrency cap of four. No automatic retries multiply paid calls.
The product call has its own `ISSUE_PRODUCT_TIMEOUT_S` budget (default 10 seconds).
Clients calling both stages should allow more than their combined budgets.

No complaint text, evidence, provider response, or key is intentionally logged or
persisted by this service. There is no response cache or database. Provider
retention policies are separate. `/metrics` exposes outcome counts and latency;
`/healthz` reports configuration/catalog readiness, not a live provider probe.
The container binds to localhost in Compose. Put authentication and request rate
limits at your gateway before exposing it publicly.

## Docker Compose

In `platform/.env`, configure `ISSUE_*` variables (see `.env.example`). Then:

```bash
cd platform
# Standalone: supply the predicted team to /classify-issue.
docker compose --profile issues up --build -d issue-api
# Combined: also start the existing product ensemble and its dependencies.
docker compose --profile ensemble --profile issues up --build -d ensemble-api issue-api
```

The issue API listens on `127.0.0.1:8081`. Plain `docker compose up` does not
start it. The image includes only service dependencies and the two mapping CSVs,
not complaint data or trained models.

## Validation and evaluation

```bash
.venv-issue/bin/python -m pytest platform/services/issue-api/tests -q
```

Tests use a simulated HTTP provider and product model to exercise the real
request, prompt, parsing, validation, and error-handling paths without paid calls.
They are functional tests, not evidence of issue accuracy.

Before choosing a production model, evaluate a fixed validation set twice:
first with reference products, then with predicted products. Report issue
accuracy and macro-F1, classified coverage, ambiguous/mismatch rates, invalid
output rate, latency and cost. Include abstentions in end-to-end accounting;
report accuracy among classified cases separately. Keep the test split untouched
while tuning prompts and model choice. This distinguishes issue-selection errors
from errors inherited from the product model.

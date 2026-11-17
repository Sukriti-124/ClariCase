"""HTTP wire contract for ClariCase model-serving services.

This module is imported by the router and every model API service
(svm-api, embed-api, ensemble-api, and future seq2seq-api / agentic-api).
Changes here affect the public API of the platform — treat as a stable
contract. Fields added after Phase 1 are optional, so a Phase 1 client
reading a Phase 2 response, or the reverse, still validates.

Routing scenarios (`scenario`), set by ensemble-api:
  auto       gate score cleared the threshold for the chosen team
  assisted   models agree on the team but the score is below the threshold
  contested  the models disagree on the team
  degraded   a model service was unavailable; the primary routed alone
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Scenario = Literal["auto", "assisted", "contested", "degraded"]

# Explanation state (Req 17.2):
#   pending      generation queued or running
#   ready        an Explanation is stored
#   failed       generation ended without a stored Explanation
#   unavailable  explanation generation is disabled for this request
ExplanationStatus = Literal["pending", "ready", "failed", "unavailable"]


class IssueProb(BaseModel):
    issue: str
    prob: float = Field(ge=0.0, le=1.0)


class PredictRequest(BaseModel):
    text: str = Field(min_length=10)
    request_id: str | None = None


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    team: str
    team_confidence: float = Field(ge=0.0, le=1.0)
    top_issues: list[IssueProb]
    auto_route: bool
    request_id: str
    latency_ms: int = Field(ge=0)

    # ---- Added for the ensemble (optional; absent in Phase 1 responses) ----
    # This service's own probability for every team, keyed by team id.
    team_probs: dict[str, float] | None = None
    # Probabilities of further models hosted in the same service, keyed by
    # model name (e.g. svm-api also serves baseline_model_2).
    extra_team_probs: dict[str, dict[str, float]] | None = None
    scenario: Scenario | None = None
    # ensemble-api: the stacker's probability for every team, after the
    # label-shift correction when it is active.
    stack_team_probs: dict[str, float] | None = None
    # ensemble-api, when agentic-api is on: what the LLM was asked and chose.
    # Keys: team_candidates, team_choice, rescued, issue_candidates,
    # issue_choice. Absent when the LLM was not consulted.
    agentic: dict | None = None
    # ensemble-api: state of the explanation for this request, fetched later
    # from GET /explain/{request_id}. Absent when the service does not
    # produce explanations (Req 17.2).
    explanation_status: ExplanationStatus | None = None

    @field_validator("team_probs", "stack_team_probs")
    @classmethod
    def _probs_in_range(cls, v):
        if v is not None:
            for k, p in v.items():
                if not 0.0 <= p <= 1.0:
                    raise ValueError(f"probability for {k} out of range: {p}")
        return v


class IssuesRequest(BaseModel):
    """Issue ranking for a team chosen elsewhere (ensemble-api asks svm-api
    for the stacker's team when it differs from svm-api's own)."""
    text: str = Field(min_length=10)
    team: str
    top_k: int = Field(default=3, ge=1, le=10)
    request_id: str | None = None


class IssuesResponse(BaseModel):
    model_name: str
    model_version: str
    team: str
    top_issues: list[IssueProb]
    request_id: str
    latency_ms: int = Field(ge=0)


class ChooseRequest(BaseModel):
    """agentic-api: pick one of the ML's candidates for a complaint."""
    text: str = Field(min_length=10)
    task: Literal["team", "issue"]
    candidates: list[str] = Field(min_length=1, max_length=5)
    # Required for task == "issue": the team the candidates belong to.
    team: str | None = None
    request_id: str | None = None


class ChooseResponse(BaseModel):
    model_name: str
    model_version: str
    task: Literal["team", "issue"]
    choice: str
    parsed: bool
    cached: bool
    request_id: str
    latency_ms: int = Field(ge=0)


class ExplainSectionsRequest(BaseModel):
    """Internal: ensemble-api asks svm-api / embed-api for their explanation
    sections for a complaint it has already routed."""
    text: str = Field(min_length=1)
    masked_text: str
    chosen_team: str
    served_issues: list[str] = []
    request_id: str


class ExplainSectionsResponse(BaseModel):
    """Section JSON keyed by model name. Kept as plain dicts so the Platform
    images do not depend on src.explain; ensemble-api validates them with
    src.explain.record."""
    sections: dict[str, dict]
    model_versions: dict[str, str]


class ExplainResponse(BaseModel):
    """GET /explain/{request_id} (Req 17.6): only request_id and
    explanation_status are required."""
    request_id: str
    explanation_status: ExplanationStatus
    consumer_summary: str | None = None
    agent_summary: str | None = None
    cited_facts: list[dict] | None = None
    source: Literal["llm", "template"] | None = None
    validation: dict | None = None
    reason: str | None = None
    # Agent role only: routing outcome and sections, and the audit trail.
    decision_record: dict | None = None
    audit: list[dict] | None = None

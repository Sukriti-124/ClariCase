import pytest
from pydantic import ValidationError

from claricase_platform_shared.contracts import (
    IssueProb,
    PredictRequest,
    PredictResponse,
)


def test_predict_request_accepts_valid_text():
    req = PredictRequest(text="I have a problem with my credit card bill today.")
    assert req.text.startswith("I have")
    assert req.request_id is None


def test_predict_request_rejects_short_text():
    with pytest.raises(ValidationError):
        PredictRequest(text="short")


def test_predict_request_round_trips_request_id():
    req = PredictRequest(text="A" * 20, request_id="abc-123")
    assert req.request_id == "abc-123"


def test_predict_response_requires_all_fields():
    resp = PredictResponse(
        model_name="improved_model_1",
        model_version="sha256:deadbeef",
        team="T01",
        team_confidence=0.82,
        top_issues=[IssueProb(issue="I001", prob=0.55)],
        auto_route=True,
        request_id="abc-123",
        latency_ms=42,
    )
    assert resp.auto_route is True
    assert resp.top_issues[0].prob == 0.55


def test_predict_response_rejects_bad_confidence():
    with pytest.raises(ValidationError):
        PredictResponse(
            model_name="m",
            model_version="v",
            team="T01",
            team_confidence=1.5,  # out of range
            top_issues=[],
            auto_route=False,
            request_id="r",
            latency_ms=1,
        )


def test_phase1_response_still_validates_without_new_fields():
    resp = PredictResponse(
        model_name="m", model_version="v", team="T01", team_confidence=0.5,
        top_issues=[], auto_route=False, request_id="r", latency_ms=1)
    assert resp.team_probs is None and resp.scenario is None
    assert "team_probs" in resp.model_dump()


def test_new_optional_fields_round_trip_and_validate():
    from claricase_platform_shared.contracts import IssuesRequest, IssuesResponse
    resp = PredictResponse(
        model_name="m", model_version="v", team="T01", team_confidence=0.5,
        top_issues=[], auto_route=False, request_id="r", latency_ms=1,
        team_probs={"T01": 0.7, "T02": 0.3},
        extra_team_probs={"baseline_model_2": {"T01": 0.6, "T02": 0.4}},
        scenario="contested")
    assert PredictResponse(**resp.model_dump()).scenario == "contested"
    with pytest.raises(ValidationError):
        PredictResponse(**{**resp.model_dump(), "scenario": "maybe"})
    with pytest.raises(ValidationError):
        PredictResponse(**{**resp.model_dump(), "team_probs": {"T01": 1.2}})
    req = IssuesRequest(text="A" * 20, team="T02")
    assert req.top_k == 3
    out = IssuesResponse(model_name="m", model_version="v", team="T02",
                         top_issues=[IssueProb(issue="T02_I001", prob=0.4)],
                         request_id="r", latency_ms=2)
    assert out.top_issues[0].issue == "T02_I001"

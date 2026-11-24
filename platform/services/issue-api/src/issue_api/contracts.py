from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from claricase_platform_shared.contracts import PredictResponse


class ComplaintRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    # A character cap, not a token estimate. Reject rather than silently crop.
    text: str = Field(min_length=10, max_length=20000)
    request_id: str | None = Field(default=None, max_length=128, pattern=r"^[\w.-]+$")

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if len(value.strip()) < 10:
            raise ValueError("complaint must contain at least 10 non-padding characters")
        return value


class ClassifyRequest(ComplaintRequest):
    team: str = Field(pattern=r"^T\d{2}$")


class Decision(BaseModel):
    """Untrusted provider output: shape validation is followed by catalog checks."""
    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["classified", "ambiguous", "insufficient_context", "product_mismatch"]
    issue_id: str | None
    candidate_issue_ids: list[str] = Field(max_length=5)
    evidence: list[str] = Field(max_length=3)
    reason: str = Field(min_length=1, max_length=600)

    @field_validator("evidence")
    @classmethod
    def bounded_quotes(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 500 for value in values):
            raise ValueError("evidence must contain short, nonempty quotes")
        return values

    @model_validator(mode="after")
    def consistent(self):
        if len(set(self.candidate_issue_ids)) != len(self.candidate_issue_ids):
            raise ValueError("duplicate candidates")
        if self.status == "classified":
            if not self.issue_id or self.candidate_issue_ids or not self.evidence:
                raise ValueError("classified requires one issue and evidence, no alternatives")
        else:
            if self.issue_id is not None:
                raise ValueError("unresolved decisions cannot assign an issue")
            if self.status == "ambiguous":
                if len(self.candidate_issue_ids) < 2 or not self.evidence:
                    raise ValueError("ambiguous requires alternatives and evidence")
            elif self.candidate_issue_ids:
                raise ValueError("candidates are only permitted for ambiguous decisions")
            if self.status == "product_mismatch" and not self.evidence:
                raise ValueError("product mismatch requires evidence")
        return self


class IssueResult(Decision):
    team: str
    team_name: str
    product: str
    issue_label: str | None
    review_required: bool
    model: str
    backend: str
    catalog_version: str
    prompt_version: str
    request_id: str
    latency_ms: int


class ProductIssueResponse(BaseModel):
    team_prediction: PredictResponse
    issue_result: IssueResult
    # The LLM is not calibrated for autonomous routing yet.
    review_required: bool = True

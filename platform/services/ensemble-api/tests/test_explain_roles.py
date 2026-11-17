"""Property test for role gating on GET /explain/{request_id}."""
import hmac
from dataclasses import replace
from itertools import count

from hypothesis import HealthCheck, given, settings, strategies as st

from src.explain.narration import Explanation, ValidationSummary
from src.explain.store import StoreEntry, _Slot

_ids = count()
_tok = st.text(alphabet=st.characters(min_codepoint=33, max_codepoint=126),
               max_size=12)
_AGENT_KEYS = {"agent_summary", "cited_facts", "decision_record", "audit"}
_BASE = {"request_id", "explanation_status", "consumer_summary"}


# Feature: model-explainability, Property 36: Role gating on the Explain_Endpoint
@settings(max_examples=100, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(token=st.one_of(st.none(), st.just(""), _tok),
       header=st.one_of(st.none(), st.just(""), _tok),
       same=st.booleans())
def test_role_gating(client, token, header, same):
    """**Validates: Requirements 15.6, 15.7**"""
    cfg_token = "" if token is None else token
    if same and header is not None and token:
        header = token
    client.app.state.explain_cfg = replace(client.app.state.explain_cfg,
                                           agent_token=cfg_token)
    ex = Explanation(consumer_summary="c", agent_summary="a", source="template",
                     validation=ValidationSummary(accepted=True),
                     record_hash="h", created_at="t")
    rid = f"p36-{next(_ids)}"
    store = client.app.state.explain_store
    store._data[rid] = _Slot(StoreEntry(status="ready", explanation=ex,
                                        inserted_at=store._clock()))
    headers = {} if header is None else {"X-ClariCase-Agent-Token": header}
    r = client.get(f"/explain/{rid}", headers=headers)
    assert r.status_code == 200
    body = r.json()
    agent = bool(cfg_token) and header is not None and hmac.compare_digest(
        header.encode(), cfg_token.encode())
    if agent:
        assert _AGENT_KEYS <= set(body)
    else:
        assert set(body) <= _BASE
        assert not (_AGENT_KEYS & set(body))

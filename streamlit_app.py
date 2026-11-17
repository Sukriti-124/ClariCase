"""ClariCase complaint intake.

Run: streamlit run streamlit_app.py
"""

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.app.pii import mask
from src.app.storage import (LocalStore, SupabaseStore, build_record,
                             normalise_tracking_id)
from src.improved_model_1.predict import load_model, predict

MIN_WORDS = 5

# How many teams to show, by the top team's confidence:
# above 80% -> 1, 50-80% -> top 2, below 50% -> top 3. Issues always show
# the top ISSUES_SHOWN.
SINGLE_ABOVE = 0.80
PAIR_FROM = 0.50
ISSUES_SHOWN = 3

# Prediction takes well under a second once the model is loaded; each step
# stays on screen this long so the progress is readable.
STEP_PAUSE_SEC = 2
# How long the fully ticked checklist stays before the result appears.
DONE_PAUSE_SEC = 1

st.set_page_config(page_title="ClariCase", page_icon=":material/forum:",
                   layout="centered")

st.markdown("""
<style>
  .block-container { padding-top: 2.5rem; max-width: 760px; }
  .cc-brand { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em;
              color: #4A4785; margin-bottom: 0.4rem; display: inline-block;
              border-bottom: 3px solid #A9A4CF; padding-bottom: 0.1rem; }
  .cc-tagline { font-size: 1.05rem; color: #475569; margin-bottom: 1.4rem; }
  .cc-steps { display: flex; gap: 0.6rem; flex-wrap: wrap;
              margin-bottom: 1.6rem; }
  .cc-step { flex: 1 1 0; min-width: 150px; background: #FFFFFF;
             border: 1px solid #E5E5EC; border-radius: 12px;
             padding: 0.75rem 0.9rem; }
  .cc-step-num { display: inline-block; width: 1.5rem; height: 1.5rem;
                 border-radius: 50%; background: #E7E5F2; color: #4A4785;
                 font-size: 0.8rem; font-weight: 700; text-align: center;
                 line-height: 1.5rem; margin-right: 0.4rem; }
  .cc-step-title { font-weight: 600; color: #26253D; }
  .cc-step-text { font-size: 0.85rem; color: #64748B; margin-top: 0.25rem; }
  .cc-badge { display: inline-block; padding: 0.2rem 0.7rem;
              border-radius: 999px; font-size: 0.8rem; font-weight: 700;
              letter-spacing: 0.02em; }
  .cc-routed { background: #DCFCE7; color: #166534; }
  .cc-review { background: #FEF3C7; color: #92400E; }
  .cc-received { background: #E2E8F0; color: #334155; }
  .cc-label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.06em; color: #64748B; margin: 0.9rem 0 0.2rem; }
  .cc-team { font-size: 1.45rem; font-weight: 700; color: #26253D; }
  .cc-issue { font-size: 1.05rem; font-weight: 600; color: #26253D; }
  .cc-note { font-size: 0.95rem; color: #334155; margin-top: 0.6rem; }
  .cc-footer { text-align: center; font-size: 0.8rem; color: #94A3B8;
               margin-top: 2.5rem; }
  /* Hide Streamlit's generic "Running..." indicator; submits show steps. */
  [data-testid="stStatusWidget"] { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_store():
    try:
        cfg = st.secrets["supabase"]
        return SupabaseStore(cfg["url"], cfg["key"])
    except (KeyError, FileNotFoundError):
        return LocalStore()


store = get_store()

log = logging.getLogger(__name__)
EXPLAIN_CACHE_DB = (Path(__file__).resolve().parent / "data" / "cache"
                    / "explain" / "llm_cache.sqlite")
AGENT_DRIVERS_SHOWN = 10


def _secrets():
    try:
        return st.secrets
    except Exception:
        return None


@st.cache_resource
def get_explainer():
    from src.agentic.llm import PromptCache
    from src.explain.cache import ExplanationCache
    from src.explain.config import load_config
    from src.explain.narrator import Narrator
    cfg = load_config(os.environ, _secrets())
    return Narrator(cfg, prompt_cache=PromptCache(EXPLAIN_CACHE_DB),
                    explanation_cache=ExplanationCache())


def agent_view_enabled() -> bool:
    raw = os.environ.get("CLARICASE_AGENT_VIEW")
    if not raw:
        try:
            raw = st.secrets.get("CLARICASE_AGENT_VIEW")
        except Exception:
            raw = None
    return str(raw or "false").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- helpers
def candidates_to_show(candidates: list[dict]) -> list[dict]:
    top = candidates[0]["confidence"]
    n = 1 if top > SINGLE_ABOVE else 2 if top >= PAIR_FROM else 3
    return candidates[:n]


def issue_name(label: str) -> str:
    # "Debt collection :: Communication tactics" -> "Communication tactics"
    return label.split(" :: ", 1)[-1]


def badge(status: str) -> str:
    css = {"Routed": "cc-routed", "Under review": "cc-review"}.get(
        status, "cc-received")
    return f'<span class="cc-badge {css}">{status}</span>'


def label(text: str) -> None:
    st.markdown(f'<div class="cc-label">{text}</div>', unsafe_allow_html=True)


def show_candidates(items: list[tuple[str, float]], css: str) -> None:
    # One candidate: shown as a heading. Several: confidence bars.
    if len(items) == 1:
        st.markdown(f'<div class="{css}">{items[0][0]}</div>',
                    unsafe_allow_html=True)
        return
    for name, conf in items:
        st.progress(min(max(conf, 0.0), 1.0), text=f"{name} — {conf:.0%}")


# ---------------------------------------------------------------- callbacks
# Both run before the rerun, so the input boxes can be cleared on every press.
def handle_submit() -> None:
    # Only captures the text; the work runs in process_complaint() so the
    # page can show progress steps while it happens.
    text = st.session_state["complaint_text"].strip()
    st.session_state["complaint_text"] = ""
    if len(text.split()) < MIN_WORDS:
        st.session_state["submit_result"] = {"kind": "too_short"}
        return
    st.session_state["submit_result"] = None
    # Redact before anything downstream sees it: the raw text is never
    # predicted on, stored or displayed. Length is checked first so a
    # complaint is not rejected for the words masking removes.
    st.session_state["pending_text"] = mask(text)


STEPS = ("Analyzing complaint", "Identifying issue", "Assigning team")


def render_steps(placeholder, current: int) -> None:
    # Steps before `current` are done, `current` is in progress, rest pending.
    rows = []
    for i, name in enumerate(STEPS):
        if i < current:
            rows.append(f":material/check_circle: {name}")
        elif i == current:
            rows.append(f":material/progress_activity: **{name}…**")
        else:
            rows.append(f":gray[:material/radio_button_unchecked: {name}]")
    placeholder.container(border=True).markdown("  \n".join(rows))


def process_complaint(text: str) -> dict:
    placeholder = st.empty()
    render_steps(placeholder, 0)
    prediction = predict(text, model=get_model())
    time.sleep(STEP_PAUSE_SEC)

    render_steps(placeholder, 1)
    time.sleep(STEP_PAUSE_SEC)

    render_steps(placeholder, 2)
    record = build_record(text, prediction)
    try:
        store.add(record)
    except Exception:
        placeholder.empty()
        return {"kind": "error"}
    time.sleep(STEP_PAUSE_SEC)

    render_steps(placeholder, len(STEPS))
    time.sleep(DONE_PAUSE_SEC)
    placeholder.empty()
    return {"kind": "sent", "record": record, "prediction": prediction}


def _explain(record: dict, prediction: dict) -> dict:
    # Returns {"explanation", "record", "audit"}; explanation is None when
    # nothing could be produced. Never raises.
    from src.explain.labels import display_label
    from src.explain.pipeline import explain_streamlit
    bundle = get_model()
    text = record["complaint_text"]          # already masked
    issues = prediction["issue_top_k"][:3]
    outcome = {
        "chosen_team": {"id": prediction["predicted_team_id"],
                        "name": prediction["predicted_team_name"]},
        "top_issues": [{"rank": n, "issue_id": i["issue_id"],
                        "label": i["issue_label"],
                        "display_label": display_label(i["issue_label"]),
                        "prob": i["confidence"]}
                       for n, i in enumerate(issues, 1)],
        "issue_order": "probability",
        "auto_route": prediction["route"] == "auto",
        "route": prediction["route"],
        "scenario": "primary_only",
    }
    try:
        result = explain_streamlit(
            request_id=str(uuid.uuid4()), raw_text=text, masked_text=text,
            vectorizer=bundle["vectorizer"], hsvm=bundle["model"],
            team_classes=bundle["team_classes"],
            served_issues=[i["issue_id"] for i in issues],
            confidence=prediction["confidence"],
            global_threshold=bundle.get("global_threshold"),
            outcome=outcome,
            model_versions={"improved_model_1": prediction["model_name"]},
            narrator=get_explainer())
    except Exception:
        log.exception("explanation failed")
        return {"explanation": None, "record": None, "audit": []}
    if result.error or result.narration is None:
        log.warning("explanation unavailable: %s", result.error)
        return {"explanation": None, "record": result.record, "audit": []}
    return {"explanation": result.narration.explanation,
            "record": result.record, "audit": result.narration.audit}


def render_why_panel(record: dict, prediction: dict) -> None:
    key = record["tracking_id"]
    cache = st.session_state.setdefault("explanations", {})
    with st.container(border=True):
        label("Why this team?")
        st.caption("This explanation covers the main routing model only "
                   "(Improved Model 1).")
        st.markdown(f"**Team:** {prediction['predicted_team_name']}")
        issues = prediction["issue_top_k"][:ISSUES_SHOWN]
        st.markdown("**Issues:** " + "; ".join(
            f"{n}. {issue_name(i['issue_label'])}"
            for n, i in enumerate(issues, 1)))

        if key not in cache:
            with st.spinner("Explaining this decision…"):
                cache[key] = _explain(record, prediction)
            entry = cache[key]
            if entry["explanation"] is not None and entry["record"] is not None:
                try:
                    store.update(key, {
                        "decision_record": entry["record"].model_dump(mode="json"),
                        "explanation": entry["explanation"].model_dump(mode="json"),
                        "explanation_audit": [a.model_dump(mode="json")
                                              for a in entry["audit"]]})
                except Exception:
                    log.exception("could not store explanation")
                    entry["not_saved"] = True
        entry = cache[key]
        expl = entry["explanation"]
        if expl is None:
            st.info("Explanation unavailable.", icon=":material/info:")
            return
        st.write(expl.consumer_summary)
        st.caption("Written by AI" if expl.source == "llm"
                   else "Standard explanation")
        if entry.get("not_saved"):
            st.caption("Explanation not saved.")

        if agent_view_enabled():
            st.markdown("**Agent summary**")
            st.write(expl.agent_summary)
            sec = entry["record"].sections["improved_model_1"]
            drivers = sec.drivers[:AGENT_DRIVERS_SHOWN]
            if drivers:
                st.markdown("**Top team drivers (Improved Model 1)**")
                for d in drivers:
                    name = d.text or d.label or d.kind
                    st.markdown(f"- {name}: {d.contribution:+.3f}")


def reset_on_tab_change() -> None:
    for key in ("submit_result", "track_result", "pending_text"):
        st.session_state.pop(key, None)
    st.session_state["complaint_text"] = ""
    st.session_state["tracking_input"] = ""


def handle_track() -> None:
    raw = st.session_state["tracking_input"].strip()
    st.session_state["tracking_input"] = ""
    if not raw:
        st.session_state["track_result"] = None
        return
    tracking_id = normalise_tracking_id(raw)
    try:
        found = store.get(tracking_id)
    except Exception:
        st.session_state["track_result"] = {"kind": "error"}
        return
    st.session_state["track_result"] = (
        {"kind": "not_found", "id": tracking_id} if found is None
        else {"kind": "found", "row": found})


# ---------------------------------------------------------------- header
st.markdown('<div class="cc-brand">ClariCase</div>'
            '<div class="cc-tagline">Tell us about a problem with a financial '
            'product or service, in your own words. We\'ll get it to the team '
            'that handles it.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="cc-steps">
  <div class="cc-step"><span class="cc-step-num">1</span>
    <span class="cc-step-title">Describe it</span>
    <div class="cc-step-text">No forms or categories to pick.</div></div>
  <div class="cc-step"><span class="cc-step-num">2</span>
    <span class="cc-step-title">We route it</span>
    <div class="cc-step-text">Matched to the right team and issue.</div></div>
  <div class="cc-step"><span class="cc-step-num">3</span>
    <span class="cc-step-title">Track it</span>
    <div class="cc-step-text">Check progress with your tracking ID.</div></div>
</div>
""", unsafe_allow_html=True)

submit_tab, track_tab = st.tabs([":material/edit_note: Submit a complaint",
                                 ":material/search: Track my complaint"],
                                key="active_tab",
                                on_change=reset_on_tab_change)

# ---------------------------------------------------------------- submit
with submit_tab:
    with st.form("complaint", border=True):
        st.text_area("What happened?", key="complaint_text", height=200,
                     placeholder="For example: I was charged an overdraft fee "
                                 "even though I had enough money in my "
                                 "account. I'd like the fee refunded.")
        st.caption(":material/lock: For your privacy, details such as account, "
                   "card and phone numbers are hidden automatically before "
                   "your complaint is saved.")
        st.form_submit_button("Submit complaint", type="primary",
                              icon=":material/send:", on_click=handle_submit,
                              width="stretch")

    pending = st.session_state.pop("pending_text", None)
    if pending:
        st.session_state["submit_result"] = process_complaint(pending)

    result = st.session_state.get("submit_result")
    if result is None:
        pass
    elif result["kind"] == "too_short":
        st.warning(f"Please describe your complaint in at least {MIN_WORDS} "
                   f"words so we can route it correctly.",
                   icon=":material/info:")
    elif result["kind"] == "error":
        st.error("We couldn't save your complaint. Please try again in a "
                 "moment.", icon=":material/error:")
    else:
        record, prediction = result["record"], result["prediction"]
        teams = candidates_to_show(prediction["top_k"])
        issues = prediction["issue_top_k"][:ISSUES_SHOWN]
        routed = prediction["route"] == "auto"

        with st.container(border=True):
            st.markdown(badge(record["status"]), unsafe_allow_html=True)
            if routed:
                st.markdown('<div class="cc-note">Your complaint has been sent '
                            'to the team that handles it.</div>',
                            unsafe_allow_html=True)
                label("Sent to")
            else:
                st.markdown('<div class="cc-note">Your complaint has been '
                            'received. A specialist will review it to confirm '
                            'the right team.</div>', unsafe_allow_html=True)
                label("Most likely team" if len(teams) == 1
                      else "Most likely teams")
            show_candidates([(t["team_name"], t["confidence"]) for t in teams],
                            "cc-team")

            label("Issue" if len(issues) == 1 else "Possible issues")
            show_candidates([(issue_name(i["issue_label"]), i["confidence"])
                             for i in issues], "cc-issue")
            if len(teams) > 1:
                st.caption(f"Issues shown are for the {teams[0]['team_name']}.")

            label("Your tracking ID")
            st.code(record["tracking_id"], language=None)
            st.caption("Save this ID to check on your complaint in the "
                       "**Track my complaint** tab.")

        render_why_panel(record, prediction)

# ---------------------------------------------------------------- track
with track_tab:
    with st.form("track", border=True):
        st.text_input("Tracking ID", key="tracking_input",
                      placeholder="CC-XXXXXX")
        st.form_submit_button("Check status", icon=":material/search:",
                              on_click=handle_track, width="stretch")

    tracked = st.session_state.get("track_result")
    if tracked is None:
        pass
    elif tracked["kind"] == "error":
        st.error("We couldn't look up your complaint. Please try again in a "
                 "moment.", icon=":material/error:")
    elif tracked["kind"] == "not_found":
        st.error(f"No complaint found with tracking ID {tracked['id']}.",
                 icon=":material/search_off:")
    else:
        found = tracked["row"]
        submitted_at = datetime.fromisoformat(found["submitted_at"])
        with st.container(border=True):
            st.markdown(f"{badge(found['status'])}&nbsp;&nbsp;"
                        f"<code>{found['tracking_id']}</code>",
                        unsafe_allow_html=True)
            # Complaints saved before route/issue existed have None here.
            label("Most likely team" if found.get("route") == "review"
                  else "Sent to")
            st.markdown(f'<div class="cc-team">{found["team_name"]}</div>',
                        unsafe_allow_html=True)
            if found.get("issue_label"):
                label("Most likely issue")
                st.markdown(f'<div class="cc-issue">'
                            f'{issue_name(found["issue_label"])}</div>',
                            unsafe_allow_html=True)
            label("Submitted")
            st.write(f"{submitted_at:%B %d, %Y at %H:%M} UTC")
            with st.expander("Your complaint"):
                st.write(found["complaint_text"])
                st.caption("Personal details are shown as XXXX for your "
                           "privacy.")
            stored = found.get("explanation")
            summary = stored.get("consumer_summary") \
                if isinstance(stored, dict) else None
            label("Why this team?")
            if summary:
                st.write(summary)
            else:
                st.write("No explanation is available for this complaint.")

# ---------------------------------------------------------------- footer
footer = ("Clear-cut complaints are routed automatically; the rest are "
          "reviewed by a person.")
if not store.persistent:
    footer += " · Local mode: complaints are saved on this machine only."
st.markdown(f'<div class="cc-footer">{footer}</div>', unsafe_allow_html=True)

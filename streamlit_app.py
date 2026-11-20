"""ClariCase complaint routing console.

Three tabs: route a single complaint, route a CSV of complaints in bulk, and
track a complaint by its tracking ID.

Run: streamlit run streamlit_app.py
"""

import html
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app import bulk
from src.app.bulk import issue_name
from src.app.pii import mask
from src.app.storage import (LocalStore, SupabaseStore, build_record,
                             normalise_tracking_id)
from src.improved_model_1.predict import load_model, predict

MIN_WORDS = bulk.MIN_WORDS

# How many teams to show, by the top team's confidence:
# above 80% -> 1, 50-80% -> top 2, below 50% -> top 3. Issues always show
# the top ISSUES_SHOWN.
SINGLE_ABOVE = 0.80
PAIR_FROM = 0.50
ISSUES_SHOWN = 3

# Prediction takes well under a second once the model is loaded; each step
# stays on screen this long so the progress is readable.
STEP_PAUSE_SEC = 1
# How long the fully ticked checklist stays before the result appears.
DONE_PAUSE_SEC = 0.5

# Bulk rows are saved in chunks; each chunk is saved whole or not at all.
SAVE_CHUNK = 500
PHRASES_SHOWN = 5

st.set_page_config(page_title="ClariCase", page_icon=":material/forum:",
                   layout="wide")

# ---------------------------------------------------------------- styles
# Cards are keyed containers (st.container(key="card_…")), which Streamlit
# renders with the class "st-key-card_…".
st.markdown("""
<style>
  .block-container { max-width: 1240px; padding-top: 1.2rem;
                     padding-bottom: 3rem; }
  [data-testid="stStatusWidget"] { visibility: hidden; }
  [data-testid="stHeader"] { background: transparent; }

  /* Header */
  .cc-top { display: flex; align-items: center; justify-content: space-between;
            gap: 16px; flex-wrap: wrap; padding: 6px 0 18px;
            border-bottom: 1px solid #E4E4EE; margin-bottom: 8px; }
  .cc-logo { display: flex; align-items: center; gap: 10px; }
  .cc-mark { width: 34px; height: 34px; border-radius: 9px; background: #17162E;
             color: #fff; display: flex; align-items: center;
             justify-content: center; font-family: 'Space Grotesk', sans-serif;
             font-weight: 700; font-size: 17px; }
  .cc-word { font-family: 'Space Grotesk', sans-serif; font-weight: 700;
             font-size: 21px; letter-spacing: -0.01em; color: #17162E; }
  .cc-pill { display: inline-flex; align-items: center; gap: 6px;
             font-size: 12px; font-weight: 600; padding: 6px 11px;
             border-radius: 999px; }
  .cc-pill-green { color: #166534; background: #E8F7EE; }
  .cc-pill-indigo { color: #3730A3; background: #EEF0FF; }

  /* Tabs as a nav bar */
  [data-baseweb="tab-list"] { gap: 6px; }
  [data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
  button[data-baseweb="tab"] { padding: 9px 16px !important;
                               border-radius: 10px; height: auto; }
  button[data-baseweb="tab"] p { font-size: 15px; font-weight: 500;
                                 color: #4A4A66; }
  button[data-baseweb="tab"][aria-selected="true"] { background: #EEF0FF; }
  button[data-baseweb="tab"][aria-selected="true"] p { color: #4338CA;
                                                       font-weight: 600; }

  /* Page intro */
  .cc-eyebrow { margin: 18px 0 4px; font-size: 13px; font-weight: 600;
                color: #4338CA; letter-spacing: 0.05em;
                text-transform: uppercase; }
  .cc-h1 { margin: 0; font-family: 'Space Grotesk', sans-serif;
           font-size: 32px; font-weight: 700; line-height: 1.15;
           letter-spacing: -0.02em; color: #17162E; }
  .cc-sub { margin: 8px 0 22px; font-size: 15px; color: #5B5B76;
            max-width: 660px; }

  /* Cards */
  [class*="st-key-card_"] { background: #FFFFFF; border: 1px solid #E4E4EE;
                            border-radius: 16px; padding: 24px 26px; }
  .cc-card-title { font-size: 17px; font-weight: 600; color: #17162E;
                   margin: 0; }
  .cc-row { display: flex; justify-content: space-between; align-items: center;
            gap: 12px; flex-wrap: wrap; }
  .cc-label { font-size: 12px; font-weight: 600; color: #5B5B76;
              text-transform: uppercase; letter-spacing: 0.05em;
              margin: 18px 0 8px; }
  .cc-muted { font-size: 13px; color: #5B5B76; }
  .cc-badge { display: inline-flex; align-items: center; font-size: 12px;
              font-weight: 700; letter-spacing: 0.03em; padding: 5px 12px;
              border-radius: 999px; text-transform: uppercase; }
  .cc-routed { background: #DCFCE7; color: #166534; }
  .cc-review { background: #FEF3C7; color: #92400E; }
  .cc-received { background: #E2E8F0; color: #334155; }

  /* Team and bars */
  .cc-team { display: flex; align-items: center; justify-content: space-between;
             gap: 12px; }
  .cc-team-name { font-family: 'Space Grotesk', sans-serif; font-size: 22px;
                  font-weight: 600; color: #17162E; letter-spacing: -0.01em; }
  .cc-team-conf { font-family: 'Space Grotesk', sans-serif; font-size: 24px;
                  font-weight: 700; color: #4338CA; text-align: right;
                  line-height: 1.1; }
  .cc-track { height: 8px; background: #EEEEF4; border-radius: 999px;
              overflow: hidden; margin-top: 12px; }
  .cc-fill { height: 100%; background: #4338CA; border-radius: 999px; }
  .cc-bar-row { display: grid; grid-template-columns: minmax(0, 1fr) 34% 46px;
                gap: 12px; align-items: center; margin: 9px 0; }
  .cc-bar-row .cc-track { margin-top: 0; height: 6px; }
  .cc-bar-name { font-size: 14px; color: #17162E; overflow: hidden;
                 text-overflow: ellipsis; white-space: nowrap; }
  .cc-bar-val { font-size: 13px; color: #5B5B76; text-align: right; }
  .cc-chip { display: inline-block; font-size: 13px; padding: 6px 12px;
             margin: 0 6px 8px 0; border-radius: 999px; background: #F6F6FA;
             border: 1px solid #E4E4EE; color: #17162E; }

  /* Steps under the form */
  .cc-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
              gap: 12px; margin-top: 16px; }
  .cc-step { background: #fff; border: 1px solid #E4E4EE; border-radius: 14px;
             padding: 14px 16px; }
  .cc-step-num { font-size: 12px; font-weight: 700; color: #4338CA; }
  .cc-step-title { font-size: 14px; font-weight: 600; margin-top: 4px;
                   color: #17162E; }
  .cc-step-text { font-size: 13px; color: #5B5B76; margin-top: 2px;
                  line-height: 1.45; }

  /* Empty state */
  .cc-empty { text-align: center; padding: 48px 12px; color: #5B5B76; }
  .cc-empty-icon { width: 52px; height: 52px; border-radius: 14px;
                   background: #EEF0FF; color: #4338CA; margin: 0 auto 14px;
                   display: flex; align-items: center; justify-content: center; }

  /* Bulk stepper and KPIs */
  .cc-stepper { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 18px;
                padding: 0; list-style: none; }
  .cc-stepper li { display: flex; align-items: center; gap: 8px;
                   font-size: 14px; font-weight: 600; background: #fff;
                   border: 1px solid #E4E4EE; border-radius: 999px;
                   padding: 7px 14px 7px 7px; color: #5B5B76; }
  .cc-stepper li span { width: 22px; height: 22px; border-radius: 50%;
                        background: #EEEEF4; font-size: 12px;
                        display: inline-flex; align-items: center;
                        justify-content: center; }
  .cc-stepper li.done { color: #166534; }
  .cc-stepper li.done span { background: #DCFCE7; }
  .cc-stepper li.now { background: #4338CA; border-color: #4338CA;
                       color: #fff; }
  .cc-stepper li.now span { background: #fff; color: #4338CA; }
  .cc-kpis { display: grid; grid-template-columns: repeat(auto-fit,
             minmax(140px, 1fr)); gap: 12px; margin-top: 14px; }
  .cc-kpi { border: 1px solid #EEEEF4; border-radius: 12px; padding: 14px 16px; }
  .cc-kpi-num { font-family: 'Space Grotesk', sans-serif; font-size: 28px;
                font-weight: 700; margin-top: 2px; color: #17162E; }
  .cc-file { display: flex; align-items: center; gap: 14px;
             border: 1.5px dashed #C7C9EE; background: #FAFAFF;
             border-radius: 14px; padding: 14px 16px; margin: 4px 0 14px; }

  /* Track timeline */
  .cc-tl { list-style: none; margin: 4px 0 0; padding: 0; }
  .cc-tl li { display: flex; gap: 14px; }
  .cc-tl-rail { display: flex; flex-direction: column; align-items: center; }
  .cc-tl-dot { width: 22px; height: 22px; border-radius: 50%; color: #fff;
               display: inline-flex; align-items: center;
               justify-content: center; flex: 0 0 auto; font-size: 12px; }
  .cc-tl-line { width: 2px; flex: 1 1 auto; min-height: 20px; }
  .cc-tl-body { padding-bottom: 16px; }
  .cc-tl-title { font-size: 14px; font-weight: 600; color: #17162E; }

  .cc-footer { display: flex; justify-content: space-between; gap: 12px;
               flex-wrap: wrap; font-size: 13px; color: #5B5B76;
               border-top: 1px solid #E4E4EE; margin-top: 40px;
               padding-top: 16px; }

  @media (max-width: 640px) {
    .cc-steps { grid-template-columns: 1fr; }
    .cc-h1 { font-size: 26px; }
  }
</style>
""", unsafe_allow_html=True)

ICON_LOCK = ('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
             'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
             'stroke-linejoin="round" aria-hidden="true"><rect x="4" y="11" '
             'width="16" height="10" rx="2"></rect><path d="M8 11V7a4 4 0 0 1 '
             '8 0v4"></path></svg>')
ICON_CHECK = ('<svg width="12" height="12" viewBox="0 0 24 24" fill="none" '
              'stroke="currentColor" stroke-width="3" stroke-linecap="round" '
              'stroke-linejoin="round" aria-hidden="true"><path d="M20 6L9 '
              '17l-5-5"></path></svg>')
ICON_SHIELD = ('<svg width="13" height="13" viewBox="0 0 24 24" fill="none" '
               'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" '
               'stroke-linejoin="round" aria-hidden="true"><path d="M12 22s8-4 '
               '8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path><path d="M9 12l2 2 '
               '4-4"></path></svg>')
ICON_ROUTE = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
              'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
              'stroke-linejoin="round" aria-hidden="true"><path d="M22 2L11 '
              '13"></path><path d="M22 2l-7 20-4-9-9-4 20-7z"></path></svg>')
ICON_FILE = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
             'stroke="#4338CA" stroke-width="2" stroke-linecap="round" '
             'stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 '
             '0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><path '
             'd="M14 2v6h6"></path><path d="M8 13h8"></path><path d="M8 '
             '17h8"></path></svg>')


# ---------------------------------------------------------------- resources
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
esc = html.escape


def candidates_to_show(candidates: list[dict]) -> list[dict]:
    top = candidates[0]["confidence"]
    n = 1 if top > SINGLE_ABOVE else 2 if top >= PAIR_FROM else 3
    return candidates[:n]


def badge(status: str) -> str:
    css, text = {"Routed": ("cc-routed", "Auto-routed"),
                 "Under review": ("cc-review", "Needs review")}.get(
        status, ("cc-received", status))
    return f'<span class="cc-badge {css}">{esc(text)}</span>'


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def bar_rows(items: list[tuple[str, float]], strong_first: bool = True) -> str:
    rows = []
    for i, (name, conf) in enumerate(items):
        pct = round(min(max(conf, 0.0), 1.0) * 100)
        color = "#4338CA" if (i == 0 or not strong_first) else "#A5A8E0"
        weight = 600 if (i == 0 and strong_first) else 400
        rows.append(
            f'<div class="cc-bar-row"><span class="cc-bar-name" '
            f'style="font-weight:{weight}">{esc(name)}</span>'
            f'<div class="cc-track"><div class="cc-fill" style="width:{pct}%;'
            f'background:{color}"></div></div>'
            f'<span class="cc-bar-val">{pct}%</span></div>')
    return "".join(rows)


def intro(eyebrow: str, title: str, sub: str) -> None:
    html_block(f'<div class="cc-eyebrow">{eyebrow}</div>'
               f'<div class="cc-h1">{title}</div>'
               f'<div class="cc-sub">{sub}</div>')


# ---------------------------------------------------------------- single
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


STEPS = ("Masking personal details", "Identifying team and issue",
         "Saving and assigning a tracking ID")


def render_steps(placeholder, current: int) -> None:
    # Steps before `current` are done, `current` is in progress, rest pending.
    rows = []
    for i, name in enumerate(STEPS):
        if i < current:
            rows.append(f":green[:material/check_circle:] {name}")
        elif i == current:
            rows.append(f":material/progress_activity: **{name}…**")
        else:
            rows.append(f":gray[:material/radio_button_unchecked: {name}]")
    # A fresh key per step: each redraw is a new element in the same run.
    with placeholder.container(key=f"card_steps_{current}"):
        html_block('<div class="cc-card-title">Routing your complaint</div>')
        st.markdown("  \n".join(rows))


def process_complaint(text: str) -> dict:
    placeholder = st.empty()
    render_steps(placeholder, 0)
    time.sleep(STEP_PAUSE_SEC)

    render_steps(placeholder, 1)
    prediction = predict(text, model=get_model())
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


def render_result(record: dict, prediction: dict) -> None:
    teams = candidates_to_show(prediction["top_k"])
    issues = prediction["issue_top_k"][:ISSUES_SHOWN]
    routed = prediction["route"] == "auto"

    with st.container(key="card_result"):
        parts = [f'<div class="cc-row"><div class="cc-card-title">Routing '
                 f'decision</div>{badge(record["status"])}</div>']
        if routed:
            parts.append('<div class="cc-muted" style="margin-top:6px">Sent '
                         'automatically to the team that handles it.</div>')
        else:
            parts.append('<div class="cc-muted" style="margin-top:6px">Below '
                         'the routing threshold, so a specialist will confirm '
                         'the team.</div>')

        if len(teams) == 1:
            top = teams[0]
            pct = round(top["confidence"] * 100)
            parts.append(
                f'<div class="cc-label">{"Sent to" if routed else "Most likely team"}</div>'
                f'<div class="cc-team"><div class="cc-team-name">'
                f'{esc(top["team_name"])}</div><div><div class="cc-team-conf">'
                f'{pct}%</div><div class="cc-muted" style="text-align:right">'
                f'confidence</div></div></div>'
                f'<div class="cc-track"><div class="cc-fill" '
                f'style="width:{pct}%"></div></div>')
        else:
            parts.append('<div class="cc-label">Most likely teams</div>'
                         + bar_rows([(t["team_name"], t["confidence"])
                                     for t in teams]))

        parts.append('<div class="cc-label">'
                     + ("Likely issue" if len(issues) == 1 else "Likely issues")
                     + '</div>'
                     + bar_rows([(issue_name(i["issue_label"]), i["confidence"])
                                 for i in issues]))
        if len(teams) > 1:
            parts.append(f'<div class="cc-muted">Issues shown are for the '
                         f'{esc(teams[0]["team_name"])}.</div>')
        parts.append('<div class="cc-label">Tracking ID</div>')
        html_block("".join(parts))
        st.code(record["tracking_id"], language=None)
        st.caption("Save this ID to check progress in the **Track** tab.")


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
            narrator=get_explainer(),
            # Lets the privacy check confirm each cited n-gram is derivable
            # from the masked text with the model's own vectorizer; without
            # it a cruder fallback rejects valid drivers.
            vectorizers={"improved_model_1": bundle["vectorizer"]})
    except Exception:
        log.exception("explanation failed")
        return {"explanation": None, "record": None, "audit": []}
    if result.error or result.narration is None:
        log.warning("explanation unavailable: %s", result.error)
        return {"explanation": None, "record": result.record, "audit": []}
    return {"explanation": result.narration.explanation,
            "record": result.record, "audit": result.narration.audit}


def _phrases(decision_record) -> list[str]:
    # Positive text drivers of the chosen team, already checked by the
    # privacy guard to come from the masked text.
    try:
        sec = decision_record.sections["improved_model_1"]
    except (AttributeError, KeyError, TypeError):
        return []
    out = []
    for d in sec.drivers or []:
        if d.kind == "ngram" and d.text and d.contribution > 0 \
                and d.text not in out:
            out.append(d.text)
        if len(out) == PHRASES_SHOWN:
            break
    return out


def render_why_panel(record: dict, prediction: dict) -> None:
    key = record["tracking_id"]
    cache = st.session_state.setdefault("explanations", {})
    with st.container(key="card_why"):
        if key not in cache:
            html_block('<div class="cc-card-title">Why this team?</div>')
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
            st.rerun()

        entry = cache[key]
        expl = entry["explanation"]
        if expl is None:
            html_block('<div class="cc-card-title">Why this team?</div>')
            st.info("An explanation isn't available for this decision.",
                    icon=":material/info:")
            return

        source = ("Written by AI · validated" if expl.source == "llm"
                  else "Validated explanation")
        html_block(f'<div class="cc-row"><div class="cc-card-title">Why this '
                   f'team?</div><span class="cc-pill cc-pill-indigo">'
                   f'{ICON_SHIELD}{source}</span></div>')
        st.write(expl.consumer_summary)
        phrases = _phrases(entry["record"])
        if phrases:
            html_block('<div class="cc-label">Phrases that drove the decision'
                       '</div>' + "".join(f'<span class="cc-chip">{esc(p)}'
                                          f'</span>' for p in phrases))
        st.caption("Every fact shown is checked against the decision record. "
                   "Covers the main routing model (Improved Model 1).")
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


def render_single_tab() -> None:
    intro("New complaint", "Route a single complaint",
          "Paste the customer's words as written. ClariCase hides personal "
          "details, picks the team and issue, and explains why.")
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        with st.container(key="card_composer"):
            html_block('<div class="cc-card-title" style="margin-bottom:10px">'
                       'Complaint details</div>')
            with st.form("complaint", border=False):
                st.text_area(
                    "What happened?", key="complaint_text", height=220,
                    placeholder="For example: I was charged an overdraft fee "
                                "even though I had enough money in my account. "
                                "I'd like the fee refunded.")
                st.caption(":material/lock: Account, card and phone numbers "
                           "are replaced with XXXX before anything is saved.")
                st.form_submit_button("Route complaint", type="primary",
                                      icon=":material/send:",
                                      on_click=handle_submit, width="stretch")
        html_block("""
<div class="cc-steps">
  <div class="cc-step"><div class="cc-step-num">01</div>
    <div class="cc-step-title">Mask</div>
    <div class="cc-step-text">Personal details hidden first.</div></div>
  <div class="cc-step"><div class="cc-step-num">02</div>
    <div class="cc-step-title">Route</div>
    <div class="cc-step-text">Team and issue, with confidence.</div></div>
  <div class="cc-step"><div class="cc-step-num">03</div>
    <div class="cc-step-title">Explain</div>
    <div class="cc-step-text">A checked reason for every decision.</div></div>
</div>""")

    with right:
        pending = st.session_state.pop("pending_text", None)
        if pending:
            st.session_state["submit_result"] = process_complaint(pending)

        result = st.session_state.get("submit_result")
        if result is None:
            with st.container(key="card_empty"):
                html_block(f'<div class="cc-empty"><div class="cc-empty-icon">'
                           f'{ICON_ROUTE}</div><div class="cc-card-title">'
                           f'Your routing decision appears here</div>'
                           f'<div class="cc-muted" style="margin-top:6px">'
                           f'Team, issue, confidence, a tracking ID and the '
                           f'reason behind it.</div></div>')
        elif result["kind"] == "too_short":
            st.warning(f"Please describe the complaint in at least {MIN_WORDS} "
                       f"words so it can be routed correctly.",
                       icon=":material/info:")
        elif result["kind"] == "error":
            st.error("We couldn't save this complaint. Please try again in a "
                     "moment.", icon=":material/error:")
        else:
            render_result(result["record"], result["prediction"])
            render_why_panel(result["record"], result["prediction"])


# ---------------------------------------------------------------- bulk
def stepper(current: int) -> str:
    names = ("Upload", "Map columns", "Review results", "Export")
    items = []
    for i, name in enumerate(names, 1):
        if i < current:
            items.append(f'<li class="done"><span>{ICON_CHECK}</span>{name}</li>')
        elif i == current:
            items.append(f'<li class="now" aria-current="step"><span>{i}</span>'
                         f'{name}</li>')
        else:
            items.append(f'<li><span>{i}</span>{name}</li>')
    return f'<ol class="cc-stepper" aria-label="Progress">{"".join(items)}</ol>'


def run_bulk(df: pd.DataFrame, text_col: str, ref_col: str | None) -> dict:
    texts = df[text_col].astype(str).tolist()
    refs = df[ref_col].astype(str).tolist() if ref_col else None
    model = get_model()
    progress = st.progress(0.0, text="Routing complaints…")

    def on_progress(done: int, total: int) -> None:
        if done == total or done % 25 == 0:
            progress.progress(done / total,
                              text=f"Routing complaints… {done:,} of {total:,}")

    start = time.perf_counter()
    records, results, skipped = bulk.route_rows(
        texts, refs, lambda t: predict(t, model=model), on_progress)

    progress.progress(1.0, text="Saving…")
    not_saved: set[str] = set()
    for i in range(0, len(records), SAVE_CHUNK):
        chunk = records[i:i + SAVE_CHUNK]
        try:
            store.add_many(chunk)
        except Exception:
            log.exception("bulk save failed")
            not_saved.update(r["tracking_id"] for r in chunk)
    for r in results:
        if r["tracking_id"] in not_saved:
            r["tracking_id"] = "Not saved"
    elapsed = time.perf_counter() - start
    progress.empty()
    return {"results": results, "skipped": skipped, "elapsed": elapsed,
            "not_saved": len(not_saved), "total": len(texts)}


def render_bulk_summary(run: dict) -> None:
    results = run["results"]
    s = bulk.summarise(results)
    auto_pct = f"{s['auto'] / s['routed']:.0%} of routed" if s["routed"] else "—"
    team_total = len(s["teams"])
    with st.container(key="card_summary"):
        html_block(
            f'<div class="cc-row"><div class="cc-card-title">Batch summary</div>'
            f'<span class="cc-muted">Processed in {run["elapsed"]:.1f} s</span>'
            f'</div><div class="cc-kpis">'
            f'<div class="cc-kpi"><div class="cc-muted">Routed</div>'
            f'<div class="cc-kpi-num">{s["routed"]:,}</div><div class="cc-muted">'
            f'of {run["total"]:,} rows</div></div>'
            f'<div class="cc-kpi"><div class="cc-muted">Auto-routed</div>'
            f'<div class="cc-kpi-num" style="color:#166534">{s["auto"]:,}</div>'
            f'<div class="cc-muted">{auto_pct}</div></div>'
            f'<div class="cc-kpi"><div class="cc-muted">Needs review</div>'
            f'<div class="cc-kpi-num" style="color:#92400E">{s["review"]:,}</div>'
            f'<div class="cc-muted">for a specialist</div></div>'
            f'<div class="cc-kpi"><div class="cc-muted">Teams involved</div>'
            f'<div class="cc-kpi-num">{team_total}</div><div class="cc-muted">'
            f'of 11</div></div></div>')
        if s["teams"]:
            top = s["teams"][0][1]
            shown = s["teams"][:6]
            rows = []
            for name, n in shown:
                w = round(n / top * 100)
                rows.append(
                    f'<div class="cc-bar-row" style="grid-template-columns:'
                    f'minmax(0,240px) minmax(0,1fr) 48px"><span '
                    f'class="cc-bar-name">{esc(name)}</span><div '
                    f'class="cc-track"><div class="cc-fill" style="width:{w}%">'
                    f'</div></div><span class="cc-bar-val" style="font-weight:'
                    f'600;color:#17162E">{n:,}</span></div>')
            rest = sum(n for _, n in s["teams"][6:])
            if rest:
                rows.append(f'<div class="cc-muted">Other teams: {rest:,}</div>')
            html_block('<div class="cc-label">By team</div>' + "".join(rows))


def render_bulk_results(run: dict) -> None:
    results = run["results"]
    s = bulk.summarise(results)
    with st.container(key="card_results"):
        head, actions = st.columns([1, 1], vertical_alignment="center")
        with head:
            html_block('<div class="cc-card-title">Results</div>')
        with actions:
            st.download_button("Export CSV", bulk.export_csv(results),
                               file_name="claricase_routed.csv",
                               mime="text/csv", type="primary",
                               icon=":material/download:", width="stretch")
        f_col, q_col = st.columns([1.2, 1], vertical_alignment="bottom")
        with f_col:
            options = {f"All {s['routed']:,}": "all",
                       f"Auto-routed {s['auto']:,}": "Routed",
                       f"Needs review {s['review']:,}": "Under review"}
            choice = st.segmented_control(
                "Filter by status", list(options), default=list(options)[0],
                key="bulk_filter", label_visibility="collapsed")
        with q_col:
            query = st.text_input("Search", key="bulk_query",
                                  placeholder="Search reference, team or text",
                                  label_visibility="collapsed")
        wanted = options.get(choice or list(options)[0], "all")
        df = pd.DataFrame(results)
        if df.empty:
            st.info("No rows were routed.", icon=":material/info:")
            return
        if wanted != "all":
            df = df[df["status"] == wanted]
        if query:
            q = query.strip().lower()
            mask_rows = df[["reference", "team", "issue", "complaint",
                            "tracking_id"]].apply(
                lambda col: col.astype(str).str.lower().str.contains(
                    q, regex=False)).any(axis=1)
            df = df[mask_rows]
        df = df.assign(confidence=(df["confidence"] * 100).round(0),
                       status=df["status"].map({"Routed": "Auto-routed",
                                                "Under review": "Needs review"}))
        st.dataframe(
            df[["reference", "complaint", "team", "issue", "confidence",
                "status", "tracking_id"]],
            hide_index=True, width="stretch", height=420,
            column_config={
                "reference": st.column_config.TextColumn("Reference"),
                "complaint": st.column_config.TextColumn("Complaint",
                                                         width="large"),
                "team": st.column_config.TextColumn("Team"),
                "issue": st.column_config.TextColumn("Issue"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%d%%"),
                "status": st.column_config.TextColumn("Status"),
                "tracking_id": st.column_config.TextColumn("Tracking ID"),
            })
        st.caption(f"Showing {len(df):,} of {s['routed']:,} routed complaints. "
                   f"Complaint text is shown with personal details masked.")


def render_bulk_tab() -> None:
    head, link = st.columns([3, 1], vertical_alignment="bottom")
    with head:
        intro("Batch", "Route complaints in bulk",
              f"Upload a CSV with one complaint per row (up to "
              f"{bulk.MAX_ROWS:,}). Every row is masked, routed and given a "
              f"tracking ID. Low-confidence rows are marked for review.")
    with link:
        st.download_button("CSV template", bulk.TEMPLATE_CSV,
                           file_name="claricase_template.csv", mime="text/csv",
                           icon=":material/description:", width="stretch")

    types = ["csv"] + (["xlsx"] if bulk.excel_supported() else [])
    upload = st.session_state.get("bulk_upload")
    run = st.session_state.get("bulk_run")
    sig = (upload.name, upload.size) if upload is not None else None
    if run is not None and run.get("sig") != sig:
        st.session_state.pop("bulk_run", None)
        run = None
    html_block(stepper(1 if upload is None else 2 if run is None else 3))

    left, right = st.columns([1, 1.6], gap="large")
    with left:
        with st.container(key="card_file"):
            html_block('<div class="cc-card-title" style="margin-bottom:8px">'
                       'File</div>')
            st.file_uploader("Upload complaints", type=types, key="bulk_upload",
                             label_visibility="collapsed")
            if upload is None:
                st.caption("Needs a column with the complaint text. A "
                           "reference column is optional and is kept in the "
                           "export.")
                df = None
            else:
                try:
                    df = bulk.read_upload(upload.name, upload.getvalue())
                except bulk.BulkError as exc:
                    st.error(str(exc), icon=":material/error:")
                    df = None
            if df is not None:
                html_block(f'<div class="cc-file">{ICON_FILE}<div><div '
                           f'style="font-weight:600;font-size:14px">'
                           f'{esc(upload.name)}</div><div class="cc-muted">'
                           f'{len(df):,} rows · {len(df.columns)} columns</div>'
                           f'</div></div>')
                cols = list(df.columns)
                guess = bulk.guess_text_column(df)
                text_col = st.selectbox("Complaint text column", cols,
                                        index=cols.index(guess))
                ref_guess = bulk.guess_ref_column(df, text_col)
                ref_opts = ["(none)"] + [c for c in cols if c != text_col]
                ref_choice = st.selectbox(
                    "Your reference column", ref_opts,
                    index=ref_opts.index(ref_guess) if ref_guess else 0)
                ref_col = None if ref_choice == "(none)" else ref_choice
                if st.button(f"Route {len(df):,} complaints", type="primary",
                             icon=":material/send:", width="stretch"):
                    out = run_bulk(df, text_col, ref_col)
                    out["sig"] = sig
                    st.session_state["bulk_run"] = out
                    st.rerun()
                if run is not None:
                    if run["skipped"]:
                        n = len(run["skipped"])
                        st.warning(f"{n:,} row{'s' if n != 1 else ''} skipped: "
                                   f"fewer than {MIN_WORDS} words.",
                                   icon=":material/info:")
                    if run["not_saved"]:
                        st.error(f"{run['not_saved']:,} complaints couldn't be "
                                 f"saved and have no tracking ID. Route the "
                                 f"file again to retry.",
                                 icon=":material/error:")

    with right:
        if run is None:
            with st.container(key="card_bulk_empty"):
                html_block(f'<div class="cc-empty"><div class="cc-empty-icon">'
                           f'{ICON_ROUTE}</div><div class="cc-card-title">'
                           f'Results appear here</div><div class="cc-muted" '
                           f'style="margin-top:6px">A summary by team, every '
                           f'routed row with its tracking ID, and a CSV '
                           f'export.</div></div>')
        else:
            render_bulk_summary(run)

    if run is not None:
        render_bulk_results(run)


# ---------------------------------------------------------------- track
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


def timeline(found: dict, submitted: str) -> str:
    done, todo = "#166534", "#C9C9D6"
    route = found.get("route")
    team = esc(found["team_name"])
    conf = found.get("confidence")
    conf_txt = f"{conf:.0%} confidence" if conf is not None else ""
    steps = [("Received", submitted, done),
             ("Personal details masked", "Before saving", done)]
    if route == "auto":
        steps += [("Auto-routed", conf_txt, done),
                  (f"With the {team}", "Awaiting the team's response", done)]
    elif route == "review":
        steps += [("Sent for specialist review", conf_txt, done),
                  ("Team confirmed", f"Most likely the {team}", todo)]
    else:
        steps += [(f"With the {team}", "", done)]
    items = []
    for i, (title, sub, color) in enumerate(steps):
        last = i == len(steps) - 1
        line = "transparent" if last else (
            "#BBE5C8" if steps[i + 1][2] == done else "#E4E4EE")
        mark = ICON_CHECK if color == done else ""
        items.append(
            f'<li><div class="cc-tl-rail"><span class="cc-tl-dot" '
            f'style="background:{color}">{mark}</span><span class="cc-tl-line" '
            f'style="background:{line}"></span></div><div class="cc-tl-body">'
            f'<div class="cc-tl-title">{title}</div><div class="cc-muted">'
            f'{esc(sub)}</div></div></li>')
    return f'<ol class="cc-tl" aria-label="Timeline">{"".join(items)}</ol>'


def render_track_tab() -> None:
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        html_block('<div style="text-align:center"><div class="cc-eyebrow">'
                   'Status</div><div class="cc-h1">Track a complaint</div>'
                   '<div class="cc-sub" style="margin-left:auto;margin-right:'
                   'auto">Enter the tracking ID shown after submission or in '
                   'your bulk export.</div></div>')
        with st.container(key="card_track_form"):
            with st.form("track", border=False):
                c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
                with c1:
                    st.text_input("Tracking ID", key="tracking_input",
                                  placeholder="CC-XXXXXX",
                                  label_visibility="collapsed")
                with c2:
                    st.form_submit_button("Check status", type="primary",
                                          icon=":material/search:",
                                          on_click=handle_track,
                                          width="stretch")

        tracked = st.session_state.get("track_result")
        if tracked is None:
            return
        if tracked["kind"] == "error":
            st.error("We couldn't look up this complaint. Please try again in "
                     "a moment.", icon=":material/error:")
            return
        if tracked["kind"] == "not_found":
            st.error(f"No complaint found with tracking ID {tracked['id']}.",
                     icon=":material/search_off:")
            return

        found = tracked["row"]
        submitted_at = datetime.fromisoformat(found["submitted_at"])
        submitted = f"{submitted_at:%B %d, %Y at %H:%M} UTC"
        with st.container(key="card_track_result"):
            html_block(
                f'<div class="cc-row"><div><div class="cc-card-title" '
                f'style="font-family:ui-monospace,Menlo,monospace;'
                f'letter-spacing:0.04em">{esc(found["tracking_id"])}</div>'
                f'<div class="cc-muted" style="margin-top:2px">Submitted '
                f'{submitted}</div></div>{badge(found["status"])}</div>'
                f'<hr style="border:0;border-top:1px solid #EEEEF4;'
                f'margin:18px 0">')
            t_col, d_col = st.columns([1, 1.3], gap="large")
            with t_col:
                html_block(timeline(found, submitted))
            with d_col:
                team_label = ("Most likely team" if found.get("route") == "review"
                              else "Team")
                parts = [f'<div class="cc-label" style="margin-top:0">'
                         f'{team_label}</div><div class="cc-team-name" '
                         f'style="font-size:20px">{esc(found["team_name"])}'
                         f'</div>']
                if found.get("issue_label"):
                    parts.append(f'<div class="cc-label">Issue</div><div '
                                 f'style="font-size:15px">'
                                 f'{esc(issue_name(found["issue_label"]))}</div>')
                html_block("".join(parts))
                stored = found.get("explanation")
                summary = stored.get("consumer_summary") \
                    if isinstance(stored, dict) else None
                html_block('<div class="cc-label">Why this team?</div>')
                st.write(summary or "No explanation is available for this "
                                    "complaint.")
                with st.expander("Complaint text"):
                    st.write(found["complaint_text"])
                    st.caption("Personal details are shown as XXXX.")


# ---------------------------------------------------------------- page
def reset_on_tab_change() -> None:
    for key in ("submit_result", "track_result", "pending_text"):
        st.session_state.pop(key, None)
    st.session_state["complaint_text"] = ""
    st.session_state["tracking_input"] = ""


html_block(f'<div class="cc-top"><div class="cc-logo"><div class="cc-mark">C'
           f'</div><span class="cc-word">ClariCase</span></div>'
           f'<span class="cc-pill cc-pill-green">{ICON_LOCK}Personal details '
           f'masked</span></div>')

single_tab, bulk_tab, track_tab = st.tabs(
    [":material/edit_note: Single complaint",
     ":material/upload_file: Bulk upload",
     ":material/search: Track"],
    key="active_tab", on_change=reset_on_tab_change)

with single_tab:
    render_single_tab()
with bulk_tab:
    render_bulk_tab()
with track_tab:
    render_track_tab()

footer = ("Clear-cut complaints are routed automatically; the rest go to a "
          "specialist for review.")
if not store.persistent:
    footer += " · Local mode: complaints are saved on this machine only."
html_block(f'<div class="cc-footer"><span>{footer}</span><span>ClariCase'
           f'</span></div>')

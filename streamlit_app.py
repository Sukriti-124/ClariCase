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

from src.app import bulk, theme
from src.app.bulk import issue_name
from src.app.highlight import highlight
from src.app.pii import mask
from src.app.storage import build_record, make_store, normalise_tracking_id
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

# The dark band reaches this far down; cards start on it and overlap it.
st.markdown(theme.css(band_px=330), unsafe_allow_html=True)


# ---------------------------------------------------------------- resources
@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_store():
    return make_store(st.secrets)


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


def status_mark(status: str) -> str:
    css, text = {"Routed": ("cc-s-ok", "Auto-routed"),
                 "Under review": ("cc-s-warn", "Needs review"),
                 "Assigned": ("cc-s-ok", "Assigned")}.get(
        status, ("", status))
    return f'<span class="cc-status {css}">{esc(text)}</span>'


def panel_title(title: str, note: str = "") -> str:
    note_html = f"<span>{note}</span>" if note else ""
    return f'<div class="cc-ph">{title}{note_html}</div>'


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def bar_rows(items: list[tuple[str, float]]) -> str:
    rows = []
    for i, (name, conf) in enumerate(items):
        pct = round(min(max(conf, 0.0), 1.0) * 100)
        rows.append(
            f'<div class="cc-bar-row{" top" if i == 0 else ""}"><span '
            f'class="cc-bar-name">{esc(name)}</span><div class="cc-track">'
            f'<div class="cc-fill" style="width:{pct}%"></div></div>'
            f'<span class="cc-bar-val">{pct}%</span></div>')
    return "".join(rows)


def intro(title: str, sub: str) -> None:
    html_block(f'<div class="cc-h1">{title}</div><div class="cc-sub">{sub}</div>')


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


STEPS = ("Masking personal details", "Identifying the team",
         "Assigning the issue", "Saving and assigning a tracking ID")


def render_steps(placeholder, current: int) -> None:
    # Steps before `current` are done, `current` is in progress, rest pending.
    items = []
    for i, name in enumerate(STEPS):
        state = "done" if i < current else "now" if i == current else "todo"
        items.append(f'<li class="{state}">{name}</li>')
    # A fresh key per step: each redraw is a new element in the same run.
    with placeholder.container(key=f"card_steps_{current}"):
        html_block(panel_title("Processing")
                   + f'<ul class="cc-proc">{"".join(items)}</ul>')


def process_complaint(text: str) -> dict:
    placeholder = st.empty()
    render_steps(placeholder, 0)
    time.sleep(STEP_PAUSE_SEC)

    # One model call gives both; the steps show the team, then the issue.
    render_steps(placeholder, 1)
    prediction = predict(text, model=get_model())
    time.sleep(STEP_PAUSE_SEC)

    render_steps(placeholder, 2)
    time.sleep(STEP_PAUSE_SEC)

    render_steps(placeholder, 3)
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
    top = teams[0]

    also = ""
    if len(teams) > 1:
        others = ", ".join(f'{esc(t["team_name"])} ({round(t["confidence"] * 100)}%)'
                           for t in teams[1:])
        also = f'<div class="cc-also">Also possible: {others}</div>'
    note = ("" if routed else
            '<div class="cc-also">Below the routing threshold, so a specialist '
            'will confirm the team.</div>')
    html_block(
        f'<div class="cc-result-top"><div class="cc-team">'
        f'<div class="cc-avatar">{theme.team_code(top.get("team_id"), top["team_name"])}</div>'
        f'<div><div class="cc-team-label">{"Sent to" if routed else "Most likely team"}</div>'
        f'<div class="cc-team-name">{esc(top["team_name"])}</div>'
        f'<div class="cc-meta">{status_mark(record["status"])}</div>'
        f'</div></div>'
        f'{theme.gauge(top["confidence"], theme.ACCENT if routed else theme.WARN)}'
        f'</div>{also}{note}'
        '<div class="cc-label">Tracking ID</div>')
    st.code(record["tracking_id"], language=None)
    html_block(
        '<div class="cc-muted cc-track-note">Copy this ID to check progress on '
        'the Track tab.</div>'
        f'<div class="cc-label">{"Likely issue" if len(issues) == 1 else "Likely issues"}</div>'
        + bar_rows([(issue_name(i["issue_label"]), i["confidence"]) for i in issues]))


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
        # The phrase is the driver's label: the vectorizer feature name,
        # "word__<n-gram>" or "char__<fragment>". Only whole words are shown.
        label = d.text or d.label or ""
        if label.startswith("char__"):
            continue
        phrase = label.removeprefix("word__")
        if d.kind == "ngram" and phrase and d.contribution > 0 \
                and phrase not in out:
            out.append(phrase)
        if len(out) == PHRASES_SHOWN:
            break
    return out


def render_why_panel(record: dict, prediction: dict) -> None:
    key = record["tracking_id"]
    cache = st.session_state.setdefault("explanations", {})
    with st.container():
        if key not in cache:
            html_block('<div class="cc-label">Why this team</div>')
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
            html_block('<div class="cc-label">Why this team</div><div '
                       'class="cc-muted">No explanation is available for '
                       'this decision.</div>')
            return

        source = ("AI-written" if expl.source == "llm" else "Standard wording")
        phrases = _phrases(entry["record"])
        parts = ['<div class="cc-why"><div class="cc-label">Why this team</div>'
                 f'<p>{esc(expl.consumer_summary)}</p>']
        if phrases:
            parts.append('<div class="cc-label">Phrases that drove the '
                         'decision</div><div class="cc-excerpt">'
                         + highlight(record["complaint_text"], phrases)
                         + '</div>')
        parts.append(f'<div class="cc-src">{theme.ICON_CHECK}{source} · every '
                     f'statement checked against the decision record</div></div>')
        html_block("".join(parts))
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


def empty_preview() -> str:
    """A greyed-out result card, so the layout is visible before routing."""
    bars = "".join(
        f'<div class="cc-bar-row{" top" if i == 0 else ""}"><div '
        f'class="cc-ghost-line" style="width:{w}%"></div><div class="cc-track">'
        f'<div class="cc-fill" style="width:{f}%"></div></div><span '
        f'class="cc-bar-val">—</span></div>'
        for i, (w, f) in enumerate([(70, 80), (52, 18), (44, 8)]))
    return (
        '<div class="cc-ghost"><div class="cc-result-top"><div class="cc-team">'
        '<div class="cc-avatar">··</div><div><div class="cc-team-label">'
        'Sent to</div><div class="cc-team-name">Team name</div>'
        '<div class="cc-meta"><span class="cc-status">Status</span>'
        '<span class="cc-id">CC-······</span></div></div></div>'
        + theme.gauge(0.0).replace("<b>0%</b>", "<b>—</b>") +
        '</div><div class="cc-label">Likely issues</div>' + bars
        + '<div class="cc-label">Why this team</div>'
          '<div class="cc-ghost-line" style="width:96%"></div>'
          '<div class="cc-ghost-line" style="width:88%"></div>'
          '<div class="cc-ghost-line" style="width:60%"></div></div>'
          '<div class="cc-ghost-cap">Route a complaint to see the team, '
          'confidence, tracking ID and the reasoning.</div>')


def render_single_tab() -> None:
    intro("Customer complaint",
          "Paste the customer's words as written. ClariCase hides personal "
          "details, picks the team and issue, and explains why.")
    left, right = st.columns([1, 1.05], gap="medium")

    with left:
        with st.container(key="card_composer"):
            html_block(panel_title("Complaint"))
            with st.form("complaint", border=False):
                st.text_area(
                    "What happened?", key="complaint_text", height=250,
                    label_visibility="collapsed",
                    placeholder="For example: I was charged an overdraft fee "
                                "even though I had enough money in my account. "
                                "I'd like the fee refunded.")
                note, btn = st.columns([1.4, 1], vertical_alignment="center")
                with note:
                    html_block(f'<div class="cc-note">{theme.ICON_LOCK}<span>'
                               f'Personal details become <span class="cc-x">'
                               f'XXXX</span> before saving</span></div>')
                with btn:
                    st.form_submit_button("Route", type="primary",
                                          icon=":material/arrow_forward:",
                                          on_click=handle_submit,
                                          width="stretch")

    with right:
        pending = st.session_state.pop("pending_text", None)
        if pending:
            st.session_state["submit_result"] = process_complaint(pending)

        result = st.session_state.get("submit_result")
        if result is None:
            with st.container(key="card_empty"):
                html_block(empty_preview())
        elif result["kind"] == "too_short":
            with st.container(key="card_short"):
                st.warning(f"Please describe the complaint in at least "
                           f"{MIN_WORDS} words so it can be routed correctly.")
        elif result["kind"] == "error":
            with st.container(key="card_error"):
                st.error("We couldn't save this complaint. Please try again "
                         "in a moment.")
        else:
            with st.container(key="card_result"):
                render_result(result["record"], result["prediction"])
                render_why_panel(result["record"], result["prediction"])


# ---------------------------------------------------------------- bulk
def stepper(current: int) -> str:
    names = ("Upload", "Map columns", "Review results", "Export")
    items = []
    for i, name in enumerate(names, 1):
        state = "done" if i < current else "now" if i == current else ""
        current_attr = ' aria-current="step"' if i == current else ""
        items.append(f'<li class="{state}"{current_attr}>{i}. {name}</li>')
    return f'<ol class="cc-crumbs" aria-label="Progress">{"".join(items)}</ol>'


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
    with st.container(key="card_summary"):
        html_block(
            panel_title("Batch summary", f"Processed in {run['elapsed']:.1f} s")
            + '<div class="cc-stats" style="margin-top:14px">'
            + theme.stat("Routed", f"{s['routed']:,}", f"of {run['total']:,} rows")
            + theme.stat("Auto-routed", f"{s['auto']:,}", auto_pct)
            + theme.stat("Needs review", f"{s['review']:,}", "for a specialist")
            + theme.stat("Teams", f"{len(s['teams'])}", "of 11")
            + '</div>')
        if s["teams"]:
            top = s["teams"][0][1]
            rows = []
            for name, n in s["teams"][:6]:
                w = round(n / top * 100)
                rows.append(
                    f'<div class="cc-bar-row" style="grid-template-columns:'
                    f'minmax(0,240px) minmax(0,1fr) 48px"><span '
                    f'class="cc-bar-name">{esc(name)}</span><div '
                    f'class="cc-track"><div class="cc-fill" style="width:{w}%">'
                    f'</div></div><span class="cc-bar-val">{n:,}</span></div>')
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
            html_block('<div class="cc-ph" style="border:0;padding:0">Results'
                       '</div>')
        with actions:
            st.download_button("Export CSV", bulk.export_csv(results),
                               file_name="claricase_routed.csv",
                               mime="text/csv", type="primary",
                               width="stretch")
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
            st.info("No rows were routed.")
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
        intro("Bulk upload",
              f"Upload a CSV with one complaint per row (up to "
              f"{bulk.MAX_ROWS:,}). Every row is masked, routed and given a "
              f"tracking ID. Low-confidence rows are marked for review.")
    with link:
        st.download_button("CSV template", bulk.TEMPLATE_CSV,
                           file_name="claricase_template.csv", mime="text/csv",
                           width="stretch")

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
            html_block(panel_title("File"))
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
                    st.error(str(exc))
                    df = None
            if df is not None:
                html_block(f'<div class="cc-muted"><span class="cc-mono" '
                           f'style="color:{theme.INK}">{esc(upload.name)}</span>'
                           f' · {len(df):,} rows · {len(df.columns)} columns'
                           f'</div>')
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
                             width="stretch"):
                    out = run_bulk(df, text_col, ref_col)
                    out["sig"] = sig
                    st.session_state["bulk_run"] = out
                    st.rerun()
                if run is not None:
                    if run["skipped"]:
                        n = len(run["skipped"])
                        st.warning(f"{n:,} row{'s' if n != 1 else ''} skipped: "
                                   f"fewer than {MIN_WORDS} words.")
                    if run["not_saved"]:
                        st.error(f"{run['not_saved']:,} complaints couldn't be "
                                 f"saved and have no tracking ID. Route the "
                                 f"file again to retry.")

    with right:
        if run is None:
            with st.container(key="card_bulk_empty"):
                html_block(panel_title("Batch summary")
                           + '<div class="cc-empty">No file routed yet. A '
                             'summary by team, every routed row with its '
                             'tracking ID, and a CSV export will appear '
                             'here.</div>')
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
    done, todo = theme.ACCENT, theme.LINE
    route = found.get("route")
    team = esc(found["team_name"])
    conf = found.get("confidence")
    conf_txt = f"{conf:.0%} confidence" if conf is not None else ""
    steps = [("Received", submitted, done),
             ("Personal details masked", "Before saving", done)]
    if route == "auto":
        steps += [("Auto-routed", conf_txt, done),
                  (f"With the {team}", "Awaiting the team's response", done)]
    elif route == "review" and found.get("status") == "Assigned":
        reviewed = found.get("reviewed_at")
        when = (f"{datetime.fromisoformat(str(reviewed)):%b %d, %Y, %H:%M} UTC"
                if reviewed else "")
        steps += [("Sent for specialist review", conf_txt, done),
                  ("Reviewed by a specialist", when, done),
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
            theme.ACCENT_SOFT if steps[i + 1][2] == done else theme.LINE)
        items.append(
            f'<li><div class="cc-tl-rail"><span class="cc-tl-dot" '
            f'style="background:{color}"></span><span class="cc-tl-line" '
            f'style="background:{line}"></span></div><div class="cc-tl-body">'
            f'<div class="cc-tl-title">{title}</div><div class="cc-muted">'
            f'{esc(sub)}</div></div></li>')
    return f'<ol class="cc-tl" aria-label="Timeline">{"".join(items)}</ol>'


def render_track_tab() -> None:
    intro("Track a complaint",
          "Enter the tracking ID shown after submission or in the bulk export.")
    form_col, _ = st.columns([1.2, 1])
    with form_col, st.container(key="card_track_form"):
        with st.form("track", border=False):
            c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
            with c1:
                st.text_input("Tracking ID", key="tracking_input",
                              placeholder="CC-XXXXXX",
                              label_visibility="collapsed")
            with c2:
                st.form_submit_button("Look up", type="primary",
                                      on_click=handle_track, width="stretch")

    tracked = st.session_state.get("track_result")
    if tracked is None:
        return
    if tracked["kind"] == "error":
        st.error("We couldn't look up this complaint. Please try again in a "
                 "moment.")
        return
    if tracked["kind"] == "not_found":
        st.error(f"No complaint found with tracking ID {tracked['id']}.")
        return

    found = tracked["row"]
    submitted_at = datetime.fromisoformat(found["submitted_at"])
    submitted = f"{submitted_at:%b %d, %Y, %H:%M} UTC"
    with st.container(key="card_track_result"):
        html_block(f'<div class="cc-ph"><span class="cc-id">'
                   f'{esc(found["tracking_id"])}</span><span>Submitted '
                   f'{submitted}</span></div>')
        t_col, d_col = st.columns([1, 1.4], gap="large")
        with t_col:
            html_block('<div class="cc-label">History</div>'
                       + timeline(found, submitted))
        with d_col:
            team_label = ("Most likely team"
                          if found.get("status") == "Under review" else "Team")
            rows = [("Status", status_mark(found["status"])),
                    (team_label, f'<b>{esc(found["team_name"])}</b>')]
            if found.get("issue_label"):
                rows.append(("Issue", esc(issue_name(found["issue_label"]))))
            items = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
            html_block(f'<dl class="cc-kv" style="margin-top:14px">{items}</dl>')
            stored = found.get("explanation")
            summary = stored.get("consumer_summary") \
                if isinstance(stored, dict) else None
            html_block('<div class="cc-label">Why this team</div>')
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


html_block(theme.top_bar())

single_tab, bulk_tab, track_tab = st.tabs(
    ["Single complaint", "Bulk upload", "Track"],
    key="active_tab", on_change=reset_on_tab_change)

with single_tab:
    render_single_tab()
with bulk_tab:
    render_bulk_tab()
with track_tab:
    render_track_tab()

html_block('<div class="cc-footer"><span>Clear-cut complaints are routed '
           'automatically; the rest go to a specialist for review.</span>'
           f'<span>{"" if store.persistent else "Local mode: complaints are "
                    "saved on this machine only · "}ClariCase</span></div>')

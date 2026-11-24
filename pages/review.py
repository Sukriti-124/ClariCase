"""ClariCase review queue: assign "needs review" complaints to a team.

Served at /review, behind the same admin sign-in as /admin. Complaints the
model routed below the confidence threshold wait here until an admin reads
them and picks the team. The model's original choice is kept alongside the
admin's (model_team_id / model_team_name).
"""

from datetime import datetime, timezone

import streamlit as st

from src.app import eda, theme
from src.app.admin_common import (check_access, get_store, html_block, load,
                                  nav, nav_slot)
from src.app.bulk import issue_name
from src.app.highlight import highlight
from src.app.storage import REVIEW_NOTE_MAX
from src.app.theme import esc, stat

st.set_page_config(page_title="ClariCase · Review", page_icon=":material/fact_check:",
                   layout="wide")
st.markdown(theme.css(band_px=262), unsafe_allow_html=True)

QUEUE_SHOWN = 200
RECENT_SHOWN = 50
store = get_store()


@st.cache_data
def team_names() -> dict[str, str]:
    return eda.team_names()


def table_key() -> str:
    # A new key resets the table's row selection after an assignment.
    return f"queue_{st.session_state.get('queue_version', 0)}"


def assign(tracking_id: str, team_id: str, team_name: str, note: str) -> None:
    try:
        store.assign(tracking_id, team_id, team_name, note)
    except Exception:
        st.session_state["assign_msg"] = ("error", "We couldn't save that "
                                          "assignment. Please try again.")
        return
    load.clear()
    st.session_state["queue_version"] = st.session_state.get("queue_version", 0) + 1
    st.session_state["assign_msg"] = (
        "ok", f"{tracking_id} assigned to the {team_name}.")


# ---------------------------------------------------------------- page
html_block(theme.top_bar("Admin"))
nav_area = nav_slot()

if not check_access(store):
    st.stop()

try:
    data = load()
except Exception:
    st.error("We couldn't load complaints from storage. Please try again.")
    st.stop()

waiting_all = eda.pending_review(data)
nav(nav_area, len(waiting_all))

head, refresh = st.columns([4, 1], vertical_alignment="bottom")
with head:
    html_block('<div class="cc-h1">Review queue</div>'
               '<div class="cc-sub">Complaints the model wasn\'t sure about. '
               'Read each one, pick the team, and assign it. The customer sees '
               'the new team on the Track tab.</div>')
with refresh:
    if st.button("Refresh", width="stretch"):
        load.clear()
        st.rerun()

msg = st.session_state.pop("assign_msg", None)
if msg:
    (st.toast if msg[0] == "ok" else st.error)(msg[1])

names = team_names()
assigned = (data[data["status"] == "Assigned"] if not data.empty
            else data)
today = datetime.now(timezone.utc).date()
changed = (assigned["team_id"] != assigned["model_team_id"]).sum() \
    if not assigned.empty else 0
assigned_today = (assigned["reviewed_at"].dt.date == today).sum() \
    if not assigned.empty else 0

with st.container(key="card_review_stats"):
    html_block(
        '<div class="cc-stats">'
        + stat("Waiting", f"{len(waiting_all):,}", "to assign")
        + stat("Assigned today", f"{assigned_today:,}", f"{today:%b %d, %Y}")
        + stat("Assigned in total", f"{len(assigned):,}", "by an admin")
        + stat("Team changed", f"{changed / len(assigned):.0%}"
               if len(assigned) else "—", "admin picked a different team")
        + '</div>')

left, right = st.columns([1.25, 1], gap="medium")

# ---- the queue
with left, st.container(key="card_review_queue"):
    html_block('<div class="cc-ph">Waiting for a team</div>'
               '<div class="cc-muted" style="margin-top:8px">Least confident '
               'first. Select a row to review it.</div>')
    if waiting_all.empty:
        st.caption("Nothing waiting. Every flagged complaint has a team.")
        selected = None
    else:
        model_teams = sorted(waiting_all["team_name"].unique())
        pick = st.multiselect("Model's team", model_teams,
                              placeholder="All teams",
                              label_visibility="collapsed")
        queue = waiting_all[waiting_all["team_name"].isin(pick)] if pick \
            else waiting_all
        queue = queue.head(QUEUE_SHOWN).reset_index(drop=True)
        event = st.dataframe(
            queue.assign(confidence=(queue["confidence"] * 100).round(0))[
                ["tracking_id", "confidence", "team_name", "issue",
                 "complaint_text"]],
            key=table_key(), on_select="rerun", selection_mode="single-row",
            hide_index=True, width="stretch", height=430,
            column_config={
                "tracking_id": st.column_config.TextColumn("Tracking ID"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%d%%"),
                "team_name": st.column_config.TextColumn("Model's team"),
                "issue": st.column_config.TextColumn("Issue"),
                "complaint_text": st.column_config.TextColumn(
                    "Complaint", width="large"),
            })
        rows = event.selection.rows
        selected = queue.iloc[rows[0] if rows else 0] if len(queue) else None
        st.caption(f"Showing {len(queue):,} of {len(waiting_all):,}"
                   + ("" if rows or selected is None
                      else " · the first row is open"))

# ---- the selected complaint
with right, st.container(key="card_review_detail"):
    if selected is None:
        html_block('<div class="cc-ph">Complaint</div>'
                   '<div class="cc-muted" style="margin-top:10px">Select a '
                   'complaint from the queue to review it.</div>')
    else:
        tid = selected["tracking_id"]
        submitted = selected["submitted_at"]
        html_block(
            f'<div class="cc-ph"><span class="cc-id">{esc(tid)}</span>'
            f'<span>Submitted {submitted:%b %d, %Y, %H:%M} UTC</span></div>'
            f'<div class="cc-result-top" style="margin-top:14px">'
            f'<div class="cc-team"><div class="cc-avatar">'
            f'{theme.team_code(selected["team_id"], selected["team_name"])}</div>'
            f'<div><div class="cc-team-label">Model\'s suggestion</div>'
            f'<div class="cc-team-name">{esc(selected["team_name"])}</div>'
            f'<div class="cc-meta"><span class="cc-status cc-s-warn">'
            f'Needs review</span></div></div></div>'
            f'{theme.gauge(float(selected["confidence"]), theme.WARN)}</div>'
            + (f'<dl class="cc-kv" style="margin-top:14px"><dt>Issue</dt>'
               f'<dd>{esc(selected["issue"])}</dd></dl>'
               if selected["issue"] else "")
            + '<div class="cc-label">Complaint</div>'
            f'<div class="cc-excerpt" style="margin-bottom:14px">'
            f'{highlight(selected["complaint_text"], [])}</div>')

        ids = list(names) or [selected["team_id"]]
        label_of = {**names, selected["team_id"]: selected["team_name"]}
        default = ids.index(selected["team_id"]) \
            if selected["team_id"] in ids else 0
        with st.form(f"assign_{tid}", border=False):
            team_id = st.selectbox(
                "Assign to", ids, index=default,
                format_func=lambda t: label_of.get(t, t)
                + ("  (model's suggestion)" if t == selected["team_id"] else ""))
            note = st.text_area("Note (optional)", key=f"note_{tid}",
                                max_chars=REVIEW_NOTE_MAX, height=80,
                                placeholder="Why this team, for the record")
            go = st.form_submit_button("Assign to team", type="primary",
                                       icon=":material/check:",
                                       width="stretch")
        if go:
            assign(tid, team_id, label_of.get(team_id, team_id), note)
            st.rerun()
        st.caption("Assigning marks the complaint as Assigned and keeps the "
                   "model's original suggestion on record.")

# ---- recently assigned
with st.container(key="card_review_recent"):
    html_block('<div class="cc-ph">Recently assigned</div>')
    if assigned.empty:
        st.caption("No assignments yet.")
    else:
        recent = assigned.sort_values("reviewed_at", ascending=False) \
            .head(RECENT_SHOWN)
        recent = recent.assign(
            changed=recent["team_id"] != recent["model_team_id"],
            review_note=recent["review_note"].fillna(""))
        st.dataframe(
            recent[["tracking_id", "reviewed_at", "model_team_name",
                    "team_name", "changed", "review_note"]],
            hide_index=True, width="stretch",
            height=min(38 + 35 * len(recent), 320),
            column_config={
                "tracking_id": st.column_config.TextColumn("Tracking ID"),
                "reviewed_at": st.column_config.DatetimeColumn(
                    "Assigned", format="MMM D, YYYY HH:mm"),
                "model_team_name": st.column_config.TextColumn("Model's team"),
                "team_name": st.column_config.TextColumn("Assigned to"),
                "changed": st.column_config.CheckboxColumn("Changed"),
                "review_note": st.column_config.TextColumn("Note",
                                                           width="large"),
            })

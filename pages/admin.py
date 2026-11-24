"""ClariCase admin dashboard: what is coming in and how it is being routed.

Served at /admin. It is not linked from the complaint pages (sidebar
navigation is hidden in .streamlit/config.toml). When complaints are stored
in Supabase it requires the password in secrets ([admin] password = "...");
in local mode it opens without one.
"""

import html

import altair as alt
import pandas as pd
import streamlit as st

from src.app import eda, theme
from src.app.admin_common import (CACHE_TTL_S, check_access, get_store,
                                  html_block, load, nav, nav_slot)
from src.app.theme import stat

st.set_page_config(page_title="ClariCase · Admin", page_icon=":material/insights:",
                   layout="wide")

AUTO, REVIEW, UNKNOWN = theme.ACCENT, "#D08A1E", "#B6BEC7"
TRAIN, LIVE = theme.ACCENT_SOFT, theme.ACCENT
ROUTING_SCALE = alt.Scale(domain=["Auto-routed", "Needs review", "Unknown"],
                          range=[AUTO, REVIEW, UNKNOWN])

st.markdown(theme.css(band_px=262), unsafe_allow_html=True)


store = get_store()


# ---------------------------------------------------------------- data
@st.cache_data
def reference():
    return (eda.training_team_counts(), eda.team_names(),
            eda.routing_threshold())


def card(key: str, title: str, sub: str = ""):
    box = st.container(key=f"card_{key}")
    box.markdown(f'<div class="cc-ph">{title}</div>'
                 + (f'<div class="cc-muted" style="margin-top:8px">{sub}</div>'
                    if sub else ""), unsafe_allow_html=True)
    return box


def chart(c: alt.Chart) -> None:
    st.altair_chart(c.configure(font="IBM Plex Sans", background="#FFFFFF")
                    .configure_view(strokeWidth=0)
                    .configure_axis(labelFontSize=12, titleFontSize=12,
                                    titleFontWeight=500, labelColor=theme.MUTED,
                                    titleColor=theme.MUTED, gridColor="#EAEEF2",
                                    domainColor=theme.LINE,
                                    tickColor=theme.LINE)
                    .configure_legend(orient="top", title=None,
                                      labelFontSize=12),
                    width="stretch")


# ---------------------------------------------------------------- page
html_block(theme.top_bar("Admin"))
nav_area = nav_slot()

if not check_access(store):
    st.stop()

head, refresh = st.columns([4, 1], vertical_alignment="bottom")
with head:
    html_block('<div class="cc-h1">Complaint analytics</div>'
               '<div class="cc-sub">What is coming in, where it is routed, and '
               'how confident the model is. Complaint text is shown masked.'
               '</div>')
with refresh:
    if st.button("Refresh data", width="stretch"):
        load.clear()
        st.rerun()

try:
    data = load()
except Exception:
    st.error("We couldn't load complaints from storage. Please try again.")
    st.stop()

nav(nav_area, eda.kpis(data)["pending"] if not data.empty else None)

if data.empty:
    st.info("No complaints yet. Submit one, or upload a CSV on the Bulk upload "
            "tab, and it will show up here.")
    st.stop()

train_counts, names, threshold = reference()

# ---- filters
first = data["submitted_at"].min().date()
last = data["submitted_at"].max().date()
filters_card = st.container(key="card_filters")
f1, f2, f3, f4 = filters_card.columns([1.3, 1.7, 1.8, 0.9],
                                      vertical_alignment="bottom")
with f1:
    picked = st.date_input("Date range", value=(first, last),
                           min_value=first, max_value=last)
with f2:
    teams = st.multiselect("Teams", sorted(data["team_name"].unique()),
                           placeholder="All teams")
with f3:
    routing = st.segmented_control(
        "Routing", ["All", "Auto-routed", "Needs review"], default="All")
with f4:
    grain = st.selectbox("Group by", ["Day", "Week", "Month"])

start, end = (picked if isinstance(picked, tuple) and len(picked) == 2
              else (first, last))
dates = data["submitted_at"].dt.date
df = data[(dates >= start) & (dates <= end)]
if teams:
    df = df[df["team_name"].isin(teams)]
if routing and routing != "All":
    df = df[df["routing"] == routing]

if df.empty:
    st.warning("No complaints match these filters.")
    st.stop()

# ---- headline numbers
k = eda.kpis(df)
cov = eda.explanation_coverage(df)
filters_card.markdown(
    '<div class="cc-stats" style="margin-top:10px;padding-top:14px;'
    'border-top:1px solid #EDF1F4">'
    + stat("Complaints", f"{k['total']:,}", f"{start:%b %d} – {end:%b %d, %Y}")
    + stat("Last 7 days", f"{k['last_7_days']:,}", "submitted")
    + stat("Auto-routed", f"{k['auto_rate']:.0%}", f"{k['auto']:,} complaints")
    + stat("Needs review", f"{k['review']:,}",
           f"{k['pending']:,} still to assign")
    + stat("Avg confidence", f"{k['avg_confidence']:.0%}",
           f"threshold {threshold:.0%}" if threshold else "top team")
    + stat("Explained", f"{cov['with_explanation'] / cov['total']:.0%}",
           f"{cov['ai_written']:,} AI-written · {cov['standard']:,} standard")
    + '</div>', unsafe_allow_html=True)

freq = {"Day": "D", "Week": "W", "Month": "M"}[grain]

# ---- volume and teams
left, right = st.columns(2, gap="medium")
with left:
    with card("volume", "Complaints over time",
              f"Per {grain.lower()}, split by routing outcome"):
        v = eda.volume(df, freq)
        chart(alt.Chart(v).mark_bar().encode(
            x=alt.X("period:T", title=None),
            y=alt.Y("complaints:Q", title="Complaints", stack=True),
            color=alt.Color("routing:N", scale=ROUTING_SCALE),
            tooltip=[alt.Tooltip("period:T", title=grain),
                     "routing:N", "complaints:Q"]).properties(height=300))
with right:
    with card("teams", "Complaints by team",
              "Auto-routed vs sent for review"):
        t = eda.by_team(df)
        order = list(dict.fromkeys(t["team_name"]))
        chart(alt.Chart(t).mark_bar().encode(
            y=alt.Y("team_name:N", sort=order, title=None,
                    axis=alt.Axis(labelLimit=220)),
            x=alt.X("complaints:Q", title="Complaints", stack=True),
            color=alt.Color("routing:N", scale=ROUTING_SCALE),
            tooltip=["team_name:N", "routing:N", "complaints:Q"])
            .properties(height=300))

# ---- confidence and issues
left, right = st.columns(2, gap="medium")
with left:
    with card("confidence", "Model confidence",
              "How sure the model is about the chosen team. Complaints right "
              "of the line are routed automatically."):
        hist = alt.Chart(df[["confidence"]]).mark_bar(
            color=AUTO, opacity=0.85).encode(
            x=alt.X("confidence:Q", bin=alt.Bin(extent=[0, 1], step=0.05),
                    title="Confidence", axis=alt.Axis(format="%")),
            y=alt.Y("count():Q", title="Complaints"),
            tooltip=[alt.Tooltip("count():Q", title="Complaints")])
        layers = [hist]
        if threshold:
            rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(
                color=REVIEW, strokeWidth=2, strokeDash=[5, 4]).encode(x="t:Q")
            label = alt.Chart(pd.DataFrame({"t": [threshold],
                                            "txt": [f"threshold {threshold:.0%}"]})
                              ).mark_text(align="right", dx=-6, dy=-130,
                                          color=REVIEW, fontSize=12).encode(
                x="t:Q", text="txt:N")
            layers += [rule, label]
        chart(alt.layer(*layers).properties(height=300))
with right:
    with card("issues", "Top issues", "Most common issues across the selection"):
        iss = eda.top_issues(df, n=10)
        chart(alt.Chart(iss).mark_bar(color=AUTO).encode(
            y=alt.Y("issue:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=280)),
            x=alt.X("complaints:Q", title="Complaints"),
            tooltip=["issue:N", "complaints:Q"]).properties(height=300))

# ---- team mix and text
left, right = st.columns(2, gap="medium")
with left:
    with card("mix", "Team mix: live vs training",
              "A big gap means incoming complaints differ from what the model "
              "learned from, so routing quality may shift."):
        if train_counts:
            mix = eda.team_mix(df, train_counts, names)
            chart(alt.Chart(mix).mark_bar().encode(
                y=alt.Y("team_name:N", title=None, sort=sorted(names.values()),
                        axis=alt.Axis(labelLimit=220)),
                yOffset="source:N",
                x=alt.X("share:Q", title="Share of complaints",
                        axis=alt.Axis(format="%")),
                color=alt.Color("source:N", scale=alt.Scale(
                    domain=["Training data", "Live complaints"],
                    range=[TRAIN, LIVE])),
                tooltip=["team_name:N", "source:N",
                         alt.Tooltip("share:Q", format=".1%")])
                .properties(height=360))
        else:
            st.caption("Training mix unavailable.")
with right:
    with card("text", "Complaint text",
              "Length in words, and the most frequent phrases (masked details "
              "removed)"):
        words = df[["words"]].assign(words=df["words"].clip(upper=600))
        chart(alt.Chart(words).mark_bar(color=TRAIN).encode(
            x=alt.X("words:Q", bin=alt.Bin(maxbins=30),
                    title="Words per complaint (600+ grouped)"),
            y=alt.Y("count():Q", title="Complaints"),
            tooltip=[alt.Tooltip("count():Q", title="Complaints")])
            .properties(height=170))
        st.caption(f"Median {int(df['words'].median()):,} words · "
                   f"longest {int(df['words'].max()):,}")
        phrases = eda.top_phrases(df["complaint_text"], n=15)
        if phrases:
            html_block("".join(f'<span class="cc-tag">{html.escape(p)}<b>{n:,}'
                               f'</b></span>' for p, n in phrases))

# ---- review queue
with card("queue", "Review queue",
          "Complaints below the routing threshold that nobody has assigned "
          "yet, least confident first"):
    st.page_link("pages/review.py", label="Review and assign",
                 icon=":material/arrow_forward:")
    q = eda.review_queue(df)
    if q.empty:
        st.caption("Nothing waiting for review in this selection.")
    else:
        st.dataframe(
            q.assign(confidence=(q["confidence"] * 100).round(0)),
            hide_index=True, width="stretch", height=320,
            column_config={
                "tracking_id": st.column_config.TextColumn("Tracking ID"),
                "submitted_at": st.column_config.DatetimeColumn(
                    "Submitted", format="MMM D, YYYY HH:mm"),
                "team_name": st.column_config.TextColumn("Most likely team"),
                "issue": st.column_config.TextColumn("Issue"),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%d%%"),
                "complaint_text": st.column_config.TextColumn(
                    "Complaint", width="large"),
            })
        st.caption(f"Showing {len(q):,} of {k['pending']:,}.")

st.download_button("Download filtered data (CSV)",
                   eda.export_frame(df).to_csv(index=False).encode("utf-8"),
                   file_name="claricase_complaints.csv", mime="text/csv")
st.caption(f"Data refreshes every {CACHE_TTL_S} seconds · up to "
           f"{len(data):,} most recent complaints loaded.")

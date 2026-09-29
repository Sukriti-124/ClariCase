"""ClariCase complaint intake.

Run: streamlit run streamlit_app.py
"""

from datetime import datetime

import streamlit as st

from src.app.storage import (LocalStore, SupabaseStore, build_record,
                             normalise_tracking_id)
from src.baseline_model_2.predict import load_model, predict

MIN_WORDS = 5

st.set_page_config(page_title="ClariCase", layout="centered")


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

st.title("ClariCase")
st.write("Tell us about your problem with a financial product or service. "
         "We'll send it straight to the team that handles it.")
if not store.persistent:
    st.info("Local mode: complaints are saved to data/app/complaints.db on "
            "this machine. Add Supabase secrets to store them online.")

def handle_submit() -> None:
    # Runs before the rerun, so the text box can be cleared on success.
    # On a warning or error the text is kept so the user can fix and resend.
    text = st.session_state["complaint_text"].strip()
    if len(text.split()) < MIN_WORDS:
        st.session_state["submit_result"] = {"kind": "too_short"}
        return
    record = build_record(text, predict(text, model=get_model()))
    try:
        store.add(record)
    except Exception:
        st.session_state["submit_result"] = {"kind": "error"}
        return
    st.session_state["submit_result"] = {"kind": "sent", "record": record}
    st.session_state["complaint_text"] = ""


submit_tab, track_tab = st.tabs(["Submit a complaint", "Track my complaint"])

with submit_tab:
    with st.form("complaint"):
        st.text_area("What happened?", key="complaint_text", height=220,
                     placeholder="Describe the problem in your own words: "
                                 "what happened, when, and what you'd like "
                                 "done about it.")
        st.form_submit_button("Submit complaint", type="primary",
                              on_click=handle_submit)

    # Kept until the next submit, so the tracking ID stays visible.
    result = st.session_state.get("submit_result")
    if result is None:
        pass
    elif result["kind"] == "too_short":
        st.warning(f"Please describe your complaint in at least "
                   f"{MIN_WORDS} words so we can route it correctly.")
    elif result["kind"] == "error":
        st.error("We couldn't save your complaint. Please try again "
                 "in a moment.")
    else:
        record = result["record"]
        st.success(f"Your complaint has been sent to the "
                   f"**{record['team_name']}**.")
        st.write("Your tracking ID:")
        st.code(record["tracking_id"], language=None)
        st.caption("Save this ID. You can use it on the "
                   "**Track my complaint** tab to check on your complaint.")

with track_tab:
    with st.form("track"):
        raw_id = st.text_input("Tracking ID", placeholder="CC-XXXXXX")
        looked_up = st.form_submit_button("Check status")

    if looked_up and raw_id.strip():
        tracking_id = normalise_tracking_id(raw_id)
        try:
            found = store.get(tracking_id)
        except Exception:
            st.error("We couldn't look up your complaint. Please try again "
                     "in a moment.")
        else:
            if found is None:
                st.error(f"No complaint found with tracking ID {tracking_id}.")
            else:
                submitted_at = datetime.fromisoformat(found["submitted_at"])
                st.markdown(f"**Status:** {found['status']}  \n"
                            f"**Sent to:** {found['team_name']}")
                st.caption(f"Submitted {submitted_at:%B %d, %Y at %H:%M} UTC")
                with st.expander("Your complaint"):
                    st.write(found["complaint_text"])

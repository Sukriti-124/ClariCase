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

submit_tab, track_tab = st.tabs(["Submit a complaint", "Track my complaint"])

with submit_tab:
    with st.form("complaint"):
        text = st.text_area("What happened?", height=220,
                            placeholder="Describe the problem in your own "
                                        "words: what happened, when, and "
                                        "what you'd like done about it.")
        st.caption("Please don't include account numbers, Social Security "
                   "numbers, or other personal identifiers.")
        submitted = st.form_submit_button("Submit complaint", type="primary")

    if submitted:
        text = text.strip()
        if len(text.split()) < MIN_WORDS:
            st.warning(f"Please describe your complaint in at least "
                       f"{MIN_WORDS} words so we can route it correctly.")
        else:
            record = build_record(text, predict(text, model=get_model()))
            try:
                store.add(record)
            except Exception:
                st.error("We couldn't save your complaint. Please try again "
                         "in a moment.")
            else:
                st.success(f"Your complaint has been sent to the "
                           f"**{record['team_name']}**.")
                st.write("Your tracking ID:")
                st.code(record["tracking_id"], language=None)
                st.caption("Save this ID. You can use it on the "
                           "**Track my complaint** tab to check on your "
                           "complaint.")

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

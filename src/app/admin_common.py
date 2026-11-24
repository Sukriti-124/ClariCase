"""Shared pieces of the admin pages (/admin and /review): storage, sign-in
and the navigation between them. Signing in on one page signs in on both,
because the flag lives in st.session_state."""

from __future__ import annotations

import hmac
import os

import streamlit as st

from src.app import eda
from src.app.storage import make_store

CACHE_TTL_S = 60

PAGES = (("pages/admin.py", "Analytics"), ("pages/review.py", "Review queue"))


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_resource
def get_store():
    return make_store(st.secrets)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner="Loading complaints…")
def load():
    """All recent complaints as a frame. Shared by both admin pages, so
    load.clear() after an assignment refreshes both."""
    return eda.to_frame(get_store().fetch_recent())


def admin_password() -> str | None:
    pw = os.environ.get("CLARICASE_ADMIN_PASSWORD")
    if not pw:
        try:
            pw = st.secrets["admin"]["password"]
        except Exception:
            pw = None
    return pw or None


def check_access(store) -> bool:
    if st.session_state.get("admin_ok"):
        return True
    password = admin_password()
    if password is None:
        if not store.persistent:
            return True                 # local mode: data stays on this machine
        st.error("The admin pages are turned off: no admin password is "
                 "configured. Add `[admin] password = \"...\"` to the app's "
                 "secrets.")
        return False
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid, st.container(key="card_login"):
        html_block('<div class="cc-ph">Admin sign-in</div><div '
                   'class="cc-muted" style="margin-top:10px">Enter the admin '
                   'password to view complaints.</div>')
        with st.form("admin_login", border=False):
            entered = st.text_input("Password", type="password")
            ok = st.form_submit_button("Sign in", type="primary",
                                       width="stretch")
        if ok:
            if hmac.compare_digest(entered.encode(), password.encode()):
                st.session_state["admin_ok"] = True
                st.rerun()
            st.error("Incorrect password.")
    return False


def nav_slot():
    """Reserve the page-link row in the brand bar; call right after top_bar
    and fill it with nav() once the data (and pending count) is loaded."""
    return st.container(key="cc_nav", horizontal=True, gap="small",
                        horizontal_alignment="right")


def nav(slot, pending: int | None = None) -> None:
    with slot:
        for page, label in PAGES:
            if page.endswith("review.py") and pending:
                label = f"{label} · {pending:,}"
            st.page_link(page, label=label)

# frontend/app.py
import os, requests
import streamlit as st
from dotenv import load_dotenv
from datetime import time as dtime

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Astro Assistant", page_icon="✨", layout="centered")

def api_post(path, json=None, token=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.post(f"{API_URL}{path}", json=json, headers=h, timeout=20)
    r.raise_for_status()
    return r.json()

def api_get(path, token=None):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(f"{API_URL}{path}", headers=h, timeout=20)
    r.raise_for_status()
    return r.json()

def ensure_state():
    for k,v in {
        "token": None, "username": None, "have_charts": False,
        "messages": []
    }.items():
        if k not in st.session_state: st.session_state[k]=v
ensure_state()

st.title("✨ Astro Assistant")

if not st.session_state["token"]:
    tab_login, tab_signup = st.tabs(["Login", "Create account"])

    with tab_login:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            s = st.form_submit_button("Login")
        if s:
            try:
                data = api_post("/auth/login", {"username": u, "password": p})
                st.session_state["token"] = data["access_token"]
                st.session_state["username"] = u
                st.success("Logged in.")
            except requests.HTTPError as e:
                st.error(e.response.text)

    with tab_signup:
        with st.form("signup"):
            u = st.text_input("Username", key="su_u")
            p = st.text_input("Password", type="password", key="su_p")
            s = st.form_submit_button("Create account")
        if s:
            try:
                data = api_post("/auth/signup", {"username": u, "password": p})
                st.session_state["token"] = data["access_token"]
                st.session_state["username"] = u
                st.success("Account created & logged in.")
            except requests.HTTPError as e:
                st.error(e.response.text)
    st.stop()

# Logged-in UI
st.sidebar.write(f"Logged in as **{st.session_state['username']}**")
if st.sidebar.button("Log out"):
    for k in ["token","username","have_charts","messages"]:
        st.session_state[k]=None if k in ["token","username"] else []
    st.experimental_rerun()

# Check chart status
status = api_get("/charts/static", token=st.session_state["token"])
st.session_state["have_charts"] = status.get("ready", False)

if not st.session_state["have_charts"]:
    st.info("No birth charts on file. Please provide your birth details and current info.")
    with st.form("first_time"):
        name = st.text_input("Full name")
        birth_place = st.text_input("Birth place (city, country)")
        birth_date = st.date_input("Birth date")
        birth_time = st.time_input("Birth time", value=dtime(12,0))
        current_place = st.text_input("Current location (city, country)")
        current_local_dt = st.text_input("Current local date & time (ISO 8601, optional)")

        submitted = st.form_submit_button("Save & Generate")
    if submitted:
        try:
            payload = {
                "name": name,
                "birth_place": birth_place,
                "birth_date": birth_date.isoformat(),
                "birth_time": birth_time.strftime("%H:%M"),
                "current_place": current_place,
                "current_local_dt": current_local_dt or None
            }
            api_post("/profile/init", payload, token=st.session_state["token"])
            st.success("Saved. Chart computation started in background.")
        except requests.HTTPError as e:
            st.error(e.response.text)
else:
    st.success("Birth charts are on file. Provide your current info if you wish.")
    with st.form("returning"):
        current_place = st.text_input("Current location (city, country)")
        current_local_dt = st.text_input("Current local date & time (ISO 8601, optional)")
        sub = st.form_submit_button("Save current info")
    if sub:
        try:
            api_post("/session/update", {"current_place": current_place, "current_local_dt": current_local_dt or None},
                     token=st.session_state["token"])
            st.success("Saved.")
        except requests.HTTPError as e:
            st.error(e.response.text)

# Poll chart readiness (optional button)
col1, col2 = st.columns(2)
if col1.button("Refresh chart status"):
    status = api_get("/charts/static", token=st.session_state["token"])
    st.session_state["have_charts"] = status.get("ready", False)
if status.get("ready") and status.get("charts"):
    with st.expander("View stored static charts (JSON)"):
        st.json(status["charts"])

st.markdown("---")
st.header("Astrologer Chat")
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_msg = st.chat_input("Ask me anything about your chart or timing...")
if user_msg:
    st.session_state["messages"].append({"role":"user","content":user_msg})
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = api_post("/chat", {"messages": st.session_state["messages"]},
                                token=st.session_state["token"])
                reply = resp["reply"]
            except requests.HTTPError as e:
                reply = f"(backend error) {e.response.text}"
            st.markdown(reply)
    st.session_state["messages"].append({"role":"assistant","content":reply})

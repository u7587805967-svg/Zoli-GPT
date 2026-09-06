import streamlit as st

_DEFAULTS = {
    "messages": [],
    "user_logged_in": False,
    "user_data": None,
    "selected_model": "llama-3.3-70b",
    "chat_history": []
}

for key, value in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value
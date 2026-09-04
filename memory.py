import streamlit as st

def init_memory():
    """Inicializálja a memóriát, ha még nem létezik."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

def add_message(role: str, content: str):
    """Új üzenetet ment el a memóriába (role: 'user' vagy 'assistant')."""
    init_memory()
    st.session_state.messages.append({"role": role, "content": content})

def get_messages():
    """Visszaadja a teljes beszélgetési előzményt az LLM számára."""
    init_memory()
    return st.session_state.messages

def render_history():
    """Kirajzolja az eddigi üzeneteket a Streamlit felületre."""
    init_memory()
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def clear_memory():
    """Kiüríti a teljes memóriát."""
    st.session_state.messages = []
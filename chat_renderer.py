import streamlit as st
from image_handler import render_smart_content

# 1. Elmentjük az eredeti Streamlit markdown függvényt egy biztonságos változóba
_real_markdown = st.markdown

def smart_markdown(body, *args, **kwargs):
    """
    Ez a függvény váltja fel a st.markdown-t. 
    Ha szöveget kap, átadja az image_handler-nek a képek feldolgozására.
    """
    if isinstance(body, str):
        render_smart_content(body)
    else:
        _real_markdown(body, *args, **kwargs)

def setup_image_renderer():
    """Beállítja a felülbírálást biztonságosan."""
    st.markdown = smart_markdown
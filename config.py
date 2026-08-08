import streamlit as st
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    DB_FILE: str = "zoli_gpt_local.db"
    ADMIN_USERNAME: str = st.secrets.get("ADMIN_USERNAME", "default_admin_fallback") 
    TIMEZONE: str = "Europe/Budapest"
    PIXABAY_API_KEY: str = st.secrets.get("PIXABAY_API_KEY", "") 
    MAX_HISTORY_CHARS: int = 4000
    RAG_SIMI_THRESHOLD: float = 0.15
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 300
    HUNGARIAN_STOPWORDS = frozenset([
        "a", "az", "egy", "be", "ki", "le", "fel", "meg", "el", "at", "es", "hogy", 
        "nem", "sem", "vagy", "is", "csak", "mint", "volt", "lesz", "cikk", "alatt"
    ])

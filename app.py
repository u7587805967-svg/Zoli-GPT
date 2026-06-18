import streamlit as st
import os
import datetime
import httpx
import re
import io
import asyncio
import time
import base64
import pytz
import sqlite3
import urllib.parse
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from contextlib import contextmanager
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
from duckduckgo_search import DDGS
from PIL import Image
from pypdf import PdfReader
import docx
from docx import Document
from groq import Groq

# --- ⚙️ 1. GLOBÁLIS SZEMÉLYES KONFIGURÁCIÓ ---
@dataclass(frozen=True)
class AppConfig:
    DB_FILE: str = "zoli_gpt_local.db"
    ADMIN_USERNAME: str = "beni-252514569690023"  # <--- Frissített, egyedi admin azonosító
    TIMEZONE: str = "Europe/Budapest"
    PIXABAY_API_KEY: str = "56302786-02377baa984d7697c0b5cc4e1"
    MAX_HISTORY_CHARS: int = 4000
    RAG_SIMI_THRESHOLD: float = 0.25
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 300
    HUNGARIAN_STOPWORDS = frozenset([
        "a", "az", "egy", "be", "ki", "le", "fel", "meg", "el", "at", "es", "hogy", 
        "nem", "sem", "vagy", "is", "csak", "mint", "volt", "lesz", "cikk", "alatt"
    ])

st.set_page_config(page_title="Zoli GPT ", page_icon="🚭", layout="centered")

# --- 📱 URL PARAMÉTER ALAPÚ FELHASZNÁLÓ KEZELÉS ---
query_params = st.query_params
url_user = query_params.get("user", "vendeg").lower().strip()

# --- 🎨 UI / UX PRÉMIUM STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0d0f16; color: #f1f5f9; }
    section[data-testid="stSidebar"] { background-color: #121520 !important; border-right: 1px solid #1e293b; }
    .agent-status {
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 16px;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .status-rag { background-color: rgba(99, 102, 241, 0.12); border: 1px solid #6366f1; color: #818cf8; }
    .status-web { background-color: rgba(6, 182, 212, 0.12); border: 1px solid #06b6d4; color: #22d3ee; }
    .status-gen { background-color: rgba(16, 185, 129, 0.12); border: 1px solid #10b981; color: #34d399; }
    .stButton>button, .stDownloadButton>button { border-radius: 6px !important; font-weight: 500; }
    .action-row { display: flex; gap: 8px; margin-top: 5px; flex-wrap: wrap; align-items: center; }
    .monitor-card { background-color: #161a27; border: 1px solid #242b3d; padding: 15px; border-radius: 8px; margin-bottom: 10px; }
    .tag-style { background-color: #1e293b; color: #94a3b8; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; display: inline-block; margin-right: 5px; margin-top: 5px; border: 1px solid #334155; }
    .meta-metrics { font-size: 11px; color: #64748b; margin-top: 4px; display: block; }
    </style>
""", unsafe_allow_html=True)

st.title("🚭 Zoli GPT")
st.caption(f"Bejelentkezve mint: **{url_user}**")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

# --- 🛠️ 2. ADATBÁZIS INFRASTRUKTÚRA ---
class DatabaseRepository:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_file, check_same_thread=False, timeout=10.0)
        try: yield conn
        finally: conn.close()

    def _init_schema(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, role TEXT, content TEXT, type TEXT, caption TEXT, timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS document_vectors (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, doc_name TEXT, chunk_text TEXT, embedding BLOB, file_size TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS latency_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, duration REAL, timestamp TEXT)''')
            conn.commit()

    def fetch_history(self, username: str) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role, content, type, caption FROM chat_history WHERE username=? ORDER BY id ASC", (username,))
            return [{"role": r[0], "content": r[1], "type": r[2], "caption": r[3]} for r in cursor.fetchall()]

    def log_message(self, username: str, role: str, content: str, msg_type: str = "text", caption: str = ""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO chat_history (username, role, content, type, caption, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                           (username, role, content, msg_type, caption, datetime.datetime.now().isoformat()))
            conn.commit()

    def purge_chat_only(self, username: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE username=?", (username,))
            conn.commit()

    def get_all_users(self) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT username FROM chat_history")
            return [r[0] for r in cursor.fetchall() if r[0]]

    def get_system_stats(self, username: str) -> dict:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chat_history WHERE username=?", (username,))
            h_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(DISTINCT doc_name) FROM document_vectors WHERE username=?", (username,))
            d_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM document_vectors WHERE username=?", (username
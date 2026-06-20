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
    ADMIN_USERNAME: str = "BeNi-252514569690023"  # <--- A te pontos felhasználóneved
    TIMEZONE: str = "Europe/Budapest"
    PIXABAY_API_KEY: str = st.secrets.get("PIXABAY_API_KEY", "56302786-02377baa984d7697c0b5cc4e1")
    MAX_HISTORY_CHARS: int = 2000  # <--- EZT ÍRTAM ÁT 2000-RE A SZIGORÚBB TÖMÖRÍTÉSHEZ
    RAG_SIMI_THRESHOLD: float = 0.25
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 300
    HUNGARIAN_STOPWORDS = frozenset([
        "a", "az", "egy", "be", "ki", "le", "fel", "meg", "el", "at", "es", "hogy", 
        "nem", "sem", "vagy", "is", "csak", "mint", "volt", "lesz", "cikk", "alatt"
    ])

st.set_page_config(page_title="Zoli GPT ", page_icon="🚭", layout="centered")

# --- ⚙️ INICIALIZÁLÁS ÉS BIZTONSÁGI SORREND ---
cfg = AppConfig()

# Biztonsági retesz: minden futáskor alaphelyzetbe állítjuk, ha beragadt volna
if "generating" not in st.session_state:
    st.session_state.generating = False
else:
    st.session_state.generating = False

# --- BEJELENTKEZÉSI ÁLLAPOT INICIALIZÁLÁSA ---
if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None

# --- 📱 URL PARAMÉTER ALAPÚ FELHASZNÁLÓ KEZELÉS (HA NINCS SESSION) ---
if not st.session_state.logged_in_user:
    query_params = st.query_params
    url_user = query_params.get("user", "").lower().strip()
    if url_user:
        st.session_state.logged_in_user = url_user

# --- BEJELENTKEZŐ FELÜLET (CSAK FELHASZNÁLÓNÉV) ---
if not st.session_state.logged_in_user:
    st.markdown("""
        <style>
        .stApp { background-color: #0d0f16; color: #f1f5f9; }
        .login-box { background-color: #121520; padding: 30px; border-radius: 10px; border: 1px solid #1e293b; margin-top: 50px; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🚭 Zoli GPT")
    st.subheader("Bejelentkezés")
    
    with st.container():
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        input_username = st.text_input("Felhasználónév:", placeholder="Írd be a felhasználóneved...")
        if st.button("Belépés", type="primary", use_container_width=True):
            cleaned_input = input_username.lower().strip()
            if cleaned_input:
                st.session_state.logged_in_user = cleaned_input
                st.rerun()
            else:
                st.error("Kérlek, adj meg egy érvényes felhasználónevet!")
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Alapértelmezetten a bejelentkezett felhasználó az aktív chat partner
active_chat_user = st.session_state.logged_in_user

# --- 🔄 MÓDOSÍTÁS: A 'szemelyes' felhasználónak már NINCS admin joga, csak a beni-nek maradt meg ---
is_admin = (active_chat_user == cfg.ADMIN_USERNAME.lower().strip())

# Biztosítjuk, hogy az admin által kiválasztott célszemély adatai töltődjenek be a renderelés előtt
if is_admin and "admin_selected_user" in st.session_state:
    active_chat_user = st.session_state.admin_selected_user

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
st.caption(f"Bejelentkezve mint: **{st.session_state.logged_in_user}**")

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
            cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, role TEXT, content TEXT, type TEXT, caption TEXT, timestamp TEXT, thread_id TEXT DEFAULT 'default')''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS document_vectors (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, doc_name TEXT, chunk_text TEXT, embedding BLOB, file_size TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS latency_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, duration REAL, timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS token_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, tokens INTEGER, cost REAL, timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS system_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT, timestamp TEXT)''')
            try:
                cursor.execute("ALTER TABLE chat_history ADD COLUMN thread_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError: pass
            
            try:
                cursor.execute("ALTER TABLE token_logs ADD COLUMN username TEXT")
            except sqlite3.OperationalError: pass
            try:
                cursor.execute("ALTER TABLE token_logs ADD COLUMN tokens INTEGER")
            except sqlite3.OperationalError: pass
            try:
                cursor.execute("ALTER TABLE token_logs ADD COLUMN cost REAL")
            except sqlite3.OperationalError: pass
            try:
                cursor.execute("ALTER TABLE token_logs ADD COLUMN timestamp TEXT")
            except sqlite3.OperationalError: pass
            
            conn.commit()

    def fetch_history(self, username: str, thread_id: str = "default") -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role, content, type, caption FROM chat_history WHERE username=? AND thread_id=? ORDER BY id ASC", (username, thread_id))
            return [{"role": r[0], "content": r[1], "type": r[2], "caption": r[3]} for r in cursor.fetchall()]

    def log_message(self, username: str, role: str, content: str, msg_type: str = "text", caption: str = "", thread_id: str = "default"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO chat_history (username, role, content, type, caption, timestamp, thread_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (username, role, content, msg_type, caption, datetime.datetime.now().isoformat(), thread_id))
            conn.commit()

    def purge_chat_only(self, username: str, thread_id: str = "default"):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chat_history WHERE username=? AND thread_id=?", (username, thread_id))
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
            cursor.execute("SELECT COUNT(*) FROM document_vectors WHERE username=?", (username,))
            c_count = cursor.fetchone()[0]
            return {"history": h_count, "docs": d_count, "chunks": c_count}

    def log_latency(self, duration: float):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO latency_logs (duration, timestamp) VALUES (?, ?)",
                           (duration, datetime.datetime.now().isoformat()))
            conn.commit()

    def fetch_latencies(self) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT duration, timestamp FROM latency_logs ORDER BY id ASC")
            return [{"duration": r[0], "timestamp": r[1]} for r in cursor.fetchall()]

    def fetch_threads(self, username: str) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM chat_history WHERE username=?", (username,))
            threads = [r[0] for r in cursor.fetchall() if r[0]]
            if "default" not in threads:
                threads.insert(0, "default")
            return threads

    def log_tokens(self, username: str, tokens: int, model: str):
        cost = tokens * 0.0000006
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO token_logs (username, tokens, cost, timestamp) VALUES (?, ?, ?, ?)",
                           (username, tokens, cost, datetime.datetime.now().isoformat()))
            conn.commit()

    def fetch_token_stats(self) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username, tokens, cost, timestamp FROM token_logs ORDER BY id ASC")
            return [{"username": r[0], "tokens": r[1], "cost": r[2], "timestamp": r[3]} for r in cursor.fetchall()]

    def fetch_user_documents(self, username: str) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT doc_name, file_size FROM document_vectors WHERE username=?", (username,))
            return [{"doc_name": r[0], "file_size": r[1]} for r in cursor.fetchall()]

    def delete_document(self, username: str, doc_name: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_vectors WHERE username=? AND doc_name=?", (username, doc_name))
            conn.commit()

    def log_alert(self, message: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO system_alerts (message, timestamp) VALUES (?, ?)", (message, datetime.datetime.now().isoformat()))
            conn.commit()

    def fetch_latest_alert(self) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT message FROM system_alerts ORDER BY id DESC LIMIT 1")
            res = cursor.fetchone()
            return res[0] if res else ""

    def fetch_user_activity(self) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username, COUNT(*), MAX(timestamp) FROM chat_history GROUP BY username ORDER BY MAX(timestamp) DESC")
            return [{"username": r[0], "count": r[1], "last_active": r[2]} for r in cursor.fetchall()]


# --- 🧠 3. ASZINKRON AI MOTOR ---
class AsyncAIEngine:
    def __init__(self, db_repo: DatabaseRepository, config: AppConfig):
        self.db = db_repo
        self.config = config

    @staticmethod
    def get_available_models() -> list:
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.2-11b-vision-preview", "llama-3.2-3b-preview", "llama-3.2-11b-text-preview"]

    def compute_simple_tfidf_vector(self, text: str) -> list:
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        words = cleaned.split()
        freq = {}
        for w in words:
            if w not in self.config.HUNGARIAN_STOPWORDS:
                freq[w] = freq.get(w, 0) + 1
        return freq

    def smart_chunk_text(self, text: str, max_size: int, overlap: int) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n\n', '\n'))
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            if current_length + sentence_len > max_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                backlap = []
                backlap_len = 0
                for s in reversed(current_chunk):
                    if backlap_len + len(s) < overlap:
                        backlap.insert(0, s)
                        backlap_len += len(s)
                    else:
                        break
                current_chunk = backlap
                current_length = backlap_len
            
            current_chunk.append(sentence)
            current_length += sentence_len
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def ingest_document(self, text: str, doc_name: str, username: str, text_model: str, file_size_str: str):
        if not text: return
        chunks = self.smart_chunk_text(text, self.config.CHUNK_SIZE, self.config.CHUNK_OVERLAP)
        
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_vectors WHERE username=? AND doc_name=?", (username, doc_name))
            p_bar = st.progress(0, text="📚 Személyes emlékek indexelése...")
            for idx, chunk in enumerate(chunks):
                freq_map = self.compute_simple_tfidf_vector(chunk)
                cursor.execute("INSERT INTO document_vectors (username, doc_name, chunk_text, embedding, file_size) VALUES (?, ?, ?, ?, ?)",
                               (username, doc_name, chunk, json.dumps(freq_map).encode('utf-8'), file_size_str))
                p_bar.progress((idx + 1) / len(chunks))
            conn.commit()
            p_bar.empty()

    def query_vector_db_with_metadata(self, query_text: str, username: str, text_model: str) -> list:
        scored = []
        rows = []
        
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT doc_name, chunk_text, embedding FROM document_vectors WHERE username=?", (username,))
            rows = cursor.fetchall()
                
        cleaned_query = re.sub(r'[^\w\s]', '', query_text.lower())
        q_words = [w.strip() for w in cleaned_query.split() if len(w) > 2 and w not in self.config.HUNGARIAN_STOPWORDS]
        
        if q_words and rows:
            for doc_name, chunk_text, emb_blob in rows:
                try:
                    freq_map = json.loads(emb_blob.decode('utf-8'))
                except Exception:
                    freq_map = {}
                
                matches = sum(freq_map.get(word, 0) for word in q_words if word in freq_map)
                if matches > 0:
                    score = min(0.35 + (0.05 * matches), 0.95)
                    scored.append({"text": chunk_text, "score": score, "source": doc_name})
                        
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:3]

    def safe_ollama_chat_stream(self, model: str, messages: list, username: str = None):
        if not GROQ_API_KEY:
            st.error("❌ Hiányzó Groq API kulcs!")
            yield "Hiba: Nincs konfigurálva API kulcs."
            return
        try:
            client = Groq(api_key=GROQ_API_KEY)
            stream = client.chat.completions.create(
                model=model, 
                messages=messages, 
                stream=True, 
                timeout=60.0,
                max_tokens=800
            )
            
            estimated_tokens = 0
            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    estimated_tokens += max(1, len(content) // 4)
                    yield content
                    
            if username and estimated_tokens > 0:
                self.db.log_tokens(username, estimated_tokens, model)
        except Exception as e:
            yield f"Szerver hiba: {e}"

    def text_to_speech(self, text: str) -> bytes:
        if not text: return None
        try:
            clean_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
            clean_text = re.sub(r'[#\*_`\-\>\+\=\[\]\(\)]', '', clean_text).strip()
            if not clean_text: return None
            
            import edge_tts
            import asyncio
            import threading

            def run_async(coro):
                result = []
                def run():
                    result.append(asyncio.run(coro))
                thread = threading.Thread(target=run)
                thread.start()
                thread.join()
                return result[0]

            async def generate_audio():
                communicate = edge_tts.Communicate(clean_text[:1000], "hu-HU-TamasNeural")
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                return audio_data

            return run_async(generate_audio())
        except Exception:
            return None

    def search_web_sync(self, query: str) -> str:
        all_results = []
        try:
            with DDGS() as ddgs:
                res_text = ddgs.text(query, max_results=8)
                if res_text:
                    all_results.extend(res_text)
                try:
                    res_news = ddgs.news(query, max_results=5)
                    if res_news:
                        all_results.extend(res_news)
                except Exception: pass
        except Exception: pass
            
        if not all_results: return ""
            
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url_key = r.get('href') or r.get('url') or r.get('title', '')
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                unique_results.append(r)
                
        return "\n---\n".join([f"Forrás: {r.get('title', 'Nincs cím')}\nKivonat: {r.get('body', r.get('snippet', ''))}" for r in unique_results[:3]])

    def generate_image(self, query: str, text_model: str) -> str:
        clean_query = query.lower()
        stop_words = ["generálj", "generál", "képet", "kép", "egy", "a", "az", "mutass", "rajzolj", "rajzol", "ról", "ről", "-"]
        for word in stop_words:
            clean_query = re.sub(r'\b' + word + r'\b', '', clean_query)
        clean_query = re.sub(r'[^\w\s]', '', clean_query).strip()
        if not clean_query: return None
        
        en_query = clean_query
        if GROQ_API_KEY:
            try:
                client = Groq(api_key=GROQ_API_KEY)
                res = client.chat.completions.create(
                    model=text_model, 
                    messages=[{"role": "user", "content": f"Translate the following prompt to English for an image generator. Output ONLY the English translation, no quotes, no extra text: {clean_query}"}], 
                    timeout=10.0
                )
                translated = res.choices[0].message.content.strip().replace('"', '').replace("'", "")
                if translated:
                    en_query = translated
            except Exception:
                en_query = clean_query
                
        return f"https://image.pollinations.ai/p/{urllib.parse.quote(en_query)}?width=1024&height=1024&seed={int(time.time())}&model=flux&enhance=true"

    def generate_video(self, query: str, text_model: str) -> str:
        clean_query = query.lower()
        stop_words = ["generálj", "generál", "videót", "videó", "egy", "a", "az", "mutass", "készíts", "rajzolj", "rajzol", "ról", "ről", "-"]
        for word in stop_words:
            clean_query = re.sub(r'\b' + word + r'\b', '', clean_query)
        clean_query = re.sub(r'[^\w\s]', '', clean_query).strip()
        if not clean_query: return None
        try:
            client = Groq(api_key=GROQ_API_KEY)
            res = client.chat.completions.create(model=text_model, messages=[{"role": "user", "content": f"Translate to English in 5 words max, no quotes: {clean_query}"}], timeout=10.0)
            en_query = res.choices[0].message.content.strip().replace('"', '').replace("'", "")
        except Exception: en_query = clean_query
        return f"https://textmevideo-m97v.pollinations.ai/{urllib.parse.quote(en_query)}"

    def post_process_text(self, text: str, text_model: str, mode: str) -> str:
        prompts = {"translate": f"Translate to English:\n\n{text}", "summary": f"Készíts összefoglalót magyarul:\n\n{text}"}
        try:
            client = Groq(api_key=GROQ_API_KEY)
            res = client.chat.completions.create(model=text_model, messages=[{"role": "user", "content": prompts[mode]}], timeout=20.0)
            return res.choices[0].message.content
        except Exception as e: return f"Hiba: {e}"

    def validate_url_safety(self, text: str) -> str:
        return re.sub(r'(http://\S+)', '⚠️ [NEM BIZTONSÁGOS LINKEK ELTÁVOLÍTVA]', text)

    def anonymize_gdpr(self, text: str) -> str:
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED EMAIL]', text)
        return re.sub(r'\+?[0-9]{2,4}[-\s]?([0-9]{2,4}[-\s]?){2,3}[0-9]{2,4}', '[REDACTED PHONE]', text)

    def execute_python_sandbox(self, code: str) -> str:
        import sys
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        try:
            exec(code, {}, {})
            sys.stdout = old_stdout
            return redirected_output.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            return f"Hiba a futtatás során: {e}"

# --- INICIALIZÁLÁS UTÓLAGOS INFRASTRUKTÚRA ---
db_repo = DatabaseRepository(cfg.DB_FILE)
ai_engine = AsyncAIEngine(db_repo, cfg)

if "voice_text" not in st.session_state: st.session_state.voice_text = ""
if "mute_voice" not in st.session_state: st.session_state.mute_voice = False

# ---🧠 HOSSZÚTÁVÚ TÖMÖRÍTETT MEMÓRIA LOGIKA 🧠---
def get_clean_history(history, max_chars, text_model=None):
    truncated = []
    old_messages = []
    curr = 0
    for h in reversed(history):
        if h.get("type") == "text":
            if curr + len(h["content"]) <= max_chars:
                truncated.insert(0, {"role": h["role"], "content": h["content"]})
                curr += len(h["content"])
            else:
                old_messages.insert(0, f"{h['role']}: {h['content']}")
    
    compressed = ""
    if old_messages and text_model and GROQ_API_KEY:
        try:
            client = Groq(api_key=GROQ_API_KEY)
            old_txt = "\n".join(old_messages[-10:])
            res = client.chat.completions.create(
                model=text_model,
                messages=[{"role": "user", "content": f"Készíts egy max 2-3 mondatos tömör összefoglalót az alábbi korábbi beszélgetésekből (preferenciák, fontos infók): \n\n{old_txt}"}],
                timeout=5.0
            )
            compressed = res.choices[0].message.content.strip()
        except Exception: pass
    return truncated, compressed

# --- ⚙️ OLDALSÁV ---
with st.sidebar:
    st.header("⚙️ Beállítások")
    
    if "current_thread" not in st.session_state:
        st.session_state.current_thread = "default"
        
    user_threads = db_repo.fetch_threads(active_chat_user)
    st.subheader("🧵 Csevegési szálak")
    selected_thread = st.selectbox("Válassz szálat:", user_threads, index=user_threads.index(st.session_state.current_thread) if st.session_state.current_thread in user_threads else 0)
    if selected_thread != st.session_state.current_thread:
        st.session_state.current_thread = selected_thread
        st.rerun()
        
    new_thread_name = st.text_input("➕ Új szál neve:", placeholder="pl. Munka, Programozás...")
    if st.button("Új szál létrehozása", use_container_width=True):
        cleaned_thread = new_thread_name.strip()
        if cleaned_thread and cleaned_thread not in user_threads:
            st.session_state.current_thread = cleaned_thread
            db_repo.log_message(active_chat_user, "system", f"Szál létrehozva: {cleaned_thread}", "text", thread_id=cleaned_thread)
            st.success(f"Szál elindítva: {cleaned_thread}")
            time.sleep(0.5)
            st.rerun()

    with st.expander("🤖 AI Modell Beállítások", expanded=True):
        st.subheader("📋 Rendszer Szerepkör Sablonok")
        persona = st.selectbox("AI Mód", ["Chat&Web keresés", "Code-olás", "Számolás", "Zoli mód"])
        persona_prompts = {
            "Chat&Web keresés": "Te egy precíz, professzionális személyes asszisztens vagy. A neved: Zoli. Válaszolj rendkívül tömören, lényegretörően, felesleges udvariassági körök nélkül.",
            "Code-olás": "Te egy Mérnök vagy. Tiszta kódot írsz markdown kódblokkokban. Csak a lényeges magyarázatot írd le, röviden.",
            "Számolás": "Használj standard szöveges formázást a képletekhez. Precízen számolsz. Légy rövid.",
            "Zoli mód": "Mindent elrontasz, semmit sem tudsz kiszámolni helyes végeredménnyel. soha nem tudsz helyes választ adni."
        }    
        st.subheader("🤖 AI Modellek")
        models = ai_engine.get_available_models()
        TEXT_MODEL = st.selectbox("Fő LLM Modell", models, index=1 if models else None) # <--- EZT ÍRTAM ÁT 1-RE HOGY A LLAMA-3.1-8B-INSTANT LEGYEN AZ ALAPÉRTELMEZETT
    
    with st.expander("📂 Média és Dokumentumok", expanded=False):
        st.subheader("📂 Fájlok és Képek Feltöltése")
        uploaded_file = st.file_uploader("Indexelés (txt, pdf, docx, csv, xlsx) / Kép elemzés (png, jpg)", type=["txt", "pdf", "docx", "csv", "xlsx", "png", "jpg", "jpeg"])
        if uploaded_file and f"idx_{uploaded_file.name}" not in st.session_state:
            ext = uploaded_file.name.split(".")[-1].lower()
            content = ""
            size_kb = f"{len(uploaded_file.getvalue()) / 1024:.1f} KB"
            
            if ext == "txt": content = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")).read()
            elif ext == "pdf": content = "\n".join([p.extract_text() or "" for p in PdfReader(io.BytesIO(uploaded_file.read())).pages])
            elif ext == "docx": content = "\n".join([p.text for p in docx.Document(io.BytesIO(uploaded_file.read())).paragraphs])
            elif ext in ["csv", "xlsx"]:
                try:
                    df = pd.read_csv(io.BytesIO(uploaded_file.getvalue())) if ext == "csv" else pd.read_excel(io.BytesIO(uploaded_file.getvalue()))
                    st.session_state.last_df = df
                    content = f"Fájl: {uploaded_file.name}\nOszlopok: {list(df.columns)}\nStatisztika:\n{df.describe().to_string()}\nAdat minta:\n{df.head(15).to_markdown() if hasattr(df, 'to_markdown') else df.head(15).to_string()}"
                    st.sidebar.dataframe(df.head(3))
                except Exception as e: st.sidebar.error(f"Táblázat hiba: {e}")
            elif ext in ["png", "jpg", "jpeg"]:
                st.session_state.active_vision_image = uploaded_file.getvalue()
                st.sidebar.image(st.session_state.active_vision_image, caption="📸 Kép készen áll az elemzésre.", use_container_width=True)
                st.session_state[f"idx_{uploaded_file.name}"] = True
                st.sidebar.success("Kép sikeresen betöltve!")

            if content:
                ai_engine.ingest_document(content, uploaded_file.name, active_chat_user, TEXT_MODEL, size_kb)
                st.session_state[f"idx_{uploaded_file.name}"] = True
                st.sidebar.success(f"✅ Mentve ({size_kb})")

    with st.expander("🎙️ Hangvezérlés", expanded=False):
        st.subheader("🎙️ Hang rögzítése")
        st.checkbox("📟 Walkie-Talkie mód (Azonnali válasz & hang)", key="walkie_talkie", value=False)
        audio = mic_recorder(start_prompt="🎙️ Hang rögzítése", stop_prompt="🛑 Megállítás", just_once=True, key="voice_input")
        
        if st.session_state.get("voice_playing", False):
            if st.button("🛑 Félbeszakítás / Némítás", type="primary", use_container_width=True):
                st.session_state.mute_voice = True
                st.session_state.voice_playing = False
                st.rerun()

    st.markdown("---")
    if st.button("🚪 Kijelentkezés", use_container_width=True):
        st.session_state.logged_in_user = None
        if "admin_selected_user" in st.session_state:
            del st.session_state.admin_selected_user
        st.query_params.clear()
        st.rerun()

chat_history = db_repo.fetch_history(active_chat_user, thread_id=st.session_state.get("current_thread", "default"))

def inject_copy_button(text: str, unique_key: str):
    escaped = base64.b64encode(text.encode('utf-8')).decode('utf-8')
    js = f"""<script>function copy_{unique_key}() {{ navigator.clipboard.writeText(atob("{escaped}")); var btn = document.getElementById("btn_{unique_key}"); btn.innerText = "📋 Másolva!"; setTimeout(function() {{ btn.innerText = "📋 Másolás"; }}, 2000); }}</script><button id="btn_{unique_key}" onclick="copy_{unique_key}()" style="background-color: #1e1b4b; color: #a5b4fc; border: 1px solid #312e81; padding: 6px 14px; font-size: 12px; cursor: pointer; border-radius: 6px; font-weight:500;">📋 Másolás</button>"""
    st.components.v1.html(js, height=38)

def generate_docx_download(text: str) -> bytes:
    doc = Document()
    doc.add_heading('Zoli GPT Személyes Jegyzet Export', 0)
    doc.add_paragraph(text)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()

if audio:
    with st.spinner("🗨️ Hangjegyzet feldolgozása..."):
        try:
            st.session_state.mute_voice = False
            if GROQ_API_KEY:
                client = Groq(api_key=GROQ_API_KEY)
                translation = client.audio.transcriptions.create(
                    file=("audio.wav", audio['bytes']),
                    model="whisper-large-v3-turbo",
                    language="hu"
                )
                transcribed_text = translation.text.strip() if translation.text else ""
                if transcribed_text:
                    processed_voice = ai_engine.anonymize_gdpr(ai_engine.validate_url_safety(transcribed_text))
                    st.session_state.voice_text = processed_voice
                    
                    if st.session_state.get("walkie_talkie", False):
                        current_tid = st.session_state.get("current_thread", "default")
                        db_repo.log_message(active_chat_user, "user", processed_voice, thread_id=current_tid)
                        
                        system_prompt = persona_prompts.get(persona, "Te egy precíz asszisztens vagy.")
                        messages = [{"role": "system", "content": system_prompt}]
                        current_thread_hist = db_repo.fetch_history(active_chat_user, thread_id=current_tid)
                        for msg in current_thread_hist[-6:]:
                            if msg["type"] == "text":
                                messages.append({"role": msg["role"], "content": msg["content"]})
                        
                        full_resp = ""
                        for chunk in ai_engine.safe_ollama_chat_stream(TEXT_MODEL, messages, username=active_chat_user):
                            full_resp += chunk
                        
                        db_repo.log_message(active_chat_user, "assistant", full_resp, "text", thread_id=current_tid)
                        st.session_state.voice_text = ""
                        st.rerun()
            else:
                st.error("Groq API kulcs hiányzik a hangfeldolgozáshoz!")
        except Exception as e: st.error(f"Groq Whisper hiba: {e}")

# --- 📑 INTERFACE TABS ---
tabs_headers = ["💬 Chat", "📊 Személyes Statisztika"]
if is_admin:
    tabs_headers.append("👑 Globális Adminisztráció")

tabs = st.tabs(tabs_headers)
tab_chat = tabs[0]
tab_monitor = tabs[1]

# --- 📊 SZEMÉLYES STATISZTIKA TAB ---
with tab_monitor:
    st.subheader(f"📈 {active_chat_user} Statisztikái")
    stats = db_repo.get_system_stats(active_chat_user)
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1: st.markdown(f'<div class="monitor-card">💬 <b>Összes gondolat:</b><br><span style="font-size:20px;color:#10b981;">{stats["history"]} db</span></div>', unsafe_allow_html=True)
    with col_m2: st.markdown(f'<div class="monitor-card">📄 <b>Saját fájlok:</b><br><span style="font-size:20px;color:#6366f1;">{stats["docs"]} db</span></div>', unsafe_allow_html=True)
    with col_m3: st.markdown(f'<div class="monitor-card">🧩 <b>Információ egységek:</b><br><span style="font-size:20px;color:#06b6d4;">{stats["chunks"]} db</span></div>', unsafe_allow_html=True)

    st.markdown("### 🗂️ Saját indexelt fájljaim")
    user_docs = db_repo.fetch_user_documents(active_chat_user)
    if user_docs:
        for doc in user_docs:
            d_col1, d_col2 = st.columns([4, 1])
            with d_col1:
                st.markdown(f"📄 **{doc['doc_name']}** ({doc['file_size']})")
            with d_col2:
                if st.button("🗑️ Törlés", key=f"del_doc_{doc['doc_name']}", use_container_width=True):
                    db_repo.delete_document(active_chat_user, doc['doc_name'])
                    st.success(f"Törölve: {doc['doc_name']}")
                    time.sleep(0.5)
                    st.rerun()
    else:
        st.info("Nincsenek feltöltött dokumentumaid.")

# --- 👑 GLOBÁLIS ADMINISZTRÁCIÓ TAB ---
if is_admin:
    with tabs[2]:
        st.subheader("👑 Globális Rendszerfelügyelet")
        st.info(f"Sikeres adminisztrátori belépés. Azonosított fiók: {st.session_state.logged_in_user}")
        
        st.markdown("---")
        st.markdown("### 👥 Felhasználói Fiók Kiválasztása")
        all_users = db_repo.get_all_users()
        if st.session_state.logged_in_user not in all_users:
            all_users.append(st.session_state.logged_in_user)
        
        if "admin_selected_user" not in st.session_state:
            st.session_state.admin_selected_user = st.session_state.logged_in_user

        selected_user = st.selectbox(
            "Felhasználó Chat megtekintése:", 
            all_users, 
            index=all_users.index(st.session_state.admin_selected_user) if st.session_state.admin_selected_user in all_users else 0,
            key="global_admin_user_selector"
        )
        
        if selected_user != st.session_state.admin_selected_user:
            st.session_state.admin_selected_user = selected_user
            st.rerun()
            
        st.info(f"Jelenleg **{active_chat_user}** chatjét látod.")
        st.markdown("---")
        
        st.markdown("### 📢 Rendszerértesítés Küldése")
        new_alert = st.text_input("Új értesítés szövege:", placeholder="pl. Karbantartás ma este...")
        if st.button("Értesítés kiküldése", use_container_width=True):
            if new_alert.strip():
                db_repo.log_alert(new_alert.strip())
                st.success("Értesítés sikeresen elmentve!")
                time.sleep(0.5)
                st.rerun()

        st.markdown("### 📋 Felhasználói Aktivitási Napló (Audit Log)")
        activity = db_repo.fetch_user_activity()
        if activity:
            df_act = pd.DataFrame(activity)
            df_act.columns = ["Felhasználónév", "Üzenetek száma", "Utolsó aktivitás"]
            st.dataframe(df_act, use_container_width=True)
        else:
            st.info("Még nincs rögzített felhasználói aktivitás.")

        st.markdown("---")
        st.markdown("### 👥 Felhasználó Kezelés")
        if st.button(f"🗑️ '{st.session_state.admin_selected_user}' beszélgetésének véglegen törlése", type="primary"):
            db_repo.purge_chat_only(st.session_state.admin_selected_user, thread_id=st.session_state.get("current_thread", "default"))
            st.success(f"{st.session_state.admin_selected_user} előzményei törölve!")
            time.sleep(1)
            st.rerun()

        st.markdown("### 📋 Rendszer Válaszidő (Latency) Monitor")
        latencies = db_repo.fetch_latencies()
        if latencies:
            df_lat = pd.DataFrame(latencies)
            df_lat['timestamp'] = pd.to_datetime(df_lat['timestamp'])
            df_lat = df_lat.set_index('timestamp')
            st.line_chart(df_lat['duration'], y_label="Válaszidő (másodperc)")
        else:
            st.info("Még nincs rögzített válaszidő adat az adatbázisban.")

        st.markdown("### 🪙 Token- és Költségfigyelő (Groq Usage)")
        token_stats = db_repo.fetch_token_stats()
        if token_stats:
            df_tok = pd.DataFrame(token_stats)
            df_tok['timestamp'] = pd.to_datetime(df_tok['timestamp'])
            
            total_tokens = df_tok['tokens'].sum()
            total_cost = df_tok['cost'].sum()
            
            col_t1, col_t2 = st.columns(2)
            with col_t1: st.metric("Összes felhasznált token", f"{total_tokens:,} db")
            with col_t2: st.metric("Becsült összköltség", f"${total_cost:.4f}")
            
            st.dataframe(df_tok.tail(30), use_container_width=True)
        else:
            st.info("Még nincs rögzített token használati adat.")

# --- 💬 CHAT INTERFACE ---
with tab_chat:
    alert = db_repo.fetch_latest_alert()
    if alert:
        st.warning(f"📢 **Rendszerértesítés:** {alert}")

    col_left, col_right = st.columns([5, 2])
    with col_right:
        if st.button("🗑️ Beszélgetés ürítése", use_container_width=True):
            db_repo.purge_chat_only(active_chat_user, thread_id=st.session_state.get("current_thread", "default"))
            st.rerun()

    for idx, msg in enumerate(chat_history):
        with st.chat_message(msg["role"]):
            if msg.get("type") == "image": st.image(msg["content"], caption=msg.get("caption"))
            elif msg.get("type") == "video": st.video(msg["content"])
            else:
                content = msg["content"]
                st.write(content)
                if msg["role"] == "assistant":
                    if not st.session_state.mute_voice and idx == len(chat_history) - 1:
                        audio_data = ai_engine.text_to_speech(content)
                        if audio_data: st.audio(audio_data, format="audio/mp3")

                    python_codes = re.findall(r'```python\s*(.*?)\s*```', content, re.DOTALL)

                    with st.container():
                        cols_layout = [1.2, 1.2, 1, 1, 1, 1] if python_codes else [1.2, 1.2, 1, 1]
                        cols = st.columns(cols_layout)
                        with cols[0]:
                            inject_copy_button(content, f"h_{idx}")
                        with cols[1]:
                            st.download_button("📄 Word-be", data=generate_docx_download(content), file_name=f"jegyzet_{idx}.docx", key=f"docx_{idx}", use_container_width=True)
                        with cols[2]:
                            if st.button("🇬🇧 En", key=f"trans_{idx}", use_container_width=True): 
                                st.toast(f"🔤 **Fordítás:**\n\n{ai_engine.post_process_text(content, TEXT_MODEL, 'translate')}", icon="🇬🇧")
                        with cols[3]:
                            if st.button("📝 Össz", key=f"sum_{idx}", use_container_width=True): 
                                st.toast(f"📝 **Összefoglaló:**\n\n{ai_engine.post_process_text(content, TEXT_MODEL, 'summary')}", icon="📝")
                        if python_codes:
                            with cols[4]:
                                if st.button("⚡ Run", key=f"run_{idx}", use_container_width=True):
                                    out = ai_engine.execute_python_sandbox(python_codes[0])
                                    st.info(f"💻 **Kód kimenet:**\n```\n{out}\n```")
                            with cols[5]:
                                st.download_button("🐍 .py", data=python_codes[0], file_name=f"script_{idx}.py", key=f"py_{idx}", use_container_width=True)

    default_input = st.session_state.voice_text if st.session_state.voice_text else ""
    
    user_input = st.chat_input("Kérdezz bármit...", key="chat_input_field", disabled=st.session_state.generating)
    if default_input and not user_input:
        user_input = default_input
        st.session_state.voice_text = ""

    if user_input:
        st.session_state.generating = True
        st.session_state.mute_voice = False
        user_input = ai_engine.anonymize_gdpr(ai_engine.validate_url_safety(user_input))
        st.chat_message("user").write(user_input)
        db_repo.log_message(active_chat_user, "user", user_input, thread_id=st.session_state.get("current_thread", "default"))

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            # Try...finally blokk, ami garantálja, hogy a kód lefutása VAGY hibája után a zár feloldódik
            try:
                if any(w in user_input.lower() for w in ["kép", "generál", "rajzol", "mutass"]) and not any(w in user_input.lower() for w in ["videó", "video", "elemzés", "elemezd"]):
                    with st.spinner("🎨 AI Képgenerálás..."):
                        url = ai_engine.generate_image(user_input, TEXT_MODEL)
                        if url:
                            st.image(url, caption=f"✨ Kép: {user_input}", use_container_width=True)
                            db_repo.log_message(active_chat_user, "assistant", url, "image", caption=user_input, thread_id=st.session_state.get("current_thread", "default"))
                elif any(w in user_input.lower() for w in ["videó", "video", "animáció", "mozgás"]):
                    with st.spinner("🎬 AI Videógenerálás..."):
                        url = ai_engine.generate_video(user_input, TEXT_MODEL)
                        if url:
                            st.video(url)
                            db_repo.log_message(active_chat_user, "assistant", url, "video", thread_id=st.session_state.get("current_thread", "default"))
                else:
                    start_time = time.perf_counter()
                    
                    system_prompt = persona_prompts.get(persona, "Te egy precíz asszisztens vagy.")
                    context_addition = ""
                    web_sources_text = ""
                    
                    web_triggers = ["keress rá", "mi történt", "hírek", "időjárás", "ma", "aktualitás"]
                    if any(w in user_input.lower() for w in web_triggers):
                        st.toast("🔍 Webes keresés indítása a friss adatokért...", icon="🌐")
                        with st.spinner("🌐 Böngészés a weben..."):
                            web_results = ai_engine.search_web_sync(user_input)
                            if web_results:
                                context_addition = f"\n\nFONTOS KONTEXTUS A WEBRŐL:\n{web_results}"
                                
                                sources = [line.replace('Forrás: ', '') for line in web_results.split('\n---\n') if line.startswith('Forrás:')]
                                if sources:
                                    web_sources_text = "\n\n---\n**🌐 Felhasznált források:**\n" + "\n".join([f"- {s}" for s in set(sources)])

                    cleaned_hist, compressed_summary = get_clean_history(chat_history, max_chars=cfg.MAX_HISTORY_CHARS, text_model=TEXT_MODEL)
                    
                    full_system_content = system_prompt + context_addition
                    if compressed_summary:
                        full_system_content += f"\n\nKorábbi beszélgetések tömörített memóriája:\n{compressed_summary}"

                    messages = [{"role": "system", "content": full_system_content}]
                    
                    for msg in cleaned_hist:
                        messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    messages.append({"role": "user", "content": user_input})
                    
                    full_response = ""
                    with st.spinner("Gondolkodom..."):
                        for chunk in ai_engine.safe_ollama_chat_stream(TEXT_MODEL, messages, username=active_chat_user):
                            full_response += chunk
                            response_placeholder.markdown(full_response + "▌")
                    
                    if web_sources_text:
                        full_response += web_sources_text
                        
                    response_placeholder.markdown(full_response)
                    
                    end_time = time.perf_counter()
                    db_repo.log_latency(end_time - start_time)
                    
                    db_repo.log_message(active_chat_user, "assistant", full_response, "text", thread_id=st.session_state.get("current_thread", "default"))
            
            finally:
                # Kényszerített feloldás a végén
                st.session_state.generating = False
                st.rerun()
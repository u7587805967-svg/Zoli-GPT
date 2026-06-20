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
    MAX_HISTORY_CHARS: int = 4000
    RAG_SIMI_THRESHOLD: float = 0.15
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 300
    HUNGARIAN_STOPWORDS = frozenset([
        "a", "az", "egy", "be", "ki", "le", "fel", "meg", "el", "at", "es", "hogy", 
        "nem", "sem", "vagy", "is", "csak", "mint", "volt", "lesz", "cikk", "alatt"
    ])

st.set_page_config(page_title="Zoli GPT ", page_icon="🚭", layout="centered")

# --- ⚙️ INICIALIZÁLÁS ÉS BIZTONSÁGI SORREND ---
cfg = AppConfig()

if "generating" not in st.session_state:
    st.session_state.generating = False

if "last_usage" not in st.session_state:
    st.session_state.last_usage = None

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
            cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, role TEXT, content TEXT, type TEXT, caption TEXT, timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS document_vectors (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, doc_name TEXT, chunk_text TEXT, embedding BLOB, file_size TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS latency_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, duration REAL, timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS token_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS broadcast_system (id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT, is_active INTEGER)''')
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

    # --- ÚJ PRÉMIUM INFRASTRUKTÚRA FUNKCIÓK ---
    def fetch_user_documents(self, username: str) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT doc_name, file_size FROM document_vectors WHERE username=?", (username,))
            return [{"name": r[0], "size": r[1]} for r in cursor.fetchall()]

    def delete_specific_document(self, username: str, doc_name: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_vectors WHERE username=? AND doc_name=?", (username, doc_name))
            conn.commit()

    def log_token_usage(self, username: str, prompt_t: int, compl_t: int):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO token_logs (username, prompt_tokens, completion_tokens, timestamp) VALUES (?, ?, ?, ?)",
                           (username, prompt_t, compl_t, datetime.datetime.now().isoformat()))
            conn.commit()

    def fetch_token_stats(self, username: str) -> dict:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(prompt_tokens), SUM(completion_tokens) FROM token_logs WHERE username=?", (username,))
            res = cursor.fetchone()
            return {"prompt": res[0] or 0, "completion": res[1] or 0}

    def set_active_broadcast(self, msg: str, active: int):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM broadcast_system")
            cursor.execute("INSERT INTO broadcast_system (message, is_active) VALUES (?, ?)", (msg, active))
            conn.commit()

    def fetch_active_broadcast(self) -> str:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT message FROM broadcast_system WHERE is_active=1 LIMIT 1")
            res = cursor.fetchone()
            return res[0] if res else ""

    def fetch_global_sessions(self) -> list:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT username, MAX(timestamp) as last_act, COUNT(*) as interactions 
                FROM chat_history GROUP BY username ORDER BY last_act DESC
            """)
            return [{"Felhasználó": r[0], "Utolsó Aktivitás": r[1], "Interakciók": r[2]} for r in cursor.fetchall()]

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
            if w not in self.config.HUNGARIAN_STOPWORDS and len(w) > 1:
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

    # --- PRÉMIUM MATEMATIKAI KOSZINUSZ-HASONLÓSÁG VEKTOROS KERESŐ ---
    def query_vector_db_with_metadata(self, query_text: str, username: str, text_model: str) -> list:
        scored = []
        rows = []
        
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT doc_name, chunk_text, embedding FROM document_vectors WHERE username=?", (username,))
            rows = cursor.fetchall()
                
        cleaned_query = re.sub(r'[^\w\s]', '', query_text.lower())
        q_words = [w.strip() for w in cleaned_query.split() if len(w) > 1 and w not in self.config.HUNGARIAN_STOPWORDS]
        
        if q_words and rows:
            q_freq = {}
            for w in q_words:
                q_freq[w] = q_freq.get(w, 0) + 1
            all_vocab = set(q_freq.keys())
            
            for doc_name, chunk_text, emb_blob in rows:
                try:
                    freq_map = json.loads(emb_blob.decode('utf-8'))
                except Exception:
                    freq_map = {}
                
                if not freq_map: continue
                
                vocab = all_vocab.union(freq_map.keys())
                v1 = np.array([q_freq.get(w, 0) for w in vocab])
                v2 = np.array([freq_map.get(w, 0) for w in vocab])
                
                dot_prod = np.dot(v1, v2)
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                
                if norm_v1 > 0 and norm_v2 > 0:
                    cosine_sim = float(dot_prod / (norm_v1 * norm_v2))
                    if cosine_sim >= self.config.RAG_SIMI_THRESHOLD:
                        scored.append({"text": chunk_text, "score": cosine_sim, "source": doc_name})
                        
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:3]

    def safe_ollama_chat_stream(self, model: str, messages: list):
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
                stream_options={"include_usage": True}
            )
            for chunk in stream:
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    st.session_state.last_usage = {
                        "prompt": chunk.usage.prompt_tokens,
                        "completion": chunk.usage.completion_tokens
                    }
                if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Szerver hiba: {e}"

    # --- FEJLETT TEXT-TO-SPEECH PARAMÉTEREZÉS ---
    def text_to_speech(self, text: str, voice: str = "hu-HU-TamasNeural", rate: str = "+0%") -> bytes:
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
                communicate = edge_tts.Communicate(clean_text[:1000], voice, rate=rate)
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
                
        return "\n---\n".join([f"Forrás: {r.get('title', 'Nincs cím')}\nKivonat: {r.get('body', r.get('snippet', ''))}" for r in unique_results[:12]])

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

# --- INICIALIZÁLÁS UTÓLAGOS INFRASTRUKTÚRA ---
db_repo = DatabaseRepository(cfg.DB_FILE)
ai_engine = AsyncAIEngine(db_repo, cfg)

if "voice_text" not in st.session_state: st.session_state.voice_text = ""
if "mute_voice" not in st.session_state: st.session_state.mute_voice = False

# --- RENDSERÜZENET (BROADCAST) MEGJELENÍTÉSE ---
active_broadcast = db_repo.fetch_active_broadcast()
if active_broadcast:
    st.warning(f"📢 **Rendszerértesítés:** {active_broadcast}")

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
    
    if is_admin:
        st.markdown("---")
        with st.expander("👑 Adminisztrációs Panel", expanded=False):
            all_users = db_repo.get_all_users()
            if st.session_state.logged_in_user not in all_users:
                all_users.append(st.session_state.logged_in_user)
            
            if "admin_selected_user" not in st.session_state:
                st.session_state.admin_selected_user = st.session_state.logged_in_user

            selected_user = st.selectbox(
                "Felhasználó Chat megtekintése:", 
                all_users, 
                index=all_users.index(st.session_state.admin_selected_user) if st.session_state.admin_selected_user in all_users else 0
            )
            
            if selected_user != st.session_state.admin_selected_user:
                st.session_state.admin_selected_user = selected_user
                st.rerun()
                
            active_chat_user = st.session_state.admin_selected_user
            st.info(f"Jelenleg **{active_chat_user}** chatjét látod.")
        st.markdown("---")

    with st.expander("🤖 AI Modell Beállítások", expanded=True):
        st.subheader("📋 Rendszer Szerepkör Sablonok")
        persona = st.selectbox("AI Mód", ["Chat&Web keresés", "Code-olás", "Számolás", "Zoli mód", "Egyedi mód"])
        
        custom_system_prompt = ""
        if persona == "Egyedi mód":
            custom_system_prompt = st.text_area("Egyedi Rendszer Szerepkör (System Prompt):", "Te egy precíz személyes asszisztens vagy, akit Zolinak hívnak.")
            
        persona_prompts = {
            "Chat&Web keresés": "Te egy precíz, professzionális személyes asszisztens vagy. A neved: Zoli.",
            "Code-olás": "Te egy Mérnök vagy. Tiszta kódot írsz markdown kódblokkokban. A neved: Zoli.",
            "Számolás": "Használj standard szöveges formázást a képletekhez. Precízen számolsz. A neved: Zoli.",
            "Zoli mód": "Mindent elrontasz, semmit sem tudsz kiszámolni helyes végeredménnyel. soha nem tudsz helyes választ adni. A neved: Zoli.",
            "Egyedi mód": custom_system_prompt
        }    
        st.subheader("🤖 AI Modellek")
        models = ai_engine.get_available_models()
        TEXT_MODEL = st.selectbox("Fő LLM Modell", models, index=0 if models else None)
    
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

        # --- PRÉMIUM DOKUMENTUM MANÁZSER / LISTÁZÓ ÉS TÖRLŐ ---
        st.markdown("---")
        st.subheader("🗂️ Indexelt Fájlok Kezelése")
        saved_docs = db_repo.fetch_user_documents(active_chat_user)
        if saved_docs:
            for doc in saved_docs:
                col_doc_n, col_doc_d = st.columns([4, 1])
                with col_doc_n:
                    st.caption(f"📄 {doc['name']} ({doc['size']})")
                with col_doc_d:
                    if st.button("🗑️", key=f"del_doc_{doc['name']}", help="Fájl végleges törlése az indexből"):
                        db_repo.delete_specific_document(active_chat_user, doc['name'])
                        st.toast(f"Törölve: {doc['name']}", icon="🗑️")
                        st.rerun()
        else:
            st.caption("Nincs mentett dokumentum.")

    with st.expander("🎙️ Hangvezérlés", expanded=False):
        st.subheader("🎙️ Hang rögzítése")
        audio = mic_recorder(start_prompt="🎙️ Hang rögzítése", stop_prompt="🛑 Megállítás", just_once=True, key="voice_input")
        
        # --- FEJLETT VOICE PARAMÉTEREK ---
        st.markdown("---")
        voice_char = st.selectbox("TTS Karakter", ["Férfi (Tamás)", "Női (Noémi)"])
        voice_speed = st.select_slider("TTS Olvasási sebesség", options=["-20%", "-10%", "+0%", "+10%", "+20%"], value="+0%")
        selected_voice = "hu-HU-TamasNeural" if voice_char == "Férfi (Tamás)" else "hu-HU-NoemiNeural"
        
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

chat_history = db_repo.fetch_history(active_chat_user)

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
                    st.session_state.voice_text = ai_engine.anonymize_gdpr(ai_engine.validate_url_safety(transcribed_text))
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

# --- 📊 SZEMÉLYES STATISZTIKA TAB (TOKEN ÉS KÖLTSÉGBECSLES KIEGÉSZÍTÉSSEL) ---
with tab_monitor:
    st.subheader(f"📈 {active_chat_user} Statisztikái")
    stats = db_repo.get_system_stats(active_chat_user)
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1: st.markdown(f'<div class="monitor-card">💬 <b>Összes gondolat:</b><br><span style="font-size:20px;color:#10b981;">{stats["history"]} db</span></div>', unsafe_allow_html=True)
    with col_m2: st.markdown(f'<div class="monitor-card">📄 <b>Saját fájlok:</b><br><span style="font-size:20px;color:#6366f1;">{stats["docs"]} db</span></div>', unsafe_allow_html=True)
    with col_m3: st.markdown(f'<div class="monitor-card">🧩 <b>Információ egységek:</b><br><span style="font-size:20px;color:#06b6d4;">{stats["chunks"]} db</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🪙 API Erőforrás Felhasználás és Költségek")
    token_stats = db_repo.fetch_token_stats(active_chat_user)
    total_tokens = token_stats["prompt"] + token_stats["completion"]
    estimated_cost = (total_tokens / 1_000_000) * 0.15 # Átlagolt $0.15 / 1M token becslés
    
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1: st.markdown(f'<div class="monitor-card">📥 <b>Prompt Token:</b><br><span style="font-size:18px;color:#a5b4fc;">{token_stats["prompt"]} tk</span></div>', unsafe_allow_html=True)
    with col_t2: st.markdown(f'<div class="monitor-card">📤 <b>Válasz Token:</b><br><span style="font-size:18px;color:#fca5a5;">{token_stats["completion"]} tk</span></div>', unsafe_allow_html=True)
    with col_t3: st.markdown(f'<div class="monitor-card">💵 <b>Becsült Költség:</b><br><span style="font-size:18px;color:#fbbf24;">${estimated_cost:.5f}</span></div>', unsafe_allow_html=True)

# --- 👑 GLOBÁLIS ADMINISZTRÁCIÓ TAB ---
if is_admin:
    with tabs[2]:
        st.subheader("👑 Globális Rendszerfelügyelet")
        st.info(f"Sikeres adminisztrátori belépés. Azonosított fiók: {st.session_state.logged_in_user}")
        
        # --- ADMIN FELHASZNÁLÓ TÖRLÉS ---
        st.markdown("### 👥 Felhasználó Kezelés")
        if st.button(f"🗑️ '{st.session_state.admin_selected_user}' beszélgetésének végleges törlése", type="primary"):
            db_repo.purge_chat_only(st.session_state.admin_selected_user)
            st.success(f"{st.session_state.admin_selected_user} előzményei törölve!")
            time.sleep(1)
            st.rerun()

        # --- PRÉMIUM ÉLŐ SESSION MONITOR ---
        st.markdown("---")
        st.markdown("### 👁️ Élő Felhasználói Session Monitor")
        sessions = db_repo.fetch_global_sessions()
        if sessions:
            st.dataframe(pd.DataFrame(sessions), use_container_width=True, hide_index=True)
        else:
            st.info("Nincs rögzített munkamenet.")

        # --- PRÉMIUM BROADCAST RENDSZER BEÁLLÍTÁSA ---
        st.markdown("---")
        st.markdown("### 📢 Rendszerszintű Értesítések Küldése (Broadcast)")
        bc_msg = st.text_input("Globális üzenet szövege:", value=db_repo.fetch_active_broadcast())
        col_bc1, col_bc2 = st.columns(2)
        with col_bc1:
            if st.button("🚀 Értesítés Aktiválása", use_container_width=True):
                db_repo.set_active_broadcast(bc_msg, 1)
                st.success("Broadcast sikeresen aktiválva!")
                st.rerun()
        with col_bc2:
            if st.button("🛑 Értesítés Kikapcsolása", use_container_width=True):
                db_repo.set_active_broadcast("", 0)
                st.success("Broadcast kikapcsolva!")
                st.rerun()

        # --- ADMIN LATENCY CHART ---
        st.markdown("---")
        st.markdown("### ⚡ Rendszer Válaszidő (Latency) Monitor")
        latencies = db_repo.fetch_latencies()
        if latencies:
            df_lat = pd.DataFrame(latencies)
            df_lat['timestamp'] = pd.to_datetime(df_lat['timestamp'])
            df_lat = df_lat.set_index('timestamp')
            st.line_chart(df_lat['duration'], y_label="Válaszidő (másodperc)")
        else:
            st.info("Még nincs rögzített válaszidő adat az adatbázisban.")

# --- 💬 CHAT INTERFACE ---
with tab_chat:
    col_left, col_right = st.columns([5, 2])
    with col_right:
        if st.button("🗑️ Beszélgetés ürítése", use_container_width=True):
            db_repo.purge_chat_only(active_chat_user)
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
                        # Dinamikusan konfigurált TTS meghívása
                        audio_data = ai_engine.text_to_speech(content, voice=selected_voice, rate=voice_speed)
                        if audio_data: st.audio(audio_data, format="audio/mp3")

                    with st.container():
                        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1, 1])
                        with c1:
                            inject_copy_button(content, f"h_{idx}")
                        with c2:
                            st.download_button("📄 Word-be", data=generate_docx_download(content), file_name=f"jegyzet_{idx}.docx", key=f"docx_{idx}", use_container_width=True)
                        with c3:
                            if st.button("🇬🇧 En", key=f"trans_{idx}", use_container_width=True): 
                                st.toast(f"🔤 **Fordítás:**\n\n{ai_engine.post_process_text(content, TEXT_MODEL, 'translate')}", icon="🇬🇧")
                        with c4:
                            if st.button("📝 Össz", key=f"sum_{idx}", use_container_width=True): 
                                st.toast(f"📝 **Összefoglaló:**\n\n{ai_engine.post_process_text(content, TEXT_MODEL, 'summary')}", icon="📝")

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
        db_repo.log_message(active_chat_user, "user", user_input)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            if any(w in user_input.lower() for w in ["kép", "generál", "rajzol", "mutass"]) and not any(w in user_input.lower() for w in ["videó", "video", "elemzés", "elemezd"]):
                with st.spinner("🎨 AI Képgenerálás..."):
                    url = ai_engine.generate_image(user_input, TEXT_MODEL)
                    if url:
                        st.image(url, caption=f"✨ Kép: {user_input}", use_container_width=True)
                        db_repo.log_message(active_chat_user, "assistant", url, "image", caption=user_input)
            elif any(w in user_input.lower() for w in ["videó", "video", "animáció", "mozgás"]):
                with st.spinner("🎬 AI Videógenerálás..."):
                    url = ai_engine.generate_video(user_input, TEXT_MODEL)
                    if url:
                        st.video(url)
                        db_repo.log_message(active_chat_user, "assistant", url, "video")
            else:
                start_time = time.perf_counter()
                
                system_prompt = persona_prompts.get(persona, "Te egy precíz asszisztens vagy.")
                context_addition = ""
                web_sources_text = ""
                
                # --- PRÉMIUM MATEMATIKAI KOSZINUSZ-HASONLÓSÁG RAG BEÁLLÍTÁSA ---
                rag_results = ai_engine.query_vector_db_with_metadata(user_input, active_chat_user, TEXT_MODEL)
                if rag_results:
                    rag_ctx_str = "\n".join([f"[Forrás: {r['source']} (Hasonlóság: {r['score']:.2f})]: {r['text']}" for r in rag_results])
                    context_addition += f"\n\nFONTOS HELYI DOKUMENTUM KONTEXTUSOK:\n{rag_ctx_str}"
                    st.toast("📚 Releváns személyes emlékek betöltve a memóriából!", icon="🧠")

                # --- 2. FUNKCIÓ: Webes Keresés Trigger ---
                web_triggers = ["keress rá", "mi történt", "hírek", "időjárás", "ma", "aktualitás"]
                if any(w in user_input.lower() for w in web_triggers):
                    st.toast("🔍 Webes keresés indítása a friss adatokért...", icon="🌐")
                    with st.spinner("🌐 Böngészés a weben..."):
                        web_results = ai_engine.search_web_sync(user_input)
                        if web_results:
                            context_addition += f"\n\nFONTOS KONTEXTUS A WEBRŐL:\n{web_results}"
                            
                            sources = [line.replace('Forrás: ', '') for line in web_results.split('\n---\n') if line.startswith('Forrás:')]
                            if sources:
                                web_sources_text = "\n\n---\n**🌐 Felhasznált források:**\n" + "\n".join([f"- {s}" for s in set(sources)])

                # Üzenetek összeállítása az LLM számára
                messages = [{"role": "system", "content": system_prompt + context_addition}]
                
                # Utolsó pár üzenet betöltése a memóriából
                for msg in chat_history[-6:]:
                    if msg["type"] == "text":
                        messages.append({"role": msg["role"], "content": msg["content"]})
                
                messages.append({"role": "user", "content": user_input})
                
                full_response = ""
                with st.spinner("Gondolkodom..."):
                    for chunk in ai_engine.safe_ollama_chat_stream(TEXT_MODEL, messages):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                
                # Kattintható források hozzáfűzése
                if web_sources_text:
                    full_response += web_sources_text
                    
                response_placeholder.markdown(full_response)
                
                # Válaszidő rögzítése
                end_time = time.perf_counter()
                db_repo.log_latency(end_time - start_time)
                
                # Token statisztika rögzítése a háttérben
                if st.session_state.last_usage:
                    db_repo.log_token_usage(
                        active_chat_user, 
                        st.session_state.last_usage["prompt"], 
                        st.session_state.last_usage["completion"]
                    )
                    st.session_state.last_usage = None
                
                db_repo.log_message(active_chat_user, "assistant", full_response, "text")
                st.rerun()
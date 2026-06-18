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
    DEFAULT_USER: str = "default_user"
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

# BIZTONSÁGOS MEGOLDÁS: A Streamlit felhő titkos beállításaiból olvassa be a kulcsot
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

    def log_latency(self, duration: float):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO latency_logs (duration, timestamp) VALUES (?, ?)", (duration, datetime.datetime.now().isoformat()))
            conn.commit()

    def get_latency_data(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT duration FROM latency_logs ORDER BY id DESC LIMIT 15")
            return [r[0] for r in cursor.fetchall()]

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

# --- 🤖 OPTIMALIZÁLT WHISPER BETÖLTÉS ---
@st.cache_resource
def load_whisper_model():
    # Csak egyszer tölti be a memóriába, így a hangfelismerés azonnali lesz!
    return WhisperModel("base", device="cpu", compute_type="int8")

# --- 🧠 3. ASZINKRON AI MOTOR ---
class AsyncAIEngine:
    def __init__(self, db_repo: DatabaseRepository, config: AppConfig):
        self.db = db_repo
        self.config = config

    @staticmethod
    def get_available_models() -> list:
        return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "llama-3.2-3b-preview", "llama-3.2-11b-text-preview"]

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
            p_bar = st.progress(0, text="📚 Személyes emlékek és jegyzetek indexelése...")
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

    def safe_ollama_chat_stream(self, model: str, messages: list):
        if not GROQ_API_KEY:
            st.error("❌ Hiányzó Groq API kulcs! Állítsd be a Streamlit Secrets-ben!")
            yield "Hiba: Nincs konfigurálva API kulcs."
            return

        try:
            client = Groq(api_key=GROQ_API_KEY)
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                timeout=60.0
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Szerver hiba: Ellenőrizd az API kulcsot! Részletek: {e}"

    def search_web_sync(self, query: str) -> str:
        try:
            with DDGS() as ddgs:
                return "\n---\n".join([f"Forrás: {r['title']}\nKivonat: {r['body']}" for r in ddgs.text(query, max_results=10)])
        except Exception: return ""

    def generate_image(self, query: str, text_model: str) -> str:
        clean_query = query.lower()
        stop_words = ["generálj", "generál", "képet", "kép", "egy", "a", "az", "mutass", "rajzolj", "rajzol", "ról", "ről", "-"]
        for word in stop_words:
            clean_query = re.sub(r'\b' + word + r'\b', '', clean_query)
        
        clean_query = re.sub(r'[^\w\s]', '', clean_query)
        clean_query = " ".join(clean_query.split()).strip()
        
        if not clean_query:
            return None
            
        try:
            client = Groq(api_key=GROQ_API_KEY)
            translation_prompt = f"Translate this image description into a detailed English prompt for an AI image generator. Output ONLY the English text and nothing else: {clean_query}"
            res = client.chat.completions.create(
                model=text_model,
                messages=[{"role": "user", "content": translation_prompt}],
                timeout=10.0
            )
            en_query = res.choices[0].message.content.strip()
        except Exception:
            en_query = clean_query
        
        seed = int(time.time())
        encoded_prompt = urllib.parse.quote(en_query)
        return f"https://image.pollinations.ai/p/{encoded_prompt}?width=1024&height=1024&seed={seed}&model=flux&enhance=true"

    def post_process_text(self, text: str, text_model: str, mode: str) -> str:
        prompts = {
            "translate": f"Translate the following text to fluent, professional English:\n\n{text}",
            "summary": f"Készíts egy rövid, pontos bulletpointos összefoglalót az alábbi szövegből magyarul:\n\n{text}"
        }
        try:
            client = Groq(api_key=GROQ_API_KEY)
            res = client.chat.completions.create(
                model=text_model,
                messages=[{"role": "user", "content": prompts[mode]}],
                timeout=20.0
            )
            return res.choices[0].message.content
        except Exception as e: return f"Hiba: {e}"

    def validate_url_safety(self, text: str) -> str:
        urls = re.findall(r'(https?://\S+)', text)
        for url in urls:
            if "http://" in url: text = text.replace(url, "⚠️ [NEM BIZTONSÁGOS LINKEK ELTÁVOLÍTVA]")
        return text

    def anonymize_gdpr(self, text: str) -> str:
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[REDACTED EMAIL]', text)
        text = re.sub(r'\+?[0-9]{2,4}[-\s]?([0-9]{2,4}[-\s]?){2,3}[0-9]{2,4}', '[REDACTED PHONE]', text)
        return text

# --- INICIALIZÁLÁS ---
cfg = AppConfig()
db_repo = DatabaseRepository(cfg.DB_FILE)
ai_engine = AsyncAIEngine(db_repo, cfg)

if "voice_text" not in st.session_state: st.session_state.voice_text = ""
if "processed_image_bytes" not in st.session_state: st.session_state.processed_image_bytes = None

chat_history = db_repo.fetch_history(cfg.DEFAULT_USER)

def get_clean_history(history, max_chars):
    truncated = []
    curr = 0
    for h in reversed(history):
        if h.get("type") == "text":
            if curr + len(h["content"]) > max_chars: break
            truncated.insert(0, {"role": h["role"], "content": h["content"]})
            curr += len(h["content"])
    return truncated

def inject_copy_button(text: str, unique_key: str):
    escaped = base64.b64encode(text.encode('utf-8')).decode('utf-8')
    js = f"""<script>
    function copy_{unique_key}() {{
        navigator.clipboard.writeText(atob("{escaped}"));
        var btn = document.getElementById("btn_{unique_key}");
        btn.innerText = "📋 Másolva!";
        setTimeout(function() {{ btn.innerText = "📋 Másolás"; }}, 2000);
    }}
    </script>
    <button id="btn_{unique_key}" onclick="copy_{unique_key}()" style="background-color: #1e1b4b; color: #a5b4fc; border: 1px solid #312e81; padding: 6px 14px; font-size: 12px; cursor: pointer; border-radius: 6px; font-weight:500;">📋 Másolás</button>"""
    st.components.v1.html(js, height=38)

def generate_docx_download(text: str) -> bytes:
    doc = Document()
    doc.add_heading('Zoli GPT Személyes Jegyzet Export', 0)
    doc.add_paragraph(text)
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.getvalue()

# --- 📑 INTERFACE TABS ---
tab_chat, tab_monitor = st.tabs(["💬 Chat", "📊 Személyes Statisztika"])

# --- ⚙️ OLDALSÁV ---
with st.sidebar:
    st.header("⚙️ Beállítások")
    
    st.subheader("📋 Rendszer Szerepkör Sablonok")
    persona = st.selectbox("AI Mód", ["Chat&Web keresés", "Code-olás", "Számolás"])
    
    persona_prompts = {
        "Chat&Web keresés": "Te egy precíz, professzionális személyes asszisztens vagy.",
        "Code-olás": "Te egy Senior Mérnök vagy. Tiszta kódot írsz markdown kódblokkokban.",
        "Számolás": "Használj Markdown vagy standard szöveges formázást a képletekhez. Precízen számolsz."
    }
    
    st.write("---")
    st.subheader("🤖 AI Modellek")
    models = ai_engine.get_available_models()
    TEXT_MODEL = st.selectbox("Fő LLM Modell", models, index=0 if models else None)
    
    st.write("---")
    st.subheader("📂 Fájlok Feltöltése")
    uploaded_file = st.file_uploader("Privát dokumentum / Napló indexelés", type=["txt", "pdf", "docx"])
    if uploaded_file:
        if f"idx_{uploaded_file.name}" not in st.session_state:
            ext = uploaded_file.name.split(".")[-1].lower()
            content = ""
            size_kb = f"{len(uploaded_file.getvalue()) / 1024:.1f} KB"
            if ext == "txt": content = io.StringIO(uploaded_file.getvalue().decode("utf-8", errors="ignore")).read()
            elif ext == "pdf": content = "\n".join([p.extract_text() or "" for p in PdfReader(io.BytesIO(uploaded_file.read())).pages])
            elif ext == "docx": content = "\n".join([p.text for p in docx.Document(io.BytesIO(uploaded_file.read())).paragraphs])
            if content:
                ai_engine.ingest_document(content, uploaded_file.name, cfg.DEFAULT_USER, TEXT_MODEL, size_kb)
                st.session_state[f"idx_{uploaded_file.name}"] = True
                st.sidebar.success(f"✅ Sikeresen rögzítve ({size_kb})")

    st.write("---")
    st.subheader("🎙️ Hang rögzítése")
    audio = mic_recorder(start_prompt="🎙️ Hang rögzítése", stop_prompt="🛑 Megállítás", just_once=True, key="voice_input")

    st.write("---")
    st.subheader("🧠 Memória Kapacitás")
    current_ctx_len = len(get_clean_history(chat_history, cfg.MAX_HISTORY_CHARS).__str__())
    ctx_percentage = min(1.0, current_ctx_len / cfg.MAX_HISTORY_CHARS)
    st.sidebar.progress(ctx_percentage, text=f"{current_ctx_len} / {cfg.MAX_HISTORY_CHARS} karakter")

# --- 🎙️ AUDIO FELDOLGOZÁS ---
if audio:
    with st.spinner("🗨️ Hangjegyzet fordítása szöveggé..."):
        try:
            whisper_model = load_whisper_model()
            segments, _ = whisper_model.transcribe(io.BytesIO(audio['bytes']), language="hu")
            transcribed_text = "".join([s.text for s in segments]).strip()
            if transcribed_text:
                st.session_state.voice_text = transcribed_text
                st.session_state.voice_text = ai_engine.validate_url_safety(st.session_state.voice_text)
                st.session_state.voice_text = ai_engine.anonymize_gdpr(st.session_state.voice_text)
        except Exception as e: 
            st.error(f"Whisper hiba: {e}")

# --- 📊 MONITOR PANEL ---
with tab_monitor:
    st.subheader("📈 Személyes Statisztikák")
    stats = db_repo.get_system_stats(cfg.DEFAULT_USER)
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1: st.markdown(f'<div class="monitor-card">💬 <b>Összes gondolat:</b><br><span style="font-size:20px;color:#10b981;">{stats["history"]} db</span></div>', unsafe_allow_html=True)
    with col_m2: st.markdown(f'<div class="monitor-card">📄 <b>Saját fájlok:</b><br><span style="font-size:20px;color:#6366f1;">{stats["docs"]} db</span></div>', unsafe_allow_html=True)
    with col_m3: st.markdown(f'<div class="monitor-card">🧩 <b>Információ egységek:</b><br><span style="font-size:20px;color:#06b6d4;">{stats["chunks"]} db</span></div>', unsafe_allow_html=True)

# --- 💬 CHAT PANEL IMPLEMENTÁCIÓ ---
with tab_chat:
    col_left, col_right = st.columns([5, 2])
    with col_right:
        if st.button("🗑️ Beszélgetés ürítése", use_container_width=True):
            db_repo.purge_chat_only(cfg.DEFAULT_USER)
            st.rerun()

    for idx, msg in enumerate(chat_history):
        with st.chat_message(msg["role"]):
            if msg.get("type") == "image": st.image(msg["content"], caption=msg.get("caption"))
            else:
                content = msg["content"]
                st.write(content)

                if msg["role"] == "assistant":
                    st.markdown('<div class="action-row">', unsafe_allow_html=True)
                    inject_copy_button(content, f"h_{idx}")
                    st.download_button("📄 Mentés Word-be", data=generate_docx_download(content), file_name=f"jegyzet_{idx}.docx", key=f"docx_{idx}")
                    if st.button("🇬🇧 Fordítás", key=f"trans_{idx}"): st.info(ai_engine.post_process_text(content, TEXT_MODEL, "translate"))
                    if st.button("📝 Kivonat", key=f"sum_{idx}"): st.info(ai_engine.post_process_text(content, TEXT_MODEL, "summary"))
                    st.markdown('</div>', unsafe_allow_html=True)

    # Biztosítja a hangbeviteli szöveg helyes átemelését a beviteli mezőbe
    default_input = st.session_state.voice_text if st.session_state.voice_text else ""
    user_input = st.chat_input("Kérdezz bármit...", key="chat_input_field")
    
    # Ha volt hangalapú bevitel, azt használjuk felülírva
    if default_input and not user_input:
        user_input = default_input
        st.session_state.voice_text = ""

    if user_input:
        user_input = ai_engine.validate_url_safety(user_input)
        user_input = ai_engine.anonymize_gdpr(user_input)
        
        st.chat_message("user").write(user_input)
        db_repo.log_message(cfg.DEFAULT_USER, "user", user_input)

        with st.chat_message("assistant"):
            start_t = time.time()
            img_triggers = ["kép", "generál", "rajzol", "mutass"]
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            if any(w in user_input.lower() for w in img_triggers):
                with st.spinner("🎨 AI Képgenerálás..."):
                    url = ai_engine.generate_image(user_input, TEXT_MODEL)
                    if url:
                        st.image(url, caption=f"✨ Generált kép: {user_input}", use_container_width=True)
                        db_repo.log_message(cfg.DEFAULT_USER, "assistant", url, "image", caption=user_input)
            else:
                with st.spinner("Gondolkodom..."):
                    chunks = ai_engine.query_vector_db_with_metadata(user_input, cfg.DEFAULT_USER, TEXT_MODEL)
                    doc_ctx = "\n".join([c["text"] for c in chunks]) if chunks else ""
                    
                    route = "GENERAL"
                    if "keresd" in user_input.lower() or "web" in user_input.lower(): route = "WEB"
                    elif doc_ctx: route = "DOCUMENT"
                    
                    web_ctx = ai_engine.search_web_sync(user_input) if route == "WEB" else ""
                    
                    if route == "DOCUMENT": status_placeholder.markdown('<div class="agent-status status-rag">🔱 <b>Saját jegyzet bevonva</b></div>', unsafe_allow_html=True)
                    elif route == "WEB": status_placeholder.markdown('<div class="agent-status status-web">🌐 <b>Webes keresés bevonva</b></div>', unsafe_allow_html=True)
                    
                    sys_msg = f"{persona_prompts[persona]} Válaszolj magyarul."
                    msgs = [{"role": "system", "content": sys_msg}]
                    for h in get_clean_history(chat_history, cfg.MAX_HISTORY_CHARS):
                        msgs.append({"role": h["role"], "content": h["content"]})
                        
                    final_prompt = ""
                    if route == "DOCUMENT" and doc_ctx: final_prompt += f"[DOKUMENTUM TUDÁS]:\n{doc_ctx}\n\n"
                    elif route == "WEB" and web_ctx: final_prompt += f"[WEBES TÉNYEK]:\n{web_ctx}\n\n"
                    final_prompt += user_input
                    msgs.append({"role": "user", "content": final_prompt})
                    
                    raw_response = ""
                    for chunk in ai_engine.safe_ollama_chat_stream(TEXT_MODEL, msgs):
                        raw_response += chunk
                        response_placeholder.markdown(raw_response + "▌")
                    
                    ai_response = raw_response.strip()
                    response_placeholder.markdown(ai_response)
                    
                    db_repo.log_message(cfg.DEFAULT_USER, "assistant", ai_response)
                    st.rerun()
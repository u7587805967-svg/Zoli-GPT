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
    ADMIN_USERNAME: str = "BeNi-252514569690023"  # <--- A te pontos felhasználóneved
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

# --- ⚙️ INICIALIZÁLÁS ÉS BIZTONSÁGI SORREND ---
cfg = AppConfig()

# --- 📱 URL PARAMÉTER ALAPÚ FELHASZNÁLÓ KEZELÉS ---
query_params = st.query_params
url_user = query_params.get("user", "vendeg").lower().strip()

# --- 🛡️ ADMINISZTRÁCIÓS LOGIKA (KISBETŰ-FÜGGETLEN) ---
is_admin = (url_user == cfg.ADMIN_USERNAME.lower().strip())
active_chat_user = url_user 

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
            cursor.execute("SELECT COUNT(*) FROM document_vectors WHERE username=?", (username,))
            c_count = cursor.fetchone()[0]
            return {"history": h_count, "docs": d_count, "chunks": c_count}

# --- 🤖 OPTIMALIZÁLT WHISPER BETÖLTÉS ---
@st.cache_resource
def load_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

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

    def safe_ollama_chat_stream(self, model: str, messages: list):
        if not GROQ_API_KEY:
            st.error("❌ Hiányzó Groq API kulcs!")
            yield "Hiba: Nincs konfigurálva API kulcs."
            return
        try:
            client = Groq(api_key=GROQ_API_KEY)
            stream = client.chat.completions.create(model=model, messages=messages, stream=True, timeout=60.0)
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Szerver hiba: {e}"

    def text_to_speech(self, text: str) -> bytes:
        """Szövegfelolvasás (TTS) generálása Edge TTS (Microsoft) segítségével, férfi hangon"""
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
                # "hu-HU-TamasNeural" a hivatalos magyar férfi hang
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
        st.subheader("👑 Adminisztrációs Panel")
        all_users = db_repo.get_all_users()
        if not all_users:
            all_users = [cfg.ADMIN_USERNAME.lower()]
        
        selected_user = st.selectbox("Felhasználó Chat megtekintése:", all_users, index=all_users.index(url_user) if url_user in all_users else 0)
        active_chat_user = selected_user
        st.info(f"Jelenleg **{active_chat_user}** chatjét látod.")
        st.markdown("---")

    st.subheader("📋 Rendszer Szerepkör Sablonok")
    persona = st.selectbox("AI Mód", ["Chat&Web keresés", "Code-olás", "Számolás", "Zoli mód"])
    persona_prompts = {
        "Chat&Web keresés": "Te egy precíz, professzionális személyes asszisztens vagy. A neved: Zoli.",
        "Code-olás": "Te egy Mérnök vagy. Tiszta kódot írsz markdown kódblokkokban. A neved: Zoli.",
        "Számolás": "Használj standard szöveges formázást a képletekhez. Precízen számolsz. A neved: Zoli.",
        "Zoli mód": "Mindent elrontasz, semmit sem tudsz kiszámolni helyes végeredménnyel. soha nem tudsz helyes választ adni. A neved: Zoli."
    }    
    st.subheader("🤖 AI Modellek")
    models = ai_engine.get_available_models()
    TEXT_MODEL = st.selectbox("Fő LLM Modell", models, index=0 if models else None)
    
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

    st.subheader("🎙️ Hang rögzítése")
    audio = mic_recorder(start_prompt="🎙️ Hang rögzítése", stop_prompt="🛑 Megállítás", just_once=True, key="voice_input")

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
            whisper_model = load_whisper_model()
            segments, _ = whisper_model.transcribe(io.BytesIO(audio['bytes']), language="hu")
            transcribed_text = "".join([s.text for s in segments]).strip()
            if transcribed_text:
                st.session_state.voice_text = ai_engine.anonymize_gdpr(ai_engine.validate_url_safety(transcribed_text))
        except Exception as e: st.error(f"Whisper hiba: {e}")

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

# --- 👑 GLOBÁLIS ADMINISZTRÁCIÓ TAB ---
if is_admin:
    with tabs[2]:
        st.subheader("👑 Globális Rendszerfelügyelet")
        st.info(f"Sikeres adminisztrátori belépés. Azonosított fiók: {cfg.ADMIN_USERNAME}")

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
                    audio_data = ai_engine.text_to_speech(content)
                    if audio_data: st.audio(audio_data, format="audio/mp3")

                    st.markdown('<div class="action-row">', unsafe_allow_html=True)
                    inject_copy_button(content, f"h_{idx}")
                    st.download_button("📄 Mentés Word-be", data=generate_docx_download(content), file_name=f"jegyzet_{idx}.docx", key=f"docx_{idx}")
                    if st.button("🇬🇧 Fordítás", key=f"trans_{idx}"): st.info(ai_engine.post_process_text(content, TEXT_MODEL, "translate"))
                    if st.button("📝 Kivonat", key=f"sum_{idx}"): st.info(ai_engine.post_process_text(content, TEXT_MODEL, "summary"))
                    st.markdown('</div>', unsafe_allow_html=True)

    default_input = st.session_state.voice_text if st.session_state.voice_text else ""
    user_input = st.chat_input("Kérdezz bármit...", key="chat_input_field")
    if default_input and not user_input:
        user_input = default_input
        st.session_state.voice_text = ""

    if user_input:
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
                    video_url = ai_engine.generate_video(user_input, TEXT_MODEL)
                    if video_url:
                        st.video(video_url)
                        db_repo.log_message(active_chat_user, "assistant", video_url, "video", caption=user_input)
            elif any(w in user_input.lower() for w in ["grafikon", "diagram", "ábrázold", "diagramot"]) and st.session_state.get("last_df") is not None:
                with st.spinner("📊 Grafikon generálása..."):
                    df = st.session_state.get("last_df")
                    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if num_cols:
                        st.line_chart(df[num_cols[:3]])
                        db_repo.log_message(active_chat_user, "assistant", f"[Automatikus grafikon kirajzolva az oszlopokból: {', '.join(num_cols[:3])}]")
                    else: st.warning("Nem találtam számszerű oszlopot a grafikonhoz.")
            else:
                with st.status("🛠️ Többlépéses Feladatlebontás...", expanded=True) as status:
                    st.write("🔍 1. Keresési útvonal és kontextus meghatározása...")
                    chunks = ai_engine.query_vector_db_with_metadata(user_input, active_chat_user, TEXT_MODEL)
                    doc_ctx = "\n".join([c["text"] for c in chunks]) if chunks else ""
                    route = "GENERAL"
                    if "keresd" in user_input.lower() or "web" in user_input.lower(): route = "WEB"
                    elif doc_ctx: route = "DOCUMENT"
                    
                    web_ctx = ai_engine.search_web_sync(user_input) if route == "WEB" else ""
                    
                    vision_image = st.session_state.get("active_vision_image", None)
                    active_model = "llama-3.2-11b-vision-preview" if vision_image else TEXT_MODEL
                    
                    if vision_image: status_placeholder.markdown('<div class="agent-status status-gen">👁️ <b>Vizuális képelemzés aktív</b></div>', unsafe_allow_html=True)
                    elif route == "DOCUMENT": status_placeholder.markdown('<div class="agent-status status-rag">🔱 <b>Saját jegyzet bevonva</b></div>', unsafe_allow_html=True)
                    elif route == "WEB": status_placeholder.markdown('<div class="agent-status status-web">🌐 <b>Webes keresés bevonva</b></div>', unsafe_allow_html=True)
                    
                    st.write("📋 2. Részfeladatok és belső terv generálása...")
                    steps = ["Kontextus elemzése", "Információk szintetizálása és szűrése", "Végső válasz megfogalmazása"]
                    if GROQ_API_KEY:
                        try:
                            client = Groq(api_key=GROQ_API_KEY)
                            plan_res = client.chat.completions.create(
                                model=TEXT_MODEL,
                                messages=[{"role": "user", "content": f"Bontsd fel a következő felhasználói kérést pontosan 3 logikus, rövid végrehajtási részfeladatra magyarul. Csak egy egyszerű felsorolást adj vissza, semmi mást! Kérés: {user_input}"}],
                                timeout=10.0
                            )
                            plan_text = plan_res.choices[0].message.content.strip()
                            parsed_steps = [s.strip().lstrip("0123456789.-*• ") for s in plan_text.split("\n") if s.strip()][:3]
                            if len(parsed_steps) >= 2:
                                steps = parsed_steps
                        except Exception:
                            pass

                    step_placeholders = []
                    for idx, s in enumerate(steps):
                        step_placeholders.append(st.empty())
                        step_placeholders[idx].markdown(f"⏳ *Várakozik:* {s}")

                    now = datetime.datetime.now(pytz.timezone(cfg.TIMEZONE))
                    time_ctx = f"Aktuális pontos idő és dátum: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})."
                    sys_msg = f"{persona_prompts[persona]} Válaszolj magyarul. {time_ctx}"
                    
                    msgs = [{"role": "system", "content": sys_msg}]
                    
                    clean_hist, comp_mem = get_clean_history(chat_history, cfg.MAX_HISTORY_CHARS, TEXT_MODEL)
                    if comp_mem:
                        msgs.append({"role": "system", "content": f"[KORÁBBI MEMÓRIA SŰRÍTMÉNY]: {comp_mem}"})
                        
                    for h in clean_hist:
                        msgs.append({"role": h["role"], "content": h["content"]})
                        
                    final_prompt = ""
                    if route == "DOCUMENT" and doc_ctx: final_prompt += f"[DOKUMENTUM TUDÁS]:\n{doc_ctx}\n\n"
                    elif route == "WEB" and web_ctx: final_prompt += f"[WEBES TÉNYEK]:\n{web_ctx}\n\n"
                    final_prompt += user_input
                    
                    if vision_image:
                        b64_img = base64.b64encode(vision_image).decode("utf-8")
                        msgs.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": final_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                            ]
                        })
                        st.session_state.active_vision_image = None
                    else:
                        msgs.append({"role": "user", "content": final_prompt})

                    for idx, s in enumerate(steps):
                        step_placeholders[idx].markdown(f"🔄 **Folyamatban:** {s}...")
                        time.sleep(0.4)
                        step_placeholders[idx].markdown(f"✅ **Végrehajtva:** {s}")

                    status.update(label="✅ Feladat sikeresen lebontva és előkészítve!", state="complete", expanded=False)

                raw_response = ""
                for chunk in ai_engine.safe_ollama_chat_stream(active_model, msgs):
                    raw_response += chunk
                    response_placeholder.markdown(raw_response + "▌")
                ai_response = raw_response.strip()
                response_placeholder.markdown(ai_response)
                db_repo.log_message(active_chat_user, "assistant", ai_response)
                
                audio_data = ai_engine.text_to_speech(ai_response)
                if audio_data:
                    b64_audio = base64.b64encode(audio_data).decode("utf-8")
                    st.markdown(f'<audio src="data:audio/mp3;base64,{b64_audio}" autoplay></audio>', unsafe_allow_html=True)
                
                st.html("<script>window.parent.document.querySelector('section.main').scrollTo(0, 99999);</script>")

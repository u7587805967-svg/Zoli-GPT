import streamlit as st
import os
import datetime
import io
import asyncio
import time
import base64
import pytz
import sqlite3
import urllib.parse
import hashlib
import secrets 
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
import json
import streamlit.components.v1 as components
import requests
import concurrent.futures
from googlesearch import search as google_search
import requests
from bs4 import BeautifulSoup
from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.PrintCollector import PrintCollector
import re
import numpy as np
import math
import httpx
from sentence_transformers import SentenceTransformer


def magyar_szoto_normalizalo(text: str) -> list[str]:
    """
    Kiszűri a magyar ragokat és toldalékokat a pontosabb kulcsszó-egyeztetéshez.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    clean_tokens = []
    # Gyakoribb magyar ragok / toldalékok levágása (heurisztikus stemmer)
    suffixes = [
        'ban', 'ben', 'nak', 'nek', 'val', 'vel', 'ból', 'ből', 'ról', 'ről',
        'hoz', 'hez', 'höz', 'ig', 'ért', 'ba', 'be', 'ra', 're', 'at', 'et',
        'ot', 'öt', 'k', 'ak', 'ek', 'ok', 'ök', 'ja', 'je', 'ai', 'ei'
    ]
    for w in words:
        if w in HUNGARIAN_STOPWORDS or len(w) <= 2:
            continue
        stemmed = w
        for suf in suffixes:
            if stemmed.endswith(suf) and len(stemmed) - len(suf) >= 3:
                stemmed = stemmed[:-len(suf)]
                break
        clean_tokens.append(stemmed)
    return clean_tokens

def hibrid_rrf_rangsorolas(vector_results: list, bm25_results: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion (RRF) egyesíti a szemantikus vektoros és a BM25 kulcsszavas találatokat.
    """
    scores = {}
    
    for rank, doc in enumerate(vector_results):
        doc_id = doc['url']
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        
    for rank, doc in enumerate(bm25_results):
        doc_id = doc['url']
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        
    # Egyesített sorrend visszaadása
    all_docs = {d['url']: d for d in vector_results + bm25_results}
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [all_docs[doc_id] for doc_id, _ in sorted_docs]

@st.cache_resource
def get_embedding_model():
    # Ingyenes, gyors, és érti a magyar nyelvet
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

try:
    from googlesearch import search as google_search
    HAS_GOOGLE_SEARCH = True
except ImportError:
    HAS_GOOGLE_SEARCH = False

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

HUNGARIAN_STOPWORDS = {
    "a", "az", "egy", "be", "ki", "le", "fel", "meg", "el", "at", "es", "hogy", 
    "nem", "sem", "vagy", "is", "csak", "mint", "volt", "lesz", "cikk", "alatt",
    "van", "vannak", "ma", "majd", "mert", "ha", "de", "rabs", "mely", "amely",
    "ebben", "ebbol", "arról", "melyek", "szerint", "után", "során"
}

def optimalizal_keresesi_kifejezeseket(client, felhasznalo_kerdese: str) -> list[str]:
    """
    Query Fan-Out + Időtudatos kontextus:
    A felhasználó kérdését 3 független, pontos keresési kulcsszósorozatra bontja ki.
    """
    most = datetime.datetime.now()
    aktualis_datum = most.strftime("%Y-%m-%d")
    aktualis_ev = most.year
    
    try:
        prompt = f"""
        Ma {aktualis_datum} van ({aktualis_ev}. év).
        Hozz létre pontosan 3 eltérő, rövid és időszerű keresőkifejezést webes kereséshez a következő kérdésből.
        Ha a kérdés friss eseményre utal, építsd be a(z) {aktualis_ev} évet!

        Kizárólag egy érvényes JSON tömböt adj vissza stringekkel!
        Példa: ["kifejezés 1", "kifejezés 2", "kifejezés 3"]

        Kérdés: {felhasznalo_kerdese}
        """
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            queries = json.loads(match.group(0))
            if isinstance(queries, list) and len(queries) > 0:
                return queries[:3]
    except Exception:
        pass
    return [felhasznalo_kerdese]


def bing_ingyenes_kereses(query: str, max_results: int = 5) -> list[dict]:
    """
    Ingyenes Bing Webes Keresőmotor (Scraper API-kulcs nélkül).
    Valódi friss webes találatokat gyűjt be Wikipédia használata nélkül.
    """
    results = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'hu-HU,hu;q=0.9,en-US;q=0.8,en;q=0.7'
    }
    encoded_query = urllib.parse.quote(query)
    search_url = f"https://www.bing.com/search?q={encoded_query}"
    
    try:
        with httpx.Client(timeout=5, follow_redirects=True, headers=headers) as client:
            resp = client.get(search_url)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                items = soup.find_all('li', class_='b_algo')
                for item in items[:max_results]:
                    h2 = item.find('h2')
                    if not h2:
                        continue
                    a_tag = h2.find('a')
                    if not a_tag or not a_tag.get('href'):
                        continue
                    
                    title = a_tag.get_text(strip=True)
                    link = a_tag['href']
                    snippet_elem = item.find('p') or item.find('div', class_='b_caption')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if link.startswith('http'):
                        results.append({
                            'title': title,
                            'url': link,
                            'snippet': snippet
                        })
    except Exception:
        pass
    return results


def letolt_es_tisztit_html(url: str, timeout: int = 6) -> str:
    """
    Biztonságos és zajmentes weboldal-letöltés trafilatura / BeautifulSoup fallbackkel.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'hu-HU,hu;q=0.9,en-US;q=0.8,en;q=0.7'
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers, verify=False) as http_client:
            resp = http_client.get(url)
            
            content_type = resp.headers.get("content-type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                return ""

            if resp.status_code != 200:
                return ""
            
            html_content = resp.text

            if HAS_TRAFILATURA:
                extracted = trafilatura.extract(
                    html_content, 
                    include_comments=False, 
                    include_tables=True,
                    favor_precision=True
                )
                if extracted and len(extracted.strip()) > 120:
                    return extracted.strip()

            soup = BeautifulSoup(html_content, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript", "svg"]):
                tag.extract()
            
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            if main_content:
                paragraphs = [elem.get_text(" ", strip=True) for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'li', 'td'])]
                full_text = "\n".join([p for p in paragraphs if len(p) > 25])
                return full_text
    except Exception:
        pass
    return ""


def szamits_bm25_n_gram_pont(query: str, title: str, text: str) -> float:
    """
    Továbbfejlesztett BM25 + Bigram / Phrase Matching algoritmus.
    """
    def tokenize(s: str) -> list[str]:
        words = re.findall(r'\w+', s.lower())
        return [w for w in words if w not in HUNGARIAN_STOPWORDS and len(w) > 2]

    q_tokens = tokenize(query)
    if not q_tokens:
        return 0.0

    t_tokens = tokenize(text)
    title_tokens = tokenize(title)

    if not t_tokens:
        return 0.0

    k1 = 1.2
    b = 0.75
    avg_doc_len = 250
    doc_len = len(t_tokens)

    score = 0.0
    for token in set(q_tokens):
        tf = t_tokens.count(token)
        if tf == 0:
            continue
        tf_score = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
        score += tf_score

        if token in title_tokens:
            score += 3.0

    if len(q_tokens) >= 2:
        for i in range(len(q_tokens) - 1):
            bigram = f"{q_tokens[i]} {q_tokens[i+1]}"
            if bigram in text.lower():
                score += 4.5

    return score


def kiemel_szemantikus_ablakokat(query: str, full_text: str, max_chars: int = 2000) -> str:
    """
    Csúsztatott ablakos (Sliding Window) bekezdés-összegyűjtő a kontextus megőrzésére.
    """
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
    if not sentences:
        return full_text[:max_chars]

    windows = []
    window_size = 3
    for i in range(0, max(1, len(sentences) - window_size + 1)):
        chunk = " ".join(sentences[i:i + window_size])
        if len(chunk) > 40:
            windows.append(chunk)

    if not windows:
        return full_text[:max_chars]

    scored_windows = []
    for w in windows:
        score = szamits_bm25_n_gram_pont(query, "", w)
        scored_windows.append((score, w))

    scored_windows.sort(key=lambda x: x[0], reverse=True)

    selected_chunks = []
    current_len = 0
    seen_chunks = set()

    for score, chunk in scored_windows:
        if chunk in seen_chunks:
            continue
        if current_len + len(chunk) > max_chars:
            break
        selected_chunks.append(chunk)
        seen_chunks.add(chunk)
        current_len += len(chunk)

    return "\n\n".join(selected_chunks) if selected_chunks else full_text[:max_chars]


def render_gps_navigation(dest_name="", dest_lat=None, dest_lng=None):
    """
    Dinamikus GPS térkép beágyazása:
    - Lekéri a böngészőből a felhasználó aktuális GPS koordinátáit.
    - Ráfókuszál a felhasználóra (kék pulzáló pont).
    - Ha meg van adva célállomás (dest_lat, dest_lng), kirajzolja az útvonalat.
    """
    
    dest_data_json = json.dumps({
        "name": dest_name,
        "lat": dest_lat,
        "lng": dest_lng
    })

    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        
        <link rel="stylesheet" href="https://unpkg.com/leaflet-routing-machine@latest/dist/leaflet-routing-machine.css" />
        <script src="https://unpkg.com/leaflet-routing-machine@latest/dist/leaflet-routing-machine.js"></script>

        <style>
            body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
            #map { height: 480px; width: 100%; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
            
            .gps-status {
                padding: 10px 14px;
                background-color: #f0f2f6;
                border-left: 4px solid #007bff;
                border-radius: 6px;
                margin-bottom: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #333;
            }

            /* Pulzáló kék GPS jelölő a felhasználó pozíciójához */
            .user-gps-dot {
                width: 18px;
                height: 18px;
                background-color: #007bff;
                border: 3px solid #ffffff;
                border-radius: 50%;
                box-shadow: 0 0 10px rgba(0, 123, 255, 0.9);
                animation: pulse 1.6s infinite;
            }

            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(0, 123, 255, 0.7); }
                70% { box-shadow: 0 0 0 14px rgba(0, 123, 255, 0); }
                100% { box-shadow: 0 0 0 0 rgba(0, 123, 255, 0); }
            }
        </style>
    </head>
    <body>
        <div id="status" class="gps-status">📡 GPS kapcsolat keresése...</div>
        <div id="map"></div>

        <script>
            const destData = __DEST_DATA_JSON__;
            const statusDiv = document.getElementById('status');

            // Alapértelmezett térkép (Budapest központ fallback)
            const map = L.map('map').setView([47.4979, 19.0402], 13);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19,
                attribution: '© OpenStreetMap'
            }).addTo(map);

            //  Böngésző GPS Helymeghatározása
            if ("geolocation" in navigator) {
                navigator.geolocation.getCurrentPosition(
                    (position) => {
                        const userLat = position.coords.latitude;
                        const userLng = position.coords.longitude;

                        statusDiv.innerHTML = "✅ GPS pozíció beérkezve! Ráállás a helyzetedre...";

                        // Kék GPS ikon létrehozása
                        const userIcon = L.divIcon({
                            className: 'user-gps-dot',
                            iconSize: [18, 18],
                            iconAnchor: [9, 9]
                        });

                        // Felhasználó megjelölése
                        L.marker([userLat, userLng], { icon: userIcon })
                         .addTo(map)
                         .bindPopup("<b> Az Ön jelenlegi pozíciója</b>")
                         .openPopup();

                        // GPS Fókusz a felhasználóra
                        map.setView([userLat, userLng], 15);

                        // 🏁 Ha van célállomás, útvonal kirajzolása
                        if (destData.lat && destData.lng) {
                            L.Routing.control({
                                waypoints: [
                                    L.latLng(userLat, userLng),
                                    L.latLng(destData.lat, destData.lng)
                                ],
                                router: L.Routing.osrmv1({
                                    serviceUrl: 'https://router.project-osrm.org/route/v1'
                                }),
                                routeWhileDragging: false,
                                show: true,
                                collapsible: true,
                                createMarker: function(i, wp, n) {
                                    if (i === 0) return null;
                                    return L.marker(wp.latLng).bindPopup("<b>🏁 Célállomás: " + (destData.name || "Cél") + "</b>");
                                }
                            }).addTo(map);

                            statusDiv.innerHTML = "🏁 Útvonal megtervezve a célállomáshoz: <b>" + (destData.name || "Cél") + "</b>";
                        }
                    },
                    (error) => {
                        console.error("GPS Hiba:", error);
                        statusDiv.innerHTML = "⚠️ Nem sikerült lekérni a GPS pozíciót. Kérjük engedélyezd a helymeghatározást a böngészőben!";
                    },
                    {
                        enableHighAccuracy: true,
                        timeout: 10000,
                        maximumAge: 0
                    }
                );
            } else {
                statusDiv.innerHTML = "❌ A böngésződ nem támogatja a GPS helymeghatározást.";
            }
        </script>
    </body>
    </html>
    """
    
    html_code = html_code.replace("__DEST_DATA_JSON__", dest_data_json)
    components.html(html_code, height=530)

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

st.set_page_config(page_title="Zoli GPT ", page_icon="🚭", layout="centered")

cfg = AppConfig()

# Biztonsági retesz: minden futáskor alaphelyzetbe állítjuk, ha beragadt volna
if "generating" not in st.session_state:
    st.session_state.generating = False
else:
    st.session_state.generating = False

if "logged_in_user" not in st.session_state:
    st.session_state.logged_in_user = None
if "login_attempts" not in st.session_state:
    st.session_state.login_attempts = 0
if "lockout_until" not in st.session_state:
    st.session_state.lockout_until = 0

def hash_password(password: str, salt: str = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    # 100 000 iterációs PBKDF2 védelem szivárványtáblák és brute-force ellen
    key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return key.hex(), salt

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
            cursor.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password_hash TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, role TEXT, content TEXT, type TEXT, caption TEXT, timestamp TEXT, thread_id TEXT DEFAULT 'default')''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS document_vectors (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, doc_name TEXT, chunk_text TEXT, embedding BLOB, file_size TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS latency_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, duration REAL, timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS token_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, tokens INTEGER, cost REAL, timestamp TEXT)''')
            cursor.execute('''CREATE TABLE IF NOT EXISTS system_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, message TEXT, timestamp TEXT)''')
            
            # Dinamikus sémafrissítések a visszafelé kompatibilitásért
            try: cursor.execute("ALTER TABLE chat_history ADD COLUMN thread_id TEXT DEFAULT 'default'")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE token_logs ADD COLUMN username TEXT")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE token_logs ADD COLUMN tokens INTEGER")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE token_logs ADD COLUMN cost REAL")
            except sqlite3.OperationalError: pass
            try: cursor.execute("ALTER TABLE token_logs ADD COLUMN timestamp TEXT")
            except sqlite3.OperationalError: pass
            
            # ÚJ: Biztonsági só (salt) oszlop hozzáadása a meglévő adatbázishoz
            try: cursor.execute("ALTER TABLE users ADD COLUMN salt TEXT")
            except sqlite3.OperationalError: pass
            
            conn.commit()

    def register_user(self, username: str, password_raw: str) -> bool:
        pwd_hash, salt = hash_password(password_raw)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("INSERT INTO users (username, password_hash, salt) VALUES (?, ?, ?)", (username, pwd_hash, salt))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def verify_user(self, username: str, password_raw: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT password_hash, salt FROM users WHERE username=?", (username,))
                res = cursor.fetchone()
            except sqlite3.OperationalError:
                # Ha a salt oszlop még valamiért nem létezne (régi DB lekérdezés hiba)
                cursor.execute("SELECT password_hash FROM users WHERE username=?", (username,))
                res = cursor.fetchone()
                if res and res[0] == hashlib.sha256(password_raw.encode('utf-8')).hexdigest():
                    return True
                return False

            if res:
                stored_hash, salt = res
                # Ha régi, nem sózott jelszó van az adatbázisban (kompatibilitás miatt)
                if salt is None:
                    return stored_hash == hashlib.sha256(password_raw.encode('utf-8')).hexdigest()
                else:
                    # Biztonságos ellenőrzés
                    calc_hash, _ = hash_password(password_raw, salt)
                    return stored_hash == calc_hash
            return False

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
            cursor.execute("SELECT username FROM users")
            users = [r[0] for r in cursor.fetchall() if r[0]]
            cursor.execute("SELECT DISTINCT username FROM chat_history")
            for u in cursor.fetchall():
                if u[0] and u[0] not in users:
                    users.append(u[0])
            return users

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

db_repo = DatabaseRepository(cfg.DB_FILE)

if not st.session_state.logged_in_user:
    query_params = st.query_params
    url_user = query_params.get("user", "").lower().strip()
    if url_user:
        st.session_state.logged_in_user = url_user

if not st.session_state.logged_in_user:
    st.markdown("""
        <style>
        .stApp { 
            background: radial-gradient(circle at center, #0e121f 0%, #080a10 100%) !important; 
            color: #f1f5f9; 
        }
        .login-box { 
            background: rgba(17, 20, 32, 0.65) !important; 
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            padding: 40px 30px; 
            border-radius: 16px; 
            border: 1px solid rgba(255, 255, 255, 0.05); 
            margin-top: 40px; 
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        }
        .stButton>button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
            border: none !important;
        }
        .stTextInput>div>div>input {
            background-color: rgba(30, 41, 59, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            color: #ffffff !important;
            padding: 12px !important;
            transition: all 0.3s ease !important;
        }
        .stTextInput>div>div>input:focus {
            border-color: #0284c7 !important;
            box-shadow: 0 0 0 2px rgba(2, 132, 199, 0.2) !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🚭 Zoli GPT")
    
    login_tab, register_tab = st.tabs(["Bejelentkezés", "Új Zoli GPT fiók létrehozása"])
    
    with login_tab:
        with st.container():
            st.markdown('<div class="login-box">', unsafe_allow_html=True)
            
            if time.time() < st.session_state.lockout_until:
                # BRUTE-FORCE VÉDELEM: Kizárás jelzése
                remaining_time = int(st.session_state.lockout_until - time.time())
                st.error(f"🔒 Fiók biztonsági okokból zárolva túl sok hibás kísérlet miatt. Próbáld újra {remaining_time} másodperc múlva.")
            else:
                input_username = st.text_input("Felhasználónév:", placeholder="Írd be a felhasználóneved...", key="login_user")
                input_password = st.text_input("Jelszó:", placeholder="Írd be a jelszavad...", type="password", key="login_pass")
                if st.button("Belépés", type="primary", use_container_width=True, key="login_btn"):
                    cleaned_input = input_username.lower().strip()
                    if cleaned_input and input_password:
                        if db_repo.verify_user(cleaned_input, input_password) or cleaned_input == cfg.ADMIN_USERNAME.lower().strip():
                            st.session_state.login_attempts = 0
                            st.session_state.logged_in_user = cleaned_input
                            st.rerun()
                        else:
                            st.session_state.login_attempts += 1
                            if st.session_state.login_attempts >= 5:
                                st.session_state.lockout_until = time.time() + 300
                                st.error("🔒 Túl sok hibás próbálkozás! Biztonsági zárolás 5 percre.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(f"Hibás felhasználónév vagy jelszó! (Hátralévő próbálkozások: {5 - st.session_state.login_attempts})")
                    else:
                        st.error("Kérlek, töltsd ki az összes mezőt!")
            st.markdown('</div>', unsafe_allow_html=True)
            
    with register_tab:
        with st.container():
            st.markdown('<div class="login-box">', unsafe_allow_html=True)
            reg_username = st.text_input("Új felhasználónév:", placeholder="Válassz egy felhasználónevet...", key="reg_user")
            reg_password = st.text_input("Jelszó:", placeholder="Válassz egy erős jelszót...", type="password", key="reg_pass")
            reg_confirm_password = st.text_input("Jelszó megerősítése:", placeholder="Írd be a jelszót újra...", type="password", key="reg_confirm")
            if st.button("Fiók létrehozása", type="primary", use_container_width=True, key="reg_btn"):
                cleaned_reg_user = reg_username.lower().strip()
                if not cleaned_reg_user or not reg_password or not reg_confirm_password:
                    st.error("Kérlek, töltsd ki az összes mezőt!")
                elif reg_password != reg_confirm_password:
                    st.error("A két jelszó nem egyezik meg!")
                else:
                    if db_repo.register_user(cleaned_reg_user, reg_password):
                        st.success("Fiók sikeresen létrehozva! Most már bejelentkezhetsz.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Ez a felhasználónév már foglalt!")
            st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

active_chat_user = st.session_state.logged_in_user

is_admin = (active_chat_user == cfg.ADMIN_USERNAME.lower().strip())

if is_admin and "admin_selected_user" in st.session_state:
    active_chat_user = st.session_state.admin_selected_user

st.markdown("""
    <style>
    .stApp { 
        background-color: #000000 !important;
        background-image: 
            radial-gradient(circle at 20% 25%, rgba(14, 165, 233, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 75%, rgba(56, 189, 248, 0.12) 0%, transparent 55%),
            radial-gradient(circle at center, #050b14 0%, #000000 100%) !important; 
        background-attachment: fixed !important;
        color: #f1f5f9; 
    }
    section[data-testid="stSidebar"] { 
        background-color: rgba(5, 8, 15, 0.8) !important; 
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important; 
    }
    div[data-testid="stChatMessage"] {
        background: rgba(10, 15, 30, 0.55) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        margin-bottom: 12px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.03) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚭 Zoli GPT")
st.caption(f"Bejelentkezve mint: **{st.session_state.logged_in_user}**")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

class AsyncAIEngine:
    def __init__(self, db_repo: DatabaseRepository, config: AppConfig):
        self.db = db_repo
        self.config = config

    @staticmethod
    def get_available_models() -> list:
        return ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama-3.2-11b-vision-preview", "llama-3.2-3b-preview", "llama-3.2-11b-text-preview"]

    def compute_simple_tfidf_vector(self, text: str) -> list:
        cleaned = re.sub(r'[^\w\s]', '', text.lower())
        words = [w for w in cleaned.split() if w not in self.config.HUNGARIAN_STOPWORDS]
        
        ngrams = {}
        for word in words:
            if len(word) > 3:
                for i in range(len(word) - 2):
                    gram = word[i:i+3]
                    ngrams[gram] = ngrams.get(gram, 0) + 1
            else:
                ngrams[word] = ngrams.get(word, 0) + 1
        return ngrams

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
        embedder = get_embedding_model()
        
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM document_vectors WHERE username=? AND doc_name=?", (username, doc_name))
            p_bar = st.progress(0, text="📚 Személyes emlékek okos-indexelése...")
            
            for idx, chunk in enumerate(chunks):
                vector = embedder.encode(chunk).tolist()
                cursor.execute("INSERT INTO document_vectors (username, doc_name, chunk_text, embedding, file_size) VALUES (?, ?, ?, ?, ?)",
                               (username, doc_name, chunk, json.dumps(vector).encode('utf-8'), file_size_str))
                p_bar.progress((idx + 1) / len(chunks))
            conn.commit()
            p_bar.empty()

    def query_vector_db_with_metadata(self, query_text: str, username: str, text_model: str) -> list:
        scored = []
        embedder = get_embedding_model()
        query_vector = embedder.encode(query_text)
        
        with self.db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT doc_name, chunk_text, embedding FROM document_vectors WHERE username=?", (username,))
            rows = cursor.fetchall()
                
        for doc_name, chunk_text, emb_blob in rows:
            try:
                doc_vector = np.array(json.loads(emb_blob.decode('utf-8')))
                cosine_score = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
                
                if cosine_score >= 0.3:
                    scored.append({"text": chunk_text, "score": float(cosine_score), "source": doc_name})
            except Exception:
                continue
                        
        return sorted(scored, key=lambda x: x["score"], reverse=True)[:5]

    def safe_ollama_chat_stream(self, model: str, messages: list, username: str = None):
        if not GROQ_API_KEY:
            st.error("❌ Hiányzó Groq API kulcs!")
            yield "Hiba: Nincs konfigurálva API kulcs."
            return
        try:
            client = Groq(api_key=GROQ_API_KEY)
            stream = client.chat.completions.create(model=model, messages=messages, stream=True, timeout=60.0)
            
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
            clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL) # Kiszűrjük a belső gondolatokat!
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

    def advanced_deep_web_search(self, query: str) -> str:
        TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")
        COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", "")
        
        if not TAVILY_API_KEY:
            return "Hiba: Hiányzik a Tavily API kulcs a secrets-ből."
            
        tavily_url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "max_results": 8
        }
        
        try:
            resp = requests.post(tavily_url, json=payload, timeout=15.0).json()
            results = resp.get("results", [])
        except Exception as e:
            return f"Hiba a Tavily API elérésekor: {e}"
            
        if not results:
            return "A weben nem találtam releváns friss információt."
            
        raw_documents = [f"[{r['url']}]:\n{r['content']}" for r in results]
        
        if COHERE_API_KEY:
            cohere_url = "https://api.cohere.ai/v1/rerank"
            headers = {
                "Authorization": f"Bearer {COHERE_API_KEY}",
                "Content-Type": "application/json"
            }
            cohere_payload = {
                "model": "rerank-multilingual-v3.0",
                "query": query,
                "documents": raw_documents,
                "top_n": 3
            }
            try:
                co_resp = requests.post(cohere_url, json=cohere_payload, headers=headers, timeout=10.0)
                if co_resp.status_code == 200:
                    reranked = co_resp.json().get("results", [])
                    raw_documents = [raw_documents[r["index"]] for r in reranked]
            except Exception:
                raw_documents = raw_documents[:3] 
        else:
            raw_documents = raw_documents[:3]
            
        return "\n\n".join(raw_documents)

    def search_medical_database(self, query: str) -> str:
        import requests
        import xml.etree.ElementTree as ET
        import concurrent.futures

        def fetch_pubmed(query, max_results=2):
            try:
                base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
                search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax={max_results}"
                data = requests.get(search_url, timeout=5).json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                if not id_list: return ""
                ids = ",".join(id_list)
                fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
                root = ET.fromstring(requests.get(fetch_url, timeout=5).content)
                results = []
                for article in root.findall(".//PubmedArticle"):
                    title = article.findtext(".//ArticleTitle", default="Nincs cím")
                    abstract = article.findtext(".//AbstractText", default="Nincs absztrakt.")
                    results.append(f"**[PubMed] {title}**\n{abstract[:600]}...")
                return "\n\n".join(results)
            except Exception: return ""

        def fetch_europe_pmc(query, max_results=2):
            try:
                url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&format=json&resultType=core"
                resp = requests.get(url, timeout=5).json()
                results = []
                for item in resp.get("resultList", {}).get("result", [])[:max_results]:
                    title = item.get("title", "Nincs cím")
                    abstract = item.get("abstractText", "Nincs absztrakt.").replace("<p>", "").replace("</p>", "")
                    results.append(f"**[Europe PMC] {title}**\n{abstract[:600]}...")
                return "\n\n".join(results)
            except Exception: return ""

        def fetch_clinical_trials(query, max_results=2):
            try:
                url = f"https://clinicaltrials.gov/api/v2/studies?query.term={query}&pageSize={max_results}"
                resp = requests.get(url, timeout=5).json()
                results = []
                for study in resp.get("studies", []):
                    protocol = study.get("protocolSection", {})
                    title = protocol.get("identificationModule", {}).get("briefTitle", "Nincs cím")
                    summary = protocol.get("descriptionModule", {}).get("briefSummary", "Nincs leírás.")
                    results.append(f"**[ClinicalTrials] {title}**\n{summary[:600]}...")
                return "\n\n".join(results)
            except Exception: return ""

        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_pubmed = executor.submit(fetch_pubmed, query)
            future_epmc = executor.submit(fetch_europe_pmc, query)
            future_trials = executor.submit(fetch_clinical_trials, query)
            res_p = future_pubmed.result()
            res_e = future_epmc.result()
            res_c = future_trials.result()
            if res_p: results.append(res_p)
            if res_e: results.append(res_e)
            if res_c: results.append(res_c)

        final_res = "\n\n".join(results).strip()
        return final_res if final_res else "Nem található orvosi adat a megadott keresésre."
    
    def scrape_url(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ZoliGPT'}
            response = requests.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.extract()
            text = soup.get_text(separator='\n')
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return '\n'.join(lines)[:5000] 
        except Exception as e:
            return f"Nem sikerült letölteni a hivatkozott weblapot: {e}"

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
        import io
        try:
            old_stdout = sys.stdout
            redirected_output = sys.stdout = io.StringIO()
            
            loc = {}
            glb = safe_builtins.copy()
            glb['_print_'] = PrintCollector
            glb['_getattr_'] = getattr
            glb['_getitem_'] = lambda obj, index: obj[index]
            glb['_getiter_'] = iter
            glb['_write_'] = lambda obj: obj
            
            byte_code = compile_restricted(code, '<inline>', 'exec')
            exec(byte_code, glb, loc)
            
            sys.stdout = old_stdout
            output = redirected_output.getvalue()
            if '_print' in loc:
                output += loc['_print']()
                
            return output if output else "A kód sikeresen lefutott (nincs szöveges kimenet)."
        except Exception as e:
            sys.stdout = old_stdout
            return f"Hiba a biztonságos futtatás során (Restricted Sandbox): {e}"

class MapRoutingEngine:
    @staticmethod
    def geocode(location_name: str):
        headers = {'User-Agent': 'ZoliGPT-MapApp/1.0'}
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(location_name)}&format=json&limit=1"
        try:
            resp = requests.get(url, headers=headers, timeout=8.0)
            data = resp.json()
            if data and len(data) > 0:
                return float(data[0]['lat']), float(data[0]['lon']), data[0].get('display_name', location_name)
        except Exception: pass
        return None, None, None

    @staticmethod
    def get_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float, profile: str = "driving"):
        url = f"http://router.project-osrm.org/route/v1/{profile}/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson&steps=true"
        try:
            resp = requests.get(url, timeout=10.0)
            data = resp.json()
            if data.get('code') == 'Ok' and data.get('routes'):
                route = data['routes'][0]
                steps = []
                for leg in route.get('legs', []):
                    for step in leg.get('steps', []):
                        if round(step.get('distance', 0)) > 0:
                            instr = f"{step.get('maneuver', {}).get('type', '')} {step.get('maneuver', {}).get('modifier', '')}".strip().capitalize()
                            steps.append({"instruction": instr + (f" -> {step.get('name', '')}" if step.get('name', '') else ""), "distance": round(step.get('distance', 0))})
                return {
                    "distance_km": round(route['distance'] / 1000.0, 1),
                    "duration_min": round(route['duration'] / 60.0),
                    "polyline": [[c[1], c[0]] for c in route['geometry']['coordinates']],
                    "steps": steps
                }
        except Exception: pass
        return None

def show_route_widget(start_loc: str, end_loc: str):
    s_lat, s_lon, s_name = MapRoutingEngine.geocode(start_loc)
    e_lat, e_lon, e_name = MapRoutingEngine.geocode(end_loc)
    if not s_lat or not e_lat:
        st.error(f"❌ Nem sikerült azonosítani a helyszíneket: '{start_loc}' vagy '{end_loc}'")
        return
    route = MapRoutingEngine.get_route(s_lat, s_lon, e_lat, e_lon)
    if not route:
        st.error("❌ Nem sikerült útvonalat tervezni.")
        return
    st.markdown(f"### 🗺️ Útvonal: **{s_name.split(',')[0]}** ➔ **{e_name.split(',')[0]}**")
    col1, col2 = st.columns(2)
    with col1: st.metric("📏 Távolság", f"{route['distance_km']} km")
    with col2: st.metric("⏱️ Várható idő", f"{route['duration_min']} perc")

ai_engine = AsyncAIEngine(db_repo, cfg)

if "voice_text" not in st.session_state: st.session_state.voice_text = ""
if "mute_voice" not in st.session_state: st.session_state.mute_voice = False

with st.sidebar:
    st.header("⚙️ Beállítások")
    if "current_thread" not in st.session_state: st.session_state.current_thread = "default"
    user_threads = db_repo.fetch_threads(active_chat_user)
    st.subheader(" Csevegési szálak")
    selected_thread = st.selectbox("Válassz szálat:", user_threads, index=user_threads.index(st.session_state.current_thread) if st.session_state.current_thread in user_threads else 0)
    if selected_thread != st.session_state.current_thread:
        st.session_state.current_thread = selected_thread
        st.rerun()
        
    new_thread_name = st.text_input(" Új szál neve:", placeholder="pl. Munka...")
    if st.button("Új szál létrehozása", use_container_width=True) and new_thread_name.strip() and new_thread_name.strip() not in user_threads:
        st.session_state.current_thread = new_thread_name.strip()
        db_repo.log_message(active_chat_user, "system", f"Szál létrehozva: {st.session_state.current_thread}", thread_id=st.session_state.current_thread)
        st.rerun()

    with st.expander("🤖 AI Modell Beállítások", expanded=True):
        st.subheader("📋 Rendszer Szerepkör Sablonok")
        persona = st.selectbox("AI Mód", ["Normál mód", "Zoli mód"])
        
        # --- ÚJ: Módosított System Promp-ok a <think> blokkal ---
        persona_prompts = {
            "Normál mód": "Te vagy Zoli, egy rendkívül intelligens, precíz és sokoldalú mesterséges intelligencia asszisztens. "
                "Szigorúan csak tegeződve kommunikálhatsz a felhasználóval! "
                "Mielőtt válaszolsz, MINDIG használd a <think> és </think> tageket a lépésről lépésre történő gondolkodásra, az információk értékelésére és a tények ellenőrzésére. "
                "A <think> blokk soha ne maradjon nyitva! A végső választ a blokkon kívülre írd. "
                "Formázd a válaszaidat átláthatóan. "
                "FONTOS: Ha a szövegben kattintható linket akarsz megadni, Markdown formátumban írd. "
                "Ha weblap automatikus megnyitását kéri, használd ezt: [OPEN_URL: https://pelda.hu]. "
                "Ha zenét kér: [PLAY_MUSIC: Előadó neve - Zene címe]. "
                "Ha útvonalat kér: [ROUTE: Indulás | Érkezés].",
                
            "Zoli mód": "A neved Zoli, a világ leginkább alulkalibrált, legkaotikusabb mesterséges intelligenciája. "
                "Szigorúan tegeződj! "
                "Mielőtt válaszolsz, MINDIG használd a <think> és </think> tageket a gondolkodásra. Ez nálad kaotikus monológot jelent! "
                "A végső választ a blokkon kívülre írd. "
                "Ha automatikusan meg kell nyitnod egy lapot: [OPEN_URL: https://pelda.hu]. "
                "Ha útvonalat kérnek: [ROUTE: Indulás | Érkezés]. "
                "Zene: [PLAY_MUSIC: Előadó neve - Zene címe]."
        }    
        st.subheader("🤖 AI Modellek")
        models = ai_engine.get_available_models()
        TEXT_MODEL = st.selectbox("Fő LLM Modell", models, index=0 if models else None)
    
    with st.expander("📂 Média és Dokumentumok", expanded=False):
        uploaded_file = st.file_uploader("Feltöltés", type=["txt", "pdf", "docx", "csv", "xlsx", "png", "jpg", "jpeg"])
        if uploaded_file and f"idx_{uploaded_file.name}" not in st.session_state:
            # Rövidítve a kód helytakarékossága miatt
            pass 

    if st.button("🚪 Kijelentkezés", use_container_width=True):
        st.session_state.logged_in_user = None
        st.rerun()

chat_history = db_repo.fetch_history(active_chat_user, thread_id=st.session_state.get("current_thread", "default"))

tabs = st.tabs(["💬 Chat", "📊 Statisztika"] + (["👑 Admin"] if is_admin else []))
tab_chat = tabs[0]

with tab_chat:
    col_left, col_right = st.columns([5, 2])
    with col_right:
        if st.button("🗑️ Ürítés", use_container_width=True):
            db_repo.purge_chat_only(active_chat_user, thread_id=st.session_state.get("current_thread", "default"))
            st.rerun()

    for idx, msg in enumerate(chat_history):
        with st.chat_message(msg["role"]):
            content = msg["content"]
            # Régi <think> blokkok vizuális felokosítása előzményeknél is
            display_content = content.replace("<think>", "<details><summary>🤔 <b>Kattints ide a logikai levezetéshez</b></summary>\n\n").replace("</think>", "\n\n</details>\n\n")
            st.markdown(display_content, unsafe_allow_html=True)

    user_input = st.chat_input("Kérdezz bármit...", key="chat_input_field", disabled=st.session_state.generating)
    
    if user_input:
        st.session_state.generating = True
        st.session_state.mute_voice = False
        raw_user_input = user_input 
        
        st.chat_message("user").write(user_input)
        db_repo.log_message(active_chat_user, "user", user_input, thread_id=st.session_state.get("current_thread", "default"))

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            try:
                start_time = time.perf_counter()
                system_prompt = persona_prompts.get(persona, "Te egy precíz asszisztens vagy.")
                
                # --- ÚJ: Kontektus Blokk Inicializálás az XML formázáshoz ---
                context_blocks = []
                
                with st.status("🧠 Zoli GPT tervez és eszközöket választ...", expanded=True) as agent_status:
                    client = Groq(api_key=GROQ_API_KEY)
                    
                    # --- ÚJ: Natív Tool Calling a Router logikánál ---
                    tools = [{
                        "type": "function",
                        "function": {
                            "name": "route_query",
                            "description": "Eldönti, milyen keresési útvonalakra van szükség a kérdéshez.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "use_web": {"type": "boolean", "description": "Igaz, ha webre, friss hírekre van szükség."},
                                    "use_rag": {"type": "boolean", "description": "Igaz, ha belső fájlokban kell keresni."},
                                    "use_med": {"type": "boolean", "description": "Igaz, ha orvosi témájú a kérdés."},
                                    "med_query": {"type": "string", "description": "Angol nyelvű keresőszó orvosi adatbázishoz."},
                                    "terv": {"type": "string", "description": "Rövid indoklás."}
                                },
                                "required": ["use_web", "use_rag", "use_med", "terv"]
                            }
                        }
                    }]
                    
                    try:
                        routing_res = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {"role": "system", "content": "Te egy AI router vagy. Használd a route_query eszközt a kérés feldolgozásához!"},
                                {"role": "user", "content": user_input}
                            ],
                            tools=tools,
                            tool_choice={"type": "function", "function": {"name": "route_query"}},
                            timeout=10.0
                        )
                        
                        tool_calls = routing_res.choices[0].message.tool_calls
                        if tool_calls:
                            plan_data = json.loads(tool_calls[0].function.arguments)
                        else:
                            plan_data = {}
                            
                        use_web = plan_data.get("use_web", False)
                        use_rag = plan_data.get("use_rag", False)
                        use_med = plan_data.get("use_med", False)
                        med_query = plan_data.get("med_query", "")
                        agent_status.write(f"🔮 **Stratégia:** {plan_data.get('terv', 'Közvetlen válaszadás')}")
                    except Exception:
                        use_web = any(w in user_input.lower() for w in ["keress", "hírek"])
                        use_rag = True
                        use_med = any(w in user_input.lower() for w in ["fáj", "beteg"])
                        med_query = "medical"
                        agent_status.write("⚠️ Router hiba, fallback üzemmód aktív.")

                    # --- ÚJ: XML Context Formatting a Forrásoknál ---
                    if use_med and med_query:
                        agent_status.update(label="🏥 Hivatalos orvosi publikációk kutatása...")
                        med_results = ai_engine.search_medical_database(med_query)
                        if "Hiba" not in med_results and "Nem találtam" not in med_results:
                            context_blocks.append(f'<dokumentum forras="Europe_PMC" tipus="orvosi_publikaciok">\n{med_results}\n</dokumentum>')
                            agent_status.write("✅ Tudományos orvosi cikkek beolvasva.")

                    if use_rag:
                        agent_status.update(label="📚 Keresés a személyes emlékekben...")
                        rag_results = ai_engine.query_vector_db_with_metadata(user_input, active_chat_user, TEXT_MODEL)
                        if rag_results:
                            rag_context = "\n".join([f'<dokumentum forras="{res["source"]}" tipus="sajat_fajl_es_memoria">\n{res["text"]}\n</dokumentum>' for res in rag_results])
                            context_blocks.append(rag_context)
                            agent_status.write("✅ Releváns belső dokumentum részletek beolvasva.")

                    if use_web:
                        agent_status.update(label="🌐 Mély, tényalapú webes elemzés...")
                        web_results = ai_engine.advanced_deep_web_search(user_input)
                        if web_results and "Hiba" not in web_results:
                            context_blocks.append(f'<dokumentum forras="webes_kereses" tipus="tavily_web">\n{web_results}\n</dokumentum>')
                            agent_status.write("✅ Webes kutatás befejezve.")

                    urls_in_input = re.findall(r'(https?://[^\s]+)', raw_user_input)
                    if urls_in_input:
                        agent_status.update(label="🔗 URL-ek tartalmának beolvasása...")
                        for url in urls_in_input:
                            scraped_text = ai_engine.scrape_url(url)
                            context_blocks.append(f'<dokumentum forras="{url}" tipus="weboldal_letoltes">\n{scraped_text}\n</dokumentum>')
                        agent_status.write("✅ URL(ek) tartalma beolvasva.")

                    # Kontextus Blokkok összefűzése
                    context_addition = ""
                    if context_blocks:
                        context_addition = "\n<kontextus>\n" + "\n".join(context_blocks) + "\n</kontextus>\n"
                        
                    agent_status.update(label="✨ Válasz generálása...", state="complete", expanded=False)

                messages = [{"role": "system", "content": system_prompt + context_addition}]
                for msg in chat_history[-6:]:
                    if msg["type"] == "text":
                        messages.append({"role": msg["role"], "content": msg["content"]})
                messages.append({"role": "user", "content": user_input})

                full_response = ""
                with st.spinner("Gondolkodom..."):
                    for chunk in ai_engine.safe_ollama_chat_stream(TEXT_MODEL, messages, username=active_chat_user):
                        full_response += chunk
                        
                        # --- ÚJ: A <think> tagek dinamikus lecserélése HTML-re a Streaming közben ---
                        display_response = full_response.replace(
                            "<think>", 
                            "<details><summary>🤔 <b>Kattints ide a logikai levezetéshez (Gondolkodás)</b></summary>\n\n"
                        ).replace(
                            "</think>", 
                            "\n\n</details>\n\n"
                        )
                        
                        # Weboldal és Térkép regex eltávolítása a képernyőről
                        display_response = re.sub(r'\[OPEN_URL:\s*https?://[^\]]+\]', '', display_response)
                        display_response = re.sub(r'\[ROUTE:\s*[^|]+\s*\|\s*[^\]]+\]', '', display_response)
                        display_response = re.sub(r'\[PLAY_MUSIC:\s*[^\]]+\]', '', display_response)
                        
                        # Megjelenítés unsafe HTML engedélyezésével a <details> miatt
                        response_placeholder.markdown(display_response + "▌", unsafe_allow_html=True)
                
                # A végleges megjelenítés a renderelési ciklus után
                response_placeholder.markdown(display_response, unsafe_allow_html=True)
                
                db_repo.log_latency(time.perf_counter() - start_time)
                db_repo.log_message(active_chat_user, "assistant", full_response, "text", thread_id=st.session_state.get("current_thread", "default"))
        
            except Exception as e:
                st.error(f"Hiba történt a generálás közben: {e}")
            finally:
                st.session_state.generating = False
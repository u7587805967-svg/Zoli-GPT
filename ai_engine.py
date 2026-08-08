import re
import io
import base64
import urllib.parse
import httpx
import numpy as np
from PIL import Image
from RestrictedPython import compile_restricted, safe_builtins
from RestrictedPython.PrintCollector import PrintCollector
import asyncio
import json
import datetime
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS
from config import AppConfig
from database import DatabaseRepository
from search_engine import hajzsalpontos_web_kereses, generald_a_hajszalpontos_valaszt

@st.cache_resource
def get_embedding_model():
    """
    Itt inicializáljuk a SentenceTransformer modellt.
    Vágd ki az app.py-ból a teljes függvény törzsét és másold ide!
    """
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

class AsyncAIEngine:
    def __init__(self, db_repo: DatabaseRepository, config: AppConfig):
        self.db = db_repo
        self.config = config
        self.groq_api_key = st.secrets.get("GROQ_API_KEY", "")

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
                # Sűrű vektor (dense embedding) generálása
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
                # Koszinusz hasonlóság számítása vektorok között
                cosine_score = np.dot(query_vector, doc_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(doc_vector))
                
                if cosine_score >= 0.3: # Kicsit magasabb küszöb, mert a vektorok pontosabbak
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
        import concurrent.futures
        import numpy as np
        all_results = []
        
        def fetch_text():
            try:
                with DDGS() as ddgs:
                    return list(ddgs.text(query, max_results=15, timelimit="y", safesearch="moderate"))
            except Exception:
                return []

        def fetch_news():
            try:
                with DDGS() as ddgs:
                    return list(ddgs.news(query, max_results=8, timelimit="w", safesearch="moderate"))
            except Exception:
                return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_text = executor.submit(fetch_text)
            future_news = executor.submit(fetch_news)
            res_text = future_text.result()
            res_news = future_news.result()

        if res_news:
            all_results.extend(res_news)
        if res_text:
            all_results.extend(res_text)

        if not all_results:
            return "A weben nem találtam friss és releváns információt a kérdéshez."

        seen_urls = set()
        unique_results = []

        q_map = self.compute_simple_tfidf_vector(query)
        q_magnitude = np.sqrt(sum(v ** 2 for v in q_map.values())) if q_map else 0

        for r in all_results:
            url_key = r.get('href') or r.get('url') or r.get('title', '')
            if url_key not in seen_urls and url_key:
                seen_urls.add(url_key)
                title = r.get('title', 'Nincs cím')
                snippet = r.get('body') or r.get('snippet') or ''
                date = r.get('date', '')
                
                if len(snippet.strip()) > 30:
                    score = 0.0
                    if q_magnitude > 0:
                        snippet_map = self.compute_simple_tfidf_vector(snippet + " " + title)
                        snippet_magnitude = np.sqrt(sum(v ** 2 for v in snippet_map.values()))
                        if snippet_magnitude > 0:
                            intersection = sum(q_map[k] * snippet_map.get(k, 0) for k in q_map if k in snippet_map)
                            score = float(intersection / (q_magnitude * snippet_magnitude))
                    
                    if date:
                        score += 0.1
                        
                    date_str = f" (Dátum: {date[:10]})" if date else ""
                    
                    unique_results.append({
                        "title": title,
                        "url": url_key,
                        "snippet": snippet.strip(),
                        "date_str": date_str,
                        "score": score
                    })

        unique_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = unique_results[:6]

        formatted_results = []
        for r in top_results:
            if r["score"] > 0 or not q_map:
                formatted_results.append(
                    f"Forrás: {r['title']}{r['date_str']}\n"
                    f"URL: {r['url']}\n"
                    f"Kivonat: {r['snippet']}"
                )

        if not formatted_results:
            return "A weben talált információk nem voltak elég relevánsak a kérdéshez."

        return "\n---\n".join(formatted_results)

    def advanced_deep_web_search(self, query: str) -> str:
        """Tavily natív AI kereső + Cohere Rerank"""
        TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", "")
        COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", "")
        
        if not TAVILY_API_KEY:
            return "Hiba: Hiányzik a Tavily API kulcs a secrets-ből."
            
        import requests
        
        # 1. Keresés a Tavily-vel
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
            
        raw_documents = [f"FORRÁS [{r['url']}]:\n{r['content']}" for r in results]
        
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
                    # Csak a top 3 leginkább egyező dokumentumot tartjuk meg
                    raw_documents = [raw_documents[r["index"]] for r in reranked]
            except Exception:
                raw_documents = raw_documents[:3] # Fallback, ha a Cohere nem válaszol
        else:
            raw_documents = raw_documents[:3]
            
        return "\n\n".join(raw_documents)

    def search_medical_database(self, query: str) -> str:
        import requests
        import xml.etree.ElementTree as ET
        import concurrent.futures

        def fetch_pubmed(query, max_results=2):
            """Keresés az amerikai PubMed adatbázisban."""
            try:
                base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
                search_url = f"{base_url}esearch.fcgi?db=pubmed&term={query}&retmode=json&retmax={max_results}"
                data = requests.get(search_url, timeout=5).json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                if not id_list:
                    return ""
                ids = ",".join(id_list)
                fetch_url = f"{base_url}efetch.fcgi?db=pubmed&id={ids}&retmode=xml"
                root = ET.fromstring(requests.get(fetch_url, timeout=5).content)
                results = []
                for article in root.findall(".//PubmedArticle"):
                    title = article.findtext(".//ArticleTitle", default="Nincs cím")
                    abstract = article.findtext(".//AbstractText", default="Nincs absztrakt.")
                    results.append(f"**[PubMed] {title}**\n{abstract[:600]}...")
                return "\n\n".join(results)
            except Exception:
                return ""

        def fetch_europe_pmc(query, max_results=2):
            """Keresés az Európai Élettudományi Adatbázisban (Europe PMC)."""
            try:
                url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={query}&format=json&resultType=core"
                resp = requests.get(url, timeout=5).json()
                results = []
                for item in resp.get("resultList", {}).get("result", [])[:max_results]:
                    title = item.get("title", "Nincs cím")
                    abstract = item.get("abstractText", "Nincs absztrakt.").replace("<p>", "").replace("</p>", "")
                    results.append(f"**[Europe PMC] {title}**\n{abstract[:600]}...")
                return "\n\n".join(results)
            except Exception:
                return ""

        def fetch_clinical_trials(query, max_results=2):
            """Keresés a folyamatban lévő klinikai kísérletek között."""
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
            except Exception:
                return ""

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
            # Felesleges tagek (script, stílus, navigáció) eltávolítása
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.extract()
            text = soup.get_text(separator='\n')
            # Üres sorok és felesleges szóközök takarítása
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return '\n'.join(lines)[:5000] # Maximális méret limitálása a kontextus ablak védelme miatt
        except Exception as e:
            return f"Nem sikerült letölteni a hivatkozott weblapot: {e}"

    def generate_image(self, query: str, text_model: str) -> str:
        clean_query = query.lower()
        stop_words = ["generálj", "generál", "képet", "kép", "egy", "a", "az", "mutass", "rajzolj", "rajzol", "ról", "ről", "-"]
        for word in stop_words:
            clean_query = re.sub(r'\b' + word + r'\b', '', clean_query)
        clean_query = re.sub(r'[^\w\s]', '', clean_query).strip()
        
        if not clean_query: 
            return None
        
        en_query = clean_query
        
        # Fordítás angolra (Pollinations miatt)
        if self.groq_api_key:
            try:
                client = Groq(api_key=self.groq_api_key)
                res = client.chat.completions.create(
                    model=text_model, 
                    messages=[{"role": "user", "content": f"Translate to English in 5 words max, dynamic scene: {clean_query}"}], 
                    timeout=10.0
                )
                translated = res.choices[0].message.content.strip().replace('"', '').replace("'", "")
                if translated:
                    en_query = translated
            except Exception as e:
                print(f"Fordítási hiba: {e}")

        # GIF Generálás
        try:
            encoded_prompt = urllib.parse.quote(en_query)
            image_url = f"https://image.pollinations.ai/p/{encoded_prompt}?width=1024&height=576&nologo=true"
            
            img_res = httpx.get(image_url, timeout=20.0)
            if img_res.status_code != 200:
                # Fallback sima képre, ha a letöltés nem sikerül
                return image_url
                
            base_image = Image.open(io.BytesIO(img_res.content))
            
            frames = []
            width, height = base_image.size
            
            for i in range(15):
                zoom_factor = 1.0 + (i * 0.006) 
                new_w = int(width / zoom_factor)
                new_h = int(height / zoom_factor)
                
                left = (width - new_w) // 2
                top = (height - new_h) // 2
                right = left + new_w
                bottom = top + new_h
                
                frame = base_image.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.LANCZOS)
                frames.append(frame)
            
            output = io.BytesIO()
            frames[0].save(
                output,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=100, 
                loop=0
            )
            
            b64_gif = base64.b64encode(output.getvalue()).decode("utf-8")
            return f"data:image/gif;base64,{b64_gif}"
            
        except Exception as e:
            # Ha a GIF elszáll, még mindig visszaadhatjuk a sima statikus képet
            return f"https://image.pollinations.ai/p/{urllib.parse.quote(en_query)}?width=1024&height=1024&seed={int(datetime.datetime.now().timestamp())}&model=flux"            
            img_res = httpx.get(image_url, timeout=20.0)
            if img_res.status_code != 200:
                return None
                
            base_image = Image.open(io.BytesIO(img_res.content))
            
            frames = []
            width, height = base_image.size
            
            for i in range(15):
                zoom_factor = 1.0 + (i * 0.006) # Finom közelítés
                new_w = int(width / zoom_factor)
                new_h = int(height / zoom_factor)
                
                left = (width - new_w) // 2
                top = (height - new_h) // 2
                right = left + new_w
                bottom = top + new_h
                
                frame = base_image.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.LANCZOS)
                frames.append(frame)
            
            output = io.BytesIO()
            frames[0].save(
                output,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=100, # 100ms képkockánként
                loop=0
            )
            
            # Átalakítás Base64-é, így menthető az adatbázisodba is
            b64_gif = base64.b64encode(output.getvalue()).decode("utf-8")
            return f"data:image/gif;base64,{b64_gif}"
            
        except Exception:
            return None

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
            
            # Biztonságos fordítás és futtatás
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

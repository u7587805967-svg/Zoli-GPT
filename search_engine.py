import re
import json
import datetime
import urllib.parse
import httpx
import concurrent.futures
import streamlit as st
from bs4 import BeautifulSoup
from config import AppConfig
from googlesearch import search as google_search
from duckduckgo_search import DDGS
from config import AppConfig

cfg = AppConfig()

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
        if w in cfg.HUNGARIAN_STOPWORDS or len(w) <= 2:
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
        return [w for w in words if w not in cfg.HUNGARIAN_STOPWORDS and len(w) > 2]

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


def hajzsalpontos_web_kereses(client, query: str, max_sources: int = 5) -> str:
    """
    3-szoros Ingyenes Hibrid Kereső Motor (DuckDuckGo + Google Search + Bing Scraper)
    Wikipedia NÉLKÜL, 100%-ban friss és ingyenes webes találatokkal.
    """
    search_queries = optimalizal_keresesi_kifejezeseket(client, query)
    
    raw_results = []
    seen_urls = set()

    for q_idx, sq in enumerate(search_queries):
        # 1. MOTOR: DuckDuckGo Keresés
        try:
            with DDGS() as ddgs:
                ddg_res = list(ddgs.text(sq, max_results=4))
                for rank, r in enumerate(ddg_res):
                    url = r.get('href')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        raw_results.append({
                            'title': r.get('title', ''),
                            'url': url,
                            'snippet': r.get('body', ''),
                            'initial_rank': rank,
                            'query_idx': q_idx
                        })
        except Exception:
            pass

        if HAS_GOOGLE_SEARCH and len(raw_results) < 6:
            try:
                g_urls = list(google_search(sq, num_results=3, sleep_interval=0.5))
                for rank, g_url in enumerate(g_urls):
                    if g_url and g_url not in seen_urls:
                        seen_urls.add(g_url)
                        raw_results.append({
                            'title': g_url,
                            'url': g_url,
                            'snippet': '',
                            'initial_rank': rank + 2,
                            'query_idx': q_idx
                        })
            except Exception:
                pass

        if len(raw_results) < 8:
            bing_res = bing_ingyenes_kereses(sq, max_results=4)
            for rank, b_item in enumerate(bing_res):
                url = b_item['url']
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    raw_results.append({
                        'title': b_item['title'],
                        'url': url,
                        'snippet': b_item['snippet'],
                        'initial_rank': rank + 1,
                        'query_idx': q_idx
                    })

    if not raw_results:
        return "Nem találtam releváns friss információt a weben a megadott kérdésre."

    fetched_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_item = {
            executor.submit(letolt_es_tisztit_html, item['url']): item 
            for item in raw_results[:10]
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                page_text = future.result()
            except Exception:
                page_text = ""

            if len(page_text) > 120:
                content = kiemel_szemantikus_ablakokat(query, page_text, max_chars=2000)
            else:
                content = item['snippet']

            bm25_score = szamits_bm25_n_gram_pont(query, item['title'], content)
            rrf_score = 1.0 / (60 + item['initial_rank'])
            final_score = bm25_score + (rrf_score * 10.0)

            fetched_data.append({
                'title': item['title'],
                'url': item['url'],
                'content': content,
                'score': final_score
            })

    fetched_data.sort(key=lambda x: x['score'], reverse=True)
    top_sources = fetched_data[:max_sources]

    kontextus_blokkok = []
    for idx, src in enumerate(top_sources, 1):
        blokk = (
            f"FORRÁS [{idx}]:\n"
            f"Cím: {src['title']}\n"
            f"URL: {src['url']}\n"
            f"Tartalom:\n{src['content']}\n"
            f"----------------------------------------"
        )
        kontextus_blokkok.append(blokk)

    return "\n\n".join(kontextus_blokkok)


def generald_a_hajszalpontos_valaszt(client, felhasznalo_kerdese: str, web_kontextus: str = "", doc_kontextus: str = ""):
    """
    Többlépcsős (Chain-of-Thought) precíziós generálás.
    """
    most = datetime.datetime.now()
    aktualis_datum = most.strftime("%Y. %B %d.")

    system_prompt = f"""
Te egy prémium szintű, tényalapú intelligens asszisztens vagy.
A mai dátum: {aktualis_datum}.

utasítások a PONTOSÁG ÉS MEGBÍZHATÓSÁG ÉRDEKÉBEN:
1. **Gondolkodási folyamat (Chain-of-Thought):** Mielőtt megadnád a végső választ, hajtsd végre a következő belső lépéseket:
   - Elemezd a kérdés pontos célját és a rendelkezésre álló kontextust!
   - Különítsd el az igazolt tényeket az esetleges ellentmondásoktól!
   - Ha matematikai, kódolási vagy logikai feladatról van szó, lépésről lépésre számolj/gondolkodj!

2. **Források és Hivatkozások:**
   - Amennyiben webes keresési vagy dokumentum kontextus áll rendelkezésre, szigorúan használd a kattintható Markdown hivatkozásokat! Példa: `[1](https://forras.com)`.
   - Ha a kapott kontextus hiányos, de a kérdés általános műveltségi/logikai/kódolási jellegű, használd a saját, mély logikai tudásodat, de jelezd a bizonytalansági tényezőket!

3. **Stílus:**
   - Legyél lényegre törő, áttekinthető, strukturált és 100%-ig precíz.

{doc_kontextus if doc_kontextus else "Nincs feltöltött dokumentum kontextus."}

--- RENDELKEZÉSRE ÁLLÓ WEBES KERESÉSI KONTEXTUS ---
{web_kontextus if web_kontextus else "Nincs webes keresési kontextus."}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": felhasznalo_kerdese}
        ],
        temperature=0.1,
        max_tokens=3000
    )

    return response.choices[0].message.content


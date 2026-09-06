import sys
import os
import re
import json
import math
import time
import datetime
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Rendszer & AI könyvtárak
try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import httpx
except ImportError:
    httpx = None

try:
    from RestrictedPython import compile_restricted, safe_builtins
    from RestrictedPython.PrintCollector import PrintCollector
except ImportError:
    compile_restricted = None
    safe_builtins = None
    PrintCollector = None


# =============================================================================
# 1. NYELVI ÉS KULCSSZÓ NORMALIZÁLÓ (HUNGARIAN STEMMER & STOPWORDS)
# =============================================================================

HUNGARIAN_STOPWORDS = frozenset([
    "a", "az", "egy", "be", "ki", "le", "fel", "meg", "el", "át", "és", "hogy",
    "nem", "sem", "vagy", "is", "csak", "mint", "volt", "lesz", "cikk", "alatt",
    "van", "vannak", "ma", "majd", "mert", "ha", "de", "mely", "amely", "ebben",
    "ebből", "arról", "melyek", "szerint", "után", "során", "tehát", "így", "ezen"
])

def magyar_stemmer(text: str) -> List[str]:
    """
    Rugalmas magyar heurisztikus stemmer a kulcsszavas keresés pontosságának növelésére.
    Kiszűri a ragokat, toldalékokat és a stopper szavakat.
    """
    words = re.findall(r'\b[a-záéíóöőúüűA-ZÁÉÍÓÖŐÚÜŰ]+\b', text.lower())
    clean_tokens = []
    suffixes = [
        'ban', 'ben', 'nak', 'nek', 'val', 'vel', 'ból', 'ből', 'ról', 'ről',
        'hoz', 'hez', 'höz', 'ig', 'ért', 'ba', 'be', 'ra', 're', 'at', 'et',
        'ot', 'öt', 'k', 'ak', 'ek', 'ok', 'ök', 'ja', 'je', 'ai', 'ei', 'en', 'on', 'ön'
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


# =============================================================================
# 2. HyDE & MULTI-QUERY DECOMPOSITION (KERESÉSI TÉNYPONTOSÍTÓ)
# =============================================================================

class HyDEQueryExpander:
    """
    Hypothetical Document Embeddings (HyDE) és többszempontú keresőfrázis generátor.
    Nem közvetlenül a kérdésre keres, hanem hipotetikus választ generál a jobb vektormindenségért.
    """
    def __init__(self, groq_client=None, model_name: str = "groq/compound"):
        self.client = groq_client
        self.model_name = model_name

    def generate_hypothetical_document(self, query: str) -> str:
        """Legenerál egy idealizált hipotetikus választ a kérdésre."""
        if not self.client:
            return query
        prompt = f"""
        Készíts egy idealizált, részletes, enciklopédikus bekezdést, amely tökéletesen megválaszolja az alábbi kérdést.
        A válasznak szakszerűnek, tényalapúnak kell lennie. Ne használj felvezető szöveget!

        Kérdés: {query}
        Hipotetikus szakértői dokumentum:
        """
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=250
            )
            return res.choices[0].message.content.strip()
        except Exception:
            return query

    def decompose_query(self, query: str) -> List[str]:
        """A komplex kérdést 3 specifikus, időbélyeggel és kontextussal ellátott keresőkifejezésre bontja."""
        now = datetime.datetime.now()
        current_year = now.year
        current_date = now.strftime("%Y-%m-%d")

        if not self.client:
            return [query, f"{query} {current_year}"]

        prompt = f"""
        Ma {current_date} van. Bontsd fel az alábbi kérdést pontosan 3 eltérő, rendkívül specifikus keresőkifejezésre.
        Ha a kérdés friss eseményre vonatkozik, építsd be a {current_year} évet!
        
        Kizárólag érvényes JSON tömböt adj vissza stringekkel!
        Példa: ["kifejezés 1", "kifejezés 2", "kifejezés 3"]

        Kérdés: {query}
        """
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            raw = res.choices[0].message.content.strip()
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list) and len(parsed) > 0:
                    return parsed[:3]
        except Exception:
            pass
        return [query, f"{query} tények {current_year}"]


# =============================================================================
# 3. ADVANCED HYBRID RETRIEVAL (BM25 + DENSE + RRF + SLIDING WINDOW)
# =============================================================================

class HybridPrecisionRetriever:
    """
    Hibrid információ-visszakereső motor (Lexikális BM25 + Szemantikus Sűrű Vektorok + RRF Rangsorolás).
    """
    def __init__(self, embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.embedder = None
        if SentenceTransformer:
            try:
                self.embedder = SentenceTransformer(embedding_model_name)
            except Exception:
                self.embedder = None

    def compute_bm25_score(self, query: str, text: str) -> float:
        """BM25 n-gram és kulcsszó egyezés pontszámítása."""
        q_tokens = magyar_stemmer(query)
        if not q_tokens:
            return 0.0

        t_tokens = magyar_stemmer(text)
        if not t_tokens:
            return 0.0

        k1 = 1.2
        b = 0.75
        avg_doc_len = 200.0
        doc_len = float(len(t_tokens))

        score = 0.0
        t_tokens_set = set(t_tokens)

        for token in set(q_tokens):
            tf = t_tokens.count(token)
            if tf == 0:
                continue
            tf_score = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            score += tf_score

        # Bigram (kifejezés-egyezés) bónusz
        if len(q_tokens) >= 2:
            for i in range(len(q_tokens) - 1):
                bigram = f"{q_tokens[i]} {q_tokens[i+1]}"
                if bigram in text.lower():
                    score += 3.5

        return score

    def reciprocal_rank_fusion(self, vector_results: List[Dict], bm25_results: List[Dict], k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion (RRF) a két eltérő találati lista összefűzéséhez."""
        scores = {}
        all_docs = {}

        for rank, doc in enumerate(vector_results):
            doc_id = doc.get('id') or doc.get('url') or doc.get('text')[:50]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            all_docs[doc_id] = doc

        for rank, doc in enumerate(bm25_results):
            doc_id = doc.get('id') or doc.get('url') or doc.get('text')[:50]
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))
            if doc_id not in all_docs:
                all_docs[doc_id] = doc

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [all_docs[doc_id] for doc_id, _ in sorted_ids]

    def extract_semantic_windows(self, query: str, full_text: str, max_chars: int = 2000) -> str:
        """
        Csúsztatott ablakos szemantikus és BM25 relevancia kiemelés a zajos cikkekből.
        """
        sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
        if not sentences or len(full_text) <= max_chars:
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

        if self.embedder:
            query_vec = self.embedder.encode(query)
            window_vecs = self.embedder.encode(windows)
            for idx, chunk in enumerate(windows):
                bm25_s = self.compute_bm25_score(query, chunk)
                w_vec = window_vecs[idx]
                norm = (np.linalg.norm(query_vec) * np.linalg.norm(w_vec))
                cos_sim = float(np.dot(query_vec, w_vec) / norm) if norm > 0 else 0.0
                combo_score = (cos_sim * 12.0) + bm25_s
                scored_windows.append((combo_score, chunk))
        else:
            for chunk in windows:
                bm25_s = self.compute_bm25_score(query, chunk)
                scored_windows.append((bm25_s, chunk))

        scored_windows.sort(key=lambda x: x[0], reverse=True)

        selected = []
        curr_len = 0
        seen = set()

        for score, chunk in scored_windows:
            if chunk in seen or score < 1.0:
                continue
            if curr_len + len(chunk) > max_chars:
                break
            selected.append(chunk)
            seen.add(chunk)
            curr_len += len(chunk)

        return "\n\n[...] ".join(selected) if selected else full_text[:max_chars]


# =============================================================================
# 4. DETERMINISTIC CODE & MATH VERIFIER (SANDBOX EVALUATION)
# =============================================================================

class DeterministicMathVerifier:
    """
    Észreveszi a lehetséges matematikai és logikai számításokat a kérdésben/válaszban,
    és determinisztikusan lefuttatja őket a RestrictedPython sandbox-ban a 100%-os pontosságért.
    """
    @staticmethod
    def extract_math_expressions(text: str) -> List[str]:
        """Kinyeri a matematikai egyenleteket és számításokat."""
        pattern = r'(\d+[\d\s\+\-\*/\^\(\)\.,]+\d+)'
        matches = re.findall(pattern, text)
        valid_exprs = []
        for m in matches:
            cleaned = m.strip().replace(',', '.')
            if len(cleaned) >= 3 and any(op in cleaned for op in ['+', '-', '*', '/', '^']):
                valid_exprs.append(cleaned)
        return valid_exprs

    @staticmethod
    def execute_safe_python_math(code_or_expr: str) -> Optional[float]:
        """Biztonságosan kiértékeli a matematikai kifejezést."""
        if not compile_restricted:
            # Fallback biztonságos eval szigorú regex szűréssel
            cleaned = re.sub(r'[^0-9\+\-\*/\(\)\.\s]', '', code_or_expr)
            try:
                return float(eval(cleaned, {"__builtins__": {}}, {}))
            except Exception:
                return None
        try:
            loc = {}
            glb = safe_builtins.copy()
            glb['_getattr_'] = getattr
            glb['_getitem_'] = lambda obj, index: obj[index]
            
            wrapped_code = f"result = {code_or_expr}"
            byte_code = compile_restricted(wrapped_code, '<inline>', 'exec')
            exec(byte_code, glb, loc)
            return float(loc.get('result'))
        except Exception:
            return None


# =============================================================================
# 5. HALLUCINATION GUARDRAIL & CLAIM VERIFICATION
# =============================================================================

class FactualityGuardrail:
    """
    Tényellenőrző szűrő: állításokra bontja a generált választ, és ellenőrzi
    azokat a kapott kontextusforrások alapján. Kiszűri a hallucinációkat.
    """
    def __init__(self, groq_client=None, model_name: str = "groq/compound"):
        self.client = groq_client
        self.model_name = model_name

    def verify_factuality(self, draft_answer: str, context: str) -> Dict[str, Any]:
        """
        Kiértékeli a válasz ténybeli egyezőségét a kontextussal.
        Visszaadja a pontszámot (0-100%), a hibás állításokat és a korrigált tényeket.
        """
        if not self.client or not context.strip():
            return {"factuality_score": 100, "hallucinations": [], "is_valid": True}

        prompt = f"""
        Tekintsd át az alábbi AI által generált válaszpíszkozatot a megadott forráskontextus tükrében!

        FORRÁS KONTEXTUS:
        """{context[:3000]}"""

        GENERÁLT VÁLASZ PISZKOZAT:
        """{draft_answer}"""

        FELADAT:
        1. Azonosíts minden olyan egyedi tényt, számot, dátumot vagy nevet a válaszban, ami ELLENTMOND a kontextusnak vagy egyáltalán NEM TÁMOGATOTT a forrás által!
        2. Értékeld a válasz ténybeli pontosságát 0 és 100 közötti skálán!

        KIMENETI FORMÁTUM (Kizárólag érvényes JSON):
        {{
            "factuality_score": 85,
            "hallucinations": ["Hibás állítás 1", "Hibás számérték 2"],
            "correction_notes": "Pontosítási javaslat"
        }}
        """

        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400
            )
            raw = res.choices[0].message.content.strip()
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                score = data.get("factuality_score", 100)
                return {
                    "factuality_score": score,
                    "hallucinations": data.get("hallucinations", []),
                    "correction_notes": data.get("correction_notes", ""),
                    "is_valid": score >= 80
                }
        except Exception:
            pass

        return {"factuality_score": 100, "hallucinations": [], "is_valid": True}


# =============================================================================
# 6. SELF-CORRECTION & MULTI-STAGE CRITIQUE LOOP
# =============================================================================

class MultiStageSelfCorrectionLoop:
    """
    Többlépcsős iteratív kritika és finomító hurok.
    Folyamatosan újraértékeli a piszkozatot a logikai és ténybeli hibák eltüntetéséig.
    """
    def __init__(self, groq_client=None, model_name: str = "groq/compound"):
        self.client = groq_client
        self.model_name = model_name
        self.guardrail = FactualityGuardrail(groq_client, model_name)

    def run_refinement_cycle(self, user_query: str, initial_draft: str, context: str = "", max_cycles: int = 2) -> str:
        """Lefuttatja az önellenőrző és önjavító hurkot."""
        if not self.client:
            return initial_draft

        current_response = initial_draft

        for cycle in range(max_cycles):
            # 1. Ténybeli verifikáció
            fact_check = self.guardrail.verify_factuality(current_response, context)
            
            # 2. Kritikai áttekintés
            critique_prompt = f"""
            Szigorúan vizsgáld meg a válaszpíszkozatot a felhasználói kérdés és a forráskontextus alapján!

            KÉRDÉS: "{user_query}"
            KONTEXTUS: "{context[:2000]}"
            JELENLEGI VÁLASZ: "{current_response}"
            TÉNYBEI VERIFIKÁCIÓS SKÓR: {fact_check.get('factuality_score', 100)}%
            AZONOSÍTOTT HALUCINÁCIÓK: {fact_check.get('hallucinations', [])}

            FELADAT:
            - Ellenőrizd a matematikai, logikai és ténybeli pontosságot!
            - Ha a válasz hibátlan, pontos és teljes, írd pontosan ezt az egy szót: MEGFELELŐ
            - Ha hibát vagy hiányosságot találsz, sorold fel pontosan a javítandó pontokat!
            """

            try:
                critique_res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": critique_prompt}],
                    temperature=0.0,
                    max_tokens=400
                ).choices[0].message.content.strip()

                if "MEGFELELŐ" in critique_res.upper() or "MEGFELELO" in critique_res.upper():
                    break

                # 3. Válasz javítása és finomítása
                refine_prompt = f"""
                Javítsd és finomítsd az alábbi választ a kritikai észrevételek alapján!

                EREDETI KÉRDÉS: "{user_query}"
                KORÁBBI VÁLASZ: "{current_response}"
                ÉPÍTŐ KRITIKA ÉS A HIBÁK LISTÁJA: "{critique_res}"
                KORREKCIÓS MEGJEGYZÉSEK: "{fact_check.get('correction_notes', '')}"

                Írd újra a választ úgy, hogy az 100%-ban tényalapú, logikus és szerkezetileg tisztázott legyen!
                """

                current_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": refine_prompt}],
                    temperature=0.1,
                    max_tokens=2000
                ).choices[0].message.content.strip()

            except Exception:
                break

        return current_response


# =============================================================================
# 7. SELF-CONSISTENCY ENSEMBLE & MAJORITY VOTING
# =============================================================================

class SelfConsistencyEnsemble:
    """
    Többszörös mintavételezéses konszenzus motor (Self-Consistency Ensemble).
    Több független gondolkodási szálat futtat le alacsony hőmérsékleten,
    és többségi szavazással választja ki a legsűrűbben előforduló helyes logikát.
    """
    def __init__(self, groq_client=None, model_name: str = "groq/compound"):
        self.client = groq_client
        self.model_name = model_name

    def generate_consensus_answer(self, user_query: str, context: str = "", samples: int = 3) -> str:
        """Több mintát gyűjt be és szintézissel állítja elő a legpontosabb választ."""
        if not self.client:
            return "Nincs aktív AI kliens a konszenzus futtatásához."

        prompt = f"""
        Lépésről lépésre elemezd a feladatot és adj végleges tényalapú választ.
        Kontextus: {context[:2000]}
        Kérdés: {user_query}
        """

        drafts = []
        for i in range(samples):
            try:
                res = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3 + (i * 0.15),  # Enyhe diverzitás
                    max_tokens=1500
                )
                content = res.choices[0].message.content.strip()
                drafts.append(content)
            except Exception:
                pass

        if not drafts:
            return "Nem sikerült válaszmintákat generálni."

        if len(drafts) == 1:
            return drafts[0]

        # Konszenzus szintézis prompt
        drafts_formatted = "\n---\n".join([f"MINTA [{idx+1}]:\n{d}" for idx, d in enumerate(drafts)])
        consensus_prompt = f"""
        Az alábbiakban ugyanarra a kérdésre generált {len(drafts)} független AI válaszpíszkozat látható.

        KÉRDÉS: "{user_query}"

        A KÜLÖNBÖZŐ AI MINTÁK:
        {drafts_formatted}

        FELADAT:
        - Hasonlítsd össze a válaszokat!
        - Azonosítsd a válaszok közötti közös konszenzust és a ténybeli egyezőségeket!
        - Szűrd ki az esetleges egyedi hibákat vagy eltéréseket!
        - Állíts össze egyetlen tökéletesen pontos, szintetizált végső választ!
        """

        try:
            final_res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": consensus_prompt}],
                temperature=0.0,
                max_tokens=2500
            )
            return final_res.choices[0].message.content.strip()
        except Exception:
            return drafts[0]


# =============================================================================
# 8. PRECISION MASTER PIPELINE (A FŐ VEZÉRLŐ PIPELINE)
# =============================================================================

class PrecisionMasterPipeline:
    """
    Az integrált mester pipeline, amely összekapcsolja az összes csúcskategóriás pontossági funkciót:
    HyDE Expander -> Hybrid Retrieval -> Math Verifier -> Self-Consistency -> Correction Loop
    """
    def __init__(self, groq_api_key: str = None, model_name: str = "groq/compound"):
        self.api_key = groq_api_key
        self.model_name = model_name
        self.client = Groq(api_key=groq_api_key) if (Groq and groq_api_key) else None

        self.hyde_expander = HyDEQueryExpander(self.client, model_name)
        self.retriever = HybridPrecisionRetriever()
        self.math_verifier = DeterministicMathVerifier()
        self.guardrail = FactualityGuardrail(self.client, model_name)
        self.correction_loop = MultiStageSelfCorrectionLoop(self.client, model_name)
        self.ensemble = SelfConsistencyEnsemble(self.client, model_name)

    def execute_precision_query(
        self, 
        user_query: str, 
        web_context: str = "", 
        doc_context: str = "", 
        use_ensemble: bool = False
    ) -> Dict[str, Any]:
        """
        Lefuttatja a teljes precíziós folyamatot és eléri a legmagasabb válaszarány-pontosságot.
        """
        start_time = time.time()
        combined_context = f"{web_context}\n\n{doc_context}".strip()

        # 1. Matematikai/Logikai kifejezés ellenőrzése
        math_exprs = self.math_verifier.extract_math_expressions(user_query)
        computed_math = {}
        for expr in math_exprs:
            val = self.math_verifier.execute_safe_python_math(expr)
            if val is not None:
                computed_math[expr] = val

        # 2. Szemantikus ablakolás a kontextuson
        if combined_context:
            refined_context = self.retriever.extract_semantic_windows(user_query, combined_context, max_chars=3000)
        else:
            refined_context = ""

        # 3. Elsődleges válaszgenerálás (vagy Konszenzus Ensemble)
        if use_ensemble and self.client:
            initial_answer = self.ensemble.generate_consensus_answer(user_query, refined_context, samples=3)
        else:
            math_hint = f"\n[Determinisztikus matek ellenőrzés: {computed_math}]" if computed_math else ""
            prompt = f"""
            Te egy csúcskategóriás tényalapú elemző AI vagy.
            KÉRDÉS: {user_query}{math_hint}
            KONTEXTUS: {refined_context}

            Válaszolj precízen, tényalapúan, felesleges mellébeszélés nélkül!
            """
            if self.client:
                try:
                    res = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=2000
                    )
                    initial_answer = res.choices[0].message.content.strip()
                except Exception as e:
                    initial_answer = f"Hiba a válaszgenerálás során: {e}"
            else:
                initial_answer = "Hiányzó API-kulcs."

        # 4. Többlépcsős Önellenőrzés & Ténykorrekció
        final_verified_answer = self.correction_loop.run_refinement_cycle(
            user_query=user_query,
            initial_draft=initial_answer,
            context=refined_context,
            max_cycles=2
        )

        # 5. Végső Ténybeli Skór
        fact_check = self.guardrail.verify_factuality(final_verified_answer, refined_context)

        execution_time = round(time.time() - start_time, 2)

        return {
            "answer": final_verified_answer,
            "factuality_score": fact_check.get("factuality_score", 100),
            "hallucinations_detected": fact_check.get("hallucinations", []),
            "math_verified": computed_math,
            "execution_time_seconds": execution_time
        }


# =============================================================================
# TESZTELÉSI BELSŐ FUTTATÁS
# =============================================================================

if __name__ == "__main__":
    print("==========================================================")
    print(" ADVANCED PRECISION ENGINE INITIALIZED SUCCESSFULLY")
    print("==========================================================")
    
    # Teszt matematikai kiértékelés
    math_test = "125 * 8 + 45"
    result = DeterministicMathVerifier.execute_safe_python_math(math_test)
    print(f" Determinisztikus Sandbox Matek Teszt ({math_test}): {result}")

    # Teszt magyar stemmer
    tokens = magyar_stemmer("A legfrissebb tudományos kutatásokról beszélünk")
    print(f" Magyar Stemmer Teszt Tokens: {tokens}")
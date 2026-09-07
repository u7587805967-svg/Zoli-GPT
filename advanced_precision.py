import sys
import os
import re
import json
import math
import time
import datetime
import asyncio
import concurrent.futures
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

class HyDEQueryExpander:
    def __init__(self, groq_client=None):
        self.groq_client = groq_client
    def expand(self, query):
        return [query]

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
    from RestrictedPython import compile_restricted, safe_builtins
except ImportError:
    compile_restricted = None
    safe_builtins = None


# =============================================================================
# GLOBAL CACHE & SINGLETON MODELS (MEMÓRIA ÉS CPU OPTIMALIZÁLÁS)
# =============================================================================

_GLOBAL_EMBEDDER_INSTANCE = None

def get_global_embedder(model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
    """Singleton minta: A SentenceTransformer modellt csak EGYSZER tölti be a RAM-ba."""
    global _GLOBAL_EMBEDDER_INSTANCE
    if _GLOBAL_EMBEDDER_INSTANCE is None and SentenceTransformer is not None:
        try:
            _GLOBAL_EMBEDDER_INSTANCE = SentenceTransformer(model_name)
        except Exception:
            _GLOBAL_EMBEDDER_INSTANCE = None
    return _GLOBAL_EMBEDDER_INSTANCE


HUNGARIAN_STOPWORDS = frozenset([
    "a", "az", "egy", "be", "ki", "le", "fel", "meg", "el", "át", "és", "hogy",
    "nem", "sem", "vagy", "is", "csak", "mint", "volt", "lesz", "cikk", "alatt",
    "van", "vannak", "ma", "majd", "mert", "ha", "de", "mely", "amely", "ebben",
    "ebből", "arról", "melyek", "szerint", "után", "során", "tehát", "így", "ezen"
])

@lru_cache(maxsize=4096)
def magyar_stemmer_cached(word: str) -> str:
    """Egyetlen szó stemmelése memóriagyorsítótárral."""
    suffixes = (
        'ban', 'ben', 'nak', 'nek', 'val', 'vel', 'ból', 'ből', 'ról', 'ről',
        'hoz', 'hez', 'höz', 'ig', 'ért', 'ba', 'be', 'ra', 're', 'at', 'et',
        'ot', 'öt', 'k', 'ak', 'ek', 'ok', 'ök', 'ja', 'je', 'ai', 'ei', 'en', 'on', 'ön'
    )
    for suf in suffixes:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[:-len(suf)]
    return word

def magyar_stemmer(text: str) -> List[str]:
    """Gyorsított magyar heurisztikus stemmer."""
    words = re.findall(r'\b[a-záéíóöőúüűA-ZÁÉÍÓÖŐÚÜŰ]+\b', text.lower())
    clean_tokens = []
    for w in words:
        if w in HUNGARIAN_STOPWORDS or len(w) <= 2:
            continue
        clean_tokens.append(magyar_stemmer_cached(w))
    return clean_tokens


# =============================================================================
# HyDE & RETRIEVAL (OPTIMALIZÁLT)
# =============================================================================

class HybridPrecisionRetriever:
    def __init__(self, embedding_model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        # A globális Singleton modellt használja az újrapéldányosítás helyett
        self.embedder = get_global_embedder(embedding_model_name)

    def compute_bm25_score(self, query: str, text: str) -> float:
        q_tokens = magyar_stemmer(query)
        if not q_tokens:
            return 0.0

        t_tokens = magyar_stemmer(text)
        if not t_tokens:
            return 0.0

        k1, b, avg_doc_len = 1.2, 0.75, 200.0
        doc_len = float(len(t_tokens))
        score = 0.0

        for token in set(q_tokens):
            tf = t_tokens.count(token)
            if tf == 0:
                continue
            tf_score = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
            score += tf_score

        return score

    def extract_semantic_windows(self, query: str, full_text: str, max_chars: int = 2000) -> str:
        if len(full_text) <= max_chars:
            return full_text

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
        if self.embedder:
            try:
                query_vec = self.embedder.encode(query, show_progress_bar=False)
                window_vecs = self.embedder.encode(windows, show_progress_bar=False, batch_size=32)
                
                query_norm = np.linalg.norm(query_vec)
                for idx, chunk in enumerate(windows):
                    bm25_s = self.compute_bm25_score(query, chunk)
                    w_vec = window_vecs[idx]
                    norm = query_norm * np.linalg.norm(w_vec)
                    cos_sim = float(np.dot(query_vec, w_vec) / norm) if norm > 0 else 0.0
                    scored_windows.append(((cos_sim * 12.0) + bm25_s, chunk))
            except Exception:
                for chunk in windows:
                    scored_windows.append((self.compute_bm25_score(query, chunk), chunk))
        else:
            for chunk in windows:
                scored_windows.append((self.compute_bm25_score(query, chunk), chunk))

        scored_windows.sort(key=lambda x: x[0], reverse=True)

        selected, curr_len, seen = [], 0, set()
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
# DETERMINISTIC MATH & FACTUALITY
# =============================================================================

class DeterministicMathVerifier:
    @staticmethod
    def extract_math_expressions(text: str) -> List[str]:
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
        if not compile_restricted:
            cleaned = re.sub(r'[^0-9\+\-\*/\(\)\.\s]', '', code_or_expr)
            try:
                return float(eval(cleaned, {"__builtins__": {}}, {}))
            except Exception:
                return None
        try:
            loc, glb = {}, safe_builtins.copy()
            glb['_getattr_'] = getattr
            glb['_getitem_'] = lambda obj, index: obj[index]
            wrapped_code = f"result = {code_or_expr}"
            byte_code = compile_restricted(wrapped_code, '<inline>', 'exec')
            exec(byte_code, glb, loc)
            return float(loc.get('result'))
        except Exception:
            return None


class FactualityGuardrail:
    def __init__(self, groq_client=None, model_name: str = "groq/compound"):
        self.client = groq_client
        self.model_name = model_name

    def verify_factuality(self, draft_answer: str, context: str) -> Dict[str, Any]:
        if not self.client or not context.strip():
            return {"factuality_score": 100, "hallucinations": [], "is_valid": True}

        prompt = f"""
        FORRÁS KONTEXTUS:
        \"\"\"{context[:2500]}\"\"\"

        GENERÁLT VÁLASZ PISZKOZAT:
        \"\"\"{draft_answer}\"\"\"

        Értékeld a válasz ténybeli pontosságát (0-100 JSON):
        {{
            "factuality_score": 85,
            "hallucinations": []
        }}
        """
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=250
            )
            match = re.search(r'\{.*\}', res.choices[0].message.content.strip(), re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                score = data.get("factuality_score", 100)
                return {
                    "factuality_score": score,
                    "hallucinations": data.get("hallucinations", []),
                    "is_valid": score >= 85
                }
        except Exception:
            pass

        return {"factuality_score": 100, "hallucinations": [], "is_valid": True}


# =============================================================================
# PARALLEL ENSEMBLE & SELF-CORRECTION (OPTIMALIZÁLT PÁRHUZAMOSÍTÁS)
# =============================================================================

class SelfConsistencyEnsemble:
    def __init__(self, groq_client=None, model_name: str = "groq/compound"):
        self.client = groq_client
        self.model_name = model_name

    def _fetch_single_sample(self, prompt: str, temp: float) -> Optional[str]:
        """Egyetlen LLM minta lekérése (párhuzamosítható)."""
        try:
            res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=1200
            )
            return res.choices[0].message.content.strip()
        except Exception:
            return None

    def generate_consensus_answer(self, user_query: str, context: str = "", samples: int = 3) -> str:
        if not self.client:
            return "Nincs aktív AI kliens."

        prompt = f"Kontextus: {context[:2000]}\nKérdés: {user_query}"

        # PÁRHUZAMOS LETHÍVÁS ThreadPoolExecutor-ral (3x gyorsabb)
        drafts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=samples) as executor:
            futures = [
                executor.submit(self._fetch_single_sample, prompt, 0.2 + (i * 0.1))
                for i in range(samples)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    drafts.append(result)

        if not drafts:
            return "Nem sikerült válaszmintákat generálni."
        if len(drafts) == 1:
            return drafts[0]

        drafts_formatted = "\n---\n".join([f"MINTA [{idx+1}]:\n{d}" for idx, d in enumerate(drafts)])
        consensus_prompt = f"""
        Szintetizáld az alábbi {len(drafts)} választ egyetlen tökéletes, tényalapú meglátássá:
        KÉRDÉS: "{user_query}"
        {drafts_formatted}
        """

        try:
            final_res = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": consensus_prompt}],
                temperature=0.0,
                max_tokens=1500
            )
            return final_res.choices[0].message.content.strip()
        except Exception:
            return drafts[0]


class MultiStageSelfCorrectionLoop:
    def __init__(self, groq_client=None, model_name: str = "groq/compound"):
        self.client = groq_client
        self.model_name = model_name
        self.guardrail = FactualityGuardrail(groq_client, model_name)

    def run_refinement_cycle(self, user_query: str, initial_draft: str, context: str = "", max_cycles: int = 1) -> str:
        if not self.client:
            return initial_draft

        current_response = initial_draft

        for _ in range(max_cycles):
            fact_check = self.guardrail.verify_factuality(current_response, context)
            
            # KORAI KILÉPÉS: Ha a ténybeli pontosság már jó (≥90%), kihagyja a felesleges API köröket
            if fact_check.get('factuality_score', 100) >= 90:
                break

            refine_prompt = f"""
            Javítsd az alábbi választ a ténybeli hiányosságok alapján!
            KÉRDÉS: "{user_query}"
            VÁLASZ: "{current_response}"
            HIBÁK: {fact_check.get('hallucinations', [])}
            """
            try:
                current_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": refine_prompt}],
                    temperature=0.1,
                    max_tokens=1500
                ).choices[0].message.content.strip()
            except Exception:
                break

        return current_response


# =============================================================================
# PRECISION MASTER PIPELINE
# =============================================================================

class PrecisionMasterPipeline:
    def __init__(self, groq_api_key: str = None, model_name: str = "groq/compound"):
        self.api_key = groq_api_key
        self.model_name = model_name
        self.client = Groq(api_key=groq_api_key) if (Groq and groq_api_key) else None

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
        start_time = time.time()
        combined_context = f"{web_context}\n\n{doc_context}".strip()

        # 1. Matematikai kiértékelés
        math_exprs = self.math_verifier.extract_math_expressions(user_query)
        computed_math = {expr: self.math_verifier.execute_safe_python_math(expr) for expr in math_exprs if self.math_verifier.execute_safe_python_math(expr) is not None}

        # 2. Szemantikus ablakolás
        refined_context = self.retriever.extract_semantic_windows(user_query, combined_context, max_chars=2500) if combined_context else ""

        # 3. Válaszgenerálás (Párhuzamosított Ensemble vagy direkt kérés)
        if use_ensemble and self.client:
            initial_answer = self.ensemble.generate_consensus_answer(user_query, refined_context, samples=3)
        else:
            math_hint = f"\n[Determinisztikus matek: {computed_math}]" if computed_math else ""
            prompt = f"KÉRDÉS: {user_query}{math_hint}\nKONTEXTUS: {refined_context}\nVálaszolj tömören és pontosan!"
            if self.client:
                try:
                    res = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        max_tokens=1500
                    )
                    initial_answer = res.choices[0].message.content.strip()
                except Exception as e:
                    initial_answer = f"Hiba: {e}"
            else:
                initial_answer = "Hiányzó API-kulcs."

        # 4. Gyorsított korrekciós hurok (max 1 kör, korai kilépéssel)
        final_verified_answer = self.correction_loop.run_refinement_cycle(
            user_query=user_query,
            initial_draft=initial_answer,
            context=refined_context,
            max_cycles=1
        )

        return {
            "answer": final_verified_answer,
            "math_verified": computed_math,
            "execution_time_seconds": round(time.time() - start_time, 2)
        }


if __name__ == "__main__":
    print("==========================================================")
    print(" OPTIMIZED PRECISION ENGINE READY")
    print("==========================================================")
import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def rewrite_query(user_query: str) -> list[str]:
    response = client.chat.completions.create(
        model="groq/compound",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "Generálj 3 eltérő megfogalmazású keresőkifejezést a megadott kérdésből. Csak a keresőkifejezéseket add meg, új sorokkal elválasztva!"},
            {"role": "user", "content": user_query}
        ]
    )
    queries = response.choices[0].message.content.strip().split("\n")
    return [user_query] + [q.strip() for q in queries if q.strip()]

def rerank_documents(query: str, retrieved_docs: list[str], top_k: int = 3) -> list[str]:
    return retrieved_docs[:top_k]

def generate_creative_answer(query: str, context: list[str], persona_style: str = "laza, közvetlen és választékos") -> str:
    system_prompt = (
        f"Stílusod: {persona_style}.\n"
        "Feladatod a kérdés megválaszolása a megadott kontextus alapján.\n"
        "1. Használj egyedi szófordulatokat, természetes, élő nyelvezetet és karakteres megfogalmazást.\n"
        "2. Szigorúan csak a kontextusban lévő tényekre támaszkodj, ne találj ki adatokat.\n"
        "3. Ha hiányzik egy információ, sződd bele a stílusodnak megfelelően a válaszba."
    )
    formatted_context = "\n---\n".join(context)
    user_content = f"Kontextus:\n{formatted_context}\n\nKérdés: {query}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0.65,
        top_p=0.9,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content

def preserve_style_and_correct(original_answer: str, context: list[str]) -> str:
    system_prompt = (
        "Feladatod kizárólag a súlyos ténybeli hibák és a durva helyesírási tévesztések javítása.\n"
        "SZIGORÚ SZABÁLYOK:\n"
        "- NE nyúlj a szöveg stílusához, hangneméhez, egyedi szófordulataihoz vagy mondatszerkezetéhez!\n"
        "- NE cseréld ki a laza vagy egyedi kifejezéseket hivatalos/steril megfogalmazásokra!\n"
        "- Ha a válasz ténybelileg helyes, pontosan ugyanazt a szöveget add vissza változtatás nélkül."
    )
    formatted_context = "\n---\n".join(context)
    user_content = f"Kontextus:\n{formatted_context}\n\nEllenőrizendő válasz:\n{original_answer}"

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content

def run_pipeline(user_query: str, raw_database_docs: list[str]) -> str:
    search_queries = rewrite_query(user_query)
    relevant_docs = rerank_documents(user_query, raw_database_docs, top_k=3)
    
    draft_answer = generate_creative_answer(user_query, relevant_docs)
    
    final_answer = preserve_style_and_correct(draft_answer, relevant_docs)
    
    return final_answer
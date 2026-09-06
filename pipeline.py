import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rewrite_query(user_query: str) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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


def generate_creative_answer(query: str, context: list[str]) -> str:
    system_prompt = (
        "Feladatod a kérdés pontos és kifejező megválaszolása a kapott kontextus alapján.\n"
        "1. Lépésről lépésre elemezd a megadott kontextust és a kérdést.\n"
        "2. Válaszolj kifejezően, stílusosan és változatos szókinccsel, de szigorúan a kontextusban található tényekre támaszkodj.\n"
        "3. Ha a kontextus nem tartalmaz elég információt, sződd bele a válaszba stílusosan a hiány tényét ahelyett, hogy adatokat találnál ki."
    )
    
    formatted_context = "\n---\n".join(context)
    user_content = f"Kontextus:\n{formatted_context}\n\nKérdés: {query}"

    response = client.chat.completions.create(
        model="groq/compound",
        temperature=0.4,
        top_p=0.85,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    return response.choices[0].message.content


def self_correct_answer(original_answer: str, context: list[str]) -> str:
    system_prompt = (
        "Ellenőrizd a generált válasz tényeit a forráskontextus alapján.\n"
        "Ha ténybeli eltérést vagy hallucinációt találsz, javítsd ki a téves adatot.\n"
        "A válasz stílusát, hangvételét és nyelvi kifejezőerejét szigorúan hagyd érintetlenül!"
    )
    
    formatted_context = "\n---\n".join(context)
    user_content = f"Kontextus:\n{formatted_context}\n\nGenerált válasz:\n{original_answer}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,  # Szigorú, determinisztikus ellenőrzés
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
    
    final_answer = self_correct_answer(draft_answer, relevant_docs)
    
    return final_answer
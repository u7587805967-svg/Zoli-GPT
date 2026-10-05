import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ultra_core_engine import (
    AdaptiveIntentRouter,
    FastSemanticCache,
    IntentType,
    ZoliUltraEngine,
)
from ultra_retriever import UltraHybridRetriever
from ultra_web_agent import UltraWebAgent


GOLDEN_CASES = (
    {
        "id": "library-hours",
        "query": "Mikor van nyitva a könyvtár hétköznap?",
        "expected_sources": ("konyvtar.md",),
        "answerable": True,
    },
    {
        "id": "refund-request",
        "query": "Hol kell kérni a visszatérítést?",
        "expected_sources": ("vasarlas.md",),
        "answerable": True,
    },
    {
        "id": "password-change",
        "query": "Hol módosítható a tesztfiók jelszava?",
        "expected_sources": ("fiok.md",),
        "answerable": True,
    },
    {
        "id": "unsupported-fact",
        "query": "Mikor indul a Jupiteri komp?",
        "expected_sources": (),
        "answerable": False,
    },
)

GOLDEN_DOCUMENTS = {
    "konyvtar.md": "A könyvtár hétfőtől péntekig 9 és 18 óra között tart nyitva.",
    "vasarlas.md": "A visszatérítést az ügyfélszolgálaton kell kérni. A kérelemhez rendelési azonosító szükséges.",
    "fiok.md": "A tesztfiók jelszavát a fiók beállításai menüben lehet módosítani.",
}


def evaluate_retrieval(retriever, cases=GOLDEN_CASES, top_k=5):
    answerable = 0
    hits = 0
    reciprocal_ranks = []
    unanswerable = 0
    false_positives = 0

    for case in cases:
        results = retriever.search(case["query"], top_k=top_k)
        sources = [result.source for result in results]
        if case["answerable"]:
            answerable += 1
            rank = next(
                (
                    index
                    for index, source in enumerate(sources, start=1)
                    if source in case["expected_sources"]
                ),
                None,
            )
            if rank is not None:
                hits += 1
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0.0)
        else:
            unanswerable += 1
            false_positives += bool(results)

    return {
        "cases": len(cases),
        "hit_rate_at_k": hits / answerable if answerable else 0.0,
        "mrr": sum(reciprocal_ranks) / answerable if answerable else 0.0,
        "unanswerable_false_positive_rate": (
            false_positives / unanswerable if unanswerable else 0.0
        ),
    }


def run_evals(retriever=None, cases=GOLDEN_CASES, top_k=5):
    if retriever is None:
        retriever = UltraHybridRetriever()
    results = evaluate_retrieval(retriever, cases, top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


class GoldenRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        for filename, content in GOLDEN_DOCUMENTS.items():
            (root / filename).write_text(content, encoding="utf-8")
        self.retriever = UltraHybridRetriever(documents_dir=str(root))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_golden_retrieval_metrics(self):
        results = evaluate_retrieval(self.retriever)
        self.assertEqual(results["hit_rate_at_k"], 1.0)
        self.assertEqual(results["mrr"], 1.0)
        self.assertEqual(results["unanswerable_false_positive_rate"], 0.0)

    def test_unknown_query_has_no_retrieval(self):
        self.assertEqual(self.retriever.search("Jupiteri komp indulása"), [])

    def test_context_contains_source_labels(self):
        context = asyncio.run(
            self.retriever.retrieve_and_compress("Mikor van nyitva a könyvtár?")
        )
        self.assertIn("[L1]", context)
        self.assertIn("konyvtar.md", context)

    def test_database_retrieval_is_user_scoped(self):
        with tempfile.TemporaryDirectory() as folder:
            database_path = Path(folder) / "documents.db"
            import sqlite3

            with sqlite3.connect(database_path) as connection:
                connection.execute(
                    "CREATE TABLE document_vectors "
                    "(id INTEGER PRIMARY KEY, username TEXT, doc_name TEXT, "
                    "chunk_text TEXT, embedding BLOB)"
                )
                connection.executemany(
                    "INSERT INTO document_vectors (username, doc_name, chunk_text) "
                    "VALUES (?, ?, ?)",
                    [
                        ("alice", "alice.md", "Alice tesztprojektje neve Kék Hold."),
                        ("bob", "bob.md", "Bob tesztprojektje neve Vörös Híd."),
                    ],
                )
            connection.close()
            retriever = UltraHybridRetriever(
                database_path=str(database_path), username="alice"
            )
            self.assertEqual(retriever.document_count, 1)
            self.assertEqual(retriever.documents[0].source, "alice.md")


class AnswerSafetyTests(unittest.TestCase):
    @staticmethod
    def make_engine(answer):
        chunk = SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=answer))]
        )
        completions = SimpleNamespace(create=lambda **kwargs: [chunk])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        engine = ZoliUltraEngine("fake-key", None)
        engine.groq_client = client
        return engine

    def test_factual_question_routes_to_retrieval(self):
        query = "Ki Magyarország miniszterelnöke?"
        self.assertEqual(AdaptiveIntentRouter.classify(query), IntentType.RAG_SEARCH)

    def test_cache_requires_exact_query_and_intent(self):
        cache = FastSemanticCache()
        cache.set("Szia", "Szia!", IntentType.DIRECT)
        self.assertEqual(cache.get("Szia", IntentType.DIRECT), "Szia!")
        self.assertIsNone(cache.get("Szia!", IntentType.DIRECT))
        self.assertIsNone(cache.get("Szia", IntentType.RAG_SEARCH))

    def test_empty_context_causes_abstention(self):
        engine = ZoliUltraEngine("", None)

        async def get_response():
            return [
                item
                async for item in engine.stream_response(
                    "Ki Magyarország miniszterelnöke?"
                )
            ]

        response = asyncio.run(get_response())
        self.assertEqual(len(response), 1)
        self.assertIn("nem találgatok", response[0])

    def test_only_available_citations_are_accepted(self):
        engine = ZoliUltraEngine("", None)
        context = "[L1] Forrás: konyvtar.md\nA könyvtár nyitvatartása 9 és 18 óra."
        self.assertTrue(engine._valid_citations("9 és 18 óra között [L1].", context))
        self.assertFalse(engine._valid_citations("9 és 18 óra között [L2].", context))

    def test_deep_reasoning_requires_grounding(self):
        query = "Why is the sky blue?"
        self.assertEqual(AdaptiveIntentRouter.classify(query), IntentType.DEEP_REASONING)
        self.assertTrue(ZoliUltraEngine._requires_grounding(query))

    def test_end_to_end_valid_citation_has_source_footer(self):
        engine = self.make_engine("A könyvtár 9 és 18 óra között tart nyitva [L1].")

        async def get_response():
            async def local_context(query):
                return "[L1] Forrás: konyvtar.md\nA könyvtár 9 és 18 óra között tart nyitva."

            return [
                item
                async for item in engine.stream_response(
                    "Mikor van nyitva a könyvtár?", rag_context_provider=local_context
                )
            ]

        response = asyncio.run(get_response())
        self.assertIn("konyvtar.md", response[0])

    def test_end_to_end_invalid_citation_abstains(self):
        engine = self.make_engine("A könyvtár éjjel-nappal nyitva van [L2].")

        async def get_response():
            async def local_context(query):
                return "[L1] Forrás: konyvtar.md\nA könyvtár 9 és 18 óra között tart nyitva."

            return [
                item
                async for item in engine.stream_response(
                    "Mikor van nyitva a könyvtár?", rag_context_provider=local_context
                )
            ]

        response = asyncio.run(get_response())
        self.assertEqual(len(response), 1)
        self.assertIn("nem találgatok", response[0])

    def test_cached_greeting_skips_search_providers(self):
        engine = self.make_engine("Szia, örülök, hogy újra itt vagy!")

        async def get_response():
            async def unexpected_provider(query):
                raise AssertionError("Greeting must not trigger retrieval")

            first = [
                item
                async for item in engine.stream_response(
                    "Szia", unexpected_provider, unexpected_provider
                )
            ]
            second = [
                item
                async for item in engine.stream_response(
                    "Szia", unexpected_provider, unexpected_provider
                )
            ]
            return first, second

        first, second = asyncio.run(get_response())
        self.assertEqual(first, ["Szia, örülök, hogy újra itt vagy!"])
        self.assertIn("Gyorsítótár", second[0])

    def test_duckduckgo_redirect_url_is_unwrapped(self):
        html = (
            '<div class="result"><a class="result__a" '
            'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fsource">'
            'Example source</a><a class="result__snippet">A verified snippet.</a></div>'
        )
        results = UltraWebAgent.parse_search_results(html)
        self.assertEqual(results[0]["url"], "https://example.com/source")


if __name__ == "__main__":
    unittest.main()
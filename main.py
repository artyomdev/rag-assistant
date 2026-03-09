"""
Simple console interface for the RAG system v2.

Uses the full pipeline with canonicalisation, entity resolution and disambiguation.
"""
import ollama
from app.search_engine import SearchEngine
from app.reranker import Reranker
from app.config import settings

from app.canonicalizer import get_canonicalizer
from app.intent import get_intent_classifier
from app.entity_resolver import get_entity_resolver
from app.disambiguator import get_disambiguator
from app.response_builder import get_response_builder


class Assistant:
    def __init__(self):
        print("Initializing RAG v2 components...")
        self.searcher = SearchEngine()
        self.reranker = Reranker()
        
        self.canonicalizer = get_canonicalizer()
        self.intent_classifier = get_intent_classifier()
        self.entity_resolver = get_entity_resolver()
        self.disambiguator = get_disambiguator()
        self.response_builder = get_response_builder()
        
        print("✅ All components initialised")
        
    def ask(self, query: str):
        print(f"\n🔍 Question: {query}")
        
        canonical = self.canonicalizer.canonicalize(query)
        print(f"📝 Canonical: {canonical.text}")
        
        if canonical.abbreviations:
            print(f"🔤 Abbreviations: {canonical.abbreviations}")
        
        intent_result = self.intent_classifier.classify(query)
        print(f"🎯 Intent: {intent_result.intent.value} (confidence: {intent_result.confidence:.2f})")
        
        entity_resolution = self.entity_resolver.resolve(canonical)
        
        disambiguation = self.disambiguator.check_and_disambiguate(
            canonical_query=canonical,
            intent_result=intent_result,
            entity_resolution=entity_resolution
        )
        
        if disambiguation.needs_clarification:
            print(f"\n❓ {disambiguation.question}")
            for i, option in enumerate(disambiguation.options):
                print(f"  {i+1}. {option.label}")
                if option.description:
                    print(f"     ({option.description[:50]}...)")
            
            choice = input("\nEnter option number (or press Enter to continue): ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(disambiguation.options):
                selected = disambiguation.options[int(choice) - 1]
                print(f"✅ Selected: {selected.label}")
        
        entity_ids = self.entity_resolver.get_entity_ids_for_search(entity_resolution)
        
        results_with_scores = self.searcher.search_with_scores(
            canonical.text if canonical.text else query, 
            limit=settings.TOP_K_RETRIEVAL
        )
        
        candidates = [
            payload for payload, score in results_with_scores 
            if score >= settings.COSINE_THRESHOLD
        ]
        print(f"🔍 Found {len(results_with_scores)} results, after filtering (>={settings.COSINE_THRESHOLD}): {len(candidates)}")
        
        if not candidates:
            print("❌ No information was found in the knowledge base for this question.")
            return
            
        # 6. Reranking (BGE-Reranker)
        print(f"📊 Re‑ranking {len(candidates)} candidates...")
        ranked_results = self.reranker.rank(query, candidates, top_k=settings.TOP_K_RERANK)
        
        top_score = ranked_results[0].get('score', -999) if ranked_results else -999
        print(f"📈 Top score: {top_score:.2f} (threshold: {settings.SCORE_THRESHOLD})")
        
        if top_score < settings.SCORE_THRESHOLD:
            print("⚠️ Relevance is below threshold. You may want to refine your question.")
        else:
            disambiguation = self.disambiguator.check_and_disambiguate(
                canonical_query=canonical,
                intent_result=intent_result,
                entity_resolution=entity_resolution,
                search_score=top_score,
                score_threshold=settings.SCORE_THRESHOLD,
                search_results=ranked_results[:10]
            )
            if disambiguation.needs_clarification:
                print(f"\n❓ {disambiguation.question}")
                for i, option in enumerate(disambiguation.options):
                    print(f"  {i+1}. {option.label}")
                choice = input("\nEnter option number (or press Enter to show all): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(disambiguation.options):
                    selected = disambiguation.options[int(choice) - 1]
                    if selected.id != "topic_show_all":
                        print(f"✅ Selected: {selected.label}")
        
        response = self.response_builder.build_response(
            query=query,
            intent=intent_result.intent,
            ranked_results=ranked_results,
            entity_ids=entity_ids
        )
        
        print("\n" + "="*50)
        print("ANSWER:")
        print("="*50)
        print(response.answer)
        
        if response.sources:
            print("\n📚 Sources:")
            for src in response.sources:
                print(f"  - {src.name} ({src.category})")
        
        print("\n")


if __name__ == "__main__":
    bot = Assistant()
    print("\n" + "="*50)
    print("RAG Assistant v2 – with canonicalisation and entity resolution")
    print("="*50 + "\n")
    
    while True:
        q = input("Enter your question (or 'exit'): ").strip()
        if q.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if q:
            bot.ask(q)

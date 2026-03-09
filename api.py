# api.py
"""
RAG API – pipeline with query canonicalisation, intent classification,
entity resolution and deterministic disambiguation.

High‑level pipeline:
1. Canonicalise the query → more stable retrieval.
2. Classify intent → choose the answer template.
3. Resolve entities and abbreviations.
4. Run disambiguator → optional clarification questions.
5. Search + rerank → threshold check and topic analysis.
6. Build structured answer with sources.
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import logging
from pathlib import Path

from app.search_engine import SearchEngine
from app.reranker import Reranker
from app.config import settings

from app.canonicalizer import get_canonicalizer
from app.intent import get_intent_classifier, IntentType
from app.entity_resolver import get_entity_resolver, expand_abbreviations
from app.disambiguator import get_disambiguator
from app.response_builder import get_response_builder
from app.source_resolver import get_source_resolver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Expert RAG API (v2)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_path = Path(settings.DATA_DIR)
if data_path.exists():
    app.mount("/docs", StaticFiles(directory=str(data_path)), name="docs")

try:
    engine = SearchEngine()
    reranker = Reranker()
    canonicalizer = get_canonicalizer()
    intent_classifier = get_intent_classifier()
    entity_resolver = get_entity_resolver()
    disambiguator = get_disambiguator()
    response_builder = get_response_builder()
    source_resolver = get_source_resolver()
    
    logger.info("✅ All RAG v2 components successfully initialised")
except Exception as e:
    logger.error(f"❌ Error initialising components: {e}")
    raise e


class Query(BaseModel):
    """User query payload."""
    question: str
    clarification_choice_id: Optional[str] = None


class ClarificationChoice(BaseModel):
    """Chosen clarification option."""
    choice_id: str
    original_query: str
    source_name: Optional[str] = None
    substituted_query: Optional[str] = None


@app.post("/ask")
async def ask(data: Query):
    """
    Main endpoint for user questions.
    
    Returns:
    - answer: text answer (or None when clarification is required)
    - needs_clarification: whether a clarification step is needed
    - clarification: data for clarification UI
    - sources: structured sources [{id, name, link}]
    - debug: debug information
    """
    try:
        logger.info(f"❓ New query: {data.question}")
        
        # 1. QUERY CANONICALISATION
        canonical = canonicalizer.canonicalize(data.question)
        logger.info(f"📝 Canonical query: {canonical.text}")
        
        # 2. INTENT CLASSIFICATION
        intent_result = intent_classifier.classify(data.question)
        intent = intent_result.intent
        logger.info(f"🎯 Intent: {intent.value} (confidence: {intent_result.confidence:.2f})")
        
        # 3. ENTITY RESOLUTION (including abbreviations)
        entity_resolution = entity_resolver.resolve(canonical)
        
        if entity_resolution.has_abbreviations:
            abbrevs = canonical.abbreviations
            logger.info(f"🔤 Abbreviations detected: {abbrevs}")
        
        # 4. DISAMBIGUATION CHECK (before search)
        disambiguation = disambiguator.check_and_disambiguate(
            canonical_query=canonical,
            intent_result=intent_result,
            entity_resolution=entity_resolution,
            search_score=None,
            score_threshold=settings.SCORE_THRESHOLD
        )
        
        if disambiguation.needs_clarification:
            logger.info(f"❔ Clarification required: {disambiguation.reason.value}")
            return {
                "needs_clarification": True,
                "clarification": disambiguation.to_dict(),
                "answer": None,
                "sources": [],
                "context": "",
                "debug": {
                    "canonical_query": canonical.text,
                    "intent": intent.value,
                    "reason": disambiguation.reason.value
                }
            }
        
        # 5. SEARCH WITH EARLY COSINE‑SIMILARITY FILTERING
        entity_ids = entity_resolver.get_entity_ids_for_search(entity_resolution)
        
        # Expand abbreviations for better search (e.g. KPI → key performance indicator)
        search_text = canonical.text if canonical.text else data.question
        search_text = expand_abbreviations(search_text)
        logger.info(f"🔍 Search text: {search_text}")
        
        # Use search_with_scores for early filtering of weak results
        results_with_scores = engine.search_with_scores(
            search_text, 
            limit=settings.TOP_K_RETRIEVAL
        )
        
        # Filter by cosine similarity to reduce reranker load
        search_results = [
            payload for payload, score in results_with_scores 
            if score >= settings.COSINE_THRESHOLD
        ]
        logger.info(f"🔍 Found {len(results_with_scores)} results, after filtering (>={settings.COSINE_THRESHOLD}): {len(search_results)}")
        
        if not search_results:
            logger.warning("🔍 Search returned no results above cosine‑similarity threshold")
            return {
                "needs_clarification": False,
                "answer": "There is no information in the knowledge base for this query.",
                "sources": [],
                "context": "",
                "debug": {"reason": "no_search_results", "cosine_threshold": settings.COSINE_THRESHOLD}
            }
        
        # 6. RERANKING (use more results for disambiguation list, but answer from top‑K)
        top_k_disambiguate = getattr(settings, 'TOP_K_FOR_DISAMBIG', 25)
        ranked = reranker.rank(data.question, search_results, top_k=max(settings.TOP_K_RERANK, top_k_disambiguate))
        
        if not ranked:
            return {
                "needs_clarification": False,
                "answer": "An error occurred while ranking the search results.",
                "sources": [],
                "context": "",
                "debug": {"reason": "rerank_failed"}
            }
        
        # 7. QUALITY THRESHOLD CHECK
        top_score = ranked[0].get('score', -999)
        logger.info(f"📊 Top score: {top_score} (threshold: {settings.SCORE_THRESHOLD})")
        
        # Re‑run disambiguation with score information
        if top_score < settings.SCORE_THRESHOLD:
            disambiguation = disambiguator.check_and_disambiguate(
                canonical_query=canonical,
                intent_result=intent_result,
                entity_resolution=entity_resolution,
                search_score=top_score,
                score_threshold=settings.SCORE_THRESHOLD
            )
            
            if disambiguation.needs_clarification:
                logger.info("📉 Low relevance, proposing clarification")
                return {
                    "needs_clarification": True,
                    "clarification": disambiguation.to_dict(),
                    "answer": None,
                    "sources": [],
                    "context": "",
                    "debug": {
                        "top_score": top_score,
                        "threshold": settings.SCORE_THRESHOLD,
                        "reason": "low_relevance"
                    }
                }
            else:
                return {
                    "needs_clarification": False,
                    "answer": "Unfortunately, the knowledge base does not contain enough precise information to answer this question.",
                    "sources": [],
                    "context": "",
                    "debug": {
                        "top_score": top_score,
                        "threshold": settings.SCORE_THRESHOLD,
                        "reason": "below_threshold"
                    }
                }
        
        # 7b. Check for multiple relevant documents (topic‑level clarification)
        disambiguation = disambiguator.check_and_disambiguate(
            canonical_query=canonical,
            intent_result=intent_result,
            entity_resolution=entity_resolution,
            search_score=top_score,
            score_threshold=settings.SCORE_THRESHOLD,
            search_results=ranked[:top_k_disambiguate]
        )
        if disambiguation.needs_clarification:
            logger.info(f"❔ Multiple relevant documents, proposing topic clarification: {disambiguation.reason}")
            return {
                "needs_clarification": True,
                "clarification": disambiguation.to_dict(),
                "answer": None,
                "sources": [],
                "context": "",
                "debug": {
                    "canonical_query": canonical.text,
                    "intent": intent.value,
                    "reason": disambiguation.reason.value if disambiguation.reason else "multiple_topics"
                }
            }
        
        # 8. ENRICH SOURCES (for answer we use only top TOP_K_RERANK)
        top_for_response = ranked[: settings.TOP_K_RERANK]
        enriched_results = source_resolver.enrich_search_results(top_for_response)
        
        # 9. BUILD ANSWER
        response = response_builder.build_response(
            query=data.question,
            intent=intent,
            ranked_results=enriched_results,
            entity_ids=entity_ids
        )
        
        # 10. STRUCTURED SOURCES
        structured_sources = source_resolver.get_structured_sources(enriched_results)
        
        return {
            "needs_clarification": False,
            "answer": response.answer,
            "sources": structured_sources,
            "context": response.context_used,
            "debug": {
                "canonical_query": canonical.text,
                "intent": response.intent,
                "confidence": response.confidence,
                "top_score": top_score,
                "entity_ids": entity_ids,
                **response.debug_info
            }
        }

    except Exception as e:
        logger.error(f"💥 Error while processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clarify")
async def clarify(data: ClarificationChoice):
    """
    Endpoint that applies a clarification choice and runs a refined search.
    """
    try:
        logger.info(f"📌 Applying clarification: choice_id={data.choice_id}")
        
        # If a catalog option was selected – use substituted_query as a new question
        query_to_use = data.substituted_query if data.substituted_query else data.original_query
        if data.substituted_query:
            logger.info(f"📂 Catalog: using substituted query '{data.substituted_query}'")
        
        # Parse choice_id to extract entity_id, intent or source_filter
        choice_parts = data.choice_id.split('_')
        
        entity_id = None
        chosen_intent = None
        source_filter = None
        
        if choice_parts[0] == 'catalog':
            # Catalog selection — entity/intent are not set, search by substituted_query
            pass
        elif choice_parts[0] == 'abbr' and len(choice_parts) >= 3:
            # abbr_0_entityid
            entity_id = choice_parts[2] if choice_parts[1] != 'other' else None
        elif choice_parts[0] == 'intent':
            # intent_definition
            chosen_intent = choice_parts[1]
        elif choice_parts[0] == 'low' and len(choice_parts) >= 3:
            # low_rel_0_entityid
            entity_id = choice_parts[3] if choice_parts[2].isdigit() else None
        elif choice_parts[0] == 'vague':
            # vague_definition
            chosen_intent = choice_parts[1]
        elif choice_parts[0] == 'topic':
            # topic_0, topic_1, ..., topic_show_all
            if len(choice_parts) >= 2 and choice_parts[1] != 'show_all':
                # Use source_name from the request for filtering
                source_filter = data.source_name
                logger.info(f"📂 Filtering by document: {source_filter}")
        
        # Canonicalise query (catalog‑substituted or original)
        canonical = canonicalizer.canonicalize(query_to_use)
        
        # Determine intent (use chosen one or classify again)
        if chosen_intent:
            intent = IntentType(chosen_intent)
        else:
            intent_result = intent_classifier.classify(query_to_use)
            intent = intent_result.intent
        
        # Search with optional entity filter and early cosine‑similarity threshold
        search_entity_ids = [entity_id] if entity_id else None
        
        # Expand abbreviations for better search
        search_text = canonical.text if canonical.text else query_to_use
        search_text = expand_abbreviations(search_text)
        
        results_with_scores = engine.search_with_scores(
            search_text,
            limit=settings.TOP_K_RETRIEVAL,
            entity_ids=search_entity_ids
        )
        
        search_results = [
            payload for payload, score in results_with_scores 
            if score >= settings.COSINE_THRESHOLD
        ]
        
        if not search_results:
            return {
                "needs_clarification": False,
                "answer": "No information was found for the selected clarification.",
                "sources": [],
                "context": "",
                "debug": {"reason": "no_results_after_clarification", "cosine_threshold": settings.COSINE_THRESHOLD}
            }
        
        # Rerank
        ranked = reranker.rank(query_to_use, search_results, top_k=settings.TOP_K_RERANK)
        
        # Filter by document (if a topic_* option was selected)
        if source_filter:
            ranked = [r for r in ranked if r.get('source', '') == source_filter]
            if not ranked:
                return {
                    "needs_clarification": False,
                    "answer": f"The selected document \"{source_filter}\" does not contain information for your query.",
                    "sources": [],
                    "context": "",
                    "debug": {"reason": "no_results_in_source", "source_filter": source_filter, "fallback": "support"}
                }
        
        # Enrich results and build answer
        enriched_results = source_resolver.enrich_search_results(ranked)
        
        response = response_builder.build_response(
            query=query_to_use,
            intent=intent,
            ranked_results=enriched_results,
            entity_ids=search_entity_ids
        )
        
        structured_sources = source_resolver.get_structured_sources(enriched_results)
        
        return {
            "needs_clarification": False,
            "answer": response.answer,
            "sources": structured_sources,
            "context": response.context_used,
            "debug": {
                "clarified_with": data.choice_id,
                "entity_id": entity_id,
                "intent": intent.value,
                **response.debug_info
            }
        }
        
    except Exception as e:
        logger.error(f"💥 Error while processing clarification: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/entities")
async def get_entities():
    """Return a list of all known entities."""
    from app.entity_registry import get_entity_registry
    
    registry = get_entity_registry()
    entities = registry.get_all_entities()
    
    return {
        "count": len(entities),
        "entities": [
            {
                "id": e.entity_id,
                "label": e.label,
                "type": e.entity_type,
                "aliases": e.aliases[:5],
                "frequency": e.frequency
            }
            for e in entities
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9004)

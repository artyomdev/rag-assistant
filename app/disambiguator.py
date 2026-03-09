"""
Disambiguator – deterministic clarification question generator.

Clarification prompts are built from entity and intent information and are NOT
generated as free‑form LLM text. There is optional support for a document
catalog used for very short, generic queries.
"""
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from enum import Enum

from app.config import settings
from app.intent import IntentType, IntentResult, get_intent_classifier
from app.entity_resolver import EntityResolutionResult, ResolvedEntity, ResolutionConfidence, ABBREVIATION_EXPANSIONS
from app.entity_registry import Entity
from app.canonicalizer import CanonicalQuery

logger = logging.getLogger(__name__)


class DisambiguationReason(Enum):
    """Reasons why a clarification step might be required."""
    MULTIPLE_ENTITIES = "multiple_entities"
    AMBIGUOUS_INTENT = "ambiguous_intent"
    ABBREVIATION_UNCLEAR = "abbreviation_unclear"
    QUERY_TOO_VAGUE = "query_too_vague"
    LOW_RELEVANCE = "low_relevance"
    MULTIPLE_TOPICS = "multiple_topics"
    CATALOG_TOPIC = "catalog_topic"


@dataclass
class Choice:
    """Single clarification choice for the user."""
    id: str                     # Unique identifier
    label: str                  # Display text
    entity_id: Optional[str] = None    # Linked entity (if any)
    intent: Optional[str] = None       # Linked intent (if any)
    description: Optional[str] = None  # Additional description
    category: Optional[str] = None     # Grouping category (for topic_*)
    substituted_query: Optional[str] = None  # Query to use when this option is chosen
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DisambiguationResponse:
    """Structured response that describes clarification options."""
    needs_clarification: bool               # Whether clarification is required
    reason: Optional[DisambiguationReason]  # Internal reason
    question: str                           # Clarification question text
    options: List[Choice]                   # Available choices
    explanation: str                        # Technical explanation (for logs)
    original_query: str                     # Original user query
    
    def to_dict(self) -> dict:
        return {
            "needs_clarification": self.needs_clarification,
            "reason": self.reason.value if self.reason else None,
            "question": self.question,
            "options": [o.to_dict() for o in self.options],
            "explanation": self.explanation,
            "original_query": self.original_query
        }


# Human‑facing question templates for different clarification reasons
QUESTION_TEMPLATES = {
    DisambiguationReason.MULTIPLE_ENTITIES: "We found several possible topics. Refine your question or pick an option:",
    DisambiguationReason.AMBIGUOUS_INTENT: "Please clarify what you want to do or choose a question type:",
    DisambiguationReason.ABBREVIATION_UNCLEAR: "The abbreviation '{abbr}' may refer to different concepts. Write more details or choose one:",
    DisambiguationReason.QUERY_TOO_VAGUE: "The query is too generic. Please add more details or choose a topic:",
    DisambiguationReason.LOW_RELEVANCE: "No highly relevant matches were found. Refine the query or choose an option:",
    DisambiguationReason.MULTIPLE_TOPICS: "We found several document types on this topic. Refine your query or choose a document:",
    DisambiguationReason.CATALOG_TOPIC: "What exactly are you interested in? Choose an option from the catalog or clarify in text:",
}


class Disambiguator:
    """Generator of deterministic clarification questions."""
    
    def __init__(self):
        self.intent_classifier = get_intent_classifier()
        
        # Minimal query length (in lemma tokens) to treat it as specific
        self.min_query_words = 2
        
        # Max number of options (for abbreviations, intent etc.)
        self.max_options = 5
        # For document‑level clarification we allow more options
        self.max_topic_options = 10
        # How many chunks to scan when collecting unique documents
        self.disambiguate_scan_chunks = 30
        
        # Optional document catalog for very short, generic queries
        self._catalog: Dict[str, List[Dict]] = {}
        self._load_catalog()
        
        logger.info("✅ Disambiguator initialised")
    
    def _load_catalog(self) -> None:
        """Load document catalog from storage/document_catalog.json if present."""
        catalog_path = Path(settings.DB_PATH).resolve().parent / "document_catalog.json"
        if catalog_path.exists():
            try:
                data = json.loads(catalog_path.read_text(encoding="utf-8"))
                self._catalog = data.get("triggers", {})
                logger.info(f"📂 Loaded document catalog: {len(self._catalog)} triggers")
            except Exception as e:
                logger.warning(f"Failed to load document catalog: {e}")
    
    def _try_catalog_disambiguate(
        self, canonical_query: CanonicalQuery, original_query: str
    ) -> Optional[DisambiguationResponse]:
        """
        For very short queries (1–2 words) try to resolve via document catalog.
        """
        lemmas = canonical_query.lemmas
        for trigger, items in self._catalog.items():
            if not items or not isinstance(items, list):
                continue
            if any(trigger == lem or trigger in lem or lem in trigger for lem in lemmas):
                question = QUESTION_TEMPLATES[DisambiguationReason.CATALOG_TOPIC]
                options = []
                for i, entry in enumerate(items[: self.max_topic_options]):
                    if not isinstance(entry, dict):
                        continue
                    label = entry.get("label", "")
                    query = entry.get("query", "")
                    if not label:
                        continue
                    options.append(
                        Choice(
                            id=f"catalog_{trigger}_{i}",
                            label=label,
                            entity_id=None,
                            intent=None,
                            description=None,
                            category="Document catalog",
                            substituted_query=query or None,
                        )
                    )
                if options:
                    return DisambiguationResponse(
                        needs_clarification=True,
                        reason=DisambiguationReason.CATALOG_TOPIC,
                        question=question,
                        options=options,
                        explanation=f"Query matched catalog trigger \"{trigger}\"",
                        original_query=original_query,
                    )
        return None
    
    def check_and_disambiguate(
        self,
        canonical_query: CanonicalQuery,
        intent_result: IntentResult,
        entity_resolution: EntityResolutionResult,
        search_score: Optional[float] = None,
        score_threshold: float = -4.0,
        search_results: Optional[List[Dict]] = None
    ) -> DisambiguationResponse:
        """
        Check whether clarification is needed and, if so, build options.
        """
        original_query = canonical_query.original
        
        # 0. For very short queries (1–2 words) check the catalog first
        if len(canonical_query.lemmas) <= 2 and self._catalog:
            catalog_response = self._try_catalog_disambiguate(canonical_query, original_query)
            if catalog_response is not None:
                return catalog_response
        
        if entity_resolution.ambiguous:
            return self._disambiguate_abbreviations(
                entity_resolution.ambiguous,
                original_query
            )
        
        if intent_result.is_ambiguous and len(intent_result.alternative_intents) > 0:
            return self._disambiguate_intent(
                intent_result,
                original_query
            )
        
        if len(canonical_query.lemmas) < self.min_query_words and not entity_resolution.resolved:
            return self._handle_vague_query(canonical_query, original_query)
        
        if search_score is not None and search_score < score_threshold:
            return self._handle_low_relevance(
                entity_resolution,
                original_query,
                search_score
            )
        
        # 5. Multiple relevant documents (after reranking)
        min_words_skip = settings.MIN_WORDS_SKIP_MULTIPLE_TOPICS
        query_word_count = len(original_query.strip().split())
        logger.info(f"📊 Checking query length: {query_word_count} words, threshold: {min_words_skip}")
        if search_results and search_score is not None and search_score >= score_threshold:
            min_docs = 2 if len(canonical_query.lemmas) <= 2 else 3
            if self._has_multiple_relevant_topics(search_results, score_threshold, min_documents=min_docs):
                if query_word_count >= min_words_skip:
                    logger.info(
                        f"Query is long enough ({query_word_count} words >= {min_words_skip}): "
                        "skipping document‑level disambiguation and answering using all sources"
                    )
                    return DisambiguationResponse(
                        needs_clarification=False,
                        reason=None,
                        question="",
                        options=[],
                        explanation=f"Query has {query_word_count} words — answer from all relevant documents",
                        original_query=original_query,
                    )
                return self._disambiguate_by_topics(search_results, original_query)
        
        return DisambiguationResponse(
            needs_clarification=False,
            reason=None,
            question="",
            options=[],
            explanation="Query is specific enough, no clarification required",
            original_query=original_query
        )
    
    def _disambiguate_abbreviations(
        self,
        ambiguous: List[tuple],
        original_query: str
    ) -> DisambiguationResponse:
        """Generate choices for ambiguous abbreviations."""
        abbr, candidates = ambiguous[0]
        
        if abbr in ABBREVIATION_EXPANSIONS or abbr.upper() in ABBREVIATION_EXPANSIONS:
            expansion = ABBREVIATION_EXPANSIONS.get(abbr) or ABBREVIATION_EXPANSIONS.get(abbr.upper())
            logger.info(f"✅ Abbreviation '{abbr}' expanded automatically to '{expansion[0]}'")
            return DisambiguationResponse(
                needs_clarification=False,
                reason=None,
                question="",
                options=[],
                explanation=f"Abbreviation '{abbr}' auto‑expanded to '{expansion[0]}'",
                original_query=original_query
            )
        
        question = QUESTION_TEMPLATES[DisambiguationReason.ABBREVIATION_UNCLEAR].format(abbr=abbr)
        
        options = []
        for i, resolved in enumerate(candidates[:self.max_options]):
            entity = resolved.entity
            choice = Choice(
                id=f"abbr_{i}_{entity.entity_id}",
                label=entity.label,
                entity_id=entity.entity_id,
                intent=None,
                description=self._get_entity_description(entity)
            )
            options.append(choice)
        
        options.append(
            Choice(
                id="abbr_other",
                label="Something else",
                entity_id=None,
                intent=None,
                description="None of the options above fit"
            )
        )
        
        return DisambiguationResponse(
            needs_clarification=True,
            reason=DisambiguationReason.ABBREVIATION_UNCLEAR,
            question=question,
            options=options,
            explanation=f"Abbreviation '{abbr}' has {len(candidates)} plausible interpretations",
            original_query=original_query
        )
    
    def _disambiguate_intent(
        self,
        intent_result: IntentResult,
        original_query: str
    ) -> DisambiguationResponse:
        """Generate choices when the intent is ambiguous."""
        question = QUESTION_TEMPLATES[DisambiguationReason.AMBIGUOUS_INTENT]
        
        options = []
        
        main_intent = intent_result.intent
        options.append(Choice(
            id=f"intent_{main_intent.value}",
            label=self.intent_classifier.get_intent_description(main_intent),
            entity_id=None,
            intent=main_intent.value,
            description=None
        ))
        
        for alt_intent, score in intent_result.alternative_intents[:3]:
            options.append(Choice(
                id=f"intent_{alt_intent.value}",
                label=self.intent_classifier.get_intent_description(alt_intent),
                entity_id=None,
                intent=alt_intent.value,
                description=None
            ))
        
        return DisambiguationResponse(
            needs_clarification=True,
            reason=DisambiguationReason.AMBIGUOUS_INTENT,
            question=question,
            options=options,
            explanation=f"Intent is ambiguous: {main_intent.value} vs alternatives",
            original_query=original_query
        )
    
    def _handle_vague_query(
        self,
        canonical_query: CanonicalQuery,
        original_query: str
    ) -> DisambiguationResponse:
        """Handle a query that is too vague/short."""
        question = QUESTION_TEMPLATES[DisambiguationReason.QUERY_TOO_VAGUE]
        
        options = [
            Choice(
                id="vague_definition",
                label="Understand what this is",
                entity_id=None,
                intent=IntentType.DEFINITION.value,
                description=None
            ),
            Choice(
                id="vague_instruction",
                label="Get an instruction",
                entity_id=None,
                intent=IntentType.INSTRUCTION.value,
                description=None
            ),
            Choice(
                id="vague_troubleshoot",
                label="Solve a problem",
                entity_id=None,
                intent=IntentType.TROUBLESHOOTING.value,
                description=None
            ),
            Choice(
                id="vague_other",
                label="Some other question",
                entity_id=None,
                intent=IntentType.OTHER.value,
                description=None
            ),
        ]
        
        return DisambiguationResponse(
            needs_clarification=True,
            reason=DisambiguationReason.QUERY_TOO_VAGUE,
            question=question,
            options=options,
            explanation=f"Query is too short: {len(canonical_query.lemmas)} tokens",
            original_query=original_query
        )
    
    def _handle_low_relevance(
        self,
        entity_resolution: EntityResolutionResult,
        original_query: str,
        search_score: float
    ) -> DisambiguationResponse:
        """Handle the case when the best search score is below threshold."""
        question = QUESTION_TEMPLATES[DisambiguationReason.LOW_RELEVANCE]
        
        options = []
        
        for i, resolved in enumerate(entity_resolution.resolved[:3]):
            entity = resolved.entity
            options.append(Choice(
                id=f"low_rel_{i}_{entity.entity_id}",
                label=entity.label,
                entity_id=entity.entity_id,
                intent=None,
                description=self._get_entity_description(entity)
            ))
        
        if not options:
            options.append(Choice(
                id="low_rel_rephrase",
                label="Re‑phrase the query",
                entity_id=None,
                intent=None,
                description="Try asking the question in a different way"
            ))
        
        options.append(Choice(
            id="low_rel_none",
            label="None of the above",
            entity_id=None,
            intent=None,
            description=None
        ))
        
        return DisambiguationResponse(
            needs_clarification=True,
            reason=DisambiguationReason.LOW_RELEVANCE,
            question=question,
            options=options,
            explanation=f"Top score {search_score:.2f} is below threshold",
            original_query=original_query
        )
    
    def _has_multiple_relevant_topics(
        self,
        search_results: List[Dict],
        score_threshold: float,
        top_n: int = 10,
        min_documents: int = 3
    ) -> bool:
        """
        Check if there are several distinct documents with scores above threshold.
        """
        top = search_results[:top_n]
        sources_above_threshold = set()
        for r in top:
            score = r.get('score', -999)
            if score >= score_threshold:
                source = r.get('source', r.get('source_id', '')) or 'Unknown source'
                sources_above_threshold.add(source)
        return len(sources_above_threshold) >= min_documents
    
    def _disambiguate_by_topics(
        self,
        search_results: List[Dict],
        original_query: str
    ) -> DisambiguationResponse:
        """Generate document‑level choices grouped by category."""
        question = QUESTION_TEMPLATES[DisambiguationReason.MULTIPLE_TOPICS]
        
        # Collect unique (source, category, label) triples across many chunks so that
        # the relevant document appears in the list.
        seen_sources: Dict[str, tuple] = {}  # normalized_source -> (label, category, original_source)
        scan_limit = getattr(self, 'disambiguate_scan_chunks', 30)
        strict_threshold = getattr(settings, 'DISAMBIGUATION_SCORE_THRESHOLD', 0.0)
        
        for r in search_results[:scan_limit]:
            score = r.get('score', -999)
            if score < strict_threshold:
                continue
            
            raw_source = r.get('source', r.get('source_id', '')) or 'Unknown source'
            normalized = os.path.basename(raw_source) if raw_source else 'Unknown source'
            if normalized not in seen_sources:
                label = normalized if len(normalized) <= 80 else normalized[:77] + "..."
                category = (r.get('category') or '').strip() or 'General'
                seen_sources[normalized] = (label, category, raw_source)
        
        if not seen_sources and search_results:
            logger.info(f"Strict threshold {strict_threshold} filtered out all documents, using top‑3 fallback")
            for r in search_results[:3]:
                raw_source = r.get('source', r.get('source_id', '')) or 'Unknown source'
                normalized = os.path.basename(raw_source) if raw_source else 'Unknown source'
                if normalized not in seen_sources:
                    label = normalized if len(normalized) <= 80 else normalized[:77] + "..."
                    category = (r.get('category') or '').strip() or 'General'
                    seen_sources[normalized] = (label, category, raw_source)
        
        # Sort by category, then by source – useful for UI grouping
        items = [(key, seen_sources[key][0], seen_sources[key][1], seen_sources[key][2]) for key in seen_sources]
        items.sort(key=lambda x: (x[2].lower(), x[0].lower()))
        
        max_opts = getattr(self, 'max_topic_options', 10)
        options = []
        for i, (key, label, category, original_source) in enumerate(items):
            if i >= max_opts:
                break
            options.append(Choice(
                id=f"topic_{i}",
                label=label,
                entity_id=None,
                intent=None,
                description=original_source,
                category=category
            ))
        
        options.append(Choice(
            id="topic_show_all",
            label="Show all",
            entity_id=None,
            intent=None,
                description="Answer using all relevant documents",
            category=None
        ))
        
        return DisambiguationResponse(
            needs_clarification=True,
            reason=DisambiguationReason.MULTIPLE_TOPICS,
            question=question,
            options=options,
            explanation=f"Found {len(seen_sources)} document types for the query",
            original_query=original_query
        )
    
    def _get_entity_description(self, entity: Entity) -> Optional[str]:
        """Generate a short description snippet for an entity."""
        if entity.contexts:
            ctx = entity.contexts[0][:100]
            if len(entity.contexts[0]) > 100:
                ctx += "..."
            return ctx
        return None
    
    def apply_clarification(
        self,
        original_query: str,
        chosen_option: Choice
    ) -> str:
        """Apply a chosen clarification option to the original query."""
        if chosen_option.entity_id:
            return f"{original_query} [entity:{chosen_option.entity_id}]"
        if chosen_option.intent:
            return f"[intent:{chosen_option.intent}] {original_query}"
        return original_query


_disambiguator_instance = None

def get_disambiguator() -> Disambiguator:
    """Return singleton instance of Disambiguator."""
    global _disambiguator_instance
    if _disambiguator_instance is None:
        _disambiguator_instance = Disambiguator()
    return _disambiguator_instance

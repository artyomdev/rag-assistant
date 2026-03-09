"""
Entity Resolver – resolving entities and handling abbreviations in a query.

Abbreviations are interpreted through their usage context rather than pure
string expansion. A small set of domain‑agnostic abbreviations has a hardcoded
mapping for better recall (for example, KPI → key performance indicator).
"""
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from enum import Enum

from app.config import settings
from app.entity_registry import get_entity_registry, Entity
from app.canonicalizer import CanonicalQuery
from app.embedder import get_embedder

logger = logging.getLogger(__name__)


# ============================================================================
# HARDCODED MAPPING FOR IMPORTANT ABBREVIATIONS
# This mapping overrides automatic resolution for a few well‑known abbreviations
# and is intentionally small and generic for portfolio use.
# ============================================================================
ABBREVIATION_EXPANSIONS: Dict[str, List[str]] = {
    # Key performance indicators
    "KPI": ["key performance indicator", "key performance indicators"],
    "SLA": ["service level agreement", "service-level agreement"],

    # Common business / IT acronyms
    "CRM": ["customer relationship management system", "CRM system"],
    "ERP": ["enterprise resource planning system", "ERP system"],
    "SSO": ["single sign-on", "single sign on"],

    # A few lower‑case variants for robustness
    "kpi": ["key performance indicator", "key performance indicators"],
    "sla": ["service level agreement"],
    "crm": ["customer relationship management system"],
    "erp": ["enterprise resource planning system"],
}


class ResolutionConfidence(Enum):
    """Confidence level for an entity resolution result."""
    HIGH = "high"           # > 0.85 similarity
    MEDIUM = "medium"       # 0.7 - 0.85
    LOW = "low"             # 0.5 - 0.7
    AMBIGUOUS = "ambiguous" # Multiple close candidates
    NOT_FOUND = "not_found" # No match found


@dataclass
class ResolvedEntity:
    """Single resolved entity with confidence and score."""
    entity: Entity
    confidence: ResolutionConfidence
    score: float
    context_used: Optional[str]


@dataclass
class EntityResolutionResult:
    """Full resolution result for all entities in a query."""
    resolved: List[ResolvedEntity]
    ambiguous: List[Tuple[str, List[ResolvedEntity]]]
    not_found: List[str]
    has_abbreviations: bool
    needs_clarification: bool


def expand_abbreviations(text: str) -> str:
    """Expand known abbreviations inline to improve search recall.

    Example: "KPI dashboard" → "KPI dashboard (key performance indicator)".
    """
    expanded = text
    words = re.findall(r'\b([A-Za-z]+)\b', text)
    
    added_expansions: List[str] = []
    for word in words:
        expansions = ABBREVIATION_EXPANSIONS.get(word) or ABBREVIATION_EXPANSIONS.get(word.upper())
        if expansions:
            main_expansion = expansions[0]
            if main_expansion.lower() not in text.lower():
                added_expansions.append(main_expansion)
    
    if added_expansions:
        expanded = f"{text} ({' '.join(added_expansions)})"
        logger.debug(f"Abbreviation expansions applied: '{text}' → '{expanded}'")
    
    return expanded


def get_known_abbreviation_expansion(abbr: str) -> Optional[str]:
    """Return a known expansion for an abbreviation, if any."""
    expansions = ABBREVIATION_EXPANSIONS.get(abbr) or ABBREVIATION_EXPANSIONS.get(abbr.upper())
    return expansions[0] if expansions else None


class EntityResolver:
    """Entity and abbreviation resolution in user queries."""
    
    # Context templates used when building embeddings for abbreviations
    CONTEXT_TEMPLATES = [
        "create {abbr}",
        "configure {abbr}",
        "run {abbr}",
        "work with {abbr}",
        "{abbr} in the system",
    ]
    
    def __init__(self):
        self.registry = get_entity_registry()
        self.embedder = get_embedder()
        
        self.high_threshold = 0.85
        self.medium_threshold = 0.7
        self.low_threshold = 0.5
        self.ambiguity_gap = 0.1
        
        self.abbrev_pattern = re.compile(r'\b([A-Z]{2,8})\b')
        
        logger.info("✅ EntityResolver initialised")
    
    def resolve(self, canonical_query: CanonicalQuery) -> EntityResolutionResult:
        """Resolve entities in the canonical query (currently abbreviations only)."""
        resolved = []
        ambiguous = []
        not_found = []
        
        query_text = canonical_query.original
        abbreviations = canonical_query.abbreviations
        has_abbreviations = len(abbreviations) > 0
        
        for abbr in abbreviations:
            result = self._resolve_abbreviation(abbr, query_text)
            
            if result is None:
                not_found.append(abbr)
            elif isinstance(result, list):
                ambiguous.append((abbr, result))
            else:
                resolved.append(result)
        
        needs_clarification = len(ambiguous) > 0
        
        return EntityResolutionResult(
            resolved=resolved,
            ambiguous=ambiguous,
            not_found=not_found,
            has_abbreviations=has_abbreviations,
            needs_clarification=needs_clarification
        )
    
    def _resolve_abbreviation(self, abbr: str, query_context: str) -> Optional[ResolvedEntity | List[ResolvedEntity]]:
        """Resolve an abbreviation via contextual search in the entity registry."""
        direct_entity = self.registry.get_entity_by_alias(abbr)
        if direct_entity and direct_entity.frequency > 5:
            return ResolvedEntity(
                entity=direct_entity,
                confidence=ResolutionConfidence.HIGH,
                score=0.95,
                context_used="direct_alias"
            )
        
        context_embeddings = self._build_context_embeddings(abbr)
        
        candidates = []
        for ctx_text, ctx_embedding in context_embeddings:
            matches = self.registry.find_by_embedding(ctx_embedding, top_k=3)
            for entity, score in matches:
                candidates.append((entity, score, ctx_text))
        
        if not candidates:
            return None
        
        entity_scores = {}
        for entity, score, ctx in candidates:
            if entity.entity_id not in entity_scores:
                entity_scores[entity.entity_id] = {
                    'entity': entity,
                    'scores': [],
                    'contexts': []
                }
            entity_scores[entity.entity_id]['scores'].append(score)
            entity_scores[entity.entity_id]['contexts'].append(ctx)
        
        ranked = []
        for eid, data in entity_scores.items():
            avg_score = sum(data['scores']) / len(data['scores'])
            ranked.append((data['entity'], avg_score, data['contexts'][0]))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        if not ranked:
            return None
        
        top_entity, top_score, top_context = ranked[0]
        
        if top_score >= self.high_threshold:
            confidence = ResolutionConfidence.HIGH
        elif top_score >= self.medium_threshold:
            confidence = ResolutionConfidence.MEDIUM
        elif top_score >= self.low_threshold:
            confidence = ResolutionConfidence.LOW
        else:
            return None
        
        if len(ranked) > 1:
            second_entity, second_score, _ = ranked[1]
            if top_score - second_score < self.ambiguity_gap:
                return [
                    ResolvedEntity(
                        entity=entity,
                        confidence=ResolutionConfidence.AMBIGUOUS,
                        score=score,
                        context_used=ctx,
                    )
                    for entity, score, ctx in ranked[:3]
                ]
        
        return ResolvedEntity(
            entity=top_entity,
            confidence=confidence,
            score=top_score,
            context_used=top_context
        )
    
    def _build_context_embeddings(self, abbr: str) -> List[Tuple[str, List[float]]]:
        """Build contextual embeddings for an abbreviation."""
        contexts = []
        for template in self.CONTEXT_TEMPLATES:
            ctx_text = template.format(abbr=abbr)
            embedding = self.embedder.encode(ctx_text, normalize_embeddings=True).tolist()
            contexts.append((ctx_text, embedding))
        
        return contexts
    
    def get_entity_ids_for_search(self, resolution_result: EntityResolutionResult) -> List[str]:
        """Extract entity ids suitable for use as a search filter."""
        entity_ids = []
        
        for resolved in resolution_result.resolved:
            if resolved.confidence in (ResolutionConfidence.HIGH, ResolutionConfidence.MEDIUM):
                entity_ids.append(resolved.entity.entity_id)
        
        return entity_ids


_resolver_instance = None

def get_entity_resolver() -> EntityResolver:
    """Return singleton instance of EntityResolver."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = EntityResolver()
    return _resolver_instance

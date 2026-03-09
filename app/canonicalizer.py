"""
Query Canonicalizer – normalisation of user queries for more stable retrieval.

Semantically equivalent questions in different wordings should map to the same
canonical form where possible.
"""
import re
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Minimal English stop‑word list, used only for light normalisation.
STOP_WORDS = {
    "a", "an", "the",
    "and", "or", "but",
    "of", "for", "to", "in", "on", "at", "by", "with", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did",
    "this", "that", "these", "those",
    "it", "its", "as",
    "i", "you", "we", "they", "he", "she", "them", "us",
    "can", "could", "should", "would", "may", "might",
    "please", "thanks", "thank", "hi", "hello",
}

# Mapping of typical English formulations to canonical “patterns”.
QUERY_PATTERNS = {
    # Definition
    r'^what\s+is\s+': 'definition',
    r'^what\s+does\s+.+\s+mean': 'definition',
    r'^definition\s+of\s+': 'definition',

    # Instruction
    r'^how\s+to\s+': 'instruction',
    r'^how\s+do\s+i\s+': 'instruction',
    r'^step\s+by\s+step': 'instruction',
    r'^guide\s+to\s+': 'instruction',

    # Troubleshooting
    r'^why\s+is\s+.+\s+not\s+': 'troubleshooting',
    r'error\s+code': 'troubleshooting',
    r'does\s+not\s+work': 'troubleshooting',

    # Comparison
    r'^what\s+is\s+the\s+difference\s+between\s+': 'comparison',
    r'^difference\s+between\s+': 'comparison',

    # List
    r'^what\s+types\s+of\s+': 'list',
    r'^list\s+all\s+': 'list',

    # Navigation
    r'^where\s+can\s+i\s+find\s+': 'navigation',
    r'^where\s+is\s+': 'navigation',
}


@dataclass
class CanonicalQuery:
    """Canonical representation of a user query."""
    text: str                    # Normalised text used for retrieval
    original: str                # Original user query
    lemmas: List[str]            # Token/lemma list (simple lower‑cased tokens)
    detected_pattern: Optional[str]  # Detected pattern (definition, instruction, etc.)
    abbreviations: List[str]     # Extracted abbreviations
    
    def __repr__(self):
        return f"CanonicalQuery(text='{self.text}', pattern={self.detected_pattern}, abbrevs={self.abbreviations})"


class QueryCanonicalizer:
    """Normalise queries to make retrieval behaviour more stable."""
    
    def __init__(self):
        # Compile patterns once
        self.patterns = [(re.compile(p, re.IGNORECASE), intent) 
                         for p, intent in QUERY_PATTERNS.items()]
        
        # Abbreviation pattern (2–8 uppercase latin letters, e.g. KPI, SLA)
        self.abbrev_pattern = re.compile(r'\b[A-Z]{2,8}\b')
    
    def canonicalize(self, query: str) -> CanonicalQuery:
        """
        Turn a free‑form user query into a canonical form.
        """
        original = query.strip()
        
        # 1. Normalise whitespace and basic punctuation
        normalized = self._normalize_whitespace(original)
        normalized_lower = normalized.lower()
        
        # 2. Detect query pattern (if any)
        detected_pattern = self._detect_pattern(normalized_lower)
        
        # 3. Extract abbreviations (before lower‑casing)
        abbreviations = self._extract_abbreviations(normalized)
        
        # 4. Simple tokenisation / “lemmatisation”
        lemmas = self._lemmatize(normalized_lower)
        
        # 5. Remove stop‑words for retrieval
        filtered_lemmas = [l for l in lemmas if l not in STOP_WORDS]
        
        # 6. Canonical text (tokens without stop‑words, but abbreviations preserved)
        canonical_text = ' '.join(filtered_lemmas)
        
        for abbr in abbreviations:
            if abbr.lower() in canonical_text:
                canonical_text = canonical_text.replace(abbr.lower(), abbr)
        
        return CanonicalQuery(
            text=canonical_text if canonical_text else normalized_lower,
            original=original,
            lemmas=lemmas,
            detected_pattern=detected_pattern,
            abbreviations=abbreviations
        )
    
    def _normalize_whitespace(self, text: str) -> str:
        """Whitespace and basic punctuation normalisation."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip(' ?!.,;:')
        return text
    
    def _detect_pattern(self, text: str) -> Optional[str]:
        """Detect a high‑level pattern for the query using regex triggers."""
        for pattern, intent in self.patterns:
            if pattern.search(text):
                return intent
        return None
    
    def _extract_abbreviations(self, text: str) -> List[str]:
        """Extract abbreviations from text, preserving order and removing duplicates."""
        matches = self.abbrev_pattern.findall(text)
        seen: set[str] = set()
        result: List[str] = []
        for m in matches:
            if m not in seen:
                seen.add(m)
                result.append(m)
        return result
    
    def _lemmatize(self, text: str) -> List[str]:
        """
        Very lightweight “lemmatisation”: just split into alphanumeric tokens
        and lowercase them. This keeps the code language‑agnostic and dependency‑free.
        """
        return re.findall(r'[a-z0-9]+', text.lower(), re.IGNORECASE)
    
    def are_equivalent(self, query1: str, query2: str) -> bool:
        """Check whether two queries are equivalent after canonicalisation."""
        canon1 = self.canonicalize(query1)
        canon2 = self.canonicalize(query2)
        return canon1.text == canon2.text


_canonicalizer_instance = None

def get_canonicalizer() -> QueryCanonicalizer:
    """Return a singleton instance of the canonicalizer."""
    global _canonicalizer_instance
    if _canonicalizer_instance is None:
        _canonicalizer_instance = QueryCanonicalizer()
    return _canonicalizer_instance

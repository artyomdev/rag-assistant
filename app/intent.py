"""
Intent Classifier – lightweight, rule‑based intent detection.

Uses a small fixed set of intents and regex triggers instead of a heavyweight model.
"""
import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Supported user intent types for a homogeneous documentation corpus."""
    DEFINITION = "definition"           # “What is X?”
    INSTRUCTION = "instruction"         # “How to do X?” (navigation, setup, steps)
    TROUBLESHOOTING = "troubleshooting"  # Errors, “it does not work”
    OTHER = "other"                     # Everything else


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: IntentType
    confidence: float              # 0.0 - 1.0
    matched_triggers: List[str]    # Regex patterns that matched
    is_ambiguous: bool             # Whether there are several strong intent candidates
    alternative_intents: List[Tuple[IntentType, float]]  # Alternative intents with scores


# Regex triggers for each intent. OTHER has no explicit triggers (fallback).
INTENT_TRIGGERS: Dict[IntentType, List[str]] = {
    IntentType.DEFINITION: [
        r'\bwhat\s+is\s+',
        r'\bwhat\s+does\s+.+\s+mean\b',
        r'\bdefine\b',
        r'\bdefinition\s+of\b',
        r'\bexplain\s+what\b',
    ],
    
    IntentType.INSTRUCTION: [
        r'\bhow\s+to\s+',
        r'\bhow\s+do\s+i\s+',
        r'\bconfiguration\b',
        r'\bconfigure\b',
        r'\bsetup\b',
        r'\bset\s+up\b',
        r'\bstep\s+by\s+step\b',
        r'\bguide\b',
        r'\bwhere\s+can\s+i\s+find\s+',
        r'\bwhere\s+is\s+',
        r'\bwhich\s+button\b',
        r'\bmenu\b',
        r'\bworkflow\b',
        r'\bprocess\b',
        r'\bsequence\b',
        r'\bsteps\b',
        r'\bprocedure\b',
    ],
    
    IntentType.TROUBLESHOOTING: [
        r'\bdoes\s+not\s+work\b',
        r'\bnot\s+working\b',
        r'\bcan\s+not\b',
        r"\bcan't\b",
        r'\bunable\s+to\b',
        r'\berror\b',
        r'\bfailed\b',
        r'\bfails\b',
        r'\bissue\b',
        r'\bproblem\b',
        r'\bwhy\s+is\s+.+\s+not\b',
        r'\bcrash(es)?\b',
        r'\bhangs\b',
        r'\bdoes\s+not\s+open\b',
        r'\bdoes\s+not\s+save\b',
        r'\bmissing\b',
    ],
}


class IntentClassifier:
    """Simple regex‑based intent classifier."""
    
    def __init__(self, ambiguity_threshold: float = 0.3):
        """
        Args:
            ambiguity_threshold: if the difference between top‑1 and top‑2 scores
                                 is below this value, we treat the intent as ambiguous.
        """
        self.ambiguity_threshold = ambiguity_threshold
        
        # Compile regex patterns
        self.compiled_triggers: Dict[IntentType, List[re.Pattern]] = {}
        for intent, patterns in INTENT_TRIGGERS.items():
            self.compiled_triggers[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        logger.info(f"✅ IntentClassifier initialised with {len(IntentType)} intents")
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify user intent.
        """
        query_lower = query.lower().strip()
        
        scores: Dict[IntentType, Tuple[float, List[str]]] = {}
        
        for intent, patterns in self.compiled_triggers.items():
            matched = []
            for pattern in patterns:
                if pattern.search(query_lower):
                    matched.append(pattern.pattern)
            
            if matched:
                # Score = matched patterns count / total patterns for that intent
                score = len(matched) / len(patterns)
                scores[intent] = (score, matched)
        
        if not scores:
            return IntentResult(
                intent=IntentType.OTHER,
                confidence=0.5,
                matched_triggers=[],
                is_ambiguous=False,
                alternative_intents=[]
            )
        
        sorted_intents = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        
        top_intent, (top_score, top_triggers) = sorted_intents[0]
        
        is_ambiguous = False
        alternatives = []
        
        if len(sorted_intents) > 1:
            second_intent, (second_score, _) = sorted_intents[1]
            
            if top_score - second_score < self.ambiguity_threshold:
                is_ambiguous = True
            
            # Collect alternatives with score > 0.1
            for intent, (score, _) in sorted_intents[1:4]:
                if score >= 0.1:
                    alternatives.append((intent, score))
        
        # Normalise confidence
        confidence = min(1.0, top_score + 0.3)
        if is_ambiguous:
            confidence *= 0.7
        
        return IntentResult(
            intent=top_intent,
            confidence=confidence,
            matched_triggers=top_triggers,
            is_ambiguous=is_ambiguous,
            alternative_intents=alternatives
        )
    
    def get_intent_description(self, intent: IntentType) -> str:
        """Return a short, user‑friendly description of an intent."""
        descriptions = {
            IntentType.DEFINITION: "Understand what something is",
            IntentType.INSTRUCTION: "Get instructions (how to do, where to find, configuration)",
            IntentType.TROUBLESHOOTING: "Resolve an error or problem",
            IntentType.OTHER: "General question",
        }
        return descriptions.get(intent, "Question")
    
    def get_all_intents(self) -> List[IntentType]:
        """Return the list of all supported intents."""
        return list(IntentType)


_classifier_instance = None

def get_intent_classifier() -> IntentClassifier:
    """Return a singleton instance of the classifier."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance

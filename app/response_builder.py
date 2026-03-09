"""
Response Builder – structured answers with constrained LLM behaviour.

The LLM is instructed not to paraphrase aggressively and not to add information
outside of the provided context. The final answer is built from the intent and
facts extracted from the ranked context chunks.
"""
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import ollama

from app.config import settings
from app.intent import IntentType

logger = logging.getLogger(__name__)


@dataclass
class StructuredSource:
    """Single structured source item."""
    id: str
    name: str
    link: Optional[str]
    category: Optional[str]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StructuredAnswer:
    """Structured answer returned by the system."""
    answer: str
    sources: List[StructuredSource]
    intent: str
    confidence: float                    # 0–1
    context_used: str
    debug_info: Dict
    
    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "intent": self.intent,
            "confidence": self.confidence,
            "context": self.context_used,
            "debug": self.debug_info
        }


# Minimal score for a chunk to be included into the final LLM context (BGE‑Reranker logits)
MIN_CONTEXT_SCORE = -2.0

# Universal system prompt for detailed, procedural answers (all intent types)
UNIVERSAL_SYSTEM_PROMPT = """You answer questions about internal product and process documentation.

RULES:
1. Use ONLY the information from the provided context.
2. Answer the specific question – do not add unrelated background.
3. If there is a step‑by‑step instruction in the context, include ALL steps (do not shorten them).
4. Preserve exact names of buttons, menus, fields and documents.
5. Use screenshot descriptions marked as [SCREENSHOT] to give precise navigation.
6. Mention the source document at the end of the answer.
7. If the context is not sufficient, say this explicitly instead of guessing.
8. IMPORTANT: When the question is about a specific operation (for example, creating a request or calculating a price),
   do NOT include generic descriptions of the system or modules. Focus only on the requested operation.

ANSWER FORMAT:
- Short summary: 1–2 sentences with the main idea.
- Details: step‑by‑step instruction or key points.
- Source: document name.

EXAMPLE OF A GOOD ANSWER (for illustration only):

Question: How do I create a payment request to a supplier?

**Short answer:**
Open the “Purchasing” section → “Purchase documents (All)”, find the document and create a payment request based on it.

**Details:**
1. Go to the “Purchasing” section.
2. Open the “Purchase documents (All)” list.
3. Find the required document – you can filter by counterparty, document number or date.
4. Open the document → click “Create based on” → choose “Payment request”.
5. In the new form, verify and, if needed, adjust:
   - Bank accounts of the organisation and the recipient.
   - Payment date.
   - Payment purpose text.
6. Click “Save” → “Post”.

**Source:** Internal payment processing guide."""


class ResponseBuilder:
    """Builds structured LLM answers from ranked retrieval results."""
    
    def __init__(self):
        self.llm_model = settings.LLM_MODEL
        logger.info(f"✅ ResponseBuilder initialised with model {self.llm_model}")
    
    def build_response(
        self,
        query: str,
        intent: IntentType,
        ranked_results: List[Dict],
        entity_ids: List[str] = None
    ) -> StructuredAnswer:
        """Build a structured answer from query, intent and ranked results."""
        if not ranked_results:
            return self._no_results_response(query, intent)
        
        context_str, sources = self._build_context_and_sources(ranked_results)
        
        system_prompt = UNIVERSAL_SYSTEM_PROMPT
        
        user_prompt = f"""CONTEXT:
{context_str}

QUESTION: {query}

ANSWER (based only on the context above, mention sources):"""
        
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            answer = response.message.content if hasattr(response, 'message') else response['message']['content']
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = "An error occurred while generating the answer."
        
        debug_info = {
            "intent": intent.value,
            "entity_ids": entity_ids or [],
            "top_scores": [r.get('score', 0) for r in ranked_results[:5]],
            "sources_count": len(sources)
        }
        
        top_score = ranked_results[0].get('score', 0) if ranked_results else 0
        confidence = self._calculate_confidence(top_score, len(ranked_results))
        
        return StructuredAnswer(
            answer=answer,
            sources=sources,
            intent=intent.value,
            confidence=confidence,
            context_used=context_str,
            debug_info=debug_info
        )
    
    def _build_context_and_sources(
        self,
        ranked_results: List[Dict]
    ) -> tuple[str, List[StructuredSource]]:
        """Build concatenated context string and list of structured sources.

        Filters out low‑relevance chunks (score < MIN_CONTEXT_SCORE).
        """
        context_parts = []
        sources_dict = {}
        
        filtered_results = [
            r for r in ranked_results 
            if r.get('score', 0) >= MIN_CONTEXT_SCORE
        ]
        
        if not filtered_results and ranked_results:
            filtered_results = ranked_results[:1]
            logger.warning(f"All results are below threshold {MIN_CONTEXT_SCORE}, using top‑1")
        
        logger.info(f"Context: {len(filtered_results)}/{len(ranked_results)} chunks (score >= {MIN_CONTEXT_SCORE})")
        
        for r in filtered_results:
            text = r.get('text', '')
            source_name = r.get('source', 'Unknown source')
            category = r.get('category', '')
            source_id = r.get('source_id', source_name)
            source_path = r.get('source_path', '')
            
            context_parts.append(f"[Source: {source_name}]\n{text}")
            
            if source_id not in sources_dict:
                link = f"/docs/{source_id}" if source_path else None
                
                sources_dict[source_id] = StructuredSource(
                    id=source_id,
                    name=source_name,
                    link=link,
                    category=category
                )
        
        context_str = "\n\n---\n\n".join(context_parts)
        sources = list(sources_dict.values())
        
        return context_str, sources
    
    def _calculate_confidence(self, top_score: float, results_count: int) -> float:
        """Approximate confidence based on reranker score and result count."""
        # BGE-Reranker scores: > 0 high confidence, > -4 medium
        if top_score >= 0:
            base_confidence = 0.9
        elif top_score >= -2:
            base_confidence = 0.7
        elif top_score >= -4:
            base_confidence = 0.5
        else:
            base_confidence = 0.3
        
        count_bonus = min(0.1, results_count * 0.02)
        
        return min(1.0, base_confidence + count_bonus)
    
    def _no_results_response(self, query: str, intent: IntentType) -> StructuredAnswer:
        """Fallback answer when no search results are available."""
        return StructuredAnswer(
            answer="No relevant information was found in the knowledge base.",
            sources=[],
            intent=intent.value,
            confidence=0.0,
            context_used="",
            debug_info={"reason": "no_results", "fallback": "support"}
        )
    
    def build_clarification_response(
        self,
        disambiguation_data: Dict
    ) -> Dict:
        """Build an API payload for clarification responses."""
        return {
            "needs_clarification": True,
            "clarification": disambiguation_data,
            "answer": None,
            "sources": [],
            "debug": {"type": "disambiguation"}
        }


_builder_instance = None

def get_response_builder() -> ResponseBuilder:
    """Return singleton instance of ResponseBuilder."""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = ResponseBuilder()
    return _builder_instance

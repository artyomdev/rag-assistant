"""
Entity Registry – automatic extraction and clustering of entities from documents.

Entities are discovered automatically from text and clustered using embeddings;
they are not manually curated.
"""
import re
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import numpy as np

from app.config import settings
from app.embedder import get_embedder

logger = logging.getLogger(__name__)

# Path to the entity registry file
ENTITY_REGISTRY_PATH = Path(settings.DB_PATH).parent / "entity_registry.json"


@dataclass
class Entity:
    """Entity in the knowledge base."""
    entity_id: str              # Deterministic identifier (e.g. hash)
    label: str                  # Primary, most frequent surface form
    aliases: List[str]          # All observed surface forms
    entity_type: str            # ABBREV, TERM, DOCUMENT, CONCEPT
    embedding: List[float]      # Entity embedding vector
    contexts: List[str] = field(default_factory=list)  # Example usage contexts
    frequency: int = 1          # Number of occurrences
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Entity':
        return cls(**data)


@dataclass
class EntityMention:
    """A single entity mention found in text."""
    text: str                   # Mention surface text
    context: str                # Surrounding text context
    source: str                 # Source document
    entity_type: str            # Guessed type


class EntityExtractor:
    """Rule‑based entity extraction from text."""
    
    # Extraction patterns
    ABBREV_PATTERN = re.compile(r'\b([A-Z]{2,8})\b')          # Abbreviations like KPI, SLA, CRM
    QUOTED_PATTERN = re.compile(r'["“]([^"”]+)["”]')          # Terms in quotes
    
    # Abbreviations to ignore (too generic)
    IGNORE_ABBREVS = {'AND', 'OR', 'NOT', 'ALL'}
    
    def __init__(self):
        # Optional: NER model (language‑agnostic, you can plug any spaCy model here)
        self.ner_model = None
        try:
            import spacy
            try:
                self.ner_model = spacy.load("en_core_web_sm")
                logger.info("✅ SpaCy en_core_web_sm loaded for NER")
            except OSError:
                try:
                    self.ner_model = spacy.load("en_core_web_md")
                    logger.info("✅ SpaCy en_core_web_md loaded for NER")
                except OSError:
                    logger.warning("⚠️ No English spaCy model found, falling back to rule‑based extraction")
        except ImportError:
            logger.warning("⚠️ SpaCy is not installed, using rule‑based extraction only")
    
    def extract_from_text(self, text: str, source: str = "unknown") -> List[EntityMention]:
        """
        Extract entity mentions from text.
        """
        mentions = []
        
        for match in self.ABBREV_PATTERN.finditer(text):
            abbr = match.group(1)
            if abbr not in self.IGNORE_ABBREVS and len(abbr) >= 2:
                context = self._get_context(text, match.start(), match.end())
                mentions.append(EntityMention(
                    text=abbr,
                    context=context,
                    source=source,
                    entity_type="ABBREV"
                ))
        
        for match in self.QUOTED_PATTERN.finditer(text):
            term = match.group(1).strip()
            if len(term) >= 2 and len(term) <= 100:
                context = self._get_context(text, match.start(), match.end())
                mentions.append(EntityMention(
                    text=term,
                    context=context,
                    source=source,
                    entity_type="TERM"
                ))
        
        if self.ner_model:
            try:
                doc = self.ner_model(text[:10000])  # Limit for performance
                for ent in doc.ents:
                    if ent.label_ in ('ORG', 'PRODUCT', 'WORK_OF_ART', 'LOC'):
                        context = self._get_context(text, ent.start_char, ent.end_char)
                        mentions.append(EntityMention(
                            text=ent.text,
                            context=context,
                            source=source,
                            entity_type=ent.label_
                        ))
            except Exception as e:
                logger.warning(f"NER error: {e}")
        
        return mentions
    
    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Extract a window of context around the mention."""
        ctx_start = max(0, start - window)
        ctx_end = min(len(text), end + window)
        return text[ctx_start:ctx_end].strip()


class EntityClusterer:
    """Cluster entities based on their embeddings."""
    
    def __init__(self, embedder, similarity_threshold: float = 0.85):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
    
    def cluster_mentions(self, mentions: List[EntityMention]) -> List[Entity]:
        """
        Cluster mentions into entities.
        """
        if not mentions:
            return []
        
        groups: Dict[str, List[EntityMention]] = defaultdict(list)
        for m in mentions:
            key = f"{m.entity_type}:{m.text.lower().strip()}"
            groups[key].append(m)
        
        preliminary_entities = []
        for key, group in groups.items():
            entity_type = group[0].entity_type
            text_counts = defaultdict(int)
            contexts = []
            sources = set()
            
            for m in group:
                text_counts[m.text] += 1
                contexts.append(m.context)
                sources.add(m.source)
            
            label = max(text_counts.keys(), key=lambda x: text_counts[x])
            aliases = list(text_counts.keys())
            
            # Embedding from label + sample context
            embed_text = f"{label}: {contexts[0][:200]}" if contexts else label
            embedding = self.embedder.encode(embed_text, normalize_embeddings=True).tolist()
            
            entity_id = self._generate_entity_id(label, entity_type)
            
            preliminary_entities.append(
                Entity(
                    entity_id=entity_id,
                    label=label,
                    aliases=aliases,
                    entity_type=entity_type,
                    embedding=embedding,
                    contexts=contexts[:5],
                    frequency=len(group),
                )
            )
        
        merged_entities = self._merge_similar_entities(preliminary_entities)
        
        return merged_entities
    
    def _generate_entity_id(self, label: str, entity_type: str) -> str:
        """Generate a deterministic ID for an entity."""
        content = f"{entity_type}:{label.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _merge_similar_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge similar entities using cosine similarity."""
        if len(entities) <= 1:
            return entities
        
        embeddings = np.array([e.embedding for e in entities])
        
        similarities = np.dot(embeddings, embeddings.T)
        
        parent = list(range(len(entities)))
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            pi, pj = find(i), find(j)
            if pi != pj:
                parent[pi] = pj
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if similarities[i, j] > self.similarity_threshold:
                    if entities[i].entity_type == entities[j].entity_type:
                        union(i, j)
        
        clusters: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(entities)):
            clusters[find(i)].append(i)
        
        merged = []
        for indices in clusters.values():
            if len(indices) == 1:
                merged.append(entities[indices[0]])
            else:
                main = entities[indices[0]]
                all_aliases = set(main.aliases)
                all_contexts = list(main.contexts)
                total_freq = main.frequency
                
                for idx in indices[1:]:
                    e = entities[idx]
                    all_aliases.update(e.aliases)
                    all_contexts.extend(e.contexts)
                    total_freq += e.frequency
                
                # Choose label by frequency
                alias_freq = defaultdict(int)
                for idx in indices:
                    for alias in entities[idx].aliases:
                        alias_freq[alias] += entities[idx].frequency
                
                best_label = max(alias_freq.keys(), key=lambda x: alias_freq[x])
                
                merged.append(
                    Entity(
                        entity_id=main.entity_id,
                        label=best_label,
                        aliases=list(all_aliases),
                        entity_type=main.entity_type,
                        embedding=main.embedding,
                        contexts=all_contexts[:10],
                        frequency=total_freq,
                    )
                )
        
        return merged


class EntityRegistry:
    """Persistent registry of discovered entities."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.alias_index: Dict[str, str] = {}  # alias -> entity_id
        
        self.embedder = get_embedder()
        self.extractor = EntityExtractor()
        self.clusterer = EntityClusterer(self.embedder)
        
        self._load()
    
    def _load(self):
        """Load registry from disk if it exists."""
        if ENTITY_REGISTRY_PATH.exists():
            try:
                with open(ENTITY_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get('entities', []):
                        entity = Entity.from_dict(item)
                        self.entities[entity.entity_id] = entity
                        for alias in entity.aliases:
                            self.alias_index[alias.lower()] = entity.entity_id
                logger.info(f"✅ Loaded {len(self.entities)} entities from registry")
            except Exception as e:
                logger.error(f"Error loading entity registry: {e}")
    
    def _save(self):
        """Persist registry to disk."""
        ENTITY_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'entities': [e.to_dict() for e in self.entities.values()]
        }
        with open(ENTITY_REGISTRY_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved {len(self.entities)} entities to registry")
    
    def process_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Process chunks, extract entities and attach entity_ids.
        """
        all_mentions = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            source = chunk.get('metadata', {}).get('source', 'unknown')
            mentions = self.extractor.extract_from_text(text, source)
            all_mentions.extend(mentions)
        
        if all_mentions:
            new_entities = self.clusterer.cluster_mentions(all_mentions)
            
            for entity in new_entities:
                existing = self.entities.get(entity.entity_id)
                if existing:
                    existing.aliases = list(set(existing.aliases + entity.aliases))
                    existing.contexts = (existing.contexts + entity.contexts)[:10]
                    existing.frequency += entity.frequency
                else:
                    self.entities[entity.entity_id] = entity
                
                for alias in entity.aliases:
                    self.alias_index[alias.lower()] = entity.entity_id
            
            logger.info(f"📊 Processed {len(all_mentions)} mentions → {len(new_entities)} new/updated entities")
        
        updated_chunks = []
        for chunk in chunks:
            text = chunk.get('text', '')
            entity_ids = self._find_entities_in_text(text)
            
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata']['entity_ids'] = list(entity_ids)
            updated_chunks.append(chunk)
        
        # 5. Persist registry
        self._save()
        
        return updated_chunks
    
    def _find_entities_in_text(self, text: str) -> Set[str]:
        """Find entity_ids that appear in the given text."""
        entity_ids = set()
        text_lower = text.lower()
        
        for alias, entity_id in self.alias_index.items():
            if alias in text_lower:
                entity_ids.add(entity_id)
        
        return entity_ids
    
    def resolve_text(self, text: str) -> List[Entity]:
        """
        Find entities mentioned in a given text snippet.
        """
        entity_ids = self._find_entities_in_text(text)
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]
    
    def find_by_embedding(self, query_embedding: List[float], top_k: int = 5, 
                          entity_type: Optional[str] = None) -> List[Tuple[Entity, float]]:
        """
        Find entities most similar to the given embedding.
        """
        if not self.entities:
            return []
        
        candidates = list(self.entities.values())
        if entity_type:
            candidates = [e for e in candidates if e.entity_type == entity_type]
        
        if not candidates:
            return []
        
        query_vec = np.array(query_embedding)
        scores = []
        for entity in candidates:
            entity_vec = np.array(entity.embedding)
            score = float(np.dot(query_vec, entity_vec))
            scores.append((entity, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Return entity by ID."""
        return self.entities.get(entity_id)
    
    def get_entity_by_alias(self, alias: str) -> Optional[Entity]:
        """Return entity by alias."""
        entity_id = self.alias_index.get(alias.lower())
        if entity_id:
            return self.entities.get(entity_id)
        return None
    
    def get_all_entities(self) -> List[Entity]:
        """Return all entities."""
        return list(self.entities.values())
    
    def get_abbreviations(self) -> List[Entity]:
        """Return all abbreviation‑type entities."""
        return [e for e in self.entities.values() if e.entity_type == "ABBREV"]


_registry_instance = None

def get_entity_registry() -> EntityRegistry:
    """Return singleton instance of the entity registry."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = EntityRegistry()
    return _registry_instance

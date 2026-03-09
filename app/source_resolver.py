"""
Source Resolver – management of document sources.

Sources are returned in a structured form: id, name, link.
If a link cannot be resolved, the source is omitted from the result.
"""
import os
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from urllib.parse import quote

from app.config import settings

logger = logging.getLogger(__name__)

# Path to the source index
SOURCE_INDEX_PATH = Path(settings.DB_PATH).parent / "source_index.json"


@dataclass
class SourceInfo:
    """Metadata about a single source document."""
    source_id: str              # Unique ID
    name: str                   # File name
    path: str                   # Relative path from DATA_DIR
    category: str               # Category (folder)
    file_hash: str              # File hash for freshness checks
    is_available: bool          # Whether the file is available
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SourceInfo':
        return cls(**data)


class SourceResolver:
    """Manage and resolve document sources."""
    
    def __init__(self, base_url: str = "/docs"):
        """
        Args:
            base_url: Base URL used when building links to documents
        """
        self.base_url = base_url
        self.data_dir = Path(settings.DATA_DIR).resolve()
        self.sources: Dict[str, SourceInfo] = {}
        
        self._load_index()
        logger.info(f"✅ SourceResolver initialised with {len(self.sources)} sources")
    
    def _load_index(self):
        """Load source index from disk."""
        if SOURCE_INDEX_PATH.exists():
            try:
                with open(SOURCE_INDEX_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get('sources', []):
                        source = SourceInfo.from_dict(item)
                        self.sources[source.source_id] = source
                logger.info(f"📂 Loaded {len(self.sources)} sources from index")
            except Exception as e:
                logger.error(f"Error loading source index: {e}")
    
    def _save_index(self):
        """Persist source index to disk."""
        SOURCE_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'sources': [s.to_dict() for s in self.sources.values()]
        }
        with open(SOURCE_INDEX_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def register_source(self, file_path: Path) -> SourceInfo:
        """
        Register a new source file and return its metadata.
        """
        file_path = Path(file_path).resolve()
        
        source_id = self._generate_source_id(file_path)
        
        try:
            relative_path = str(file_path.relative_to(self.data_dir))
        except ValueError:
            relative_path = file_path.name
        
        category = str(Path(relative_path).parent)
        if category == ".":
            category = "General"
        
        file_hash = self._compute_file_hash(file_path)
        
        source = SourceInfo(
            source_id=source_id,
            name=file_path.name,
            path=relative_path,
            category=category,
            file_hash=file_hash,
            is_available=file_path.exists()
        )
        
        self.sources[source_id] = source
        self._save_index()
        
        return source
    
    def _generate_source_id(self, file_path: Path) -> str:
        """Generate a deterministic ID for a source based on its path."""
        try:
            relative = str(file_path.relative_to(self.data_dir))
        except ValueError:
            relative = file_path.name
        
        return hashlib.md5(relative.encode()).hexdigest()[:12]
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute file hash."""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def get_source(self, source_id: str) -> Optional[SourceInfo]:
        """Return source by ID."""
        return self.sources.get(source_id)
    
    def get_source_by_name(self, name: str) -> Optional[SourceInfo]:
        """Return source by file name."""
        for source in self.sources.values():
            if source.name == name:
                return source
        return None
    
    def resolve_source_link(self, source_id: str) -> Optional[str]:
        """
        Resolve a public URL for a source if it is available.
        """
        source = self.sources.get(source_id)
        if not source:
            return None
        
        file_path = self.data_dir / source.path
        if not file_path.exists():
            source.is_available = False
            return None
        
        source.is_available = True
        
        encoded_path = quote(source.path)
        return f"{self.base_url}/{encoded_path}"
    
    def enrich_search_results(self, results: List[Dict]) -> List[Dict]:
        """
        Enrich raw search results with source metadata and links.
        """
        enriched = []
        
        for r in results:
            source_name = r.get('source', '')
            
            source = self.get_source_by_name(source_name)
            
            if source:
                link = self.resolve_source_link(source.source_id)
                
                if link:
                    r['source_id'] = source.source_id
                    r['source_path'] = source.path
                    r['source_link'] = link
                    r['category'] = source.category
                    enriched.append(r)
                else:
                    logger.warning(f"Source not available on disk: {source_name}")
            else:
                possible_path = self.data_dir / source_name
                if possible_path.exists():
                    new_source = self.register_source(possible_path)
                    link = self.resolve_source_link(new_source.source_id)
                    if link:
                        r['source_id'] = new_source.source_id
                        r['source_path'] = new_source.path
                        r['source_link'] = link
                        r['category'] = new_source.category
                        enriched.append(r)
                else:
                    logger.warning(f"Source file not found: {source_name}")
        
        return enriched
    
    def get_structured_sources(self, results: List[Dict]) -> List[Dict]:
        """
        Build a deduplicated list of structured sources from enriched results.
        """
        sources_dict = {}
        
        for r in results:
            source_id = r.get('source_id')
            if not source_id or source_id in sources_dict:
                continue
            
            link = r.get('source_link')
            if not link:
                continue
            
            sources_dict[source_id] = {
                "id": source_id,
                "name": r.get('source', ''),
                "link": link,
                "category": r.get('category', '')
            }
        
        return list(sources_dict.values())
    
    def refresh_availability(self):
        """Refresh availability flags for all registered sources."""
        for source in self.sources.values():
            file_path = self.data_dir / source.path
            source.is_available = file_path.exists()
        
        self._save_index()
        
        available = sum(1 for s in self.sources.values() if s.is_available)
        logger.info(f"🔄 Availability updated: {available}/{len(self.sources)} sources available")


_resolver_instance = None

def get_source_resolver() -> SourceResolver:
    """Return singleton instance of SourceResolver."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = SourceResolver()
    return _resolver_instance

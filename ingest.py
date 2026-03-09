#!/usr/bin/env python3
"""
SmartIngestor – document ingestion and indexing with entity extraction.

Updated pipeline:
1. Document conversion (Docling).
2. Screenshot analysis (Vision).
3. Chunking.
4. Entity extraction and adding entity_ids.
5. Source registration.
6. Indexing into Qdrant.
"""
import os
import logging
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import timedelta

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from tqdm import tqdm

from app.search_engine import SearchEngine
from app.vision import VisionProcessor 
from app.config import settings
from app.entity_registry import get_entity_registry
from app.source_resolver import get_source_resolver

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_time(seconds: float) -> str:
    """Format duration in a human‑readable form."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        rest = int(seconds % 60)
        return f"{minutes}m {rest}s"
    else:
        return str(timedelta(seconds=int(seconds)))


class SmartIngestor:
    def __init__(self):
        from docling.document_converter import DocumentConverter
        from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
        
        self.converter = DocumentConverter()
        
        self.engine = SearchEngine()
        self.vision = VisionProcessor()
        
        self.entity_registry = get_entity_registry()
        
        self.source_resolver = get_source_resolver()
        
        self.headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=250,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info("✅ SmartIngestor initialised (with Entity Registry)")

    def _get_file_hash(self, file_path: str) -> str:
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def process_file(self, file_path: Path) -> tuple[List[Dict[str, Any]], dict]:
        """
        Process a single file and return (chunks, stats).
        
        stats keys: {time, images_count, chunks_count}
        """
        file_name = file_path.name
        file_start_time = time.time()
        stats = {"time": 0, "images_count": 0, "chunks_count": 0}
        
        try:
            relative_path = file_path.relative_to(Path(settings.DATA_DIR).resolve())
            category = str(relative_path.parent) if str(relative_path.parent) != "." else "General"
        except Exception:
            category = "External"
        
        source_info = self.source_resolver.register_source(file_path)
        
        try:
            result = self.converter.convert(str(file_path))
            doc = result.document
            markdown_text = doc.export_to_markdown()
            
            images_descriptions = []
            if hasattr(doc, 'pictures') and doc.pictures:
                stats["images_count"] = len(doc.pictures)
                for i, m_image in enumerate(doc.pictures):
                    try:
                        pil_img = m_image.get_image(doc)
                        description = self.vision.describe_image(pil_img)
                        images_descriptions.append(description)
                    except Exception as img_e:
                        logger.error(f"Image analysis error {i} in {file_name}: {img_e}")

            if images_descriptions:
                markdown_text += "\n\n### SCREENSHOT DESCRIPTION:\n"
                markdown_text += "\n".join(images_descriptions)

            header_splits = self.header_splitter.split_text(markdown_text)
            
            final_chunks = []
            for split in header_splits:
                h1 = split.metadata.get("Header_1", "General section")
                sub_chunks = self.text_splitter.split_text(split.page_content)
                
                for content in sub_chunks:
                    context_prefix = f"DOCUMENT: {file_name} | CATEGORY: {category} | SECTION: {h1}\n"
                    enriched_text = f"{context_prefix}CONTENT: {content}"
                    
                    final_chunks.append({
                        "text": enriched_text,
                        "metadata": {
                            "source": file_name,
                            "source_id": source_info.source_id,
                            "source_path": source_info.path,
                            "category": category,
                            "Header_1": h1,
                            "hash": self._get_file_hash(str(file_path)),
                            "entity_ids": []
                        }
                    })
            
            stats["chunks_count"] = len(final_chunks)
            stats["time"] = time.time() - file_start_time
            return final_chunks, stats
            
        except Exception as e:
            logger.error(f"❌ Critical error while processing file {file_name}: {e}")
            stats["time"] = time.time() - file_start_time
            return [], stats
        
    def run(self):
        data_path = Path(settings.DATA_DIR).resolve()
        supported = {'.pdf', '.docx', '.xlsx', '.txt', '.md'}
        
        files_to_process = [
            f for f in data_path.rglob("*") 
            if f.is_file() and f.suffix.lower() in supported
        ]

        total_files = len(files_to_process)
        if total_files == 0:
            logger.warning("⚠️ No files to process in the data directory")
            return
        
        logger.info(f"🚀 Starting indexing of {total_files} files...")
        print()
        
        all_chunks = []
        total_images = 0
        total_time = 0
        processed_times = []
        
        with tqdm(
            files_to_process, 
            desc="📄 Processing files",
            unit="file",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:
            for file_path in pbar:
                pbar.set_postfix_str(f"{file_path.name[:30]}...")
                
                chunks, stats = self.process_file(file_path)
                all_chunks.extend(chunks)
                
                total_images += stats["images_count"]
                total_time += stats["time"]
                processed_times.append(stats["time"])
                
                avg_time = sum(processed_times) / len(processed_times)
                remaining_files = total_files - len(processed_times)
                eta = avg_time * remaining_files
                
                pbar.set_postfix({
                    "file": f"{stats['time']:.1f}s",
                    "chunks": stats["chunks_count"],
                    "📸": stats["images_count"]
                })
        
        print()
        
        logger.info(f"📊 File processing finished in {format_time(total_time)}")
        logger.info(f"   • Files: {total_files}")
        logger.info(f"   • Chunks: {len(all_chunks)}")
        logger.info(f"   • Images: {total_images}")
        if processed_times:
            logger.info(f"   • Average time per file: {format_time(sum(processed_times)/len(processed_times))}")
        
        if all_chunks:
            logger.info(f"🔍 Extracting entities from {len(all_chunks)} chunks...")
            entity_start = time.time()
            all_chunks = self.entity_registry.process_chunks(all_chunks)
            entity_time = time.time() - entity_start
            logger.info(f"   ✓ Entity extraction finished in {format_time(entity_time)}")
            
            logger.info(f"📥 Uploading {len(all_chunks)} chunks into Qdrant...")
            index_start = time.time()
            self.engine.index_documents(all_chunks)
            index_time = time.time() - index_start
            logger.info(f"   ✓ Indexing finished in {format_time(index_time)}")
            
            entities = self.entity_registry.get_all_entities()
            abbrevs = self.entity_registry.get_abbreviations()
            
            total_pipeline_time = total_time + entity_time + index_time
            print()
            logger.info("=" * 50)
            logger.info("✅ INDEXING COMPLETED")
            logger.info("=" * 50)
            logger.info(f"   📁 Files processed: {total_files}")
            logger.info(f"   📝 Chunks created: {len(all_chunks)}")
            logger.info(f"   🔤 Entities extracted: {len(entities)} (abbreviations: {len(abbrevs)})")
            logger.info(f"   ⏱️  Total time: {format_time(total_pipeline_time)}")
            logger.info("=" * 50)


if __name__ == "__main__":
    ingestor = SmartIngestor()
    try:
        ingestor.run()
    finally:
        ingestor.engine.close()

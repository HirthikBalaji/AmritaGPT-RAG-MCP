"""
Production Ingestion Pipeline for AmritaGPT.
Coordinates universal document ingestion, idempotency, semantic chunking, and dual indexing.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

from amritagpt.config import DATA_DIR
from amritagpt.ingestion.parser import DocumentParser, ParsedDocument
from amritagpt.ingestion.chunker import SemanticChunker, EnrichedChunk
from amritagpt.indexing.metadata_store import MetadataStore
from amritagpt.indexing.vector_store import DenseVectorStore
from amritagpt.indexing.bm25_store import BM25LexicalStore


class IngestionPipeline:
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        meta_store: Optional[MetadataStore] = None,
        vector_store: Optional[DenseVectorStore] = None,
        bm25_store: Optional[BM25LexicalStore] = None
    ):
        self.data_dir = data_dir
        self.meta_store = meta_store or MetadataStore()
        self.vector_store = vector_store or DenseVectorStore()
        self.bm25_store = bm25_store or BM25LexicalStore()

    def discover_files(self) -> List[Path]:
        """Scan data directory for all ingestible files."""
        supported_exts = {".pdf", ".docx", ".xlsx", ".xls", ".pptx", ".txt", ".md", ".csv"}
        found_files = []
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in supported_exts and not f.startswith("~$"):
                    found_files.append(p)
        return found_files

    def _process_single_file(self, file_path: Path, force_reindex: bool):
        try:
            rel_path = str(file_path.relative_to(self.data_dir))
        except ValueError:
            rel_path = file_path.name

        checksum = DocumentParser.compute_checksum(file_path)
        existing = self.meta_store.get_document_by_path(rel_path)
        
        if not force_reindex and existing and existing.get("checksum") == checksum:
            return {"status": "skipped", "file": file_path.name}

        try:
            parsed = DocumentParser.parse_document(file_path, self.data_dir)
            if not parsed or not parsed.sections:
                return {"status": "skipped", "file": file_path.name}

            chunks = SemanticChunker.chunk_document(parsed)
            if not chunks:
                return {"status": "skipped", "file": file_path.name}

            return {
                "status": "success",
                "parsed": parsed,
                "chunks": chunks,
                "file": file_path.name
            }
        except Exception as e:
            return {"status": "error", "file": file_path.name, "error": str(e)}

    def run_ingestion(self, force_reindex: bool = False, max_files: Optional[int] = None) -> Dict[str, Any]:
        """Run continuous ingestion over the institutional data directory."""
        start_time = time.time()
        files = self.discover_files()
        if max_files:
            files = files[:max_files]

        print(f"AmritaGPT Ingestion Bus: Discovered {len(files)} institutional files.", flush=True)
        
        parsed_docs: List[ParsedDocument] = []
        all_new_chunks: List[EnrichedChunk] = []
        skipped_count = 0
        error_count = 0

        for idx, file_path in enumerate(files):
            res = self._process_single_file(file_path, force_reindex)
            status = res.get("status")
            if status == "success":
                parsed = res["parsed"]
                chunks = res["chunks"]
                self.meta_store.save_document(parsed, chunks)
                parsed_docs.append(parsed)
                all_new_chunks.extend(chunks)
            elif status == "skipped":
                skipped_count += 1
            elif status == "error":
                error_count += 1
                print(f"Error ingesting {res['file']}: {res.get('error')}", flush=True)

            if (idx + 1) % 25 == 0 or idx == len(files) - 1:
                print(f"Processed [{idx+1}/{len(files)}] files... ({len(all_new_chunks)} new chunks saved)", flush=True)

        # Rebuild or update indices if new chunks exist or indices missing
        indices_missing = not self.vector_store.index_path.exists() or not self.bm25_store.index_path.exists()
        rebuild_indices = force_reindex or len(all_new_chunks) > 0 or indices_missing
        
        if rebuild_indices:
            print(f"\nBuilding dense and sparse indices across institutional knowledge base...")
            with self.meta_store._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id, enriched_content FROM chunks ORDER BY chunk_id")
                all_db_chunks = cursor.fetchall()
            
            total_db_chunks = len(all_db_chunks)
            print(f"Total chunks in database to index: {total_db_chunks}")
            
            chunk_ids = [r[0] for r in all_db_chunks]
            chunk_texts = [r[1] for r in all_db_chunks]
            dirty_chunk_ids = {c.chunk_id for c in all_new_chunks}

            # 1. Build Dense Vector Index (with incremental embedding caching)
            self.vector_store.build_index(
                chunk_ids,
                chunk_texts,
                force_recompute=force_reindex,
                dirty_chunk_ids=dirty_chunk_ids
            )
            
            # 2. Build BM25 Sparse Index
            self.bm25_store.build_index(chunk_ids, chunk_texts)
            print("Dual indexing successfully completed!")
        else:
            print("All documents are up to date. Indices loaded from disk.")

        elapsed = time.time() - start_time
        return {
            "total_files_scanned": len(files),
            "new_or_updated_docs": len(parsed_docs),
            "skipped_unmodified": skipped_count,
            "errors": error_count,
            "total_chunks_indexed": len(self.vector_store.chunk_ids),
            "elapsed_seconds": round(elapsed, 2)
        }

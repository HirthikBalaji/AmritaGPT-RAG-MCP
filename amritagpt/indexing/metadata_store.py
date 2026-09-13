"""
Metadata and Document Catalog SQLite Store for AmritaGPT.
Maintains relational indexes, document lineage, ACLs, versions, and chunk catalog.
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from amritagpt.config import DB_PATH
from amritagpt.ingestion.parser import ParsedDocument
from amritagpt.ingestion.chunker import EnrichedChunk


class MetadataStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    relative_path TEXT NOT NULL UNIQUE,
                    file_type TEXT NOT NULL,
                    department TEXT NOT NULL,
                    category TEXT NOT NULL,
                    academic_year TEXT,
                    version TEXT NOT NULL,
                    access_policy TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    section_count INTEGER NOT NULL,
                    metadata_json TEXT,
                    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Chunks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    department TEXT NOT NULL,
                    category TEXT NOT NULL,
                    section_title TEXT NOT NULL,
                    page INTEGER,
                    academic_year TEXT,
                    version TEXT NOT NULL,
                    access_policy TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    enriched_content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    prev_chunk_id TEXT,
                    next_chunk_id TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
                )
            """)

            # Indexes for fast filtering
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_dept ON chunks (department)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks (category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_access ON chunks (access_policy)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_year ON chunks (academic_year)")

            # Query audit / Telemetry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    role TEXT NOT NULL,
                    detected_intent TEXT,
                    dense_hits INTEGER,
                    bm25_hits INTEGER,
                    fused_hits INTEGER,
                    final_top_k INTEGER,
                    grounding_score REAL,
                    latency_ms REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["metadata"] = json.loads(d["metadata_json"]) if d["metadata_json"] else {}
                return d
            return None

    def get_document_by_path(self, relative_path: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE relative_path = ?", (relative_path,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["metadata"] = json.loads(d["metadata_json"]) if d["metadata_json"] else {}
                return d
            return None

    def get_all_documents(self, department: Optional[str] = None, access_policy: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM documents WHERE 1=1"
            params = []
            if department:
                query += " AND department = ?"
                params.append(department)
            if access_policy:
                query += " AND access_policy = ?"
                params.append(access_policy)
            query += " ORDER BY filename"
            cursor.execute(query, params)
            rows = cursor.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["metadata"] = json.loads(item["metadata_json"]) if item["metadata_json"] else {}
                results.append(item)
            return results

    def save_document(self, doc: ParsedDocument, chunks: List[EnrichedChunk]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Delete old chunks if document exists
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.doc_id,))
            
            # Upsert document
            cursor.execute("""
                INSERT OR REPLACE INTO documents (
                    doc_id, filename, relative_path, file_type, department,
                    category, academic_year, version, access_policy, checksum,
                    section_count, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc.doc_id, doc.filename, doc.relative_path, doc.file_type, doc.department,
                doc.category, doc.academic_year, doc.version, doc.access_policy, doc.checksum,
                len(doc.sections), json.dumps(doc.metadata), datetime.utcnow().isoformat()
            ))

            # Insert chunks in batch
            chunk_records = [
                (
                    c.chunk_id, c.doc_id, c.filename, c.relative_path, c.department,
                    c.category, c.section_title, c.page, c.academic_year, c.version,
                    c.access_policy, c.raw_content, c.enriched_content, c.chunk_index,
                    c.prev_chunk_id, c.next_chunk_id, json.dumps(c.metadata)
                )
                for c in chunks
            ]
            cursor.executemany("""
                INSERT OR REPLACE INTO chunks (
                    chunk_id, doc_id, filename, relative_path, department,
                    category, section_title, page, academic_year, version,
                    access_policy, raw_content, enriched_content, chunk_index,
                    prev_chunk_id, next_chunk_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, chunk_records)
            conn.commit()

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["metadata"] = json.loads(d["metadata_json"]) if d["metadata_json"] else {}
                return d
            return None

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        if not chunk_ids:
            return []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ",".join(["?"] * len(chunk_ids))
            cursor.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
            rows = cursor.fetchall()
            id_map = {r["chunk_id"]: dict(r) for r in rows}
            results = []
            for cid in chunk_ids:
                if cid in id_map:
                    item = id_map[cid]
                    item["metadata"] = json.loads(item["metadata_json"]) if item["metadata_json"] else {}
                    results.append(item)
            return results

    def get_document_sections(self, doc_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id, section_title, page, raw_content FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,))
            return [dict(r) for r in cursor.fetchall()]

    def record_telemetry(self, query: str, role: str, intent: str, dense_hits: int, bm25_hits: int, fused_hits: int, final_k: int, grounding_score: float, latency_ms: float):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_telemetry (
                    query, role, detected_intent, dense_hits, bm25_hits,
                    fused_hits, final_top_k, grounding_score, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (query, role, intent, dense_hits, bm25_hits, fused_hits, final_k, grounding_score, latency_ms))
            conn.commit()

    def get_system_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents")
            doc_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            cursor.execute("SELECT department, COUNT(*) FROM documents GROUP BY department ORDER BY COUNT(*) DESC")
            depts = {r[0]: r[1] for r in cursor.fetchall()}
            cursor.execute("SELECT access_policy, COUNT(*) FROM documents GROUP BY access_policy")
            access_dist = {r[0]: r[1] for r in cursor.fetchall()}
            cursor.execute("SELECT COUNT(*), AVG(latency_ms), AVG(grounding_score) FROM query_telemetry")
            telemetry = cursor.fetchone()
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "departments": depts,
                "access_distribution": access_dist,
                "total_queries_served": telemetry[0] or 0,
                "avg_query_latency_ms": round(telemetry[1] or 0.0, 2),
                "avg_grounding_score": round(telemetry[2] or 0.0, 2)
            }

    def get_recent_telemetry(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM query_telemetry ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]


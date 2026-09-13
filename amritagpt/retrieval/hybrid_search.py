"""
Hybrid Retrieval Engine for AmritaGPT.
Combines Dense Vector search + BM25 Lexical search via Reciprocal Rank Fusion (RRF)
with role-based access control, precision reranking, and context assembly.
"""
from typing import List, Dict, Any, Optional, Set
import time

from amritagpt.config import DENSE_TOP_K, SPARSE_TOP_K, RRF_K, RERANK_TOP_K
from amritagpt.indexing.metadata_store import MetadataStore
from amritagpt.indexing.vector_store import DenseVectorStore
from amritagpt.indexing.bm25_store import BM25LexicalStore
from amritagpt.retrieval.query_analyzer import QueryAnalyzer, AnalyzedQuery
from amritagpt.retrieval.reranker import PrecisionReranker
from amritagpt.retrieval.context_assembler import ContextAssembler
from amritagpt.security.access_control import AccessControlManager


class HybridSearchEngine:
    def __init__(
        self,
        meta_store: Optional[MetadataStore] = None,
        vector_store: Optional[DenseVectorStore] = None,
        bm25_store: Optional[BM25LexicalStore] = None
    ):
        self.meta_store = meta_store or MetadataStore()
        self.vector_store = vector_store or DenseVectorStore()
        self.bm25_store = bm25_store or BM25LexicalStore()
        self.context_assembler = ContextAssembler(self.meta_store)

    def retrieve(
        self,
        query: str,
        role: str = "student",
        department_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        academic_year_filter: Optional[str] = None,
        top_k: int = RERANK_TOP_K
    ) -> Dict[str, Any]:
        """Perform multi-stage hybrid retrieval: Query Understanding -> Dense + BM25 -> RRF -> Reranking -> Assembly."""
        start_time = time.time()

        # Step 1: Query Understanding
        analyzed_q: AnalyzedQuery = QueryAnalyzer.analyze(query)
        dept = department_filter or analyzed_q.department
        year = academic_year_filter or analyzed_q.academic_year

        # Step 2: Access Control Security Filter (Pre-retrieval)
        allowed_chunk_ids = AccessControlManager.filter_chunk_ids_by_role(self.meta_store, role)

        # Department / Year filtering if specified
        if dept:
            with self.meta_store._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT chunk_id FROM chunks WHERE department LIKE ?", (f"%{dept}%",))
                dept_chunks = {r[0] for r in cursor.fetchall()}
                if dept_chunks:
                    allowed_chunk_ids = allowed_chunk_ids.intersection(dept_chunks)

        # Step 3: Parallel Dense + Sparse Retrieval
        expanded_query = query
        if analyzed_q.expanded_terms:
            expanded_query = query + " " + " ".join(analyzed_q.expanded_terms)

        dense_hits = self.vector_store.search(
            query=expanded_query,
            top_k=DENSE_TOP_K,
            allowed_chunk_ids=allowed_chunk_ids
        )

        bm25_hits = self.bm25_store.search(
            query=query + (f" {analyzed_q.regulation_code}" if analyzed_q.regulation_code else ""),
            top_k=SPARSE_TOP_K,
            allowed_chunk_ids=allowed_chunk_ids
        )

        # Step 4: Reciprocal Rank Fusion (RRF)
        # RRF(d) = sum( 1 / (k + rank) )
        rrf_scores: Dict[str, float] = {}
        dense_rank_map = {}
        for rank, (cid, score) in enumerate(dense_hits):
            dense_rank_map[cid] = rank + 1
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_K + rank + 1))

        bm25_rank_map = {}
        for rank, (cid, score) in enumerate(bm25_hits):
            bm25_rank_map[cid] = rank + 1
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (RRF_K + rank + 1))

        # Sort candidate chunk IDs by RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:30]

        # Fetch chunk details from metadata store
        chunks = self.meta_store.get_chunks_by_ids(sorted_cids)
        chunk_map = {c["chunk_id"]: c for c in chunks}

        candidate_items = []
        for cid in sorted_cids:
            if cid in chunk_map:
                candidate_items.append({
                    "chunk": chunk_map[cid],
                    "rrf_score": rrf_scores[cid],
                    "dense_rank": dense_rank_map.get(cid),
                    "bm25_rank": bm25_rank_map.get(cid)
                })

        # Step 5: Precision Cross-Feature Reranking
        reranked_items = PrecisionReranker.rerank(
            query=query,
            candidates=candidate_items,
            top_k=top_k,
            preferred_year=year
        )

        # Step 6: Evidence Packaging & Context Assembly
        evidence_package = self.context_assembler.assemble(
            reranked_items=reranked_items,
            expand_neighbors=True
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "role": role,
            "analyzed_query": {
                "intent": analyzed_q.intent,
                "department": analyzed_q.department,
                "program": analyzed_q.program,
                "regulation_code": analyzed_q.regulation_code,
                "academic_year": analyzed_q.academic_year,
                "is_current": analyzed_q.is_current_requested
            },
            "retrieval_stats": {
                "dense_candidates": len(dense_hits),
                "bm25_candidates": len(bm25_hits),
                "fused_candidates": len(candidate_items),
                "final_selected": len(reranked_items),
                "latency_ms": latency_ms
            },
            "evidence_package": evidence_package,
            "top_chunks": [item["chunk"] for item in reranked_items]
        }

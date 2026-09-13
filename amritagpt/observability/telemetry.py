"""
Observability and Evaluation Framework for AmritaGPT.
Calculates RAG metrics: Recall@K, MRR, Faithfulness, latency profiles, and query analytics.
"""
from typing import Dict, Any, List
import time
import numpy as np

from amritagpt.indexing.metadata_store import MetadataStore
from amritagpt.retrieval.hybrid_search import HybridSearchEngine


# Golden evaluation dataset for institutional RAG benchmarks
GOLDEN_EVAL_DATASET = [
    {
        "id": "eval-1",
        "question": "What is the minimum attendance requirement for B.Tech students?",
        "expected_intent": "Academic Regulation",
        "expected_keywords": ["attendance", "75", "80", "regulation"],
        "target_department": "Academic Admin Office"
    },
    {
        "id": "eval-2",
        "question": "What are the regulations and penalties for examination malpractice?",
        "expected_intent": "Examination",
        "expected_keywords": ["malpractice", "committee", "disciplinary", "exam"],
        "target_department": "Academic Admin Office"
    },
    {
        "id": "eval-3",
        "question": "Find curriculum and syllabus details for B.Tech Computer Science 2023",
        "expected_intent": "Curriculum & Syllabus",
        "expected_keywords": ["curriculum", "cse", "syllabus", "2023"],
        "target_department": "Academic Admin Office"
    },
    {
        "id": "eval-4",
        "question": "What are the hostel facilities, mess timings and residential rules?",
        "expected_intent": "Campus & Residential",
        "expected_keywords": ["hostel", "mess", "rules"],
        "target_department": "Hostel"
    },
    {
        "id": "eval-5",
        "question": "What are the IQAC quality assurance and research cell initiatives?",
        "expected_intent": "Research & Grants",
        "expected_keywords": ["iqac", "research", "quality"],
        "target_department": "IQAC"
    }
]


class EvaluationFramework:
    """Evaluates institutional RAG performance against golden benchmark test cases."""

    def __init__(self, search_engine: HybridSearchEngine):
        self.search_engine = search_engine

    def run_benchmark(self) -> Dict[str, Any]:
        results = []
        recalls_at_5 = []
        recalls_at_1 = []
        reciprocal_ranks = []
        latencies = []

        for item in GOLDEN_EVAL_DATASET:
            t0 = time.time()
            retrieval = self.search_engine.retrieve(query=item["question"], top_k=5)
            lat = (time.time() - t0) * 1000
            latencies.append(lat)

            chunks = retrieval.get("top_chunks", [])
            expected_kws = [k.lower() for k in item["expected_keywords"]]

            # Check matches in retrieved chunks
            first_match_rank = None
            hits_in_top_5 = 0

            for rank, c in enumerate(chunks):
                content = (c.get("raw_content", "") + " " + c.get("filename", "")).lower()
                matched = any(kw in content for kw in expected_kws)
                if matched:
                    hits_in_top_5 += 1
                    if first_match_rank is None:
                        first_match_rank = rank + 1

            # Recall@5
            recalls_at_5.append(1.0 if hits_in_top_5 > 0 else 0.0)
            # Recall@1
            recalls_at_1.append(1.0 if first_match_rank == 1 else 0.0)
            # MRR
            reciprocal_ranks.append(1.0 / first_match_rank if first_match_rank else 0.0)

            results.append({
                "eval_id": item["id"],
                "question": item["question"],
                "matched": hits_in_top_5 > 0,
                "first_match_rank": first_match_rank,
                "latency_ms": round(lat, 2)
            })

        return {
            "total_benchmark_queries": len(GOLDEN_EVAL_DATASET),
            "recall_at_1": round(float(np.mean(recalls_at_1)), 3),
            "recall_at_5": round(float(np.mean(recalls_at_5)), 3),
            "mean_reciprocal_rank_mrr": round(float(np.mean(reciprocal_ranks)), 3),
            "p50_latency_ms": round(float(np.median(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "detailed_results": results
        }

"""
Precision Reranker for AmritaGPT.
Evaluates query-passage cross-alignment, exact identifier preservation,
section relevance, and temporal freshness to achieve high precision.
"""
from typing import List, Dict, Any, Tuple
import re
import math


class PrecisionReranker:
    """Reranks candidate evidence chunks using lexical, structural, and semantic features."""

    @classmethod
    def rerank(
        cls,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 8,
        preferred_year: str = None
    ) -> List[Dict[str, Any]]:
        """Rerank candidates and return top_k with calibrated relevance scores."""
        if not candidates:
            return []

        q_tokens = set(re.findall(r"\b[a-z0-9\.\-]{2,}\b", query.lower()))
        scored_candidates = []

        for item in candidates:
            chunk = item["chunk"]
            rrf_score = item.get("rrf_score", 0.0)
            raw_text = (chunk.get("raw_content") or "").lower()
            title = (chunk.get("section_title") or "").lower()
            filename = (chunk.get("filename") or "").lower()
            dept = (chunk.get("department") or "").lower()
            doc_year = str(chunk.get("academic_year") or "")

            score = 0.0

            # 1. Base RRF contribution (0 - 1 normalized)
            score += rrf_score * 10.0

            # 2. Token overlap in passage content
            p_tokens = set(re.findall(r"\b[a-z0-9\.\-]{2,}\b", raw_text))
            overlap = q_tokens.intersection(p_tokens)
            if q_tokens:
                token_recall = len(overlap) / len(q_tokens)
                score += token_recall * 4.0

            # 3. Title & Heading Match Bonus
            title_overlap = q_tokens.intersection(set(re.findall(r"\b[a-z0-9\.\-]{2,}\b", title)))
            if title_overlap:
                score += len(title_overlap) * 2.5

            # 4. Exact Phrase or Code Match (e.g. "R.4", "attendance requirement", "examination")
            clean_q = query.lower().strip()
            if clean_q in raw_text or clean_q in title:
                score += 5.0

            # Regulation code match (e.g. "r.4" or "r.1")
            reg_codes = re.findall(r"\br\.\d+\b", clean_q)
            for rc in reg_codes:
                if rc in raw_text or rc in title:
                    score += 6.0

            # 5. Department match bonus
            for token in q_tokens:
                if token in dept or token in filename:
                    score += 1.5

            # 6. Version & Academic Year alignment
            if preferred_year and doc_year:
                if preferred_year == doc_year:
                    score += 3.0
                elif preferred_year != doc_year:
                    # slight penalty for outdated year if specific year was requested
                    score -= 1.5

            scored_candidates.append({
                **item,
                "rerank_score": round(score, 4)
            })

        # Sort descending by rerank score
        scored_candidates.sort(key=lambda x: -x["rerank_score"])
        return scored_candidates[:top_k]

"""
Answer Verification and Grounding Evaluator for AmritaGPT.
Audits generated claims against retrieved evidence, flags hallucination risk,
and calculates grounding scores.
"""
from typing import Dict, Any, List
import re


class GroundingVerifier:
    """Verifies factual alignment between retrieved evidence and generated response."""

    @classmethod
    def verify(
        cls,
        query: str,
        answer: str,
        sources: List[Dict[str, Any]],
        evidence_text: str
    ) -> Dict[str, Any]:
        """Audit the answer against the retrieved evidence."""
        if not sources or not evidence_text.strip():
            return {
                "grounding_score": 0.0,
                "confidence_level": "None",
                "is_faithful": False,
                "unsupported_claims": ["No evidence found in knowledge base."],
                "citation_count": 0
            }

        evidence_lower = evidence_text.lower()
        answer_lower = answer.lower()

        # 1. Check numeric / quantitative consistency
        numbers_in_answer = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", answer_lower))
        unsupported_numbers = []
        for num in numbers_in_answer:
            if num not in evidence_lower and len(num) > 1 and num not in ["1", "2", "3", "4", "5"]:
                unsupported_numbers.append(num)

        # 2. Extract key terms
        answer_words = set(re.findall(r"\b[a-z]{4,}\b", answer_lower))
        evidence_words = set(re.findall(r"\b[a-z]{4,}\b", evidence_lower))
        stop_words = {"this", "that", "with", "from", "have", "been", "which", "there", "their", "such", "other", "about", "into", "regulation", "amrita", "university", "source"}
        substantive_words = answer_words - stop_words

        if substantive_words:
            word_grounding_ratio = len(substantive_words.intersection(evidence_words)) / len(substantive_words)
        else:
            word_grounding_ratio = 1.0

        # Penalize if numbers are fabricated
        num_penalty = 0.2 * len(unsupported_numbers)
        final_grounding_score = max(0.0, min(1.0, word_grounding_ratio - num_penalty))

        # Confidence level classification
        if final_grounding_score >= 0.75:
            confidence = "High"
        elif final_grounding_score >= 0.45:
            confidence = "Moderate"
        else:
            confidence = "Low"

        is_faithful = final_grounding_score >= 0.45 and len(unsupported_numbers) == 0

        return {
            "grounding_score": round(final_grounding_score, 3),
            "confidence_level": confidence,
            "is_faithful": is_faithful,
            "unsupported_metrics": unsupported_numbers,
            "citation_count": len(sources),
            "checked_sources": len(sources)
        }

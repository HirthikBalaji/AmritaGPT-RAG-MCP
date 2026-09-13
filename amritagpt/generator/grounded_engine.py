"""
Grounded Answer Generator for AmritaGPT.
Produces verified, evidence-grounded answers with provenance citations
and strict adherence to institutional safety policies.
"""
from typing import Dict, Any, Optional, List
import re

from amritagpt.retrieval.hybrid_search import HybridSearchEngine
from amritagpt.generator.citation_engine import CitationEngine
from amritagpt.generator.verifier import GroundingVerifier
from amritagpt.security.access_control import PromptInjectionGuard


class GroundedAnswerEngine:
    """Institutional Grounded Response Generator."""

    def __init__(self, search_engine: Optional[HybridSearchEngine] = None):
        self.search_engine = search_engine or HybridSearchEngine()

    def generate_answer(
        self,
        question: str,
        role: str = "student",
        department: Optional[str] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Generate a verified, grounded answer backed by institutional evidence."""
        # Step 1: Hybrid Retrieval
        retrieval_res = self.search_engine.retrieve(
            query=question,
            role=role,
            department_filter=department,
            top_k=top_k
        )

        evidence_pkg = retrieval_res["evidence_package"]
        sources = evidence_pkg.get("sources", [])
        evidence_text = evidence_pkg.get("evidence_text", "")

        # Step 2: Evidence Sufficiency Check (Slide 20 & 36)
        if not sources or not evidence_text.strip() or len(sources) == 0:
            return {
                "question": question,
                "answer": "I couldn't find sufficient evidence in the institutional knowledge base to answer this query.",
                "confidence": "None",
                "grounding_score": 0.0,
                "citations": [],
                "sources": [],
                "is_sufficient": False,
                "intent": retrieval_res["analyzed_query"]["intent"],
                "role": role
            }

        # Step 3: Extract Verified Institutional Claims
        sanitized_evidence = PromptInjectionGuard.sanitize_passage(evidence_text)
        key_claims = CitationEngine.extract_key_claims(sanitized_evidence, question)

        # Step 4: Synthesize Grounded Narrative
        top_source = sources[0]
        top_dept = top_source.get("department", "Amrita University")
        top_doc = top_source.get("filename", "University Document")
        top_sec = top_source.get("section", "General")
        top_page = top_source.get("page")
        page_ref = f", Page {top_page}" if top_page else ""

        answer_lines = []
        if key_claims:
            answer_lines.append(f"Based on institutional records from **{top_doc}** ({top_dept} — {top_sec}{page_ref}):\n")
            for claim in key_claims[:4]:
                answer_lines.append(f"• {claim.strip()}")
        else:
            # Fallback to top raw evidence segment if no specific sub-claims matched
            top_content = retrieval_res["top_chunks"][0]["raw_content"].strip()
            # Take first 500 characters
            snippet = top_content[:500].rsplit(".", 1)[0] + "."
            answer_lines.append(f"According to **{top_doc}** ({top_sec}):\n\n{snippet}")

        # Add primary inline citation
        primary_citation = f"**Primary Reference:** {top_doc} (Section: {top_sec}{page_ref})"
        answer_lines.append(f"\n{primary_citation}")

        final_answer = "\n".join(answer_lines)

        # Step 5: Answer Verification & Grounding Audit (Slide 22)
        verification = GroundingVerifier.verify(
            query=question,
            answer=final_answer,
            sources=sources,
            evidence_text=sanitized_evidence
        )

        # Format citations list
        formatted_citations = [CitationEngine.format_citation(s) for s in sources[:4]]

        # Record telemetry
        self.search_engine.meta_store.record_telemetry(
            query=question,
            role=role,
            intent=retrieval_res["analyzed_query"]["intent"],
            dense_hits=retrieval_res["retrieval_stats"]["dense_candidates"],
            bm25_hits=retrieval_res["retrieval_stats"]["bm25_candidates"],
            fused_hits=retrieval_res["retrieval_stats"]["fused_candidates"],
            final_k=len(sources),
            grounding_score=verification["grounding_score"],
            latency_ms=retrieval_res["retrieval_stats"]["latency_ms"]
        )

        return {
            "question": question,
            "answer": final_answer,
            "confidence": verification["confidence_level"],
            "grounding_score": verification["grounding_score"],
            "is_faithful": verification["is_faithful"],
            "intent": retrieval_res["analyzed_query"]["intent"],
            "citations": formatted_citations,
            "sources": sources[:5],
            "is_sufficient": True,
            "role": role,
            "latency_ms": retrieval_res["retrieval_stats"]["latency_ms"]
        }

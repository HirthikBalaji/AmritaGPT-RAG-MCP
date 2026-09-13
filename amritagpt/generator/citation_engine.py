"""
Citation and Provenance Engine for AmritaGPT.
Links factual statements directly to source document coordinates (Document, Section, Page, Year).
"""
from typing import List, Dict, Any, Tuple
import re


class CitationEngine:
    """Manages citation provenance and attribution formatting."""

    @classmethod
    def format_citation(cls, source: Dict[str, Any]) -> str:
        """Create a clean institutional citation string."""
        doc = source.get("filename", "Institutional Record")
        sec = source.get("section", "General")
        page = source.get("page")
        year = source.get("academic_year")
        dept = source.get("department", "Amrita University")

        parts = [f"**Source [{source.get('source_num', 1)}]:** {doc}"]
        if sec:
            parts.append(f"Section: {sec}")
        if page:
            parts.append(f"Page {page}")
        if year:
            parts.append(f"AY: {year}")
        if dept:
            parts.append(f"({dept})")

        return " | ".join(parts)

    @classmethod
    def extract_key_claims(cls, evidence_text: str, query: str) -> List[Dict[str, Any]]:
        """Extract high-relevance sentences/clauses containing answers to the query."""
        if not evidence_text:
            return []

        q_terms = set(re.findall(r"\b[a-z0-9\.\%]{3,}\b", query.lower()))
        sentences = re.split(r"(?<=[.!?\n])\s+", evidence_text)
        scored_sentences = []

        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 25 or s_clean.startswith("--- Evidence Source") or s_clean.startswith("Document:") or s_clean.startswith("Institution:"):
                continue

            s_terms = set(re.findall(r"\b[a-z0-9\.\%]{3,}\b", s_clean.lower()))
            overlap = q_terms.intersection(s_terms)
            if overlap:
                # Bonus if sentence contains quantitative facts (%, numbers, dates, codes)
                has_metric = bool(re.search(r"(\b\d+\%|\b\d+\s+credits|\bgrade\s+[a-z]|\br\.\d+)\b", s_clean.lower()))
                score = len(overlap) + (2.0 if has_metric else 0.0)
                scored_sentences.append((score, s_clean))

        scored_sentences.sort(key=lambda x: -x[0])
        return [s[1] for s in scored_sentences[:6]]

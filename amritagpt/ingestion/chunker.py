"""
Structure-aware semantic chunker with contextual enrichment for AmritaGPT.
Splits documents along semantic boundaries while attaching rich contextual prefixes.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import re

from amritagpt.ingestion.parser import ParsedDocument, DocumentSection
from amritagpt.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, MAX_CHUNK_CHAR_LEN


@dataclass
class EnrichedChunk:
    chunk_id: str
    doc_id: str
    filename: str
    relative_path: str
    department: str
    category: str
    section_title: str
    page: Optional[int]
    academic_year: Optional[str]
    version: str
    access_policy: str  # public, student, faculty, admin
    raw_content: str
    enriched_content: str
    chunk_index: int
    prev_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticChunker:
    """Structure-aware chunker that enriches chunks with institutional hierarchy and metadata."""

    @staticmethod
    def extract_keywords(text: str) -> List[str]:
        """Extract dominant institutional keywords for lexical boosting."""
        candidates = re.findall(r"\b[A-Za-z0-9\-\.]{3,20}\b", text)
        stop_words = {
            "the", "and", "for", "with", "this", "that", "from", "shall", "will", "are",
            "have", "been", "which", "there", "their", "such", "other", "about", "into"
        }
        filtered = [w.lower() for w in candidates if w.lower() not in stop_words and not w.isdigit()]
        # Get frequency
        counts = {}
        for w in filtered:
            counts[w] = counts.get(w, 0) + 1
        sorted_kws = sorted(counts.items(), key=lambda x: -x[1])
        return [k for k, v in sorted_kws[:6]]

    @classmethod
    def split_text_semantically(cls, text: str, max_chars: int = MAX_CHUNK_CHAR_LEN) -> List[str]:
        """Split text along paragraphs, bullet points, or sentence boundaries."""
        text = text.strip()
        if len(text) <= max_chars:
            return [text]

        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for p in paragraphs:
            p_clean = p.strip()
            if not p_clean:
                continue

            # If a single paragraph is too large, split by sentences
            if len(p_clean) > max_chars:
                sentences = re.split(r"(?<=[.?!])\s+", p_clean)
                for s in sentences:
                    s_clean = s.strip()
                    if not s_clean:
                        continue
                    if current_len + len(s_clean) > max_chars and current_chunk:
                        chunks.append("\n".join(current_chunk).strip())
                        current_chunk = [s_clean]
                        current_len = len(s_clean)
                    else:
                        current_chunk.append(s_clean)
                        current_len += len(s_clean) + 1
            else:
                if current_len + len(p_clean) > max_chars and current_chunk:
                    chunks.append("\n".join(current_chunk).strip())
                    current_chunk = [p_clean]
                    current_len = len(p_clean)
                else:
                    current_chunk.append(p_clean)
                    current_len += len(p_clean) + 2

        if current_chunk:
            chunks.append("\n".join(current_chunk).strip())

        return chunks

    @classmethod
    def chunk_document(cls, doc: ParsedDocument) -> List[EnrichedChunk]:
        """Convert a ParsedDocument into a sequence of contextually enriched chunks."""
        chunks: List[EnrichedChunk] = []
        chunk_counter = 0

        for sec in doc.sections:
            sec_text = sec.content.strip()
            if not sec_text:
                continue

            # Split large sections semantically
            text_blocks = cls.split_text_semantically(sec_text)
            for b_idx, block in enumerate(text_blocks):
                chunk_counter += 1
                chunk_id = f"{doc.doc_id}_c{chunk_counter}"
                
                # Contextual Enrichment Header
                # E.g. [Amrita Vishwa Vidyapeetham] | Dept: Academic Admin Office | Doc: btech-regulations-2023 | Section: R.4 Attendance
                page_info = f" | Page: {sec.page}" if sec.page else ""
                year_info = f" | AY: {doc.academic_year}" if doc.academic_year else ""
                keywords = cls.extract_keywords(block)
                kw_str = f" | Keywords: {', '.join(keywords)}" if keywords else ""

                header = (
                    f"Institution: Amrita Vishwa Vidyapeetham\n"
                    f"Department: {doc.department} | Category: {doc.category}{year_info}\n"
                    f"Document: {doc.filename} | Section: {sec.title}{page_info}\n"
                    f"Access Policy: {doc.access_policy}\n"
                )

                enriched_text = f"{header}\nContent:\n{block}"

                enriched_chunk = EnrichedChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    relative_path=doc.relative_path,
                    department=doc.department,
                    category=doc.category,
                    section_title=sec.title,
                    page=sec.page,
                    academic_year=doc.academic_year,
                    version=doc.version,
                    access_policy=doc.access_policy,
                    raw_content=block,
                    enriched_content=enriched_text,
                    chunk_index=chunk_counter,
                    metadata={
                        **doc.metadata,
                        "section_id": sec.section_id,
                        "section_type": sec.section_type,
                        "keywords": keywords,
                    }
                )
                chunks.append(enriched_chunk)

        # Set linked list pointers for neighbor expansion
        for i in range(len(chunks)):
            if i > 0:
                chunks[i].prev_chunk_id = chunks[i - 1].chunk_id
            if i < len(chunks) - 1:
                chunks[i].next_chunk_id = chunks[i + 1].chunk_id

        return chunks

"""
Context Assembler and Evidence Packager for AmritaGPT.
Deduplicates, expands neighboring chunks, respects token budgets, and formats structured evidence.
"""
from typing import List, Dict, Any, Optional
from amritagpt.indexing.metadata_store import MetadataStore


class ContextAssembler:
    """Assembles reranked chunks into a high-signal grounded evidence package."""

    def __init__(self, meta_store: MetadataStore):
        self.meta_store = meta_store

    def assemble(
        self,
        reranked_items: List[Dict[str, Any]],
        expand_neighbors: bool = True,
        max_total_chars: int = 6000
    ) -> Dict[str, Any]:
        """Assemble structured evidence package from top reranked chunks."""
        if not reranked_items:
            return {
                "evidence_text": "",
                "sources": [],
                "chunk_count": 0,
                "is_sufficient": False
            }

        seen_chunk_ids = set()
        evidence_blocks = []
        sources = []
        current_len = 0

        for item in reranked_items:
            chunk = item["chunk"]
            cid = chunk["chunk_id"]
            if cid in seen_chunk_ids:
                continue
            seen_chunk_ids.add(cid)

            # Neighbor expansion: fetch surrounding paragraph if context is short
            content = chunk["raw_content"]
            if expand_neighbors and len(content) < 300 and chunk.get("next_chunk_id"):
                next_chunk = self.meta_store.get_chunk_by_id(chunk["next_chunk_id"])
                if next_chunk and next_chunk["chunk_id"] not in seen_chunk_ids:
                    content += "\n" + next_chunk["raw_content"]
                    seen_chunk_ids.add(next_chunk["chunk_id"])

            page_str = f" | Page: {chunk['page']}" if chunk.get("page") else ""
            year_val = chunk.get("academic_year") or chunk.get("version")
            block_text = (
                f"--- Evidence Source [{len(sources) + 1}] ---\n"
                f"Document: {chunk['filename']}\n"
                f"Department: {chunk['department']} | Category: {chunk['category']}\n"
                f"Section: {chunk['section_title']}{page_str}\n"
                f"Academic Year / Version: {year_val}\n"
                f"Content:\n{content.strip()}\n"
            )

            if current_len + len(block_text) > max_total_chars and evidence_blocks:
                break

            evidence_blocks.append(block_text)
            current_len += len(block_text)

            sources.append({
                "source_num": len(sources) + 1,
                "doc_id": chunk["doc_id"],
                "chunk_id": cid,
                "filename": chunk["filename"],
                "relative_path": chunk["relative_path"],
                "department": chunk["department"],
                "section": chunk["section_title"],
                "page": chunk.get("page"),
                "academic_year": chunk.get("academic_year"),
                "access_policy": chunk.get("access_policy"),
                "rerank_score": item.get("rerank_score", 0.0)
            })

        evidence_str = "\n".join(evidence_blocks)
        is_sufficient = len(sources) > 0

        return {
            "evidence_text": evidence_str,
            "sources": sources,
            "chunk_count": len(sources),
            "is_sufficient": is_sufficient
        }

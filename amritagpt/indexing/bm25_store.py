"""
Sparse BM25 Lexical Index for AmritaGPT using rank_bm25.
Enables high-precision keyword, code, regulation identifier, and exact-term retrieval.
"""
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional, Set
from rank_bm25 import BM25Okapi

from amritagpt.config import BM25_INDEX_PATH


class BM25LexicalStore:
    def __init__(self, index_path: Path = BM25_INDEX_PATH):
        self.index_path = index_path
        self.chunk_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize string into normalized terms, alphanumeric tokens, and codes."""
        # Preserve codes with dots and hyphens like "R.14", "23CSE101", "AMR-REG"
        tokens = re.findall(r"\b[A-Za-z0-9\.\_\-]{2,30}\b", text.lower())
        return tokens

    def build_index(self, chunk_ids: List[str], texts: List[str]):
        """Build BM25 index from chunk texts and persist."""
        if not chunk_ids:
            return
        
        self.chunk_ids = chunk_ids
        tokenized_corpus = [self.tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.save()

    def save(self):
        """Save BM25 index and chunk ID mapping to disk."""
        if self.bm25 is not None and self.chunk_ids:
            with open(self.index_path, "wb") as f:
                pickle.dump({"chunk_ids": self.chunk_ids, "bm25": self.bm25}, f)

    def load(self) -> bool:
        """Load BM25 index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunk_ids = data["chunk_ids"]
                    self.bm25 = data["bm25"]
                return True
            except Exception as e:
                print(f"Failed to load BM25 index: {e}")
                return False
        return False

    def search(
        self,
        query: str,
        top_k: int = 30,
        allowed_chunk_ids: Optional[Set[str]] = None
    ) -> List[Tuple[str, float]]:
        """Perform BM25 search returning (chunk_id, bm25_score)."""
        if self.bm25 is None or len(self.chunk_ids) == 0:
            if not self.load():
                return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Collect and filter scores
        scored_pairs = []
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            cid = self.chunk_ids[idx]
            if allowed_chunk_ids is not None and cid not in allowed_chunk_ids:
                continue
            scored_pairs.append((cid, float(score)))

        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda x: -x[1])
        return scored_pairs[:top_k]

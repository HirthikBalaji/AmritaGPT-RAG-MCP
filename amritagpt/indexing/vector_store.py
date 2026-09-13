"""
Dense Vector Index for AmritaGPT using FastEmbed and NumPy Vectorized Cosine Similarity.
Provides dense semantic search over contextually enriched chunks.
"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set
import json
import numpy as np
from fastembed import TextEmbedding

from amritagpt.config import VECTOR_INDEX_PATH, VECTOR_CHUNKS_PATH, EMBEDDING_MODEL_NAME, EMBEDDING_DIM


class DenseVectorStore:
    def __init__(self, index_path: Path = VECTOR_INDEX_PATH, chunks_path: Path = VECTOR_CHUNKS_PATH):
        self.index_path = index_path
        self.chunks_path = chunks_path
        self._embedding_model: Optional[TextEmbedding] = None
        self.chunk_ids: List[str] = []
        self.embeddings: Optional[np.ndarray] = None  # Normalized matrix shape: (N, dim)

    @property
    def model(self) -> TextEmbedding:
        if self._embedding_model is None:
            self._embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        return self._embedding_model

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Generate dense normalized embeddings for a list of texts."""
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        
        all_vecs = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            generator = self.model.embed(batch)
            vecs = np.array(list(generator), dtype=np.float32)
            all_vecs.append(vecs)
            batch_num = (i // batch_size) + 1
            if total_batches > 1 and (batch_num % 5 == 0 or batch_num == total_batches):
                print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...", flush=True)
            
        matrix = np.vstack(all_vecs)
        # L2 normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query into a normalized vector."""
        vec = list(self.model.embed([query]))[0]
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def build_index(
        self,
        chunk_ids: List[str],
        texts: List[str],
        force_recompute: bool = False,
        dirty_chunk_ids: Optional[Set[str]] = None
    ):
        """Build and persist the vector index incrementally, caching already-embedded chunks."""
        if not chunk_ids:
            return

        # Load existing vectors from disk if not in memory
        if self.embeddings is None or not self.chunk_ids:
            self.load()

        existing_emb_map: Dict[str, np.ndarray] = {}
        if not force_recompute and self.embeddings is not None and self.chunk_ids:
            num_existing = min(len(self.chunk_ids), len(self.embeddings))
            for i in range(num_existing):
                cid = self.chunk_ids[i]
                if dirty_chunk_ids is not None and cid in dirty_chunk_ids:
                    continue  # Explicitly modified/new chunk, re-embed
                existing_emb_map[cid] = self.embeddings[i]

        dim = self.embeddings.shape[1] if (self.embeddings is not None and len(self.embeddings) > 0) else EMBEDDING_DIM

        # Identify which chunks actually need embedding
        missing_indices = []
        texts_to_embed = []
        for idx, (cid, text) in enumerate(zip(chunk_ids, texts)):
            if cid not in existing_emb_map:
                missing_indices.append(idx)
                texts_to_embed.append(text)

        final_embeddings = np.zeros((len(chunk_ids), dim), dtype=np.float32)

        # Populate from existing cache
        reused_count = 0
        for idx, cid in enumerate(chunk_ids):
            if cid in existing_emb_map:
                final_embeddings[idx] = existing_emb_map[cid]
                reused_count += 1

        # Compute embeddings ONLY for missing / dirty chunks
        if texts_to_embed:
            print(f"Embedding {len(texts_to_embed)} chunks with {EMBEDDING_MODEL_NAME} (reused {reused_count} cached embeddings)...", flush=True)
            new_vectors = self.embed_texts(texts_to_embed)
            for m_idx, vec in zip(missing_indices, new_vectors):
                final_embeddings[m_idx] = vec
        else:
            print(f"All {len(chunk_ids)} chunk embeddings reused from cache. No re-embedding required.", flush=True)

        self.chunk_ids = chunk_ids
        self.embeddings = final_embeddings
        self.save()

    def save(self):
        """Save vector matrix and chunk ID mapping to disk."""
        if self.embeddings is not None and self.chunk_ids:
            np.savez_compressed(str(self.index_path), embeddings=self.embeddings)
            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump(self.chunk_ids, f)

    def load(self) -> bool:
        """Load vector matrix and chunk ID mapping if exists."""
        if self.index_path.exists() and self.chunks_path.exists():
            try:
                data = np.load(str(self.index_path))
                self.embeddings = data["embeddings"]
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunk_ids = json.load(f)
                return True
            except Exception as e:
                print(f"Failed to load vector index: {e}")
                return False
        return False

    def search(
        self,
        query: str,
        top_k: int = 30,
        allowed_chunk_ids: Optional[set] = None
    ) -> List[Tuple[str, float]]:
        """Dense similarity search returning (chunk_id, similarity_score)."""
        if self.embeddings is None or len(self.chunk_ids) == 0:
            if not self.load():
                return []

        q_vec = self.embed_query(query)
        # Cosine similarity is dot product of normalized vectors
        scores = np.dot(self.embeddings, q_vec)

        if allowed_chunk_ids is not None:
            # Mask out non-allowed chunk ids
            mask = np.array([cid in allowed_chunk_ids for cid in self.chunk_ids], dtype=bool)
            scores[~mask] = -1.0

        top_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > -0.9:  # valid candidate
                results.append((self.chunk_ids[idx], score))
        return results

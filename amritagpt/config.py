"""
Configuration settings for AmritaGPT: Institutional Intelligence Infrastructure.
"""
from pathlib import Path

# Root directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
INDEX_DIR = STORAGE_DIR / "indices"
CACHE_DIR = STORAGE_DIR / "cache"
LOGS_DIR = STORAGE_DIR / "logs"

# Ensure directories exist
for p in [STORAGE_DIR, INDEX_DIR, CACHE_DIR, LOGS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Database and file paths
DB_PATH = STORAGE_DIR / "amritagpt_meta.db"
VECTOR_INDEX_PATH = INDEX_DIR / "dense_embeddings.npz"
VECTOR_CHUNKS_PATH = INDEX_DIR / "chunks.json"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"
AUDIT_LOG_PATH = LOGS_DIR / "query_audit.jsonl"

# Ingestion settings
CHUNK_SIZE_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50
MAX_CHUNK_CHAR_LEN = 1500

# Embedding model settings
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Retrieval settings
DENSE_TOP_K = 30
SPARSE_TOP_K = 30
RRF_K = 60
RERANK_TOP_K = 8
MIN_GROUNDING_CONFIDENCE = 0.45

# Role hierarchy for Access Control (higher number = more privileges)
ROLE_HIERARCHY = {
    "public": 1,
    "student": 2,
    "faculty": 3,
    "admin": 4
}

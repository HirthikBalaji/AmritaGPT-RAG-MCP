# AmritaGPT: Institutional Intelligence Infrastructure for the AI-Native University

> *"One intelligence layer for every document, department, workflow, and AI agent."*

**AmritaGPT** is an enterprise-grade institutional intelligence infrastructure built for **Amrita Vishwa Vidyapeetham**. It converts the university's heterogeneous, distributed document ecosystem into a governed, searchable, reasoning-ready knowledge layer and exposes it universally through the **Model Context Protocol (MCP)**.

---

## 🏛️ System Architecture

AmritaGPT is structured across 6 core pillars:

```
                         ┌─────────────────────────────────────────┐
                         │           AI USERS & AGENTS             │
                         │  Students • Faculty • Admin • Research  │
                         └────────────────────┬────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────┐
                         │         AMRITAGPT MCP SERVER            │
                         │   Tools • Resources • Prompts • Auth    │
                         └────────────────────┬────────────────────┘
                                              │
                         ┌────────────────────▼────────────────────┐
                         │         HYBRID RETRIEVAL ENGINE         │
                         │                                         │
                         │   Dense Vector (BGE) + BM25 Lexical     │
                         │   Reciprocal Rank Fusion (RRF k=60)     │
                         │   Cross-Feature Precision Reranking     │
                         │   Role-Based Pre-Retrieval ACL          │
                         └────────────────────┬────────────────────┘
                                              │
                         ┌────────────────────┴────────────────────┐
                         ▼                                         ▼
              ┌─────────────────────┐                   ┌─────────────────────┐
              │    VECTOR STORE     │                   │     METADATA DB     │
              │  FastEmbed ONNX 384 │                   │   SQLite Relational │
              │  Cosine Similarity  │                   │   ACLs & Lineage    │
              └─────────────────────┘                   └─────────────────────┘
                                              ▲
                                              │
                         ┌────────────────────┴────────────────────┐
                         │      CONTINUOUS INGESTION PIPELINE      │
                         │                                         │
                         │  PDF (PDFium) • DOCX • XLSX • PPTX      │
                         │  Structure-Aware Semantic Chunking      │
                         │  Contextual Enrichment (Hierarchy/Meta) │
                         │  Checksum Deduplication (Idempotency)   │
                         └─────────────────────────────────────────┘
```

---

## 🚀 Key Architectural Innovations

### 1. Structure-Aware Semantic Chunking & Contextual Enrichment
Unlike naive fixed-token chunking, AmritaGPT preserves document hierarchy:
* **Contextual Prefixes**: Every chunk is prefixed with the institutional path:
  `[Institution: Amrita Vishwa Vidyapeetham] | Dept: Academic Admin Office | Doc: btech-regulations-2023 | Section: R.4 Attendance | Page: 4`
* **Semantic Boundaries**: Splits occur cleanly on regulation rules, sub-sections, paragraphs, or tabular lines, avoiding split meanings.
* **Neighbor Linking**: Chunks maintain `prev_chunk_id` and `next_chunk_id` pointers for dynamic context expansion during retrieval.

### 2. Multi-Stage Hybrid Retrieval
* **Dense Vector Search**: Powered by `BAAI/bge-small-en-v1.5` running on local ONNX runtime for conceptual matching.
* **Sparse Lexical Search**: Powered by `rank_bm25` for exact regulation codes (e.g. `R.4`, `R.14`), course numbers (e.g. `23CSE101`), and faculty names.
* **Reciprocal Rank Fusion (RRF)**:
  $$RRF(d) = \sum_{m \in \{dense, bm25\}} \frac{1}{k + rank_m(d)} \quad (k=60)$$
* **Precision Cross-Feature Reranking**: Evaluates query-passage cross-alignment, heading bonuses, exact identifier matches, and temporal freshness.

### 3. Institutional Access Control (RBAC) & Security
* **Pre-Retrieval Filtering**: Enforces role boundaries (`public` < `student` < `faculty` < `admin`) *before* candidates are searched, preventing unauthorized information leakage.
* **Prompt Injection Defense**: All retrieved documents are sanitized and treated strictly as untrusted data, preventing adversarial instruction override.

### 4. Grounded Generation & Citation Engine
* **Evidence-Bound Policy**: If sufficient evidence exists, answers are constructed with inline citations `[Document, Section, Page, Year]`. If evidence is insufficient, the system safely returns:
  > *"I couldn't find sufficient evidence in the institutional knowledge base."*
* **Grounding Verifier**: Automatically audits generated claims against retrieved evidence, computing a factual grounding score.

---

## 🛠️ MCP Capabilities

AmritaGPT implements the official Model Context Protocol specification:

### MCP Tools (Actions)
| Tool | Description |
| :--- | :--- |
| `search_knowledge` | Multi-stage hybrid search across institutional knowledge with access filtering. |
| `search_documents` | Search the document catalog by title, department, or category. |
| `get_document` | Read full metadata, section list, and content of an institutional document. |
| `get_document_section` | Read a specific section, regulation clause, or table from a document. |
| `search_policy` | Dedicated search for university policies, disciplinary rules, and guidelines. |
| `search_academic_regulations` | Search official B.Tech/M.Tech/PhD ordinances and attendance rules. |
| `get_latest_circular` | Retrieve recent notifications, circulars, and calendar dates. |
| `compare_document_versions` | Compare two document versions to audit curriculum or policy changes. |
| `find_related_documents` | Discover companion files within the same department or academic program. |
| `ask_institutional_question` | End-to-end grounded RAG answer generation with verified citations. |
| `get_system_stats` | Real-time observability: total documents, chunks, queries, and latency. |
| `run_rag_evaluation` | Run automated benchmark suite measuring Recall@1, Recall@5, and MRR. |

### MCP Resources (Knowledge Objects)
* `amritagpt://system/stats`: Live system health, document count, and index status.
* `amritagpt://policies/current`: Summary of currently active university regulations.
* `amritagpt://departments/list`: Catalog of university departments and repository sizes.

### MCP Prompts (Reusable Agent Workflows)
* `academic_advising`: Guide students through curriculum, prerequisites, and graduation requirements.
* `policy_compliance_check`: Audit scenarios against institutional and examination rules.

---

## 🌐 OpenAPI 3.1 REST API Gateway

AmritaGPT is **100% OpenAPI 3.1 compliant** and exposes an interactive REST API alongside its MCP capabilities:

* **Interactive Swagger UI**: `http://localhost:8000/docs`
* **ReDoc Interactive Documentation**: `http://localhost:8000/redoc`
* **OpenAPI 3.1 JSON Specification**: `http://localhost:8000/openapi.json` (also exported to [`openapi.json`](file:///C:/Users/Suz%20Machine%20Tech/RAG/openapi.json))
* **OpenAPI YAML Specification**: [`openapi.yaml`](file:///C:/Users/Suz%20Machine%20Tech/RAG/openapi.yaml)
* **Integrated MCP SSE Endpoint**: `http://localhost:8000/mcp/sse`

### REST Endpoints Reference
| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | API identity, documentation links, and MCP SSE endpoint. |
| `GET` | `/health` | Index health check (documents, chunks, vector/BM25 status). |
| `POST` | `/api/v1/search/knowledge` | Hybrid search (Dense Vector + BM25 + RRF + Reranker) with access filtering. |
| `POST` | `/api/v1/qa/ask` | End-to-end grounded generation with citations and grounding confidence. |
| `POST` | `/api/v1/policy/search` | Search university policies, disciplinary rules, and guidelines. |
| `POST` | `/api/v1/regulations/academic` | Search official B.Tech/M.Tech/PhD ordinances & attendance rules. |
| `POST` | `/api/v1/search/documents` | Search the document catalog by title, department, or keywords. |
| `GET` | `/api/v1/documents/{document_id}` | Retrieve full metadata and sections of an institutional document. |
| `POST` | `/api/v1/documents/compare` | Compare two document versions to detect policy/curriculum revisions. |
| `GET` | `/api/v1/system/stats` | Observability metrics (total documents, chunks, query latency). |
| `POST` | `/api/v1/evaluation/run` | Run automated RAG evaluation benchmark suite. |

---

## 📂 Project Structure

```
C:\Users\Suz Machine Tech\RAG\
├── amritagpt/
│   ├── __init__.py
│   ├── config.py                 # System configurations and paths
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                # FastAPI OpenAPI REST Gateway with mounted MCP
│   │   └── schemas.py            # Pydantic schemas for all requests/responses
│   ├── ingestion/
│   │   ├── parser.py             # Format-agnostic parser (PDF, DOCX, XLSX, PPTX)
│   │   ├── chunker.py            # Structure-aware semantic chunker
│   │   └── pipeline.py           # Ingestion bus with checksum deduplication
│   ├── indexing/
│   │   ├── vector_store.py       # Dense vector index (FastEmbed ONNX 384-dim)
│   │   ├── bm25_store.py         # BM25 sparse lexical index
│   │   └── metadata_store.py     # SQLite relational metadata database
│   ├── retrieval/
│   │   ├── query_analyzer.py     # Intent detection, entity extraction, temporal logic
│   │   ├── hybrid_search.py      # Hybrid Dense + BM25 search with RRF
│   │   ├── reranker.py           # Precision cross-feature reranker
│   │   └── context_assembler.py  # Evidence packager and neighbor expansion
│   ├── generator/
│   │   ├── grounded_engine.py    # Grounded response synthesis
│   │   ├── citation_engine.py    # Provenance and citation formatting
│   │   └── verifier.py           # Grounding and hallucination auditor
│   ├── security/
│   │   └── access_control.py     # Role-based ACL & prompt injection guard
│   ├── observability/
│   │   └── telemetry.py          # Metrics, query audit logs, evaluation framework
│   └── mcp/
│       └── server.py             # MCPServer exposing tools, resources, prompts
├── data/                         # Institutional documents directory (316 files)
├── storage/                      # SQLite DB, vector embeddings, BM25 indices
├── scripts/
│   ├── ingest_all.py             # CLI runner for document ingestion
│   ├── export_openapi.py         # Export openapi.json and openapi.yaml
│   ├── test_mcp_client.py        # End-to-end MCP verification suite
│   └── test_openapi_client.py    # End-to-end OpenAPI REST verification suite
├── openapi.json                  # Exported OpenAPI 3.1 JSON Specification
├── openapi.yaml                  # Exported OpenAPI 3.1 YAML Specification
├── mcp_config.json               # MCP client configuration template
├── run_server.py                 # Unified server entry point (MCP & OpenAPI)
└── README.md
```

---

## ⚡ Quick Start

### 1. Ingest Institutional Documents
```bash
python scripts/ingest_all.py
```

### 2. Run Verification Suites
```bash
# Run MCP Verification Suite
python scripts/test_mcp_client.py

# Run OpenAPI REST API Verification Suite
python scripts/test_openapi_client.py
```

### 3. Start AmritaGPT Server

#### Option A: OpenAPI REST Gateway & Swagger UI (Port 8000)
```bash
python run_server.py --mode api --port 8000
```
Then visit:
* **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **OpenAPI Spec**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
* **MCP SSE**: [http://localhost:8000/mcp/sse](http://localhost:8000/mcp/sse)

#### Option B: Standard MCP Stdio Server (for Claude Desktop / Cursor / Antigravity)
```bash
python run_server.py --mode mcp --transport stdio
```

### 4. Configure in Claude Desktop / Cursor / Antigravity
Add the following to your `claude_desktop_config.json` or MCP settings:

```json
{
  "mcpServers": {
    "amritagpt": {
      "command": "python",
      "args": [
        "path/to/run_server.py",
        "--mode",
        "mcp",
        "--transport",
        "stdio"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8"
      }
    }
  }
}
```

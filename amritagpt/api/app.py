"""
FastAPI OpenAPI REST Gateway for AmritaGPT.
Exposes institutional intelligence as OpenAPI 3.1 REST API alongside the Model Context Protocol.
"""
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Path as PathParam, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from amritagpt.api.schemas import (
    SearchKnowledgeRequest, SearchKnowledgeResponse,
    SearchDocumentsRequest, SearchDocumentsResponse, DocumentDetailResponse, DocumentSummary,
    PolicySearchRequest, AcademicRegulationsRequest, PolicyRegulationResponse,
    AskQuestionRequest, AskQuestionResponse,
    CompareVersionsRequest, CompareVersionsResponse,
    SystemStatsResponse, EvaluationMetricResponse, SourceItem
)
from amritagpt.indexing.metadata_store import MetadataStore
from amritagpt.indexing.vector_store import DenseVectorStore
from amritagpt.indexing.bm25_store import BM25LexicalStore
from amritagpt.retrieval.hybrid_search import HybridSearchEngine
from amritagpt.generator.grounded_engine import GroundedAnswerEngine
from amritagpt.observability.telemetry import EvaluationFramework
from amritagpt.security.access_control import AccessControlManager
from amritagpt.mcp.server import create_amrita_mcp_server


def create_app() -> FastAPI:
    app = FastAPI(
        title="AmritaGPT Institutional Intelligence API",
        description=(
            "**AmritaGPT** is an enterprise-grade Institutional Intelligence Infrastructure for "
            "**Amrita Vishwa Vidyapeetham**. It exposes a production RAG platform combining multimodal "
            "document ingestion, structure-aware semantic chunking, dense vector + BM25 hybrid search, "
            "cross-encoder reranking, role-based access control, and grounded answer verification.\n\n"
            "This API is **fully OpenAPI 3.1 compliant** and exposes interactive Swagger documentation, "
            "ReDoc specifications, and standard REST endpoints alongside an integrated **MCP Server**."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "AmritaGPT Intelligence Team",
            "url": "https://www.amrita.edu",
            "email": "amritagpt@amrita.edu",
        },
        license_info={
            "name": "Institutional License",
        }
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize Core Engines
    meta_store = MetadataStore()
    vector_store = DenseVectorStore()
    bm25_store = BM25LexicalStore()

    vector_store.load()
    bm25_store.load()

    search_engine = HybridSearchEngine(meta_store, vector_store, bm25_store)
    grounded_engine = GroundedAnswerEngine(search_engine)
    eval_framework = EvaluationFramework(search_engine)

    # Mount MCP SSE App directly under /mcp
    mcp_server = create_amrita_mcp_server()
    app.mount("/mcp", mcp_server.sse_app())

    # ---------------------------------------------------------------------------
    # Health & Meta Endpoints
    # ---------------------------------------------------------------------------

    @app.get("/", tags=["General"], summary="API Root Information")
    def root():
        """Returns API identity and server status."""
        return {
            "name": "AmritaGPT Institutional Intelligence API",
            "version": "1.0.0",
            "institution": "Amrita Vishwa Vidyapeetham",
            "documentation": {
                "swagger_ui": "/docs",
                "redoc": "/redoc",
                "openapi_json": "/openapi.json"
            },
            "mcp_sse_endpoint": "/mcp/sse"
        }

    @app.get("/health", tags=["General"], summary="Health Check")
    def health():
        """Service health check ensuring index availability."""
        stats = meta_store.get_system_stats()
        return {
            "status": "healthy",
            "documents_indexed": stats["total_documents"],
            "chunks_indexed": stats["total_chunks"],
            "vector_index_active": len(vector_store.chunk_ids) > 0,
            "bm25_index_active": len(bm25_store.chunk_ids) > 0
        }

    # ---------------------------------------------------------------------------
    # Knowledge & Search Endpoints
    # ---------------------------------------------------------------------------

    @app.post(
        "/api/v1/search/knowledge",
        response_model=SearchKnowledgeResponse,
        tags=["Hybrid Retrieval"],
        summary="Hybrid Knowledge Retrieval",
        description="Executes multi-stage hybrid search (Dense Vector + BM25 + RRF + Reranker) with access filtering."
    )
    def search_knowledge(req: SearchKnowledgeRequest):
        res = search_engine.retrieve(
            query=req.query,
            role=req.role,
            department_filter=req.department,
            academic_year_filter=req.academic_year,
            top_k=req.top_k
        )
        pkg = res["evidence_package"]
        sources = [
            SourceItem(
                source_num=s.get("source_num", idx + 1),
                doc_id=s.get("doc_id", ""),
                chunk_id=s.get("chunk_id", ""),
                filename=s.get("filename", ""),
                relative_path=s.get("relative_path", ""),
                department=s.get("department", "Amrita University"),
                section=s.get("section", "General"),
                page=s.get("page"),
                academic_year=s.get("academic_year"),
                access_policy=s.get("access_policy", "student"),
                rerank_score=round(s.get("rerank_score", 0.0), 3)
            )
            for idx, s in enumerate(pkg.get("sources", []))
        ]

        return SearchKnowledgeResponse(
            status="success" if pkg.get("is_sufficient") else "empty",
            query=req.query,
            role=req.role,
            analyzed_intent=res["analyzed_query"]["intent"],
            total_candidates_retrieved=res["retrieval_stats"]["fused_candidates"],
            selected_evidence_count=len(sources),
            sources=sources,
            evidence_text=pkg.get("evidence_text", "")
        )

    # ---------------------------------------------------------------------------
    # Grounded Q&A / Institutional Reasoning Endpoints
    # ---------------------------------------------------------------------------

    @app.post(
        "/api/v1/qa/ask",
        response_model=AskQuestionResponse,
        tags=["Grounded Generation"],
        summary="Ask Grounded Institutional Question",
        description="Generates an evidence-grounded answer backed strictly by institutional citations and a factual grounding score."
    )
    def ask_question(req: AskQuestionRequest):
        ans = grounded_engine.generate_answer(
            question=req.question,
            role=req.role,
            department=req.department
        )
        sources = [
            SourceItem(
                source_num=s.get("source_num", idx + 1),
                doc_id=s.get("doc_id", ""),
                chunk_id=s.get("chunk_id", ""),
                filename=s.get("filename", ""),
                relative_path=s.get("relative_path", ""),
                department=s.get("department", "Amrita University"),
                section=s.get("section", "General"),
                page=s.get("page"),
                academic_year=s.get("academic_year"),
                access_policy=s.get("access_policy", "student"),
                rerank_score=round(s.get("rerank_score", 0.0), 3)
            )
            for idx, s in enumerate(ans.get("sources", []))
        ]

        return AskQuestionResponse(
            question=req.question,
            answer=ans["answer"],
            confidence=ans["confidence"],
            grounding_score=ans["grounding_score"],
            is_faithful=ans["is_faithful"],
            intent=ans["intent"],
            citations=ans["citations"],
            sources=sources,
            is_sufficient=ans["is_sufficient"],
            role=ans["role"],
            latency_ms=ans.get("latency_ms", 0.0)
        )

    # ---------------------------------------------------------------------------
    # Policy & Regulation Endpoints
    # ---------------------------------------------------------------------------

    @app.post(
        "/api/v1/policy/search",
        response_model=PolicyRegulationResponse,
        tags=["Policies & Regulations"],
        summary="Search University Policies",
        description="Searches institutional policies, disciplinary rules, hostel regulations, and administrative guidelines."
    )
    def search_policy(req: PolicySearchRequest):
        res = search_engine.retrieve(
            query=f"policy regulation rule {req.query}",
            role=req.role,
            top_k=5
        )
        pkg = res["evidence_package"]
        sources = [
            SourceItem(
                source_num=s.get("source_num", idx + 1),
                doc_id=s.get("doc_id", ""),
                chunk_id=s.get("chunk_id", ""),
                filename=s.get("filename", ""),
                relative_path=s.get("relative_path", ""),
                department=s.get("department", "Amrita University"),
                section=s.get("section", "General"),
                page=s.get("page"),
                academic_year=s.get("academic_year"),
                access_policy=s.get("access_policy", "student"),
                rerank_score=round(s.get("rerank_score", 0.0), 3)
            )
            for idx, s in enumerate(pkg.get("sources", []))
        ]
        return PolicyRegulationResponse(
            topic=req.query,
            program=None,
            academic_year=None,
            regulations_found=len(sources),
            evidence=pkg.get("evidence_text", ""),
            sources=sources
        )

    @app.post(
        "/api/v1/regulations/academic",
        response_model=PolicyRegulationResponse,
        tags=["Policies & Regulations"],
        summary="Search Academic Regulations",
        description="Search official B.Tech/M.Tech/PhD ordinances, grading criteria, and attendance rules."
    )
    def search_academic_regulations(req: AcademicRegulationsRequest):
        full_query = f"{req.program or ''} Academic Regulations {req.query}"
        res = search_engine.retrieve(
            query=full_query,
            role="student",
            department_filter="Academic Admin Office",
            academic_year_filter=req.academic_year,
            top_k=5
        )
        pkg = res["evidence_package"]
        sources = [
            SourceItem(
                source_num=s.get("source_num", idx + 1),
                doc_id=s.get("doc_id", ""),
                chunk_id=s.get("chunk_id", ""),
                filename=s.get("filename", ""),
                relative_path=s.get("relative_path", ""),
                department=s.get("department", "Amrita University"),
                section=s.get("section", "General"),
                page=s.get("page"),
                academic_year=s.get("academic_year"),
                access_policy=s.get("access_policy", "student"),
                rerank_score=round(s.get("rerank_score", 0.0), 3)
            )
            for idx, s in enumerate(pkg.get("sources", []))
        ]
        return PolicyRegulationResponse(
            topic=req.query,
            program=req.program,
            academic_year=req.academic_year,
            regulations_found=len(sources),
            evidence=pkg.get("evidence_text", ""),
            sources=sources
        )

    # ---------------------------------------------------------------------------
    # Document Catalog Endpoints
    # ---------------------------------------------------------------------------

    @app.post(
        "/api/v1/search/documents",
        response_model=SearchDocumentsResponse,
        tags=["Document Catalog"],
        summary="Search Document Catalog",
        description="Search institutional documents by filename, department, or keywords."
    )
    def search_documents(req: SearchDocumentsRequest):
        docs = meta_store.get_all_documents(department=req.department)
        user_level = AccessControlManager.get_role_level(req.role)

        filtered = []
        q_lower = req.query.lower()
        for d in docs:
            if AccessControlManager.get_role_level(d["access_policy"]) > user_level:
                continue
            text = f"{d['filename']} {d['category']} {d['department']}".lower()
            if q_lower in text:
                filtered.append(
                    DocumentSummary(
                        doc_id=d["doc_id"],
                        filename=d["filename"],
                        department=d["department"],
                        category=d["category"],
                        academic_year=d["academic_year"],
                        version=d["version"],
                        access_policy=d["access_policy"],
                        sections=d["section_count"]
                    )
                )

        return SearchDocumentsResponse(
            query=req.query,
            total_matches=len(filtered),
            documents=filtered[:30]
        )

    @app.get(
        "/api/v1/documents/{document_id}",
        response_model=DocumentDetailResponse,
        tags=["Document Catalog"],
        summary="Get Document Details",
        description="Retrieve full metadata, section list, and content of an institutional document."
    )
    def get_document(document_id: str = PathParam(..., description="Unique document identifier")):
        doc = meta_store.get_document_by_id(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

        sections = meta_store.get_document_sections(document_id)
        return DocumentDetailResponse(
            doc_id=doc["doc_id"],
            filename=doc["filename"],
            department=doc["department"],
            category=doc["category"],
            academic_year=doc["academic_year"],
            version=doc["version"],
            access_policy=doc["access_policy"],
            total_sections=len(sections),
            sections=sections
        )

    @app.post(
        "/api/v1/documents/compare",
        response_model=CompareVersionsResponse,
        tags=["Document Catalog"],
        summary="Compare Document Versions",
        description="Compare two document versions on a topic to identify institutional revisions."
    )
    def compare_documents(req: CompareVersionsRequest):
        doc1 = meta_store.get_document_by_id(req.doc_id_1)
        doc2 = meta_store.get_document_by_id(req.doc_id_2)

        if not doc1 or not doc2:
            raise HTTPException(status_code=404, detail="One or both document IDs do not exist.")

        sec1 = meta_store.get_document_sections(req.doc_id_1)
        sec2 = meta_store.get_document_sections(req.doc_id_2)

        if req.topic:
            t = req.topic.lower()
            sec1 = [s for s in sec1 if t in s["section_title"].lower() or t in s["raw_content"].lower()]
            sec2 = [s for s in sec2 if t in s["section_title"].lower() or t in s["raw_content"].lower()]

        return CompareVersionsResponse(
            doc_1={
                "id": doc1["doc_id"],
                "filename": doc1["filename"],
                "academic_year": doc1["academic_year"],
                "sections_matched": len(sec1),
                "sample_content": [s["raw_content"][:300] for s in sec1[:2]]
            },
            doc_2={
                "id": doc2["doc_id"],
                "filename": doc2["filename"],
                "academic_year": doc2["academic_year"],
                "sections_matched": len(sec2),
                "sample_content": [s["raw_content"][:300] for s in sec2[:2]]
            },
            comparison_topic=req.topic or "Full Document"
        )

    # ---------------------------------------------------------------------------
    # Observability & Evaluation Endpoints
    # ---------------------------------------------------------------------------

    @app.get(
        "/api/v1/system/stats",
        response_model=SystemStatsResponse,
        tags=["Observability & Metrics"],
        summary="Get System Statistics",
        description="Retrieve real-time observability telemetry: document counts, chunks, query latency, and grounding score."
    )
    def system_stats():
        stats = meta_store.get_system_stats()
        return SystemStatsResponse(**stats)

    @app.post(
        "/api/v1/evaluation/run",
        response_model=EvaluationMetricResponse,
        tags=["Observability & Metrics"],
        summary="Run Automated RAG Benchmark",
        description="Executes the automated evaluation suite calculating Recall@1, Recall@5, MRR, and latency profiles."
    )
    def run_evaluation():
        eval_res = eval_framework.run_benchmark()
        return EvaluationMetricResponse(**eval_res)

    return app


# Global FastAPI application instance
app = create_app()

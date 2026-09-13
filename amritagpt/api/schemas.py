"""
Pydantic Schemas for OpenAPI 3.1 REST Endpoints of AmritaGPT.
Provides strict type validation, parameter documentation, and response models.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Search & Knowledge Schemas
# ---------------------------------------------------------------------------

class SearchKnowledgeRequest(BaseModel):
    query: str = Field(..., description="Natural language search query", example="What is the attendance requirement for B.Tech students?")
    department: Optional[str] = Field(None, description="Optional department filter", example="Academic Admin Office")
    document_type: Optional[str] = Field(None, description="Optional file type filter ('pdf', 'docx', 'xlsx')", example="pdf")
    academic_year: Optional[str] = Field(None, description="Optional academic year filter", example="2023")
    role: str = Field("student", description="Role for access control ('public', 'student', 'faculty', 'admin')", example="student")
    top_k: int = Field(5, ge=1, le=20, description="Number of verified candidate chunks to return", example=5)


class SourceItem(BaseModel):
    source_num: int = Field(..., example=1)
    doc_id: str = Field(..., example="8f9a2b3c4d5e")
    chunk_id: str = Field(..., example="8f9a2b3c4d5e_c4")
    filename: str = Field(..., example="btech-school-computing-regulations-2023.pdf")
    relative_path: str = Field(..., example="Academic Admin Office/CSE/btech-school-computing-regulations-2023.pdf")
    department: str = Field(..., example="Academic Admin Office")
    section: str = Field(..., example="R.4 Attendance")
    page: Optional[int] = Field(None, example=4)
    academic_year: Optional[str] = Field(None, example="2023")
    access_policy: Optional[str] = Field("student", example="student")
    rerank_score: float = Field(..., example=8.75)


class SearchKnowledgeResponse(BaseModel):
    status: str = Field(..., example="success")
    query: str = Field(..., example="What is the attendance requirement for B.Tech students?")
    role: str = Field(..., example="student")
    analyzed_intent: str = Field(..., example="Academic Regulation")
    total_candidates_retrieved: int = Field(..., example=18)
    selected_evidence_count: int = Field(..., example=4)
    sources: List[SourceItem] = Field(default_factory=list)
    evidence_text: str = Field(..., description="Assembled context with institutional evidence")


# ---------------------------------------------------------------------------
# Document Catalog Schemas
# ---------------------------------------------------------------------------

class SearchDocumentsRequest(BaseModel):
    query: str = Field(..., description="Search keyword in filename, category, or department", example="regulations")
    department: Optional[str] = Field(None, description="Optional department filter", example="CSE")
    role: str = Field("student", description="User role for permission filtering", example="student")


class DocumentSummary(BaseModel):
    doc_id: str = Field(..., example="a1b2c3d4e5f6")
    filename: str = Field(..., example="curriculum-and-syllabi-amrita-btech-cse-2023.pdf")
    department: str = Field(..., example="Academic Admin Office")
    category: str = Field(..., example="Academic Programs & Regulations")
    academic_year: Optional[str] = Field(None, example="2023")
    version: str = Field(..., example="2023")
    access_policy: str = Field(..., example="student")
    sections: int = Field(..., example=42)


class SearchDocumentsResponse(BaseModel):
    query: str = Field(...)
    total_matches: int = Field(...)
    documents: List[DocumentSummary] = Field(default_factory=list)


class DocumentDetailResponse(BaseModel):
    doc_id: str = Field(...)
    filename: str = Field(...)
    department: str = Field(...)
    category: str = Field(...)
    academic_year: Optional[str] = Field(None)
    version: str = Field(...)
    access_policy: str = Field(...)
    total_sections: int = Field(...)
    sections: List[Dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Policy & Academic Regulation Schemas
# ---------------------------------------------------------------------------

class PolicySearchRequest(BaseModel):
    query: str = Field(..., description="Policy question or guideline keyword", example="hostel rules and curfew")
    role: str = Field("student", description="User role for access control", example="student")


class AcademicRegulationsRequest(BaseModel):
    query: str = Field(..., description="Specific regulation clause, attendance, grading rule", example="R.14 grading system")
    program: Optional[str] = Field("B.Tech", description="Academic degree program", example="B.Tech")
    academic_year: Optional[str] = Field("2023", description="Admissions or regulation year", example="2023")


class PolicyRegulationResponse(BaseModel):
    topic: str = Field(...)
    program: Optional[str] = None
    academic_year: Optional[str] = None
    regulations_found: int = Field(...)
    evidence: str = Field(...)
    sources: List[SourceItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Grounded Q&A / Institutional Reasoning Schemas
# ---------------------------------------------------------------------------

class AskQuestionRequest(BaseModel):
    question: str = Field(..., description="Natural language question for institutional RAG", example="What is the minimum attendance required to appear for end semester examinations?")
    role: str = Field("student", description="Role of the asking user", example="student")
    department: Optional[str] = Field(None, description="Optional target department", example="Academic Admin Office")


class AskQuestionResponse(BaseModel):
    question: str = Field(...)
    answer: str = Field(..., description="Verified answer synthesized from institutional documents")
    confidence: str = Field(..., description="'High', 'Moderate', 'Low', or 'None'", example="High")
    grounding_score: float = Field(..., description="Factual alignment score between 0.0 and 1.0", example=0.95)
    is_faithful: bool = Field(..., description="Flag indicating if the answer contains unsupported claims", example=True)
    intent: str = Field(..., example="Academic Regulation")
    citations: List[str] = Field(..., description="Formatted citations with document, section, and page provenance")
    sources: List[SourceItem] = Field(default_factory=list)
    is_sufficient: bool = Field(..., description="False if knowledge base lacked sufficient evidence")
    role: str = Field(...)
    latency_ms: float = Field(..., example=120.5)


# ---------------------------------------------------------------------------
# Document Comparison Schemas
# ---------------------------------------------------------------------------

class CompareVersionsRequest(BaseModel):
    doc_id_1: str = Field(..., description="ID of first/baseline document", example="doc_id_1")
    doc_id_2: str = Field(..., description="ID of second/comparative document", example="doc_id_2")
    topic: Optional[str] = Field(None, description="Optional topic or course keyword to compare", example="machine learning")


class CompareVersionsResponse(BaseModel):
    doc_1: Dict[str, Any] = Field(...)
    doc_2: Dict[str, Any] = Field(...)
    comparison_topic: str = Field(...)


# ---------------------------------------------------------------------------
# System Observability & Evaluation Schemas
# ---------------------------------------------------------------------------

class SystemStatsResponse(BaseModel):
    total_documents: int = Field(..., example=299)
    total_chunks: int = Field(..., example=15346)
    departments: Dict[str, int] = Field(default_factory=dict)
    access_distribution: Dict[str, int] = Field(default_factory=dict)
    total_queries_served: int = Field(..., example=42)
    avg_query_latency_ms: float = Field(..., example=118.4)
    avg_grounding_score: float = Field(..., example=0.92)


class EvaluationMetricResponse(BaseModel):
    total_benchmark_queries: int = Field(..., example=5)
    recall_at_1: float = Field(..., example=0.4)
    recall_at_5: float = Field(..., example=0.6)
    mean_reciprocal_rank_mrr: float = Field(..., example=0.5)
    p50_latency_ms: float = Field(..., example=180.2)
    p95_latency_ms: float = Field(..., example=245.0)
    detailed_results: List[Dict[str, Any]] = Field(default_factory=list)

"""
AmritaGPT Model Context Protocol (MCP) Server.
Exposes institutional intelligence as standardized Tools, Resources, and Prompts
for AI agents, IDEs, and assistants across Amrita Vishwa Vidyapeetham.
"""
from typing import Optional, Dict, Any, List
import json
import asyncio

from mcp.server.mcpserver import MCPServer
from amritagpt.indexing.metadata_store import MetadataStore
from amritagpt.indexing.vector_store import DenseVectorStore
from amritagpt.indexing.bm25_store import BM25LexicalStore
from amritagpt.retrieval.hybrid_search import HybridSearchEngine
from amritagpt.generator.grounded_engine import GroundedAnswerEngine
from amritagpt.observability.telemetry import EvaluationFramework
from amritagpt.security.access_control import AccessControlManager


def create_amrita_mcp_server() -> MCPServer:
    server = MCPServer(
        name="AmritaGPT",
        version="1.0.0",
        description="Institutional Intelligence Infrastructure for Amrita Vishwa Vidyapeetham"
    )

    # Initialize shared components
    meta_store = MetadataStore()
    vector_store = DenseVectorStore()
    bm25_store = BM25LexicalStore()
    
    # Try loading existing indices
    vector_store.load()
    bm25_store.load()

    search_engine = HybridSearchEngine(meta_store, vector_store, bm25_store)
    grounded_engine = GroundedAnswerEngine(search_engine)
    eval_framework = EvaluationFramework(search_engine)

    # ---------------------------------------------------------
    # MCP TOOLS (Actions for AI Agents)
    # ---------------------------------------------------------

    @server.tool()
    def search_knowledge(
        query: str,
        department: Optional[str] = None,
        document_type: Optional[str] = None,
        academic_year: Optional[str] = None,
        role: str = "student",
        top_k: int = 5
    ) -> str:
        """Search institutional knowledge using hybrid retrieval (Dense + BM25 + RRF + Reranker).
        
        Args:
            query: The natural language search query.
            department: Optional department filter (e.g. 'CSE', 'Academic Admin Office', 'Hostel').
            document_type: Optional file type filter ('pdf', 'docx', 'xlsx').
            academic_year: Optional academic year filter ('2023', '2024', '2025', '2026').
            role: User role for access control ('student', 'faculty', 'admin', 'public').
            top_k: Number of verified chunks to return.
        """
        results = search_engine.retrieve(
            query=query,
            role=role,
            department_filter=department,
            academic_year_filter=academic_year,
            top_k=top_k
        )
        
        pkg = results["evidence_package"]
        if not pkg["is_sufficient"]:
            return json.dumps({
                "status": "not_found",
                "message": "No relevant institutional records found matching the query and access level.",
                "query": query,
                "role": role
            }, indent=2)

        return json.dumps({
            "status": "success",
            "query": query,
            "role": role,
            "analyzed_intent": results["analyzed_query"]["intent"],
            "total_candidates_retrieved": results["retrieval_stats"]["fused_candidates"],
            "selected_evidence_count": pkg["chunk_count"],
            "sources": pkg["sources"],
            "evidence_text": pkg["evidence_text"]
        }, indent=2)

    @server.tool()
    def search_documents(
        query: str,
        department: Optional[str] = None,
        role: str = "student"
    ) -> str:
        """Search the document catalog for regulations, syllabi, handbooks, or circulars by name.
        
        Args:
            query: Keyword in filename or category.
            department: Optional department name filter.
            role: User role for permission filtering.
        """
        docs = meta_store.get_all_documents(department=department)
        user_level = AccessControlManager.get_role_level(role)

        filtered = []
        q_lower = query.lower()
        for d in docs:
            if AccessControlManager.get_role_level(d["access_policy"]) > user_level:
                continue
            text = f"{d['filename']} {d['category']} {d['department']}".lower()
            if q_lower in text:
                filtered.append({
                    "doc_id": d["doc_id"],
                    "filename": d["filename"],
                    "department": d["department"],
                    "category": d["category"],
                    "academic_year": d["academic_year"],
                    "version": d["version"],
                    "access_policy": d["access_policy"],
                    "sections": d["section_count"]
                })

        return json.dumps({
            "query": query,
            "total_matches": len(filtered),
            "documents": filtered[:20]
        }, indent=2)

    @server.tool()
    def get_document(document_id: str) -> str:
        """Retrieve full details, metadata, and sections of an institutional document by ID.
        
        Args:
            document_id: Unique 12-character document identifier.
        """
        doc = meta_store.get_document_by_id(document_id)
        if not doc:
            return json.dumps({"error": f"Document '{document_id}' not found."}, indent=2)

        sections = meta_store.get_document_sections(document_id)
        return json.dumps({
            "doc_id": doc["doc_id"],
            "filename": doc["filename"],
            "department": doc["department"],
            "category": doc["category"],
            "academic_year": doc["academic_year"],
            "version": doc["version"],
            "access_policy": doc["access_policy"],
            "total_sections": len(sections),
            "sections": sections
        }, indent=2)

    @server.tool()
    def get_document_section(document_id: str, section_title_or_id: str) -> str:
        """Retrieve a specific section, regulation rule, or table from a document.
        
        Args:
            document_id: Unique document identifier.
            section_title_or_id: Title keyword or section ID (e.g. 'R.4', 'Attendance', 'doc_s1').
        """
        sections = meta_store.get_document_sections(document_id)
        matched = []
        search_term = section_title_or_id.lower()
        for s in sections:
            if search_term in s["section_title"].lower() or search_term in s["chunk_id"].lower() or search_term in s["raw_content"].lower():
                matched.append(s)

        if not matched:
            return json.dumps({"message": f"No section matching '{section_title_or_id}' found in document '{document_id}'."}, indent=2)

        return json.dumps({
            "doc_id": document_id,
            "matched_sections": matched
        }, indent=2)

    @server.tool()
    def search_policy(
        query: str,
        role: str = "student"
    ) -> str:
        """Search university policies, code of conduct, residential guidelines, and administrative rules.
        
        Args:
            query: Policy topic (e.g. 'attendance penalty', 'hostel curfew', 'malpractice punishment').
            role: User role ('student', 'faculty', 'admin').
        """
        res = search_engine.retrieve(
            query=f"policy regulation rule {query}",
            role=role,
            top_k=5
        )
        return json.dumps({
            "policy_topic": query,
            "evidence": res["evidence_package"]["evidence_text"],
            "sources": res["evidence_package"]["sources"]
        }, indent=2)

    @server.tool()
    def search_academic_regulations(
        query: str,
        program: Optional[str] = "B.Tech",
        academic_year: Optional[str] = "2023"
    ) -> str:
        """Search official B.Tech/M.Tech/PhD ordinances, grading criteria, and attendance rules.
        
        Args:
            query: Regulation topic (e.g. 'R.4 attendance', 'minimum CGPA', 'supplementary exam').
            program: Degree program ('B.Tech', 'M.Tech', 'PhD').
            academic_year: Academic year ('2023', '2024', etc.).
        """
        full_query = f"{program or ''} Academic Regulations {query}"
        res = search_engine.retrieve(
            query=full_query,
            role="student",
            department_filter="Academic Admin Office",
            academic_year_filter=academic_year,
            top_k=5
        )
        return json.dumps({
            "program": program,
            "academic_year": academic_year,
            "regulations_found": len(res["evidence_package"]["sources"]),
            "evidence": res["evidence_package"]["evidence_text"],
            "sources": res["evidence_package"]["sources"]
        }, indent=2)

    @server.tool()
    def get_latest_circular(
        department: Optional[str] = None
    ) -> str:
        """Retrieve recent circulars, notifications, and academic notices.
        
        Args:
            department: Optional department name (e.g. 'Exam_Cell', 'Academic Admin Office').
        """
        res = search_engine.retrieve(
            query="circular notice important dates notification announcement",
            department_filter=department,
            role="student",
            top_k=4
        )
        return json.dumps({
            "department": department or "All Departments",
            "circulars": res["evidence_package"]["sources"],
            "content": res["evidence_package"]["evidence_text"]
        }, indent=2)

    @server.tool()
    def compare_document_versions(
        doc_id_1: str,
        doc_id_2: str,
        topic: Optional[str] = None
    ) -> str:
        """Compare two document versions (e.g., 2019 curriculum vs 2023 curriculum or policies) on a topic.
        
        Args:
            doc_id_1: Document ID of the earlier or reference document.
            doc_id_2: Document ID of the newer or comparative document.
            topic: Topic or course code to compare.
        """
        doc1 = meta_store.get_document_by_id(doc_id_1)
        doc2 = meta_store.get_document_by_id(doc_id_2)

        if not doc1 or not doc2:
            return json.dumps({"error": "One or both document IDs do not exist."}, indent=2)

        sec1 = meta_store.get_document_sections(doc_id_1)
        sec2 = meta_store.get_document_sections(doc_id_2)

        # Filter by topic if specified
        if topic:
            t = topic.lower()
            sec1 = [s for s in sec1 if t in s["section_title"].lower() or t in s["raw_content"].lower()]
            sec2 = [s for s in sec2 if t in s["section_title"].lower() or t in s["raw_content"].lower()]

        return json.dumps({
            "doc_1": {
                "id": doc1["doc_id"],
                "filename": doc1["filename"],
                "academic_year": doc1["academic_year"],
                "sections_matched": len(sec1),
                "sample_content": [s["raw_content"][:300] for s in sec1[:2]]
            },
            "doc_2": {
                "id": doc2["doc_id"],
                "filename": doc2["filename"],
                "academic_year": doc2["academic_year"],
                "sections_matched": len(sec2),
                "sample_content": [s["raw_content"][:300] for s in sec2[:2]]
            },
            "comparison_topic": topic or "Full Document"
        }, indent=2)

    @server.tool()
    def find_related_documents(document_id: str) -> str:
        """Find related documents sharing the same department, program, or academic category.
        
        Args:
            document_id: The base document ID.
        """
        doc = meta_store.get_document_by_id(document_id)
        if not doc:
            return json.dumps({"error": f"Document '{document_id}' not found."}, indent=2)

        related = meta_store.get_all_documents(department=doc["department"])
        filtered = [
            {
                "doc_id": d["doc_id"],
                "filename": d["filename"],
                "category": d["category"],
                "academic_year": d["academic_year"]
            }
            for d in related if d["doc_id"] != document_id
        ]

        return json.dumps({
            "base_document": doc["filename"],
            "department": doc["department"],
            "related_documents_count": len(filtered),
            "related_documents": filtered[:10]
        }, indent=2)

    @server.tool()
    def ask_institutional_question(
        question: str,
        role: str = "student",
        department: Optional[str] = None
    ) -> str:
        """Ask any natural language question about Amrita Vishwa Vidyapeetham.
        Produces a grounded answer strictly backed by citations and verification checks.
        
        Args:
            question: The user's question (e.g. 'What is the attendance requirement for B.Tech students?').
            role: Access role ('student', 'faculty', 'admin', 'public').
            department: Optional target department.
        """
        ans_data = grounded_engine.generate_answer(
            question=question,
            role=role,
            department=department
        )
        return json.dumps(ans_data, indent=2)

    @server.tool()
    def get_system_stats() -> str:
        """Retrieve real-time observability metrics: total documents, chunks, latency, and grounding score."""
        stats = meta_store.get_system_stats()
        return json.dumps(stats, indent=2)

    @server.tool()
    def run_rag_evaluation() -> str:
        """Execute the automated evaluation suite calculating Recall@K, MRR, and latency."""
        eval_results = eval_framework.run_benchmark()
        return json.dumps(eval_results, indent=2)

    # ---------------------------------------------------------
    # MCP RESOURCES (Static / Structured Institutional Views)
    # ---------------------------------------------------------

    @server.resource("amritagpt://system/stats")
    def resource_system_stats() -> str:
        """System observability metrics and index status."""
        return json.dumps(meta_store.get_system_stats(), indent=2)

    @server.resource("amritagpt://policies/current")
    def resource_current_policies() -> str:
        """Summary of active policies across Amrita Vishwa Vidyapeetham."""
        docs = meta_store.get_all_documents()
        policy_docs = [
            {"filename": d["filename"], "department": d["department"], "year": d["academic_year"]}
            for d in docs if "regulation" in d["filename"].lower() or "policy" in d["filename"].lower()
        ]
        return json.dumps({"current_policies_count": len(policy_docs), "policies": policy_docs[:25]}, indent=2)

    @server.resource("amritagpt://departments/list")
    def resource_departments_list() -> str:
        """List of all university departments and their document repositories."""
        stats = meta_store.get_system_stats()
        return json.dumps(stats.get("departments", {}), indent=2)

    # ---------------------------------------------------------
    # MCP PROMPTS (Reusable Workflows for AI Agents)
    # ---------------------------------------------------------

    @server.prompt("academic_advising")
    def prompt_academic_advising(student_query: str) -> str:
        """Standardized prompt for advising students on curriculum, grades, and graduation rules."""
        return (
            f"You are an Institutional Academic Advisor for Amrita Vishwa Vidyapeetham.\n"
            f"Use the AmritaGPT MCP tools (`search_academic_regulations`, `ask_institutional_question`) "
            f"to verify facts before advising.\n"
            f"Student Query: {student_query}\n"
            f"Instructions:\n"
            f"1. Check the official Academic Regulations (e.g. attendance, pass criteria, credit limits).\n"
            f"2. Cite the exact regulation section (e.g. R.4 Attendance, R.14 Grading).\n"
            f"3. Provide empathetic and clear academic guidance."
        )

    @server.prompt("policy_compliance_check")
    def prompt_policy_compliance(scenario: str) -> str:
        """Standardized prompt for auditing institutional compliance against regulations."""
        return (
            f"You are an Institutional Compliance Officer for Amrita Vishwa Vidyapeetham.\n"
            f"Audit the following scenario using the `search_policy` and `search_academic_regulations` tools:\n"
            f"Scenario: {scenario}\n"
            f"Checklist:\n"
            f"1. Is the action permitted under current university regulations?\n"
            f"2. What are the applicable penalties or procedures?\n"
            f"3. Quote the specific document and page number."
        )

    return server


# Global instance
mcp_server = create_amrita_mcp_server()

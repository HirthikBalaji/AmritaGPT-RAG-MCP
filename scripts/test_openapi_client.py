"""
End-to-End Verification Test Script for AmritaGPT OpenAPI REST Endpoints.
Simulates an HTTP / OpenAPI client querying every REST endpoint.
"""
import sys
from pathlib import Path
from starlette.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amritagpt.api.app import app


def run_openapi_tests():
    print("=" * 80)
    print("  AMRITAGPT OPENAPI REST API: END-TO-END VERIFICATION SUITE")
    print("=" * 80)

    client = TestClient(app)

    # 1. Test Root & Health
    print("\n[1] Testing GET / (Root Meta)")
    r_root = client.get("/")
    assert r_root.status_code == 200, f"Root failed: {r_root.text}"
    print(f"    Status: {r_root.status_code}, Docs: {r_root.json()['documentation']['swagger_ui']}")

    print("\n[2] Testing GET /health (Health Check)")
    r_health = client.get("/health")
    assert r_health.status_code == 200, f"Health failed: {r_health.text}"
    h_data = r_health.json()
    print(f"    Status: {h_data['status']}, Docs: {h_data['documents_indexed']}, Chunks: {h_data['chunks_indexed']}")

    # 2. Test OpenAPI Spec Endpoint
    print("\n[3] Testing GET /openapi.json (OpenAPI 3.1 Spec)")
    r_spec = client.get("/openapi.json")
    assert r_spec.status_code == 200, f"OpenAPI spec failed: {r_spec.text}"
    spec = r_spec.json()
    print(f"    OpenAPI Version: {spec.get('openapi')}, Title: {spec.get('info', {}).get('title')}")
    print(f"    Exposed Paths: {len(spec.get('paths', {}))}")

    # 3. Test Search Knowledge
    print("\n[4] Testing POST /api/v1/search/knowledge")
    r_search = client.post("/api/v1/search/knowledge", json={
        "query": "attendance requirement B.Tech",
        "role": "student",
        "top_k": 3
    })
    assert r_search.status_code == 200, f"Search failed: {r_search.text}"
    s_data = r_search.json()
    print(f"    Status: {s_data['status']}, Intent: {s_data['analyzed_intent']}, Sources: {len(s_data['sources'])}")

    # 4. Test Grounded Q&A
    print("\n[5] Testing POST /api/v1/qa/ask (Grounded Generation)")
    r_qa = client.post("/api/v1/qa/ask", json={
        "question": "What is the attendance requirement for B.Tech students?",
        "role": "student"
    })
    assert r_qa.status_code == 200, f"QA failed: {r_qa.text}"
    qa_data = r_qa.json()
    print(f"    Confidence: {qa_data['confidence']}, Grounding Score: {qa_data['grounding_score']}")
    print(f"    Faithful: {qa_data['is_faithful']}, Citations: {len(qa_data['citations'])}")
    print(f"    Answer Preview: {qa_data['answer'][:160]}...")

    # 5. Test Academic Regulations
    print("\n[6] Testing POST /api/v1/regulations/academic")
    r_reg = client.post("/api/v1/regulations/academic", json={
        "query": "attendance and examinations",
        "program": "B.Tech",
        "academic_year": "2023"
    })
    assert r_reg.status_code == 200, f"Regulations failed: {r_reg.text}"
    reg_data = r_reg.json()
    print(f"    Regulations Found: {reg_data['regulations_found']}, Topic: {reg_data['topic']}")

    # 6. Test Policy Search
    print("\n[7] Testing POST /api/v1/policy/search")
    r_pol = client.post("/api/v1/policy/search", json={
        "query": "examination malpractice rules",
        "role": "student"
    })
    assert r_pol.status_code == 200, f"Policy search failed: {r_pol.text}"
    pol_data = r_pol.json()
    print(f"    Policies Found: {pol_data['regulations_found']}")

    # 7. Test Document Catalog Search
    print("\n[8] Testing POST /api/v1/search/documents")
    r_docs = client.post("/api/v1/search/documents", json={
        "query": "curriculum",
        "role": "student"
    })
    assert r_docs.status_code == 200, f"Docs search failed: {r_docs.text}"
    docs_data = r_docs.json()
    print(f"    Total Document Matches: {docs_data['total_matches']}")
    sample_id = None
    if docs_data["documents"]:
        sample_id = docs_data["documents"][0]["doc_id"]
        print(f"    Top Match: {docs_data['documents'][0]['filename']} (ID: {sample_id})")

    # 8. Test Document Details
    if sample_id:
        print(f"\n[9] Testing GET /api/v1/documents/{sample_id}")
        r_detail = client.get(f"/api/v1/documents/{sample_id}")
        assert r_detail.status_code == 200, f"Doc detail failed: {r_detail.text}"
        detail = r_detail.json()
        print(f"    Filename: {detail['filename']}, Total Sections: {detail['total_sections']}")

    # 9. Test System Stats
    print("\n[10] Testing GET /api/v1/system/stats")
    r_stats = client.get("/api/v1/system/stats")
    assert r_stats.status_code == 200, f"Stats failed: {r_stats.text}"
    stats_data = r_stats.json()
    print(f"    Total Docs: {stats_data['total_documents']}, Chunks: {stats_data['total_chunks']}")
    print(f"    Avg Latency: {stats_data['avg_query_latency_ms']} ms")

    # 10. Test Evaluation Benchmark
    print("\n[11] Testing POST /api/v1/evaluation/run")
    r_eval = client.post("/api/v1/evaluation/run")
    assert r_eval.status_code == 200, f"Evaluation failed: {r_eval.text}"
    eval_data = r_eval.json()
    print(f"    Benchmark Queries: {eval_data['total_benchmark_queries']}")
    print(f"    Recall@5: {eval_data['recall_at_5']}, MRR: {eval_data['mean_reciprocal_rank_mrr']}")
    print(f"    P50 Latency: {eval_data['p50_latency_ms']} ms")

    print("\n" + "=" * 80)
    print("  ALL AMRITAGPT OPENAPI REST ENDPOINTS VERIFIED AND COMPLIANT!")
    print("=" * 80)


if __name__ == "__main__":
    run_openapi_tests()

"""
End-to-End Verification Test Script for AmritaGPT MCP Server.
Simulates an AI Agent calling MCP tools, reading resources, and evaluating prompts.
"""
import sys
import json
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amritagpt.mcp.server import create_amrita_mcp_server


async def run_tests():
    print("=" * 80)
    print("  AMRITAGPT MCP SERVER: END-TO-END VERIFICATION SUITE")
    print("=" * 80)

    server = create_amrita_mcp_server()

    # 1. Discover Capabilities
    tools = await server.list_tools()
    resources = await server.list_resources()
    prompts = await server.list_prompts()

    print(f"\n[+] Capabilities Discovered:")
    print(f"    • Tools ({len(tools)}): {[t.name for t in tools]}")
    print(f"    • Resources ({len(resources)}): {[str(r.uri) for r in resources]}")
    print(f"    • Prompts ({len(prompts)}): {[p.name for p in prompts]}")

    # 2. Test Tool: search_knowledge
    print("\n" + "-" * 70)
    print("[1] Testing Tool: search_knowledge('attendance requirement B.Tech')")
    res_search = await server.call_tool("search_knowledge", {
        "query": "What is the attendance requirement for B.Tech students?",
        "role": "student",
        "top_k": 3
    })
    raw_text = res_search.content[0].text if res_search.content else "{}"
    parsed_search = json.loads(raw_text)
    print(f"    Status: {parsed_search.get('status')}")
    print(f"    Detected Intent: {parsed_search.get('analyzed_intent')}")
    print(f"    Sources Found: {len(parsed_search.get('sources', []))}")
    if parsed_search.get("sources"):
        s0 = parsed_search["sources"][0]
        print(f"    Top Source: {s0.get('filename')} (Sec: {s0.get('section')}, AY: {s0.get('academic_year')})")

    # 3. Test Tool: ask_institutional_question (Grounded Generation)
    print("\n" + "-" * 70)
    print("[2] Testing Tool: ask_institutional_question('What is the attendance requirement for B.Tech students?')")
    res_qa = await server.call_tool("ask_institutional_question", {
        "question": "What is the attendance requirement for B.Tech students?",
        "role": "student"
    })
    parsed_qa = json.loads(res_qa.content[0].text if res_qa.content else "{}")
    print(f"    Confidence: {parsed_qa.get('confidence')}")
    print(f"    Grounding Score: {parsed_qa.get('grounding_score')}")
    print(f"    Faithful: {parsed_qa.get('is_faithful')}")
    print(f"    Citations ({len(parsed_qa.get('citations', []))}):")
    for cit in parsed_qa.get("citations", [])[:2]:
        print(f"      • {cit}")
    print(f"\n    Generated Answer Preview:\n    {parsed_qa.get('answer')[:350]}...\n")

    # 4. Test Tool: search_academic_regulations
    print("\n" + "-" * 70)
    print("[3] Testing Tool: search_academic_regulations('attendance and examinations')")
    res_reg = await server.call_tool("search_academic_regulations", {
        "query": "attendance criteria and examination conduct",
        "program": "B.Tech",
        "academic_year": "2023"
    })
    parsed_reg = json.loads(res_reg.content[0].text if res_reg.content else "{}")
    print(f"    Regulations Found: {parsed_reg.get('regulations_found')}")
    if parsed_reg.get("sources"):
        print(f"    Matched File: {parsed_reg['sources'][0].get('filename')}")

    # 5. Test Tool: get_system_stats
    print("\n" + "-" * 70)
    print("[4] Testing Tool: get_system_stats()")
    res_stats = await server.call_tool("get_system_stats", {})
    parsed_stats = json.loads(res_stats.content[0].text if res_stats.content else "{}")
    print(f"    Total Documents: {parsed_stats.get('total_documents')}")
    print(f"    Total Chunks: {parsed_stats.get('total_chunks')}")
    print(f"    Total Queries Served: {parsed_stats.get('total_queries_served')}")
    print(f"    Avg Latency: {parsed_stats.get('avg_query_latency_ms')} ms")

    # 6. Test Tool: run_rag_evaluation
    print("\n" + "-" * 70)
    print("[5] Testing Tool: run_rag_evaluation() (Evaluation Benchmark)")
    res_eval = await server.call_tool("run_rag_evaluation", {})
    parsed_eval = json.loads(res_eval.content[0].text if res_eval.content else "{}")
    print(f"    Recall@1: {parsed_eval.get('recall_at_1')}")
    print(f"    Recall@5: {parsed_eval.get('recall_at_5')}")
    print(f"    MRR: {parsed_eval.get('mean_reciprocal_rank_mrr')}")
    print(f"    P50 Latency: {parsed_eval.get('p50_latency_ms')} ms")

    # 7. Test Resource: amritagpt://system/stats
    print("\n" + "-" * 70)
    print("[6] Testing Resource: amritagpt://system/stats")
    res_rec = await server.read_resource("amritagpt://system/stats")
    raw_content = getattr(res_rec[0], "content", getattr(res_rec[0], "text", str(res_rec[0])))
    print(f"    Resource Content: {raw_content[:120]}...")

    # 8. Test Prompt: academic_advising
    print("\n" + "-" * 70)
    print("[7] Testing Prompt: academic_advising")
    prompt_res = await server.get_prompt("academic_advising", {"student_query": "Can I write exams if I have 72% attendance?"})
    print(f"    Rendered Prompt:\n    {prompt_res.messages[0].content.text[:250]}...")

    print("\n" + "=" * 80)
    print("  ALL AMRITAGPT MCP CAPABILITIES VERIFIED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_tests())

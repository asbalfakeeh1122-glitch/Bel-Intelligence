import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.core.nlp_pipeline import NLPService
from tools.metrics_engine import calculate_evidence_recall, calculate_token_f1


async def run_semantic_benchmark():
    print("--- [ELITE SEMANTIC BENCHMARK START] ---")
    nlp = NLPService()
    # Isolating RAG models to save VRAM and avoid OOM on Windows
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(nlp.executor, nlp._load_rag_models)
    import torch

    torch.cuda.empty_cache()

    # Complex document with contradictory logic
    text = """
    ENVIRONMENTAL RISK REPORT 2026
    
    EXECUTIVE SUMMARY:
    The organization maintains high sustainability standards. While the 2024 goal was to reduce emissions by 40%, the 2025 status shows we failed to meet this target due to supply chain issues.
    
    SECTION 1: COMPLIANCE
    Quantum Solutions qualifies as a 'Large Emitter' under the Green Act, except in its EU operations where it is classified as a 'Carbon-Neutral Entity'.
    
    SECTION 2: PROHIBITIONS
    Standard vessels are not permitted to enter the protected reef area. However, research vessels are explicitly exempt from this restriction.
    """

    test_cases = [
        {
            "id": "NEG_01",
            "q": "Are research vessels restricted from entering the protected reef area?",
            "gt_answer": "No, research vessels are explicitly exempt from this restriction.",
            "gt_evidence": "research vessels are explicitly exempt from this restriction",
        },
        {
            "id": "CONTRADICTION_01",
            "q": "Is the organization's 2025 emissions target met?",
            "gt_answer": "No, we failed to meet this target.",
            "gt_evidence": "failed to meet this target",
        },
        {
            "id": "SEMANTIC_01",
            "q": "What is Quantum Solutions' classification in its EU operations?",
            "gt_answer": "Carbon-Neutral Entity",
            "gt_evidence": "classified as a 'Carbon-Neutral Entity'",
        },
    ]

    total_f1 = 0
    total_recall = 0

    for tc in test_cases:
        print(f"\n[TEST {tc['id']}] Q: {tc['q']}")
        res = nlp._rag_pipeline(text, tc["q"])

        f1 = calculate_token_f1(res["answer"], tc["gt_answer"])
        recall = calculate_evidence_recall(res["evidence"], tc["gt_evidence"])

        total_f1 += f1
        total_recall += recall

        print(f"A: {res['answer']}")
        print(f"E: {res['evidence']}")
        print(f"CONF: {res['confidence']}")
        print(f"METRICS -> F1: {f1:.4f}, Recall: {recall:.4f}")

    avg_f1 = total_f1 / len(test_cases)
    avg_recall = total_recall / len(test_cases)

    print("\n--- FINAL SCORES ---")
    print(f"AVERAGE TOKEN F1: {avg_f1:.4f}")
    print(f"AVERAGE EVIDENCE RECALL: {avg_recall:.4f}")

    if avg_f1 > 0.6 and avg_recall > 0.8:
        print("\n[RESULT] SEMANTIC ACCURACY TARGETS MET.")
    else:
        print("\n[RESULT] TARGETS NOT MET. ITERATION REQUIRED.")


if __name__ == "__main__":
    asyncio.run(run_semantic_benchmark())

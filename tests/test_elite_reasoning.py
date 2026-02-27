import asyncio

from app.core.nlp_pipeline import NLPService


async def run_reasoning_test():
    service = NLPService()
    print("[Test] Loading Models...")
    await service.load_models()

    scenarios = [
        {
            "name": "Social Disambiguation (Equality in Sports)",
            "text": "The new government policy focuses on gender equality and inclusivity in national sports programs. It aims to reduce discrimination and ensure equal rights for all athletes regardless of gender.",
            "categories": ["Sports", "Social", "Business"],
            "chat_q": "What is the primary social goal mentioned?",
            "expected_category": "Social",
        },
        {
            "name": "Implicit Environment (Carbon)",
            "text": "The company has committed to reducing its footprint by 20% over the next decade. We are investing in renewable solutions and tracking our emissions daily.",
            "categories": ["Environment", "Business", "Healthcare"],
            "chat_q": "How is the company helping the planet?",
            "expected_keywords": ["renewable", "emissions"],
        },
    ]

    for s in scenarios:
        print(f"\n--- Scenario: {s['name']} ---")

        # 1. Classification Test
        print(f"[Act] Analyzing categories...")
        analysis = await service.analyze(s["text"], s["categories"])
        top_cat = analysis["primary_domain"]
        print(f"[Result] Top Category: {top_cat}")

        if "expected_category" in s:
            if top_cat == s["expected_category"]:
                print("✅ Category Match!")
            else:
                print(f"❌ Category Mismatch! Expected {s['expected_category']}")

        # 2. Chat/RAG Test
        print(f"[Act] Chat Question: {s['chat_q']}")
        chat_res = await service.ask(s["text"], s["chat_q"])
        print(f"[Result] Answer: {chat_res['answer']}")
        print(f"[Result] Evidence: {chat_res['evidence']}")

        if "expected_keywords" in s:
            found = any(k in chat_res["answer"].lower() for k in s["expected_keywords"])
            if found:
                print("✅ Found expected keywords in answer!")
            else:
                print(f"❌ Missing keywords {s['expected_keywords']}")


if __name__ == "__main__":
    asyncio.run(run_reasoning_test())

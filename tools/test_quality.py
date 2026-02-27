import requests
import json
import time

url = "http://localhost:8000/api/analyze"
headers = {'Content-Type': 'application/json'}

tests = [
    {
        "name": "Strict Finance / Business",
        "payload": {
            "text": "The latest Q3 earnings report for CyberTech Dynamics reveals a massive 45% increase in year-over-year revenue, totaling $1.2 billion. This growth is primarily attributed to their new enterprise cloud security suite. The board of directors approved a $500M share buyback program.",
            "categories": ["Business", "Technology", "Legal", "Finance", "Healthcare"]
        },
        "expected_primary": ["Finance", "Business"]
    },
    {
        "name": "Strict Healthcare / Science",
        "payload": {
            "text": "Phase III clinical trials of the novel mRNA therapeutic demonstrated a 72% reduction in tumor markers among patients with advanced staged melanoma. The FDA has fast-tracked the application for emergency use.",
            "categories": ["Business", "Technology", "Science", "Finance", "Healthcare"]
        },
        "expected_primary": ["Healthcare", "Science"]
    },
    {
        "name": "Ambiguous Tech / Law",
        "payload": {
            "text": "The European Union passed legislation enforcing strict data localization requirements for cloud providers. Failure to comply with the new GDPR mandate will result in fines up to 4% of global annual revenue.",
            "categories": ["Policy", "Technology", "Legal", "Finance", "Environment"]
        },
        "expected_primary": ["Legal", "Policy"]
    },
    {
        "name": "Strict Technology / AI",
        "payload": {
            "text": "The new open-source large language model utilizes a novel sparse-attention mechanism, significantly reducing VRAM requirements during inference. Benchmarks show it outperforms proprietary models on coding tasks.",
            "categories": ["Policy", "Technology", "Legal", "Finance", "Science"]
        },
        "expected_primary": ["Technology", "Science"]
    },
    {
        "name": "Strict Sports / Business",
        "payload": {
            "text": "The Premier League champion signed a record-breaking $150M sponsorship deal. The team's star forward also secured a personal endorsement contract with a major athletic apparel brand.",
            "categories": ["Sports", "Business", "Healthcare", "Finance"]
        },
        "expected_primary": ["Sports", "Business", "Finance"]
    }
]

print("=== RUNNING QUALITY ASSURANCE BENCHMARKS ===\n")

for i, test in enumerate(tests):
    print(f"Test {i+1}: {test['name']}")
    start = time.time()
    
    try:
        response = requests.post(url, json=test['payload'], headers=headers)
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            primary = data.get("primary_domain")
            confidence = "High" if data.get("evidence_quotes") else "Low" # Proxy check
            
            print(f"  -> Latency: {duration:.2f}s")
            print(f"  -> Analyzed Primary Domain: {primary} (Expected: {test['expected_primary']})")
            print(f"  -> Secondary Domains: {data.get('secondary_domains')}")
            print(f"  -> Neural Reasoning: {data.get('reasoning')}")
            
            if primary in test['expected_primary']:
                print("  -> Veredict: PASS [OK]\n")
            else:
                print("  -> Veredict: FAIL [ERR] (Classification Drift Detected)\n")
                
        else:
             print(f"  -> Error: {response.text}\n")
             
    except Exception as e:
        print(f"  -> Connection Error: {e}\n")
    
print("=== QA COMPLETE ===")

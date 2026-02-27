import requests
import time

url = "http://localhost:8000/api/chat"
headers = {'Content-Type': 'application/json'}

context_text = """
The Global Data Protection Framework (GDPF) outlines strict regulations for entities handling sensitive information. 
Under Article 4, a "Critical Data Processor" is defined as any enterprise, regardless of revenue, that processes more than 10 million biometric records annually.
Quantum Solutions is a small enterprise founded in 2022 with only 15 employees and $2M in annual revenue. 
However, due to their specialized facial recognition software, Quantum Solutions processes approximately 15 million biometric records per year for various municipal clients.
The compliance deadline for Critical Data Processors to implement Level 3 encryption is December 31, 2026. 
Failure to meet this deadline will result in an immediate suspension of operating licenses and a fine of $500,000.
"""

tests = [
    {
        "name": "Multi-hop Reasoning",
        "question": "Does Quantum Solutions qualify as a Critical Data Processor?",
    },
    {
        "name": "Explicit Extraction (Number)",
        "question": "How many biometric records does Quantum Solutions process annually?",
    },
    {
        "name": "Explicit Extraction (Date / Penalty)",
        "question": "What is the compliance deadline and the fine for failing to meet it?",
    }
]

print("=== RUNNING REASONING & RECALL BENCHMARKS ===\n")

for i, test in enumerate(tests):
    print(f"Test {i+1}: {test['name']}")
    print(f"Q: {test['question']}")
    
    payload = {
        "context": context_text,
        "question": test['question']
    }
    
    start = time.time()
    try:
        response = requests.post(url, json=payload, headers=headers)
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            print(f"  -> Latency: {duration:.2f}s")
            print(f"  -> Intent Classified: {data.get('intent')}")
            print(f"  -> Answer: {data.get('answer')}")
            print(f"  -> Evidence: {data.get('evidence')}\n")
        else:
             print(f"  -> Error: {response.text}\n")
             
    except Exception as e:
        print(f"  -> Connection Error: {e}\n")
    
print("=== QA COMPLETE ===")

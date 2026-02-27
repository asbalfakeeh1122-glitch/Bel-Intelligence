import requests
import time

url = "http://localhost:8000/api/chat"
headers = {'Content-Type': 'application/json'}

# Generate a large document (50k+ characters)
filler_text = "The quick brown fox jumps over the lazy dog. " * 50 # ~2k characters
parts = []
for i in range(25):
    parts.append(f"Section {i}: {filler_text}")

# Inject a specific fact at the very end
parts.append("Special Note: The secret key for the high-priority vault is 'NEBULA-9' and the authorization expires on January 1, 2030.")

long_context = "\n\n".join(parts)
print(f"Generated document size: {len(long_context)} characters")

question = "What is the secret key for the vault and when does the authorization expire?"

print(f"Q: {question}")

payload = {
    "context": long_context,
    "question": question
}

start = time.time()
try:
    response = requests.post(url, json=payload, headers=headers)
    duration = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        print(f"  -> Latency: {duration:.2f}s")
        print(f"  -> Intent: {data.get('intent')}")
        print(f"  -> Answer: {data.get('answer')}")
        print(f"  -> Evidence: {data.get('evidence')}")
        
        # Verify correctness
        if "NEBULA-9" in data.get('answer') and "2030" in data.get('answer'):
            print("\n[SUCCESS] Long-context recall and composite extraction verified.")
        else:
            print("\n[FAILURE] Missing key details in answer.")
    else:
        print(f"  -> Error: {response.status_code} - {response.text}")
        
except Exception as e:
    print(f"  -> Connection Error: {e}")

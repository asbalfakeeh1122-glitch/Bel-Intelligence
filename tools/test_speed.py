import requests
import time

url = "http://localhost:8000/api/analyze"
payload = {
    "text": "The latest Q3 earnings report for CyberTech Dynamics reveals a massive 45% increase in year-over-year revenue, totaling $1.2 billion. This growth is primarily attributed to their new enterprise cloud security suite. However, legal teams are currently battling two patent infringement lawsuits related to the core encryption algorithms used in the suite. The stock price has surged by 15% following the announcement.",
    "categories": ["Business", "Technology", "Legal", "Finance"]
}
headers = {'Content-Type': 'application/json'}

print("Sending request to Neural Engine...")
start_time = time.time()
response = requests.post(url, json=payload, headers=headers)
end_time = time.time()

print(f"Status Code: {response.status_code}")
print(f"\n--- SUCCESS ---")
print(f"Latency: {end_time - start_time:.2f} seconds")

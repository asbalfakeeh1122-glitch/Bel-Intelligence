from transformers import pipeline

print("Loading QA...")
qa = pipeline("question-answering", model="valhalla/longformer-base-4096-finetuned-squadv1")
context = """
The Global Data Protection Framework (GDPF) outlines strict regulations for entities handling sensitive information. 
Under Article 4, a "Critical Data Processor" is defined as any enterprise, regardless of revenue, that processes more than 10 million biometric records annually.
Quantum Solutions is a small enterprise founded in 2022 with only 15 employees and $2M in annual revenue. 
However, due to their specialized facial recognition software, Quantum Solutions processes approximately 15 million biometric records per year for various municipal clients.
The compliance deadline for Critical Data Processors to implement Level 3 encryption is December 31, 2026. 
Failure to meet this deadline will result in an immediate suspension of operating licenses and a fine of $500,000.
"""

q1 = "Does Quantum Solutions qualify as a Critical Data Processor?"
q2 = "What is the compliance deadline and the fine for failing to meet it?"

print(f"\nQ1: {q1}")
res1 = qa(question=q1, context=context, top_k=3)
for r in res1:
    print(r)

print(f"\nQ2: {q2}")
res2 = qa(question=q2, context=context, top_k=3)
for r in res2:
    print(r)

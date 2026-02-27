---
title: Bel Intelligence
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Bel Intelligence

> **⚠️ NOTE: This project is currently a Work In Progress (WIP).** Features and documentation are actively being updated.

[![Live Demo](https://img.shields.io/badge/Live_Website-Click_Here-success?style=for-the-badge)](https://your-live-website-url-here.com)

![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-brightgreen)
![Build: Passing](https://img.shields.io/badge/Build-Passing-blue)
![Architecture: Singleton](https://img.shields.io/badge/Architecture-Singleton-orange)

A high-precision, production-grade asynchronous service for real-time NLP inference on unstructured documentation. Engineered for enterprise-level performance using the **Singleton Service Pattern** and **Neural reasoning agent**.

## Key Capabilities

- **Semantic Reasoning Agent**: Interactive RAG-based chat with intent-locked reasoning.
- **Precision Evidence Highlighting**: Verbatim neural evidence for every automated decision.
- **Multi-Label Domain Classification**: High-tier zero-shot classification with contextual boosting.
- **Quantitative Extraction**: Automated discovery of financial, statistical, and temporal insights.
- **Architecture**: Models are loaded once at startup into memory, ensuring <1s latency for subsequent operations.

## Model Inventory

| Core Task | Neural Engine | RAM | Latency |
| :--- | :--- | :--- | :--- |
| **Reasoning & QA** | `deepset/roberta-base-squad2` | ~450 MB | < 0.6s |
| **Zero-shot Engine** | `facebook/bart-large-mnli` | ~1.5 GB | ~1.5 - 3s |
| **Sentiment & XAI** | `distilbert-base-uncased` | ~300 MB | < 0.3s |
| **Summarization** | `sshleifer/distilbart-cnn-12-6`| ~1 GB   | ~1 - 2s |

## Technical Architecture

- **Service Layer**: FastAPI (Async Engine)
- **Model Management**: Singleton Pattern (Memory Efficient)
- **Explainability (XAI)**: Integrated Gradients via Captum
- **Frontend**: Gravity-Modern Vanilla CSS/JS Dashboard

## Rapid Deployment

### Docker (Production)
Ensure you have Docker installed and run:
```bash
# 1. Build world-tier stable image
docker build -t ai-doc-intelligence:1.0.0 .

# 2. Launch with healthchecks and resource limits
docker-compose up -d
```

### Monitoring & Health
Once running, you can monitor the service via:
- **Health Check**: `GET /health` (`{"status": "healthy"}`)
- **Metrics**: `GET /metrics` (Prometheus format)
- **API Docs**: `GET /docs` (OpenAPI)

## Security & Production Readiness
The service has been hardened for production environments:
- **CORS Protection**: Restricted origin policy to prevent unauthorized cross-site requests.
- **DoS Mitigation**: Forced 10MB file size limit on all uploads to protect server memory.
- **Information Masking**: Internal system traces are suppressed in production mode to prevent information leakage.
- **Resource Management**: Singleton model loading prevents redundant RAM usage.

### Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch engine
python -m uvicorn app.main:app --reload
```

##  Expert Analysis Protocol
The service automatically classifies user questions into specialized intents:
1. **NUMERIC**: Strict quantitative audit.
2. **FACT**: Verbatim factual retrieval.
3. **ANALOGY**: Abstract functional mapping.
4. **EVALUATION**: Conclusion-based assessment.
5. **EVIDENCE**: Direct neural verification.

---


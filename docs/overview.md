# AI Document Intelligence - Project Overview

## Core Purpose
This website is a production-grade **AI Document Intelligence** platform. It allows users to upload complex documents (PDF, Word, Text) and perform deep semantic analysis, automated classification, and interactive question-answering.

## Technology Stack
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/) (Python) - High-performance, asynchronous web framework.
- **Frontend**: Custom **Gravity-Modern UI** using HTML/Jinja2 templates and modern CSS.
- **NLP Engine**: Powered by `transformers`, `sentence-transformers`, and `PyTorch`. It uses state-of-the-art models for document embedding and classification.
- **Utilities**: 
  - `PyPDF2` & `python-docx` for document parsing.
  - `Captum` for Explainable AI (XAI) features.
  - `Pydantic` for strict data validation.

## Key Features
1. **Multi-Label Classification**: Analyzes documents and maps them to domains (Legal, Finance, Technical, etc.) with confidence scores.
2. **Interactive Chat (RAG)**: Users can ask questions about the document context and receive evidence-backed answers.
3. **Semantic Reasoning**: Handles complex queries that require cross-topic understanding and implicit logic.
4. **Export & Analytics**: Ability to visualize analysis profiles and export results.

## Project Structure
- `/app`: The heart of the application.
  - `/api`: RESTful endpoints for analysis and chat.
  - `/core`: The NLP pipeline logic and model management.
  - `/static` & `/templates`: The web UI assets and pages.
- `/tests`: Current validation suite for integration and unit testing.
- `/benchmarks`: Performance testing and edge-case validation.

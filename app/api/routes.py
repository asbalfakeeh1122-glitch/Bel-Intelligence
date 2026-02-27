import asyncio
import io
import json

import docx
import PyPDF2
from bs4 import BeautifulSoup
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.nlp_pipeline import logger, nlp_service
from app.schemas.request import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    FeedbackRequest,
)

router = APIRouter()


def log_analysis_event(event_type: str, details: dict):
    """Background monitoring for failed parses or low-confidence results."""
    log_file = "monitoring.jsonl"
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps({"type": event_type, **details}) + "\n")
    except Exception as e:
        logger.error(f"Monitoring Log Failure: {str(e)}")


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={500: {"model": ErrorResponse}, 422: {"description": "Validation Error"}},
    tags=["Analysis"],
    summary="Multi-Label Domain Classification",
    description="High-tier zero-shot classification with contextual boosting and numerical discovery.",
)
async def analyze_document(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    try:
        result = await nlp_service.analyze(request.text, request.categories)

        # Monitor intent consistency
        if result["primary_domain"] == "Unknown":
            background_tasks.add_task(
                log_analysis_event,
                "high_ambiguity",
                {"text_sample": request.text[:100]},
            )

        return AnalyzeResponse(**result)
    except Exception as e:
        logger.error(f"Analysis Endpoint Failure: {str(e)}")
        background_tasks.add_task(
            log_analysis_event, "analysis_failure", {"error": str(e)}
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "ANALYSIS_FAILURE",
                "detail": "Document analysis failed.",
                "status_code": 500,
            },
        )


@router.post(
    "/upload",
    responses={
        413: {"model": ErrorResponse, "description": "Document too large"},
        400: {"model": ErrorResponse, "description": "Extraction error"},
        500: {"model": ErrorResponse},
    },
    tags=["Ingestion"],
    summary="Secure Document Upload",
    description="Secure extraction from PDF, DOCX, and HTML with size limits (10MB).",
)
async def upload_document(file: UploadFile = File(...)):
    """Secure extraction from PDF, DOCX, and HTML with size limits."""
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    content = ""
    try:
        # Read the file content
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "DOC_TOO_LARGE",
                    "detail": f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024)}MB",
                    "status_code": 413,
                },
            )

        if file.filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            content = " ".join([page.extract_text() for page in reader.pages])
        elif file.filename.endswith(".docx"):
            doc = docx.Document(io.BytesIO(file_bytes))
            content = " ".join([para.text for para in doc.paragraphs])
        elif file.filename.endswith(".html") or file.filename.endswith(".htm"):
            soup = BeautifulSoup(file_bytes, "html.parser")
            content = soup.get_text(separator=" ")
        else:
            # Plain text fallback
            content = file_bytes.decode("utf-8")

        # Basic sanitization: strip excessive whitespace
        content = " ".join(content.split())

        if len(content) < 100:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "NOT_READABLE",
                    "detail": "Extracted text too short for analysis.",
                    "status_code": 400,
                },
            )

        return {"filename": file.filename, "text": content}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File Upload Failure [{file.filename}]: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PARSE_ERROR",
                "detail": f"File processing failed: {str(e)}",
                "status_code": 500,
            },
        )


@router.post(
    "/feedback",
    tags=["Monitoring"],
    summary="Submit User Feedback",
    description="Log user corrections for model alignment.",
)
async def log_feedback(request: FeedbackRequest):
    try:
        feedback_file = "feedback.jsonl"
        with open(feedback_file, "a") as f:
            f.write(json.dumps(request.dict()) + "\n")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Feedback Log Failure: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "FEEDBACK_FAILURE",
                "detail": "Feedback submission failed.",
                "status_code": 500,
            },
        )


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    tags=["Query"],
    summary="Expert Analysis Chat",
    description="Interactive RAG-based chat with intent-locked reasoning (FACT, NUMERIC, etc.).",
)
async def chat_with_document(request: ChatRequest):
    try:
        result = await nlp_service.ask(request.context, request.question)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Chat Endpoint Failure: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "QUERY_FAILURE",
                "detail": "Question analysis failed.",
                "status_code": 500,
            },
        )


@router.post(
    "/chat/stream",
    tags=["Query"],
    summary="Streaming Expert Analysis",
    description="Incremental Expert Analysis Streaming via Server-Sent Events (NDJSON).",
)
async def chat_stream(request: ChatRequest):
    """Incremental Expert Analysis Streaming."""

    async def event_generator():
        try:
            result = await nlp_service.ask(request.context, request.question)

            # Yield Intent detection
            yield json.dumps(
                {"status": "analyzing_intent", "intent": result.get("intent")}
            ) + "\n"
            await asyncio.sleep(0.5)

            # Yield the deterministic response
            yield json.dumps(result) + "\n"
        except Exception as e:
            logger.error(f"Streaming Chat Failure: {str(e)}")
            yield json.dumps(
                {"error": "Internal streaming error", "status": "failed"}
            ) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@router.post("/explain", response_model=ExplainResponse)
async def explain_sentiment(request: ExplainRequest):
    try:
        if request.target_label:
            result = await nlp_service.explain_zero_shot(
                request.text, request.target_label
            )
        else:
            result = await nlp_service.explain_sentiment(request.text)
        return ExplainResponse(**result)
    except Exception as e:
        logger.error(f"XAI Endpoint Failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Explanation generation failed.")

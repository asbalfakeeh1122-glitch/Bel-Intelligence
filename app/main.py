import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router as api_router
from app.core.nlp_pipeline import logger, nlp_service

# World-Tier: Production Versioning
VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    logger.info("Service Bootstrap: Initializing Neural Engine...")
    await nlp_service.load_models()
    yield
    # Clean up (if needed) on shutdown
    logger.info("Service Shutdown: Terminating model instances.")


app = FastAPI(
    title="BelIntelligence API",
    description="Production-grade async NLP inference service",
    version=VERSION,
    lifespan=lifespan,
)

# CORS Middleware
# In production, replace ["*"] with specific domains like ["https://yourdomain.com"]
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Tier-1 safety
    allow_headers=["*"],
)


# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    # Allow SAMEORIGIN so Hugging Face can embed the app
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    # Relax CSP for Hugging Face embedding
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' https://huggingface.co; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "frame-ancestors 'self' https://huggingface.co https://*.hf.space;"
    )
    return response


# Static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# Global Error Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Global Error: {str(exc)}",
        extra={"path": request.url.path, "method": request.method},
    )

    # Hide raw error details in production to prevent information leakage
    is_debug = os.getenv("DEBUG", "false").lower() == "true"
    detail = (
        str(exc)
        if is_debug
        else "An internal server error occurred. Please contact support."
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": detail,
            "status_code": 500,
        },
    )


@app.get("/health", tags=["Infrastructure"])
async def health_check():
    """Liveness probe for infrastructure monitoring."""
    return {"status": "healthy", "version": VERSION}


@app.get("/metrics", tags=["Infrastructure"])
async def metrics():
    """Prometheus-compatible metrics endpoint."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return JSONResponse(
        content=generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST
    )


# Standard validation error handler is built-in with FastAPI (returns 422 JSON)


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


app.include_router(api_router, prefix="/api")

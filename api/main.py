"""
api/main.py
FastAPI REST Layer — AI Style Advisor

Endpoints:
  POST /analyze  — Upload image, receive personalized style report.
  GET  /health   — Pipeline component health check.

Run:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

# Add pipeline to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from report import generate_report, pipeline_health  # noqa: E402

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

MAX_IMAGE_BYTES  = 10 * 1024 * 1024   # 10 MB
ALLOWED_FORMATS  = {"JPEG", "PNG", "WEBP"}
UPLOAD_TEMP_DIR  = Path("/tmp/style-advisor-uploads")
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

VISION_CHECKPOINT = os.getenv("VISION_CHECKPOINT", "models/vision/checkpoints/epoch_1.pt")
LLM_MODEL_PATH    = os.getenv("LLM_MODEL_PATH",    "models/llm/adapter")
LLM_BACKEND       = os.getenv("LLM_BACKEND",       "local")  # "local" | "ollama"
TOP_K_TRENDS      = int(os.getenv("TOP_K_TRENDS",  "5"))

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AI Style Advisor",
    description = "Self-hosted facial analysis and personalized style recommendations.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_methods     = ["GET", "POST"],
    allow_headers     = ["*"],
)

# ─── Response models ─────────────────────────────────────────────────────────

class TimingInfo(BaseModel):
    extraction_ms:  float
    retrieval_ms:   float
    generation_ms:  float
    total_ms:       float


class AnalyzeResponse(BaseModel):
    request_id:         str
    attributes:         Dict[str, str]
    trends_used:        List[str]
    report:             str
    processing_time_ms: float
    timing:             TimingInfo


class HealthResponse(BaseModel):
    status:         str
    vision_encoder: Dict[str, Any]
    qdrant:         Dict[str, Any]
    llm:            Dict[str, Any]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def validate_image(file: UploadFile, data: bytes) -> Image.Image:
    """
    Validate uploaded image: size, format, and readability.

    Args:
        file: The UploadFile object from FastAPI.
        data: Raw bytes read from the file.

    Returns:
        PIL Image object if valid.

    Raises:
        HTTPException: 400 if invalid.
    """
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image exceeds maximum allowed size of 10 MB "
                   f"(received {len(data) / 1024 / 1024:.1f} MB).",
        )

    try:
        img = Image.open(io.BytesIO(data))
        img.verify()   # Checks for corruption without fully decoding
        img = Image.open(io.BytesIO(data))  # Re-open after verify() closes stream
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read image file: {exc}",
        )

    fmt = (img.format or "").upper()
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format '{fmt}'. Allowed: {', '.join(ALLOWED_FORMATS)}.",
        )

    return img


def save_temp_image(img: Image.Image, request_id: str) -> Path:
    """Save a PIL Image to a temporary file and return its path."""
    temp_path = UPLOAD_TEMP_DIR / f"{request_id}.jpg"
    img.convert("RGB").save(str(temp_path), format="JPEG", quality=92)
    return temp_path


def cleanup_temp(path: Path) -> None:
    """Remove temporary upload file."""
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("Could not delete temp file %s: %s", path, exc)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze a facial image and return a personalized style report.",
    responses={
        400: {"description": "Invalid image (format, size, or corruption)."},
        422: {"description": "No file provided."},
        500: {"description": "Internal pipeline error."},
    },
)
async def analyze(
    image: UploadFile = File(..., description="Image file (JPEG / PNG / WEBP, max 10 MB)"),
) -> AnalyzeResponse:
    """
    **Upload a facial photo and receive a full personalized style report.**

    The pipeline runs three stages:
    1. Vision encoder extracts facial and style attributes.
    2. Qdrant retrieves relevant current fashion trend snippets.
    3. Fine-tuned LLM generates a markdown report.

    Returns attributes, the trends used, the full markdown report, and timing info.
    """
    request_id = str(uuid.uuid4())
    logger.info("[%s] Received /analyze request — file: %s", request_id, image.filename)

    # Read file
    try:
        data = await image.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        )

    # Validate
    pil_img = validate_image(image, data)

    # Save temp
    temp_path = save_temp_image(pil_img, request_id)

    try:
        result = generate_report(
            temp_path,
            vision_checkpoint = VISION_CHECKPOINT,
            llm_model_path    = LLM_MODEL_PATH,
            llm_backend       = LLM_BACKEND,
            top_k_trends      = TOP_K_TRENDS,
        )
    except FileNotFoundError as exc:
        logger.error("[%s] Model not found: %s", request_id, exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"A required model is not available: {exc}",
        )
    except RuntimeError as exc:
        logger.error("[%s] Pipeline error: %s", request_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )
    finally:
        cleanup_temp(temp_path)

    logger.info(
        "[%s] Report generated in %.0f ms.",
        request_id, result["timing"]["total_ms"],
    )

    return AnalyzeResponse(
        request_id         = request_id,
        attributes         = result["attributes"],
        trends_used        = result["trends_used"],
        report             = result["report"],
        processing_time_ms = result["timing"]["total_ms"],
        timing             = TimingInfo(**result["timing"]),
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Check the health of all pipeline components.",
)
async def health() -> HealthResponse:
    """
    Returns the operational status of each pipeline component:
    - Vision encoder checkpoint presence.
    - Qdrant connectivity and document count.
    - LLM model directory presence.
    """
    components = pipeline_health()
    all_ok = all(
        c.get("status") == "ok" for c in components.values()
    )
    return HealthResponse(
        status         = "ok" if all_ok else "degraded",
        vision_encoder = components.get("vision_encoder", {}),
        qdrant         = components.get("qdrant", {}),
        llm            = components.get("llm", {}),
    )


@app.get("/", include_in_schema=False)
async def root():
    return JSONResponse({"message": "AI Style Advisor API v1.0 — visit /docs for API reference."})

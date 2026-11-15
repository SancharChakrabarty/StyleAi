"""
pipeline/report.py
Full Pipeline Orchestration

Ties together all three pipeline stages:
  Stage 1 — Vision encoder   (extractor.py)
  Stage 2 — Qdrant retrieval (retriever.py)
  Stage 3 — LLM generation   (recommender.py)

Public API:
    generate_report(image_path) -> dict

Can be run standalone for end-to-end testing.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from extractor   import extract_attributes
from retriever   import FashionTrendRetriever
from recommender import generate_recommendation

logger = logging.getLogger(__name__)

# ─── Module-level singletons ──────────────────────────────────────────────────
# Shared across API requests within the same worker process.
_retriever: Optional[FashionTrendRetriever] = None


def _get_retriever() -> FashionTrendRetriever:
    """Return cached retriever, instantiating on first call."""
    global _retriever
    if _retriever is None:
        _retriever = FashionTrendRetriever()
    return _retriever


# ─── Main pipeline function ───────────────────────────────────────────────────

def generate_report(
    image_path: Union[str, Path, Image.Image],
    vision_checkpoint: str = "models/vision/checkpoints/epoch_1.pt",
    llm_model_path:    str = "models/llm/merged",
    llm_backend:       str = "local",
    top_k_trends:      int = 5,
) -> Dict[str, Any]:
    """
    End-to-end pipeline: image → personalized style report.

    Stages:
      1. Extract facial/style attributes from image (Vision Encoder)
      2. Retrieve top-k relevant fashion trend snippets (Qdrant)
      3. Generate markdown report combining attributes + trends (LLM)

    Args:
        image_path:         Path to image, Path object, or PIL Image.
        vision_checkpoint:  Path to the fine-tuned vision encoder checkpoint.
        llm_model_path:     Path to the merged LLM model directory.
        llm_backend:        "local" or "ollama".
        top_k_trends:       Number of trend documents to retrieve.

    Returns:
        {
            "attributes":     {attr: value, ...},
            "trends_used":    ["trend text 1", ...],
            "report":         "## Face Shape Analysis\n...",
            "timing": {
                "extraction_ms":  float,
                "retrieval_ms":   float,
                "generation_ms":  float,
                "total_ms":       float,
            }
        }

    Raises:
        FileNotFoundError: If any required model artifact is missing.
        RuntimeError:      If any pipeline stage fails.
        ValueError:        If the image is unreadable or invalid.
    """
    pipeline_start = time.time()

    # ── Stage 1: Extract attributes ──────────────────────────────────────────
    logger.info("[Stage 1] Extracting facial attributes …")
    t0 = time.time()
    try:
        attributes: Dict[str, str] = extract_attributes(
            image_path,
            checkpoint_path=vision_checkpoint,
        )
    except Exception as exc:
        logger.error("Stage 1 failed: %s", exc)
        raise
    extraction_ms = (time.time() - t0) * 1000
    logger.info("[Stage 1] Done in %.0f ms. Attributes: %s", extraction_ms, attributes)

    # ── Stage 2: Retrieve trends ─────────────────────────────────────────────
    logger.info("[Stage 2] Retrieving fashion trends from Qdrant …")
    t0 = time.time()
    try:
        retriever = _get_retriever()
        trends: List[str] = retriever.retrieve(attributes, top_k=top_k_trends)
    except Exception as exc:
        logger.warning("Stage 2 failed (%s) — proceeding without trends.", exc)
        trends = []
    retrieval_ms = (time.time() - t0) * 1000
    logger.info("[Stage 2] Done in %.0f ms. Retrieved %d trends.", retrieval_ms, len(trends))

    # ── Stage 3: Generate report ─────────────────────────────────────────────
    logger.info("[Stage 3] Generating style report with LLM …")
    t0 = time.time()
    enriched_input = {
        "attributes":     attributes,
        "current_trends": trends,
    }
    try:
        report: str = generate_recommendation(
            enriched_input,
            model_path=llm_model_path,
            backend=llm_backend,
        )
    except Exception as exc:
        logger.error("Stage 3 failed: %s", exc)
        raise
    generation_ms = (time.time() - t0) * 1000
    total_ms      = (time.time() - pipeline_start) * 1000
    logger.info("[Stage 3] Done in %.0f ms.", generation_ms)
    logger.info("[Pipeline] Total time: %.0f ms.", total_ms)

    return {
        "attributes":  attributes,
        "trends_used": trends,
        "report":      report,
        "timing": {
            "extraction_ms":  round(extraction_ms, 1),
            "retrieval_ms":   round(retrieval_ms, 1),
            "generation_ms":  round(generation_ms, 1),
            "total_ms":       round(total_ms, 1),
        },
    }


# ─── Health check for all stages ─────────────────────────────────────────────

def pipeline_health() -> Dict[str, Any]:
    """
    Check the availability of all pipeline components.

    Returns:
        {
            "vision_encoder":  {"status": "ok" | "missing", "checkpoint": str},
            "qdrant":          {"status": "ok" | "error", "document_count": int},
            "llm":             {"status": "ok" | "missing", "model_path": str},
        }
    """
    health: Dict[str, Any] = {}

    # Vision encoder
    checkpoint = Path("models/vision/checkpoints/epoch_1.pt")
    health["vision_encoder"] = {
        "status":     "ok" if checkpoint.exists() else "missing",
        "checkpoint": str(checkpoint),
    }

    # Qdrant
    try:
        ret = _get_retriever()
        health["qdrant"] = ret.health()
    except Exception as exc:
        health["qdrant"] = {"status": "error", "detail": str(exc)}

    # LLM
    llm_dir = Path("models/llm/merged")
    health["llm"] = {
        "status":     "ok" if llm_dir.exists() else "missing",
        "model_path": str(llm_dir),
    }

    return health


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the full AI Style Advisor pipeline")
    parser.add_argument("image_path",   help="Path to input image")
    parser.add_argument("--checkpoint", default="models/vision/checkpoints/epoch_1.pt")
    parser.add_argument("--llm",        default="models/llm/merged")
    parser.add_argument("--backend",    default="local", choices=["local", "ollama"])
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--health",     action="store_true", help="Check pipeline health only")
    args = parser.parse_args()

    if args.health:
        print(json.dumps(pipeline_health(), indent=2))
    else:
        result = generate_report(
            args.image_path,
            vision_checkpoint=args.checkpoint,
            llm_model_path=args.llm,
            llm_backend=args.backend,
            top_k_trends=args.top_k,
        )
        print("\n" + "═" * 60)
        print(result["report"])
        print("═" * 60)
        print(f"\n⏱  Timing: {json.dumps(result['timing'], indent=2)}")

"""
scripts/download_models.py

Downloads model files from Hugging Face Hub into the expected local paths.
Run once at container startup (called by Dockerfile CMD via start.sh).

Required env vars:
  HF_MODEL_REPO  — e.g. "Sanchar95/styleai-models"
  HF_TOKEN       — HF access token (only needed if repo is private)
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
HF_TOKEN      = os.environ.get("HF_TOKEN", None)

VISION_CHECKPOINT_LOCAL = Path("models/vision/checkpoints/epoch_1.pt")
ADAPTER_LOCAL_DIR       = Path("models/llm/adapter")

ADAPTER_FILES = [
    "models/llm/adapter/adapter_config.json",
    "models/llm/adapter/adapter_model.safetensors",
    "models/llm/adapter/tokenizer_config.json",
    "models/llm/adapter/tokenizer.json",
]


def download():
    if not HF_MODEL_REPO:
        logger.error("HF_MODEL_REPO env var is not set. Cannot download models.")
        sys.exit(1)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed.")
        sys.exit(1)

    kwargs = {"repo_id": HF_MODEL_REPO, "token": HF_TOKEN}

    # ── Vision checkpoint ────────────────────────────────────────────────────
    if VISION_CHECKPOINT_LOCAL.exists():
        logger.info("Vision checkpoint already present, skipping download.")
    else:
        logger.info("Downloading epoch_1.pt from %s …", HF_MODEL_REPO)
        VISION_CHECKPOINT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(
            **kwargs,
            filename      = "models/vision/checkpoints/epoch_1.pt",
            local_dir     = ".",
            local_dir_use_symlinks = False,
        )
        logger.info("Vision checkpoint saved to %s", path)

    # ── LoRA adapter files ───────────────────────────────────────────────────
    ADAPTER_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    for hf_path in ADAPTER_FILES:
        local_path = Path(hf_path)
        if local_path.exists():
            logger.info("%s already present, skipping.", local_path)
            continue
        logger.info("Downloading %s …", hf_path)
        hf_hub_download(
            **kwargs,
            filename      = hf_path,
            local_dir     = ".",
            local_dir_use_symlinks = False,
        )
        logger.info("Saved %s", local_path)

    logger.info("All model files ready.")


if __name__ == "__main__":
    download()

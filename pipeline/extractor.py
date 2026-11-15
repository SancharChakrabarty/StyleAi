"""
pipeline/extractor.py
Vision Attribute Extraction — Pipeline Stage 1
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = str(Path(__file__).parent.parent / "models" / "vision" / "checkpoints" / "epoch_1.pt")
DEFAULT_BACKBONE   = "facebook/dinov2-base"

_cached_extractor  = None
_cached_checkpoint: Optional[str] = None


def extract_attributes(
    image: Union[str, Path, Image.Image],
    checkpoint_path: str  = DEFAULT_CHECKPOINT,
    backbone_name:   str  = DEFAULT_BACKBONE,
    with_confidence: bool = False,
) -> Dict:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Vision checkpoint not found at '{checkpoint_path}'.\n"
            "Make sure epoch_1.pt is in models/vision/checkpoints/"
        )

    extractor = _get_cached_extractor(checkpoint_path, backbone_name)

    try:
        if with_confidence:
            return extractor.predict_with_confidence(image)
        return extractor.predict(image)
    except Exception as exc:
        logger.error("Attribute extraction failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Attribute extraction failed: {exc}") from exc


def _get_cached_extractor(checkpoint_path: str, backbone_name: str):
    global _cached_extractor, _cached_checkpoint

    if _cached_extractor is None or _cached_checkpoint != checkpoint_path:
        logger.info("Loading vision extractor from %s …", checkpoint_path)
        # Add models/vision to path so infer.py can import train.py
        vision_dir = str(Path(__file__).parent.parent / "models" / "vision")
        if vision_dir not in sys.path:
            sys.path.insert(0, vision_dir)
        from infer import StyleAttributeExtractor
        _cached_extractor  = StyleAttributeExtractor(checkpoint_path, backbone_name)
        _cached_checkpoint = checkpoint_path

    return _cached_extractor


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Extract style attributes from an image")
    parser.add_argument("image_path",   help="Full path to your photo, e.g. C:\\Users\\shaan\\photo.jpg")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backbone",   default=DEFAULT_BACKBONE)
    parser.add_argument("--confidence", action="store_true")
    args = parser.parse_args()

    result = extract_attributes(
        args.image_path,
        checkpoint_path=args.checkpoint,
        backbone_name=args.backbone,
        with_confidence=args.confidence,
    )
    print(json.dumps(result, indent=2))

"""
models/vision/infer.py
Vision Encoder Inference

Loads the fine-tuned DINOv2+LoRA checkpoint (epoch_1.pt) and returns
a clean attribute dict given an image path or PIL Image.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from PIL import Image
from transformers import AutoImageProcessor

from train import (
    VisionAttributeClassifier,
    BINARY_LABELS,
    HAIR_COLORS,
    ATTRIBUTES,
    ATTR_SLICES,
)

logger = logging.getLogger(__name__)


class StyleAttributeExtractor:
    def __init__(
        self,
        checkpoint_path: str,
        backbone_name:   str = "facebook/dinov2-base",
        device:          Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("Loading vision model on %s", self.device)

        # Must match Colab training: 224x224, no center crop
        self.processor = AutoImageProcessor.from_pretrained(
            backbone_name,
            size={"height": 224, "width": 224},
            crop_size={"height": 224, "width": 224},
            do_center_crop=False,
        )

        # Build model (pretrained=False — we load weights from checkpoint)
        self.model = VisionAttributeClassifier(backbone_name=backbone_name)

        # Load checkpoint — raw state_dict (no wrapper keys)
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            self.model.load_state_dict(state["model"])
        else:
            self.model.load_state_dict(state)

        self.model.to(self.device)
        self.model.eval()
        logger.info("Checkpoint loaded from %s", checkpoint_path)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, "Image.Image"]) -> Dict[str, str]:
        """
        Returns dict like:
        {
          "facial_hair": "yes",
          "eyeglasses":  "no",
          "hair_color":  "black",
          ...
        }
        """
        img = self._load_image(image)
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)

        binary_out, hair_out = self.model(pixel_values)

        result: Dict[str, str] = {}

        # Binary attributes — sigmoid threshold 0.5
        binary_probs = torch.sigmoid(binary_out[0])  # (9,)
        for i, attr in enumerate(BINARY_LABELS):
            result[attr] = "yes" if binary_probs[i].item() >= 0.5 else "no"

        # Hair color — argmax
        hair_idx         = hair_out[0].argmax().item()
        result["hair_color"] = HAIR_COLORS[hair_idx]

        return result

    @torch.no_grad()
    def predict_with_confidence(self, image: Union[str, Path, "Image.Image"]) -> Dict[str, Dict[str, float]]:
        """
        Returns per-attribute probabilities:
        {
          "facial_hair": {"no": 0.12, "yes": 0.88},
          "hair_color":  {"black": 0.72, "blonde": 0.05, ...},
          ...
        }
        """
        img = self._load_image(image)
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].to(self.device)

        binary_out, hair_out = self.model(pixel_values)

        result: Dict[str, Dict[str, float]] = {}

        binary_probs = torch.sigmoid(binary_out[0])
        for i, attr in enumerate(BINARY_LABELS):
            p = round(binary_probs[i].item(), 4)
            result[attr] = {"no": round(1 - p, 4), "yes": p}

        hair_probs = torch.softmax(hair_out[0], dim=0)
        result["hair_color"] = {
            color: round(hair_probs[j].item(), 4)
            for j, color in enumerate(HAIR_COLORS)
        }

        return result

    def _load_image(self, image) -> "Image.Image":
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif hasattr(image, "convert"):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Predict style attributes for an image")
    parser.add_argument("image_path",   help="Path to input image")
    parser.add_argument("--checkpoint", default="models/vision/checkpoints/epoch_1.pt")
    parser.add_argument("--backbone",   default="facebook/dinov2-base")
    parser.add_argument("--confidence", action="store_true")
    args = parser.parse_args()

    extractor = StyleAttributeExtractor(args.checkpoint, backbone_name=args.backbone)

    if args.confidence:
        output = extractor.predict_with_confidence(args.image_path)
    else:
        output = extractor.predict(args.image_path)

    print(json.dumps(output, indent=2))

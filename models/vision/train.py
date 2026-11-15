"""
models/vision/train.py
Vision Encoder — Model Definition

Matches the EXACT architecture trained on Colab:
- DINOv2 ViT-Base + LoRA (timm backbone)
- binary_head: Linear(768, 9)  — one output per binary attribute
- hair_head:   Linear(768, 5)  — one output per hair color class
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

import timm
from peft import LoraConfig, get_peft_model

# ── Binary attributes (9) — trained with BCEWithLogitsLoss ───────────────────
BINARY_LABELS: List[str] = [
    "facial_hair",
    "no_beard",
    "eyeglasses",
    "bald",
    "wavy_hair",
    "straight_hair",
    "high_cheekbones",
    "pale_skin",
    "male",
]

# ── Hair color classes (5) — trained with CrossEntropyLoss ───────────────────
HAIR_COLORS: List[str] = ["black", "blonde", "brown", "gray", "other"]

# ── Combined attribute schema for inference ───────────────────────────────────
# Each binary attribute: threshold sigmoid output at 0.5 → "yes" / "no"
# Hair color: argmax over 5 logits
ATTRIBUTES: Dict[str, List[str]] = {
    "facial_hair":     ["no", "yes"],
    "no_beard":        ["no", "yes"],
    "eyeglasses":      ["no", "yes"],
    "bald":            ["no", "yes"],
    "wavy_hair":       ["no", "yes"],
    "straight_hair":   ["no", "yes"],
    "high_cheekbones": ["no", "yes"],
    "pale_skin":       ["no", "yes"],
    "male":            ["no", "yes"],
    "hair_color":      ["black", "blonde", "brown", "gray", "other"],
}

# Slice map used by infer.py to decode outputs
# Binary attrs: index into BINARY_LABELS list
# Hair color:   separate head
ATTR_SLICES: Dict[str, Tuple[int, int]] = {
    attr: (i, i + 1) for i, attr in enumerate(BINARY_LABELS)
}
ATTR_SLICES["hair_color"] = (len(BINARY_LABELS), len(BINARY_LABELS) + len(HAIR_COLORS))

NUM_BINARY = len(BINARY_LABELS)   # 9
NUM_HAIR   = len(HAIR_COLORS)     # 5


class VisionAttributeClassifier(nn.Module):
    """
    Exact architecture used during Colab training.

    binary_head output shape: (B, 9)  — raw logits, use sigmoid > 0.5
    hair_head   output shape: (B, 5)  — raw logits, use argmax
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        lora_r:        int = 16,
        lora_alpha:    int = 32,
        lora_dropout:  float = 0.1,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=0,
            img_size=224,
        )

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            lora_dropout=lora_dropout,
            bias="none",
        )
        self.backbone    = get_peft_model(self.backbone, lora_cfg)
        self.binary_head = nn.Linear(768, NUM_BINARY)   # (B, 9)
        self.hair_head   = nn.Linear(768, NUM_HAIR)     # (B, 5)

    def forward(self, pixel_values: torch.Tensor):
        feats      = self.backbone(pixel_values)   # (B, 768)
        binary_out = self.binary_head(feats)       # (B, 9)
        hair_out   = self.hair_head(feats)         # (B, 5)
        return binary_out, hair_out

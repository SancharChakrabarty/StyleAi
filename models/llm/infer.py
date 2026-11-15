"""
models/llm/infer.py
LLM Recommendation Engine — Inference

Loads microsoft/phi-2 base + LoRA adapter (adapter_model.safetensors)
and generates personalized markdown style reports.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

BASE_MODEL_ID = "microsoft/phi-2"

SYSTEM_PROMPT = (
    "You are a professional stylist and grooming advisor. "
    "Given a person's facial and style attributes and current fashion trends, "
    "generate a detailed, encouraging, and personalized style report."
)

DEFAULT_GENERATION_KWARGS = dict(
    max_new_tokens     = 512,
    temperature        = 0.7,
    top_p              = 0.9,
    repetition_penalty = 1.1,
    do_sample          = True,
)


class StyleRecommender:
    """
    Loads microsoft/phi-2 base model + your LoRA adapter for style generation.

    adapter_path should point to the folder containing:
      adapter_config.json, adapter_model.safetensors, tokenizer*.json
    """

    def __init__(
        self,
        adapter_path: str = "models/llm/adapter",
        device: Optional[str] = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info("Loading tokenizer from adapter path: %s", adapter_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Loading base model: %s on %s", BASE_MODEL_ID, self.device)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype       = torch.float16,
            device_map        = "auto",
            trust_remote_code = True,
        )

        logger.info("Applying LoRA adapter from: %s", adapter_path)
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        logger.info("Model ready.")

    def _build_prompt(self, enriched_input: Dict[str, Any]) -> str:
        attrs  = enriched_input.get("attributes", {})
        trends = enriched_input.get("current_trends", [])

        attr_lines  = "\n".join(f"  {k}: {v}" for k, v in attrs.items())
        trend_lines = "\n".join(f"  - {t[:200]}" for t in trends[:5])

        return (
            f"Instruct: {SYSTEM_PROMPT}\n\n"
            f"Attributes:\n{attr_lines}\n\n"
            f"Current Trends:\n{trend_lines}\n\n"
            f"Output:"
        )

    @torch.no_grad()
    def generate(self, enriched_input: Dict[str, Any], **generation_kwargs) -> str:
        prompt = self._build_prompt(enriched_input)

        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 1024,
        ).to(self.device)

        gen_kwargs = {**DEFAULT_GENERATION_KWARGS, **generation_kwargs}

        t0 = time.time()
        output_ids = self.model.generate(
            **inputs,
            pad_token_id = self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        elapsed_ms = (time.time() - t0) * 1000

        prompt_len = inputs["input_ids"].shape[1]
        new_ids    = output_ids[0, prompt_len:]
        report     = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        logger.info("Generated report in %.0f ms (%d tokens)", elapsed_ms, len(new_ids))
        return report


class OllamaStyleRecommender:
    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434") -> None:
        self.model_name = model_name
        self.base_url   = base_url.rstrip("/")

    def generate(self, enriched_input: Dict[str, Any], **_) -> str:
        import urllib.request
        user_content = json.dumps(enriched_input, indent=2, ensure_ascii=False)
        payload = {
            "model":  self.model_name,
            "prompt": f"[INST] {SYSTEM_PROMPT}\n\n{user_content} [/INST]",
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 512},
        }
        data    = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data    = data,
            headers = {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        return result.get("response", "").strip()

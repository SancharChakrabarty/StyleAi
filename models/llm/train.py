"""
models/llm/train.py
LLM Fine-tuning Script (QLoRA via TRL SFTTrainer)

Fine-tunes Mistral-7B-Instruct-v0.3 in 4-bit quantization with QLoRA
on the synthetic style-report dataset generated in Phase 3.
After training, merges LoRA adapter weights into the base model for inference.
"""

import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────

BASE_MODEL       = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH     = "data/training_pairs/seed_pairs.jsonl"
CHECKPOINT_DIR   = "models/llm/checkpoints"
MERGED_MODEL_DIR = "models/llm/merged"

LORA_CONFIG = dict(
    r               = 16,
    lora_alpha      = 32,
    target_modules  = ["q_proj", "v_proj"],
    lora_dropout    = 0.05,
    bias            = "none",
    task_type       = TaskType.CAUSAL_LM,
)

TRAINING_CONFIG = dict(
    num_train_epochs      = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    learning_rate         = 2e-4,
    warmup_ratio          = 0.03,
    lr_scheduler_type     = "cosine",
    fp16                  = not torch.cuda.is_bf16_supported(),
    bf16                  = torch.cuda.is_bf16_supported(),
    logging_steps         = 10,
    save_strategy         = "epoch",
    save_total_limit      = 3,
    evaluation_strategy   = "no",
    max_grad_norm         = 1.0,
    report_to             = "none",
    dataloader_num_workers = 0,
    optim                 = "paged_adamw_8bit",
)

MAX_SEQ_LENGTH = 2048


# ─── BitsAndBytes 4-bit config ────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """Return 4-bit NF4 quantization config for QLoRA."""
    return BitsAndBytesConfig(
        load_in_4bit               = True,
        bnb_4bit_quant_type        = "nf4",
        bnb_4bit_use_double_quant  = True,
        bnb_4bit_compute_dtype     = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ),
    )


# ─── Chat formatting ──────────────────────────────────────────────────────────

def format_chat(example: dict, tokenizer) -> dict:
    """
    Convert HuggingFace chat-format messages list into a single formatted string
    using the tokenizer's apply_chat_template.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ─── Main training function ───────────────────────────────────────────────────

def train() -> None:
    """Load model, apply QLoRA, train, and save checkpoints + merged model."""
    logger.info("Loading tokenizer: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    # Mistral tokenizer has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info("Loading model in 4-bit quantization …")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config = get_bnb_config(),
        device_map          = "auto",
        trust_remote_code   = True,
    )

    model = prepare_model_for_kbit_training(model)

    logger.info("Applying QLoRA …")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ──
    logger.info("Loading dataset from %s", DATASET_PATH)
    raw_ds = load_dataset("json", data_files=DATASET_PATH, split="train")

    # Format to text
    raw_ds = raw_ds.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=raw_ds.column_names,
    )
    logger.info("Dataset size: %d examples", len(raw_ds))

    # ── Training args ──
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir = str(checkpoint_dir),
        **TRAINING_CONFIG,
    )

    # ── SFTTrainer ──
    trainer = SFTTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = raw_ds,
        tokenizer        = tokenizer,
        dataset_text_field = "text",
        max_seq_length   = MAX_SEQ_LENGTH,
        packing          = False,
    )

    logger.info("Starting training …")
    trainer.train()

    # Save final adapter
    final_adapter_path = checkpoint_dir / "final_adapter"
    trainer.save_model(str(final_adapter_path))
    tokenizer.save_pretrained(str(final_adapter_path))
    logger.info("Adapter saved to %s", final_adapter_path)

    # ── Merge adapter into base model ──
    merge_adapter(str(final_adapter_path))


def merge_adapter(adapter_path: str) -> None:
    """
    Merge LoRA adapter weights into the full base model and save to disk.
    The merged model is in float16 and can be loaded without BitsAndBytes.
    """
    logger.info("Merging adapter into base model …")

    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype  = torch.float16,
        device_map   = "auto",
    )
    merged_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = merged_model.merge_and_unload()

    merged_dir = Path(MERGED_MODEL_DIR)
    merged_dir.mkdir(parents=True, exist_ok=True)

    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    logger.info("Merged model saved to %s", merged_dir)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()

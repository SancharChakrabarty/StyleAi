"""Download model weights from Hugging Face Hub."""
import os, sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ROOT   = Path(__file__).parent.parent
HF_REPO     = os.getenv("HF_MODEL_REPO", "")       # YOUR_HF_USERNAME/styleai-models
HF_TOKEN    = os.getenv("HF_TOKEN", None)           # only if repo is private
SKIP_LLM    = os.getenv("SKIP_LLM_DOWNLOAD", "0") == "1"

def download_vision_checkpoint():
    dest = REPO_ROOT / "models" / "vision" / "checkpoints" / "epoch_1.pt"
    if dest.exists():
        print(f"[skip] epoch_1.pt already at {dest}")
        return
    if not HF_REPO:
        print("[warn] HF_MODEL_REPO not set — skipping vision download")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading epoch_1.pt from {HF_REPO} …")
    hf_hub_download(
        repo_id  = HF_REPO,
        filename = "models/vision/checkpoints/epoch_1.pt",
        local_dir= str(REPO_ROOT),
        token    = HF_TOKEN,
    )
    print("Done.")

def download_llm_adapter():
    # The adapter_config.json + tokenizer are already committed to the repo
    # Only the adapter_model.safetensors needs downloading (it's too large for git)
    safetensors = REPO_ROOT / "models" / "llm" / "adapter" / "adapter_model.safetensors"
    if safetensors.exists():
        print(f"[skip] adapter_model.safetensors already present")
        return
    if not HF_REPO:
        # No adapter weights → pipeline uses rule-based fallback automatically
        print("[info] No HF_MODEL_REPO — LLM will use rule-based fallback report")
        return
    print(f"Downloading LLM adapter from {HF_REPO} …")
    snapshot_download(
        repo_id       = HF_REPO,
        allow_patterns= ["models/llm/adapter/**"],
        local_dir     = str(REPO_ROOT),
        token         = HF_TOKEN,
    )
    print("Done.")

if __name__ == "__main__":
    download_vision_checkpoint()
    if not SKIP_LLM:
        download_llm_adapter()
    print("All downloads complete.")

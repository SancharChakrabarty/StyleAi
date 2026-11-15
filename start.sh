#!/bin/bash
set -e

echo "=== StyleAI startup ==="

# Download models from HF Hub if not already present
python scripts/download_models.py

echo "=== Starting API server ==="
exec uvicorn api.main:app --host 0.0.0.0 --port 7860

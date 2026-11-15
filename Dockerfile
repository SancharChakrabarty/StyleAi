# ── Hugging Face Spaces requires port 7860 ───────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/        ./api/
COPY pipeline/   ./pipeline/
COPY models/     ./models/
COPY data/       ./data/
COPY scripts/    ./scripts/
COPY start.sh    ./start.sh

RUN chmod +x start.sh

# HF Spaces runs as a non-root user — ensure dirs are writable
RUN mkdir -p models/vision/checkpoints models/llm/adapter /tmp/style-advisor-uploads \
    && chmod -R 777 models /tmp/style-advisor-uploads

ENV PYTHONPATH=/app
ENV PORT=7860

EXPOSE 7860

CMD ["./start.sh"]

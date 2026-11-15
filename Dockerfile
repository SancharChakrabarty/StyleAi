FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/        ./api/
COPY pipeline/   ./pipeline/
COPY models/     ./models/
COPY data/       ./data/
COPY scripts/    ./scripts/
COPY start.sh    ./start.sh
RUN chmod +x start.sh

RUN mkdir -p models/vision/checkpoints models/llm/adapter /tmp/style-advisor-uploads

ENV PYTHONPATH=/app
ENV PORT=8080
ENV VISION_CHECKPOINT=models/vision/checkpoints/epoch_1.pt
ENV LLM_BACKEND=local

EXPOSE 8080
CMD ["./start.sh"]

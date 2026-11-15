# AI Style Advisor

**Self-hosted · Fully offline after setup · No external AI APIs**

A complete end-to-end ML pipeline that accepts a facial photo and returns a personalized report covering facial analysis, grooming guidance, clothing color palettes, and current fashion trend recommendations — running entirely on your own hardware.

---

## 1. Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │               AI Style Advisor               │
                        │                                             │
   User Photo           │   ┌─────────────────────────────────────┐  │
       │                │   │   FastAPI  ·  api/main.py           │  │
       ▼                │   │   POST /analyze                     │  │
  [Frontend UI]─────────┼──▶│   GET  /health                      │  │
  frontend/index.html   │   └──────────────┬──────────────────────┘  │
                        │                  │                          │
                        │          pipeline/report.py                 │
                        │          generate_report()                  │
                        │                  │                          │
                        │      ┌───────────┼──────────────┐           │
                        │      │           │              │           │
                        │      ▼           ▼              ▼           │
                        │  ┌────────┐  ┌────────┐  ┌──────────┐     │
                        │  │Stage 1 │  │Stage 2 │  │ Stage 3  │     │
                        │  │Vision  │  │Qdrant  │  │  LLM     │     │
                        │  │Encoder │  │Trend DB│  │  Engine  │     │
                        │  │(DINOv2 │  │(vector │  │(Mistral  │     │
                        │  │+ LoRA) │  │search) │  │+ QLoRA)  │     │
                        │  └───┬────┘  └───┬────┘  └────┬─────┘     │
                        │      │           │             │            │
                        │      ▼           ▼             ▼            │
                        │  Attribute    Trend        Markdown         │
                        │    JSON     Snippets       Report           │
                        └─────────────────────────────────────────────┘
```

---

## 2. Hardware Requirements

| Task                              | Recommended Hardware           | Estimated Cost        |
|-----------------------------------|--------------------------------|-----------------------|
| Vision Encoder Fine-tuning        | 1× A100 40GB                   | ~$200–500 one-time    |
| LLM Fine-tuning (QLoRA 7B)       | 1× A100 or 2× RTX 3090        | ~$100–300 one-time    |
| Inference Serving (production)    | GPU with 16GB+ VRAM            | ~$300–800/month       |
| Development / Testing             | Google Colab Pro+ or RunPod    | Pay-per-use           |

> **Minimum for inference only (after training):** A GPU with ≥16 GB VRAM (RTX 3080 Ti, RTX 4080, A10G, etc.)

---

## 3. Step-by-Step Setup

### 3.1 Prerequisites

```bash
# Python 3.10 or higher
python --version

# CUDA toolkit (for GPU training/inference)
nvcc --version

# Docker (for Qdrant)
docker --version
docker compose version
```

### 3.2 Clone & Install

```bash
git clone https://github.com/your-org/ai-style-advisor.git
cd ai-style-advisor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3.3 Start Qdrant

```bash
cd docker
docker compose up -d
cd ..

# Verify Qdrant is running
curl http://localhost:6333/healthz
```

### 3.4 Ingest Fashion Trend Documents

```bash
# Ingest the 20 sample trend documents
python pipeline/retriever.py bootstrap

# Verify
python pipeline/retriever.py status
```

### 3.5 Train the Vision Encoder

You need a labeled image dataset (see Section 4). Once ready:

```bash
python models/vision/train.py \
  --backbone facebook/dinov2-base \
  --train_jsonl data/labeled/train.jsonl \
  --val_jsonl   data/labeled/val.jsonl \
  --epochs 10 \
  --batch_size 32 \
  --checkpoint_dir models/vision/checkpoints
```

To resume from a checkpoint:

```bash
python models/vision/train.py --resume
```

### 3.6 Generate LLM Training Data

```bash
python data/generate_seed_pairs.py
# Output: data/training_pairs/seed_pairs.jsonl (50 pairs)
```

### 3.7 Fine-tune the LLM

```bash
python models/llm/train.py
# Adapter saved to: models/llm/checkpoints/final_adapter/
# Merged model saved to: models/llm/merged/
```

### 3.8 Start the API Server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3.9 Open the Frontend

```
http://localhost:8000/static/index.html
```

Or serve the frontend separately:

```bash
cd frontend
python -m http.server 3000
# then open http://localhost:3000
```

---

## 4. Labeling Your Own Data (Vision Encoder)

### Dataset Format

Create a JSONL file where each line represents one labeled image:

```json
{
  "image_path": "data/raw/001.jpg",
  "labels": {
    "face_shape":           "oval",
    "skin_tone":            "medium",
    "hair_texture":         "wavy",
    "hair_length":          "short",
    "facial_hair":          "stubble",
    "eye_shape":            "almond",
    "jaw_type":             "defined",
    "cheekbone_prominence": "medium"
  }
}
```

### Attribute Reference

| Attribute              | Valid Values                                         |
|------------------------|------------------------------------------------------|
| `face_shape`           | oval, round, square, heart, oblong, diamond          |
| `skin_tone`            | fair, light, medium, olive, tan, deep                |
| `hair_texture`         | straight, wavy, curly, coily                         |
| `hair_length`          | short, medium, long                                  |
| `facial_hair`          | none, stubble, beard, mustache                       |
| `eye_shape`            | almond, round, hooded, monolid, upturned             |
| `jaw_type`             | soft, defined, sharp                                 |
| `cheekbone_prominence` | low, medium, high                                    |

### Recommended Labeling Tools

- **Label Studio** (free, self-hosted): [labelstud.io](https://labelstud.io)
- **CVAT** (free, self-hosted): [cvat.ai](https://cvat.ai)
- **Scale AI** (paid, managed): [scale.com](https://scale.com)

### Recommended Volume

- **Minimum** for meaningful training: 2,000 labeled images
- **Recommended** for production quality: 10,000–50,000 labeled images
- **Train/val split:** 90% train / 10% val

### Split your JSONL

```python
import json, random

with open('data/labeled/all.jsonl') as f:
    data = [json.loads(l) for l in f]

random.shuffle(data)
split = int(len(data) * 0.9)

with open('data/labeled/train.jsonl', 'w') as f:
    for d in data[:split]:    f.write(json.dumps(d) + '\n')

with open('data/labeled/val.jsonl', 'w') as f:
    for d in data[split:]:    f.write(json.dumps(d) + '\n')
```

---

## 5. Ingesting New Fashion Trend Documents

### Document Format

```python
{
    "text":      "Full trend description — who it suits, how to wear it ...",
    "category":  "menswear",      # menswear | womenswear | unisex | accessories | footwear
    "season":    "autumn-winter", # autumn-winter | spring-summer | all-seasons
    "style_tag": "quiet-luxury",  # slug for filtering
}
```

### Ingest from Python

```python
from pipeline.retriever import FashionTrendRetriever

retriever = FashionTrendRetriever()

new_docs = [
    {
        "text": "Utility cargo trousers in olive or khaki are back for 2025 ...",
        "category": "unisex",
        "season": "spring-summer",
        "style_tag": "utility-cargo",
    },
    # ... more documents
]

retriever.ingest(new_docs)
print(f"Collection now has {retriever.collection_count()} documents.")
```

### Ingest from CLI

```bash
# Bootstrap sample docs
python pipeline/retriever.py bootstrap

# Check status
python pipeline/retriever.py status

# Reset and re-ingest everything
python pipeline/retriever.py reset
```

### Recommended Update Cadence

Add new trend documents monthly. Fashion trend data goes stale quickly — a fresh batch every 4–6 weeks keeps the recommendations current. Source material: Vogue Runway, Hypebeast, GQ, trend forecasting services, or your own editorial team.

---

## 6. API Documentation

### `POST /analyze`

Upload a facial image and receive a personalized style report.

**Request**
```
Content-Type: multipart/form-data

image: <file>    JPEG, PNG, or WEBP — max 10 MB
```

**Response `200 OK`**
```json
{
  "request_id": "a3f4b2c1-...",
  "attributes": {
    "face_shape":           "oval",
    "skin_tone":            "medium",
    "hair_texture":         "wavy",
    "hair_length":          "short",
    "facial_hair":          "stubble",
    "eye_shape":            "almond",
    "jaw_type":             "defined",
    "cheekbone_prominence": "medium"
  },
  "trends_used": [
    "Quiet Luxury is the defining menswear aesthetic …",
    "Vintage Denim Revival: raw-selvedge and distressed …"
  ],
  "report": "## Face Shape Analysis\n\nYour face shape is **oval** …",
  "processing_time_ms": 4821.3,
  "timing": {
    "extraction_ms":  312.5,
    "retrieval_ms":   88.2,
    "generation_ms":  4420.6,
    "total_ms":       4821.3
  }
}
```

**Errors**

| Status | Meaning                                        |
|--------|------------------------------------------------|
| 400    | Invalid image (wrong format, too large, corrupt)|
| 503    | A required model is not loaded/available       |
| 500    | Internal pipeline error                        |

---

### `GET /health`

Check the status of all pipeline components.

**Response `200 OK`**
```json
{
  "status": "ok",
  "vision_encoder": {
    "status": "ok",
    "checkpoint": "models/vision/checkpoints/best.pt"
  },
  "qdrant": {
    "status": "ok",
    "collection": "fashion_trends",
    "document_count": 20
  },
  "llm": {
    "status": "ok",
    "model_path": "models/llm/merged"
  }
}
```

---

### cURL Example

```bash
curl -X POST http://localhost:8000/analyze \
  -F "image=@/path/to/photo.jpg" \
  | python -m json.tool
```

---

## 7. Environment Variables

| Variable            | Default                                    | Description                        |
|---------------------|--------------------------------------------|------------------------------------|
| `VISION_CHECKPOINT` | `models/vision/checkpoints/best.pt`        | Path to fine-tuned vision model    |
| `LLM_MODEL_PATH`    | `models/llm/merged`                        | Path to merged LLM model directory |
| `LLM_BACKEND`       | `local`                                    | `local` or `ollama`                |
| `TOP_K_TRENDS`      | `5`                                        | Number of trend docs to retrieve   |

Set via `.env` file or shell export before starting the server.

---

## 8. Running Tests

```bash
# Test vision extractor (requires checkpoint)
python pipeline/extractor.py data/raw/sample.jpg

# Test trend retrieval (requires Qdrant running)
python pipeline/retriever.py retrieve \
  --attributes '{"face_shape":"oval","skin_tone":"medium","hair_texture":"wavy","hair_length":"short","facial_hair":"stubble","eye_shape":"almond","jaw_type":"defined","cheekbone_prominence":"medium"}'

# Test LLM recommender (requires merged model)
python pipeline/recommender.py \
  --attributes '{"face_shape":"oval","skin_tone":"medium","hair_texture":"wavy","hair_length":"short","facial_hair":"none","eye_shape":"almond","jaw_type":"soft","cheekbone_prominence":"high"}'

# Test full pipeline end-to-end
python pipeline/report.py data/raw/sample.jpg

# Test FastAPI with pytest
pytest tests/
```

---

## 9. Project Structure

```
ai-style-advisor/
├── data/
│   ├── raw/                      # Raw unlabeled images
│   ├── labeled/                  # train.jsonl, val.jsonl
│   ├── training_pairs/           # seed_pairs.jsonl (LLM fine-tuning)
│   └── fashion_trends/
│       └── sample_trends.py      # 20 sample trend documents
│
├── models/
│   ├── vision/
│   │   ├── train.py              # DINOv2 + LoRA training script
│   │   ├── infer.py              # Vision encoder inference wrapper
│   │   └── checkpoints/          # Saved checkpoints (best.pt, latest.pt)
│   └── llm/
│       ├── train.py              # Mistral QLoRA fine-tuning script
│       ├── infer.py              # LLM inference wrapper
│       ├── checkpoints/          # LoRA adapter weights
│       └── merged/               # Merged model for inference
│
├── pipeline/
│   ├── extractor.py              # Stage 1: attribute extraction
│   ├── retriever.py              # Stage 2: Qdrant retrieval
│   ├── recommender.py            # Stage 3: LLM generation
│   └── report.py                 # Orchestration: all stages together
│
├── api/
│   └── main.py                   # FastAPI app (POST /analyze, GET /health)
│
├── frontend/
│   └── index.html                # Single-page upload UI with PDF export
│
├── docker/
│   └── docker-compose.yml        # Qdrant vector database
│
├── requirements.txt              # Pinned Python dependencies
└── README.md                     # This file
```

---

## 10. Key Constraints

- **No external AI API calls** — all inference runs locally after setup.
- **Offline capable** — once models are downloaded and fine-tuned, the system runs with no internet connection.
- **Python 3.10+** required across all scripts.
- **Each pipeline stage is independently testable** — run any of the pipeline scripts standalone.
- **All code includes** error handling, logging, and docstrings.
- **Vision encoder data** — labeling is the developer's responsibility (see Section 4).
- **Fashion trend data** — must be curated and ingested into Qdrant (see Section 5).

---

*AI Style Advisor — v1.0 · For internal / developer use only*

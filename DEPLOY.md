# StyleAI Deployment Guide

## Why Vercel alone won't work for the full app

Vercel serverless functions have a **250 MB compressed size limit**.  
This project's backend needs PyTorch + transformers + DINOv2 (~1–2 GB).  
The plan below splits the app into two parts:

| Part | Host | Why |
|------|------|-----|
| **Frontend** (`frontend/index.html`) | Vercel | Static HTML — zero config |
| **Backend** (`api/` + `pipeline/`) | Railway **or** Render | Full Docker support, GPU optional |
| **Model weights** (`epoch_1.pt`, LLM) | Hugging Face Hub | Free large-file storage, download at startup |

---

## Step 1 — Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/styleai.git
git push -u origin main
```

> The `.gitignore` already excludes `*.pt`, `*.bin`, `venv/`, and `__pycache__`.

---

## Step 2 — Upload model weights to Hugging Face Hub

Your `epoch_1.pt` and the LLM directory are too large for GitHub (100 MB limit).  
Store them on Hugging Face (free, private repos allowed):

```bash
pip install huggingface_hub
huggingface-cli login          # paste your HF token

# Upload vision checkpoint
huggingface-cli upload YOUR_HF_USERNAME/styleai-models \
    models/vision/checkpoints/epoch_1.pt \
    models/vision/checkpoints/epoch_1.pt

# Upload LLM (if you have it)
huggingface-cli upload YOUR_HF_USERNAME/styleai-models \
    models/llm/merged/ models/llm/merged/
```

Then set these env vars in Railway/Render (Step 3):
```
HF_MODEL_REPO=YOUR_HF_USERNAME/styleai-models
HF_TOKEN=hf_...           # only needed if repo is private
```

---

## Step 3 — Deploy the backend on Railway

Railway gives you a free tier with Docker support.

1. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**
2. Select your `styleai` repo
3. Railway auto-detects the `Dockerfile` and builds it
4. Set environment variables in Railway dashboard:

```
VISION_CHECKPOINT=models/vision/checkpoints/epoch_1.pt
LLM_BACKEND=local
LLM_MODEL_PATH=models/llm/merged
TOP_K_TRENDS=5
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

5. Railway gives you a URL like `https://styleai-production.up.railway.app`  
   → This is your `API_BASE` for the frontend

**Alternative: Render**  
Same steps — connect GitHub repo, it detects Dockerfile, set env vars under "Environment".

---

## Step 4 — Deploy the frontend on Vercel

The frontend is a single HTML file. Deploy it separately:

```bash
cd frontend
npx vercel --prod
```

Or via the Vercel dashboard:
1. **New Project → Import Git Repository → styleai**
2. Set **Root Directory** to `frontend`
3. Framework preset: **Other**
4. No build command needed

**Set the API URL** — before deploying, either:

**Option A** — Edit `frontend/index.html` directly:
```js
const API_BASE = 'https://styleai-production.up.railway.app';
```

**Option B** — Add a `frontend/config.js` (no rebuild needed):
```js
window.STYLEAI_API_URL = 'https://styleai-production.up.railway.app';
```
Then add `<script src="config.js"></script>` before the closing `</body>` in `index.html`.

---

## Step 5 — Test end-to-end

```bash
# Health check
curl https://styleai-production.up.railway.app/health

# Analyze endpoint
curl -X POST https://styleai-production.up.railway.app/analyze \
  -F "image=@your_photo.jpg"
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `FileNotFoundError: epoch_1.pt` | Checkpoint not in `models/vision/checkpoints/`. Download from HF Hub. |
| CORS error in browser | Set `ALLOWED_ORIGINS` env var on Railway to your Vercel URL |
| Railway build OOM | Upgrade Railway plan or reduce `torch` to CPU-only: `torch==2.3.0+cpu` |
| Cold start slow | Railway keeps containers warm on paid plan; Render spins down on free tier |

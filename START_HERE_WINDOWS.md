# StyleAI — Windows Setup & Run Guide

## Prerequisites
- Python 3.10 or 3.11 installed
- `epoch_1.pt` already placed in `models/vision/checkpoints/`

---

## Step 1 — Open Command Prompt in the project folder
Right-click the `Style_Ai` folder → Open in Terminal (or cd to it)

---

## Step 2 — Create virtual environment
```
python -m venv venv   (one time)
venv\Scripts\activate
```

---

## Step 3 — Install dependencies
```
pip install -r requirements.txt (one time)
```

---

## Step 4 — Bootstrap the vector store (run once)
```
cd pipeline                      (one time)
python retriever.py bootstrap    (one time)
cd ..                            (one time)
```

---

## Step 5 — Start the API server
```
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Step 6 — Open the frontend
Open `frontend/index.html` in your browser.
Or visit: http://localhost:8000/docs to test via Swagger UI.

---

## Step 7 — Test with an image
In a new terminal (with venv activated):
```
cd pipeline
python extractor.py path\to\your\photo.jpg --checkpoint ..\models\vision\checkpoints\epoch_1.pt
```

---

## Troubleshooting
- If you get a torchao error: `pip install torchao==0.16.0 peft==0.14.0`
- If port 8000 is busy: change `--port 8000` to `--port 8001`
- The first startup is slow (~30s) as the vision model loads into memory

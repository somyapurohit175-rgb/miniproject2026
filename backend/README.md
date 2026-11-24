# Backend: FastAPI + ONNX Inference + Risk Engine

This folder hosts the backend web service and inference pipeline.

## Structure
- `main.py`: FastAPI server exposing health and inference endpoints (`/api/v1/predict`).
- `requirements.txt`: Python package requirements.
- `models/` (create as needed): Directory for exported `.onnx` models.

## Running Locally
```bash
python -m venv .venv
# Activate environment (Windows PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
API Documentation will be available at: http://localhost:8000/docs

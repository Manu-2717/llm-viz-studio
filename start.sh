#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
pip install -q fastapi uvicorn httpx cairosvg huggingface_hub pydantic
npm --prefix frontend install
PYTHONPATH=backend uvicorn app.main:app --host 0.0.0.0 --port 8000 &
npm --prefix frontend run dev -- --host 0.0.0.0 --port 5173

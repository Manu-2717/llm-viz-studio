# LLMViz Studio — Explore LLM architectures visually.

Turn Hugging Face `config.json` into publication-ready 2D architecture diagrams. No weights, GPU, or API keys. LLMViz Studio is an independently maintained open-source project; it is not an official website of any upstream project.

## Quick start (dev)
```bash
./start.sh
# or manually:
pip install fastapi uvicorn httpx cairosvg huggingface_hub pydantic
PYTHONPATH=backend uvicorn app.main:app --reload --port 8000
npm --prefix frontend install && npm --prefix frontend run dev
```
Open http://localhost:5173

## Docker
```bash
docker compose up --build
# frontend http://localhost:8080  backend http://localhost:8000/api/v1/health
```

## API
- `GET /api/v1/health`
- `GET /api/v1/models/search?q=Qwen`
- `GET /api/v1/models/architecture?model=Qwen/Qwen2.5-1.5B-Instruct`
- `GET /api/v1/models/diagram?model=...` (SVG)
- `GET /api/v1/models/download?model=...&format=svg|png&scale=2`

Model IDs use query params to avoid slash-routing issues. IDs are validated (`org/model`), SVG is script-stripped, remote code is never executed, only `config.json` is fetched.

## Architecture
Frontend (React/Vite) → FastAPI → built-in diagram engine (`backend/app/engine/`: architecture parser, parameter estimation, SVG renderer vendored from an Apache-2.0 codebase and modified for LLMViz Studio) → HF Hub `config.json` only. In-memory TTL cache for specs/SVG/search. No external diagram package is required at install or runtime.

## Tests
```bash
python -m pytest backend/tests -q
npm --prefix frontend run build
```

## License / attribution
Apache-2.0 (see `LICENSE`). The diagram engine in `backend/app/engine/` is based on the Apache-2.0 `llmviz` project by Hesham Haroon (https://github.com/h9-tec/llmviz), modified and renamed for LLMViz Studio; see `NOTICE`, `THIRD_PARTY_NOTICES.md`, and `THIRD_PARTY_LICENSE.llmviz.txt`. Not affiliated with or endorsed by upstream.

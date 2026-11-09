# Phase 1 audit (2026-11-09)

## How it works today
- `backend/app/main.py`: FastAPI; `load_spec` → `app.engine.fetch.load_spec` (HF `config.json` only, never weights) → `ArchSpec`; `render_model` → `app.engine.render.block.render_model` (generic tower SVG). Downloads via CairoSVG.
- `backend/app/engine/`: vendored Apache-2.0 engine (spec/fetch/gguf + render/{block,svg,tokens,widths}). Only `spec`, `fetch`, `render/{block,svg,tokens,widths}` are imported by the website. `gguf.py` is reachable only via `ollama:`/`.gguf` sources, which the web API's `org/model` validation blocks — kept (harmless, part of engine).
- No CLI/gallery/poster/lineage/MCP/fit code exists in this repo (never vendored). Nothing web-unused to remove.

## Generic/inaccurate parts (addressed by graph overhaul)
- Tower diagram is identical in structure for every dense model; attention/MLP internals, per-tensor shapes, shard info are not shown.
- Parameter counts are estimates from config, not verified against checkpoint headers.
- No tensor-level evidence; unknown/custom architectures fail or are approximated.

## New layers added (this overhaul)
- `backend/app/graph/tensors.py` — header-only safetensors inventory via `get_safetensors_metadata` (config-only fallback).
- `backend/app/graph/mapping.py` — tensor-name → module hierarchy parser (extensible regex adapters).
- `backend/app/graph/topology.py` — verified topology adapters (Llama/Qwen2/Qwen3 dense, DeepSeek MLA+MoE) emitting the canonical graph; anything else is marked `partial`/`unknown`, never fabricated.
- `backend/app/graph/graph.py` — canonical JSON graph (single source of truth for viewer + exports).
- Old SVG/PNG download endpoints preserved unchanged.

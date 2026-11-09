# Vendored & modified from llmviz by Hesham Haroon (Apache-2.0, https://github.com/h9-tec/llmviz).
# Licensed under Apache-2.0; see NOTICE / THIRD_PARTY_NOTICES.md. Modifications: renamed to LLMViz Studio engine (app.engine), branding/scope ids updated.
"""Load a config.json from the HF Hub, a local path, or a URL — never weights."""

from __future__ import annotations

import json
from pathlib import Path

from app.engine.spec import ArchSpec, parse_config


def load_spec(source: str, token: str | None = None) -> ArchSpec:
    if source.startswith("ollama:") or source.endswith(".gguf"):
        from app.engine.gguf import load_gguf_config

        cfg, name = load_gguf_config(source)
        return parse_config(cfg, name=name)
    path = Path(source)
    if path.is_file():
        cfg = json.loads(path.read_text())
        return parse_config(cfg, name=path.stem)
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

    try:
        local = hf_hub_download(source, "config.json", token=token)
    except GatedRepoError as e:
        raise SystemExit(
            f"'{source}' is gated. Accept the license on huggingface.co, then retry with "
            f"--token or `hf auth login`. Tip: community mirrors often host the same config."
        ) from e
    except (RepositoryNotFoundError, EntryNotFoundError) as e:
        raise SystemExit(f"No config.json found for '{source}' on the Hugging Face Hub.") from e
    cfg = json.loads(Path(local).read_text())
    return parse_config(cfg, name=source.split("/")[-1])

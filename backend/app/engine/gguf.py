# Vendored & modified from llmviz by Hesham Haroon (Apache-2.0, https://github.com/h9-tec/llmviz).
# Licensed under Apache-2.0; see NOTICE / THIRD_PARTY_NOTICES.md. Modifications: renamed to LLMViz Studio engine (app.engine), branding/scope ids updated.
"""Read GGUF metadata (header only, never weights) and map it to a config.json-like dict.

Covers: local .gguf files, Ollama-installed models (ollama:<name>), and remote GGUF URLs
via ranged HTTP so only the metadata bytes are fetched.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

_TYPES = {
    0: "u8",
    1: "i8",
    2: "u16",
    3: "i16",
    4: "u32",
    5: "i32",
    6: "f32",
    7: "bool",
    8: "str",
    9: "arr",
    10: "u64",
    11: "i64",
    12: "f64",
}
_SIZES = {
    "u8": 1,
    "i8": 1,
    "u16": 2,
    "i16": 2,
    "u32": 4,
    "i32": 4,
    "f32": 4,
    "bool": 1,
    "u64": 8,
    "i64": 8,
    "f64": 8,
}
_FMT = {
    "u8": "B",
    "i8": "b",
    "u16": "H",
    "i16": "h",
    "u32": "I",
    "i32": "i",
    "f32": "f",
    "bool": "?",
    "u64": "Q",
    "i64": "q",
    "f64": "d",
}


class _Reader:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise EOFError
        b = self.data[self.pos : self.pos + n]
        self.pos += n
        return b

    def scalar(self, t: str):
        return struct.unpack("<" + _FMT[t], self.take(_SIZES[t]))[0]

    def string(self) -> str:
        n = self.scalar("u64")
        return self.take(n).decode("utf-8", errors="replace")

    def value(self, t: str, want: bool):
        """Parse (or skip) one value. Large unwanted arrays are skipped cheaply."""
        if t == "str":
            return self.string() if want else self.take(self.scalar("u64")) and None
        if t == "arr":
            et = _TYPES[self.scalar("u32")]
            n = self.scalar("u64")
            if not want:
                if et in _SIZES:
                    self.take(_SIZES[et] * n)
                else:
                    for _ in range(n):
                        self.value(et, False)
                return n  # length is free and useful (tokenizer array = vocab size)
            return [self.value(et, True) for _ in range(n)]
        return self.scalar(t)


# keys we need (per-architecture prefix is substituted for {a})
_WANTED_SUFFIXES = (
    "block_count",
    "embedding_length",
    "feed_forward_length",
    "context_length",
    "attention.head_count",
    "attention.head_count_kv",
    "attention.key_length",
    "rope.freq_base",
    "expert_count",
    "expert_used_count",
    "expert_feed_forward_length",
    "expert_shared_count",
    "vocab_size",
    "attention.layer_norm_rms_epsilon",
    "attention.layer_norm_epsilon",
    "sliding_window",
    "attention.sliding_window",
)


def parse_gguf_metadata(data: bytes) -> dict:
    r = _Reader(data)
    if r.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    r.scalar("u32")  # version
    r.scalar("u64")  # tensor count
    n_kv = r.scalar("u64")
    meta: dict = {}
    for _ in range(n_kv):
        key = r.string()
        t = _TYPES[r.scalar("u32")]
        want = key in ("general.architecture", "general.name", "general.size_label") or any(
            key.endswith(sfx) for sfx in _WANTED_SUFFIXES
        )
        v = r.value(t, want)
        if want:
            meta[key] = v
        elif key == "tokenizer.ggml.tokens" and t == "arr":
            meta["_tokenizer_vocab"] = v
    return meta


def gguf_to_config(meta: dict) -> dict:
    """Map GGUF keys onto the config.json field names the normal parser understands."""
    a = meta.get("general.architecture", "llama")

    def g(sfx, *alts):
        for k in (f"{a}.{sfx}", *[f"{a}.{s}" for s in alts]):
            if k in meta:
                return meta[k]
        return None

    cfg = {
        "model_type": a,
        "architectures": [meta.get("general.name", a)],
        "hidden_size": g("embedding_length"),
        "num_hidden_layers": g("block_count"),
        "num_attention_heads": g("attention.head_count"),
        "num_key_value_heads": g("attention.head_count_kv"),
        "intermediate_size": g("feed_forward_length"),
        "max_position_embeddings": g("context_length"),
        "rope_theta": g("rope.freq_base"),
        "vocab_size": g("vocab_size") or meta.get("_tokenizer_vocab"),
        "head_dim": g("attention.key_length"),
        "num_experts": g("expert_count"),
        "num_experts_per_tok": g("expert_used_count"),
        "moe_intermediate_size": g("expert_feed_forward_length"),
        "n_shared_experts": g("expert_shared_count"),
        "sliding_window": g("sliding_window", "attention.sliding_window"),
        # GGUF doesn't store the activation; every modern family here is gated SwiGLU
        "hidden_act": (
            "gelu" if a in ("gpt2", "gptj", "gpt_neox", "falcon", "starcoder") else "silu"
        ),
    }
    # head_count can be a per-layer list (hybrid models) — use the max (full-attn layers)
    for k in ("num_attention_heads", "num_key_value_heads"):
        if isinstance(cfg[k], list):
            cfg[k] = max(v for v in cfg[k] if v) if any(cfg[k]) else None
    if g("attention.layer_norm_epsilon") is not None:
        cfg["layer_norm_epsilon"] = g("attention.layer_norm_epsilon")
    return {k: v for k, v in cfg.items() if v is not None}


def _read_local(path: Path) -> dict:
    size = path.stat().st_size
    n = min(size, 8 * 1024 * 1024)
    while True:
        with path.open("rb") as f:
            data = f.read(n)
        try:
            return parse_gguf_metadata(data)
        except EOFError:
            if n >= size:
                raise
            n = min(size, n * 4)


def _read_remote(url: str) -> dict:
    import urllib.request

    n = 4 * 1024 * 1024
    while n <= 256 * 1024 * 1024:
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{n - 1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        try:
            return parse_gguf_metadata(data)
        except EOFError:
            n *= 4
    raise SystemExit("GGUF metadata larger than 256MB — refusing")


def _ollama_blob(name: str) -> Path:
    """Resolve an Ollama model name to its GGUF blob via the local manifest store."""
    base = Path.home() / ".ollama" / "models"
    model, _, tag = name.partition(":")
    tag = tag or "latest"
    candidates = list((base / "manifests").glob(f"**/{model}/{tag}"))
    if not candidates:
        installed = sorted(
            p.parent.name + ":" + p.name for p in (base / "manifests").glob("**/*/*")
        )
        raise SystemExit(
            f"Ollama model '{name}' not found locally. Installed: {', '.join(installed) or 'none'}"
        )
    manifest = json.loads(candidates[0].read_text())
    for layer in manifest.get("layers", []):
        if layer.get("mediaType", "").endswith("image.model"):
            return base / "blobs" / layer["digest"].replace(":", "-")
    raise SystemExit(f"no model blob in Ollama manifest for '{name}'")


def load_gguf_config(source: str) -> tuple[dict, str]:
    """Returns (config-dict, display-name) for a .gguf path/URL or ollama:<name>."""
    if source.startswith("ollama:"):
        name = source.removeprefix("ollama:")
        meta = _read_local(_ollama_blob(name))
        return gguf_to_config(meta), meta.get("general.name") or name
    if source.startswith(("http://", "https://")):
        meta = _read_remote(source)
    else:
        meta = _read_local(Path(source))
    cfg = gguf_to_config(meta)
    return cfg, meta.get("general.name") or Path(source).stem

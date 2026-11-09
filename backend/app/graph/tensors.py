"""Header-only safetensors inventory. Never downloads weight bytes."""
from __future__ import annotations
import logging, os
from dataclasses import dataclass, field

log = logging.getLogger("llmviz-studio.tensors")
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4, "U32": 4, "I64": 8, "U64": 8, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

@dataclass
class TensorInfo:
    name: str
    shape: list[int]
    dtype: str
    numel: int
    bytes: int
    shard: str
    layer: int | None = None
    module: str = "unknown"
    category: str = "unknown"

    def to_dict(self):
        return {"name": self.name, "shape": self.shape, "dtype": self.dtype,
                "numel": self.numel, "bytes": self.bytes, "shard": self.shard,
                "layer": self.layer, "module": self.module, "category": self.category}

def fetch_inventory(model_id: str) -> tuple[list[TensorInfo], bool, str | None]:
    """Returns (tensors, available, note). available=False -> config-only graph."""
    try:
        from huggingface_hub import get_safetensors_metadata
        meta = get_safetensors_metadata(model_id, token=os.getenv("HF_TOKEN") or None,
                                        timeout=float(os.getenv("SAFETENSORS_TIMEOUT", "30")))
    except Exception as e:
        log.info("safetensors metadata unavailable for %s: %s", model_id, e)
        return [], False, f"safetensors metadata unavailable: {type(e).__name__}"
    tensors: list[TensorInfo] = []
    max_tensors = int(os.getenv("MAX_TENSORS", "200000"))
    for shard, fmeta in (meta.files_metadata or {}).items():
        for name, t in (getattr(fmeta, "tensors", {}) or {}).items():
            if len(tensors) >= max_tensors:
                return tensors, True, f"truncated at {max_tensors} tensors"
            shape = list(getattr(t, "shape", []) or [])
            numel = int(getattr(t, "parameter_count", 0) or 0)
            offs = getattr(t, "data_offsets", None)
            nbytes = int(offs[1] - offs[0]) if offs and len(offs) == 2 else numel * DTYPE_BYTES.get(str(getattr(t, "dtype", "")).upper(), 0)
            tensors.append(TensorInfo(name=name, shape=shape, dtype=str(getattr(t, "dtype", "?")),
                                      numel=numel, bytes=nbytes, shard=shard))
    tensors.sort(key=lambda x: x.name)
    return tensors, len(tensors) > 0, None

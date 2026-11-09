"""Canonical architecture graph — single source of truth for viewer + exports."""
from __future__ import annotations
from pydantic import BaseModel, Field

class TensorRef(BaseModel):
    name: str
    shape: list[int] = []
    dtype: str = "?"
    numel: int = 0
    bytes: int = 0
    shard: str = ""

class GraphNode(BaseModel):
    id: str
    label: str
    kind: str  # embedding|norm|attention|mlp|moe|block|head|op|unknown
    level: int = 1  # 1 overview, 2 block, 3 tensor
    parent: str | None = None
    layer: int | None = None
    params: int = 0
    dtype: str | None = None
    in_dim: int | None = None
    out_dim: int | None = None
    confidence: str = "verified"  # verified|config|inferred|unknown
    evidence: str = ""
    tensors: list[TensorRef] = []
    collapsed_count: int | None = None  # for repeated-layer group nodes

class GraphEdge(BaseModel):
    src: str
    dst: str
    kind: str = "data"  # data|residual
    confidence: str = "verified"

class ArchGraph(BaseModel):
    model_id: str
    model_type: str
    architecture: str
    confidence: str = "verified"
    metadata_source: str = "safetensors"  # safetensors|config_only
    metadata_note: str | None = None
    total_params: int = 0
    verified_params: int = 0
    tied_note: str | None = None
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    layers: list[int] = []
    unknowns: list[str] = []

    def validate_graph(self) -> list[str]:
        ids = {n.id for n in self.nodes}
        problems = []
        for e in self.edges:
            if e.src not in ids or e.dst not in ids:
                problems.append(f"dangling edge {e.src}->{e.dst}")
        if len(ids) != len(self.nodes):
            problems.append("duplicate node ids")
        return problems

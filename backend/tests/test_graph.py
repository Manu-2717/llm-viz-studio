import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app
from app.graph.tensors import TensorInfo
from app.graph.mapping import map_tensor, annotate
from app.graph.topology import build_graph
from app.graph.export import graph_to_svg
from app.engine.spec import parse_config

client = TestClient(app, raise_server_exceptions=False)
FIX = pathlib.Path(__file__).resolve().parent / "fixtures"

def test_mapping():
    assert map_tensor("model.layers.0.self_attn.q_proj.weight") == (0, "attention", "attention", None)
    assert map_tensor("model.embed_tokens.weight")[1] == "embedding"
    assert map_tensor("lm_head.weight")[1] == "lm_head"
    assert map_tensor("model.layers.3.mlp.experts.17.gate_proj.weight")[3] == 17
    assert map_tensor("model.layers.0.self_attn.q_a_proj.weight")[1] == "attention_q_a"
    assert map_tensor("totally.custom.tensor")[1] == "unknown"

def test_graph_dense_from_tensors():
    spec = parse_config(json.loads((FIX / "qwen3_06b.json").read_text()), name="t")
    ts = [TensorInfo("model.embed_tokens.weight", [151936, 1024], "BF16", 151936*1024, 0, "model.safetensors"),
          TensorInfo("model.layers.0.input_layernorm.weight", [1024], "BF16", 1024, 2048, "model.safetensors"),
          TensorInfo("model.layers.0.self_attn.q_proj.weight", [2048, 1024], "BF16", 2048*1024, 0, "model.safetensors"),
          TensorInfo("model.layers.0.mlp.gate_proj.weight", [3072, 1024], "BF16", 3072*1024, 0, "model.safetensors")]
    g = build_graph("org/test", spec, ts, True, None)
    assert g.validate_graph() == []
    assert g.metadata_source == "safetensors" and g.verified_params > 0
    labels = " ".join(n.label for n in g.nodes)
    assert "embeddings" in labels.lower() and "Layer 0" in labels
    svg = graph_to_svg(g)
    assert "<svg" in svg and "architecture graph" in svg

def test_graph_config_only():
    spec = parse_config(json.loads((FIX / "qwen3_06b.json").read_text()), name="t")
    g = build_graph("org/test", spec, [], False, "no safetensors")
    assert g.confidence == "partial" and g.metadata_source == "config_only"
    assert not any(n.id.startswith("block:0:") for n in g.nodes)  # no fabricated internals

def test_live_graph():
    r = client.get("/api/v1/models/graph", params={"model": "Qwen/Qwen2.5-1.5B-Instruct"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["metadata_source"] == "safetensors" and d["verified_params"] > 1_000_000_000
    ids = {n["id"] for n in d["nodes"]}
    assert "embed" in ids and "blocks" in ids
    r2 = client.get("/api/v1/models/graph", params={"model": "Qwen/Qwen2.5-1.5B-Instruct", "layer": 0})
    assert r2.status_code == 200 and any("attn" in n["id"] for n in r2.json()["nodes"])
    r3 = client.get("/api/v1/models/tensors", params={"model": "Qwen/Qwen2.5-1.5B-Instruct", "layer": 0, "limit": 5})
    assert r3.status_code == 200 and r3.json()["available"] and len(r3.json()["tensors"]) == 5
    r4 = client.get("/api/v1/models/graph/download", params={"model": "Qwen/Qwen2.5-1.5B-Instruct", "format": "png", "scale": 1})
    assert r4.status_code == 200 and r4.content[:4] == b"\x89PNG"

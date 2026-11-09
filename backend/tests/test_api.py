import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app
from app.engine.spec import parse_config
from app.engine.render.block import render_model

client = TestClient(app, raise_server_exceptions=False)
FIX = pathlib.Path(__file__).resolve().parent / "fixtures"

def test_health():
    r = client.get("/api/v1/health"); assert r.status_code == 200 and r.json()["status"] == "ok"

def test_invalid_id():
    r = client.get("/api/v1/models/architecture", params={"model": "../../etc/passwd"}); assert r.status_code == 422
    r = client.get("/api/v1/models/architecture", params={"model": "notamodel"}); assert r.status_code == 422

def test_fixture_dense_and_moe_render():
    for name in ["qwen3_06b.json", "qwen3_moe.json", "llama3_8b.json"]:
        cfg = json.loads((FIX / name).read_text())
        spec = parse_config(cfg, name=name)
        svg = render_model(spec)
        assert "<svg" in svg and len(svg) > 1000
    moe = parse_config(json.loads((FIX/"qwen3_moe.json").read_text()), name="moe")
    assert moe.moe is not None and moe.total_params > moe.active_params

def test_search_shape():
    r = client.get("/api/v1/models/search", params={"q": "Qwen"})
    assert r.status_code in (200, 502)
    if r.status_code == 200:
        assert isinstance(r.json(), list)

def test_live_model():
    r = client.get("/api/v1/models/architecture", params={"model": "Qwen/Qwen2.5-1.5B-Instruct"})
    assert r.status_code == 200, r.text
    d = r.json(); assert d["hidden_size"] > 0 and d["num_layers"] > 0
    r2 = client.get("/api/v1/models/diagram", params={"model": "Qwen/Qwen2.5-1.5B-Instruct"})
    assert r2.status_code == 200 and "<svg" in r2.text
    r3 = client.get("/api/v1/models/download", params={"model": "Qwen/Qwen2.5-1.5B-Instruct", "format": "png", "scale": 1})
    assert r3.status_code == 200 and r3.content[:4] == b"\x89PNG"

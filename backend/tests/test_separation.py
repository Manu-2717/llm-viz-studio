"""Regression: Diagram and Graph systems are independent but share metadata."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app, raise_server_exceptions=False)
M = "Qwen/Qwen2.5-1.5B-Instruct"

def test_diagram_unchanged():
    r = client.get("/api/v1/models/diagram", params={"model": M})
    assert r.status_code == 200 and "<svg" in r.text and 'id="llmviz-studio-' in r.text

def test_graph_independent():
    r = client.get("/api/v1/models/graph", params={"model": M})
    assert r.status_code == 200, r.text
    d = r.json()
    assert "nodes" in d and "edges" in d and "nodes" not in [e.get("kind") for e in []]  # graph, not tower svg
    assert not any("<svg" in str(n) for n in d["nodes"])

def test_shared_metadata_both_views():
    a = client.get("/api/v1/models/architecture", params={"model": M})
    g = client.get("/api/v1/models/graph", params={"model": M})
    assert a.status_code == 200 and g.status_code == 200
    assert a.json()["model_id"] == g.json()["model_id"] == M

def test_graph_failure_does_not_break_diagram():
    r = client.get("/api/v1/models/graph", params={"model": "bad/id"})
    assert r.status_code in (404, 422, 502)
    r2 = client.get("/api/v1/models/diagram", params={"model": M})
    assert r2.status_code == 200

"""Verified topology adapters -> canonical graph. Uncertain => partial/unknown, never fabricated."""
from __future__ import annotations
from .graph import ArchGraph, GraphNode, GraphEdge, TensorRef
from .mapping import annotate

DENSE_TYPES = {"llama", "qwen2", "qwen3", "qwen", "mistral", "gemma", "gemma2", "gemma3",
               "phi3", "olmo2", "granite", "lfm2", "glm", "falcon", "starcoder", "gpt2", "gpt_oss"}
MOE_TYPES = {"deepseek_v3", "deepseek_v2", "qwen3_moe", "qwen3_next", "mixtral", "glm4_moe"}

def _tref(t) -> TensorRef:
    return TensorRef(name=t.name, shape=t.shape, dtype=t.dtype, numel=t.numel, bytes=t.bytes, shard=t.shard)

def _group(tensors, layer, module):
    gs = [t for t in tensors if t.layer == layer and t.module == module and t.category != "quant_scale"]
    return gs

def _params(ts) -> int:
    return sum(t.numel for t in ts)

def _dim(ts, idx=0):
    for t in ts:
        if len(t.shape) > idx:
            return t.shape[idx]
    return None

def build_graph(model_id: str, spec, tensors: list, available: bool, note: str | None, rep_layer: int | None = None) -> ArchGraph:
    tensors = annotate(tensors)
    mt = (spec.model_type or "").lower()
    is_moe = bool(spec.moe) or mt in MOE_TYPES
    layers = sorted({t.layer for t in tensors if t.layer is not None}) or list(range(spec.num_layers))
    g = ArchGraph(model_id=model_id, model_type=spec.model_type, architecture=spec.architecture,
                  metadata_source="safetensors" if available else "config_only", metadata_note=note,
                  total_params=spec.total_params, layers=layers)
    g.verified_params = sum(t.numel for t in tensors if t.category != "quant_scale")
    g.tied_note = ("lm_head absent from checkpoint (tied embeddings) — verified total excludes it"
                   if available and not any("lm_head" in t.name for t in tensors) else None)
    unknowns = sorted({t.name for t in tensors if t.module == "unknown"})
    g.unknowns = unknowns[:50]

    def N(id, label, kind, level=1, parent=None, layer=None, ts=None, conf="verified", ev="", in_dim=None, out_dim=None, collapsed=None):
        ts = ts or []
        g.nodes.append(GraphNode(id=id, label=label, kind=kind, level=level, parent=parent, layer=layer,
                                 params=_params(ts), dtype=ts[0].dtype if ts else None,
                                 in_dim=in_dim, out_dim=out_dim, confidence=conf, evidence=ev,
                                 tensors=[_tref(t) for t in ts[:20]], collapsed_count=collapsed))
    def E(src, dst, kind="data", conf="verified"):
        g.edges.append(GraphEdge(src=src, dst=dst, kind=kind, confidence=conf))

    emb = _group(tensors, None, "embedding")
    N("embed", "Token embeddings", "embedding", ts=emb, ev="embed_tokens.weight" if emb else "config only",
      in_dim=spec.vocab_size, out_dim=spec.hidden_size, conf="verified" if emb else "config")
    N("blocks", f"Transformer layers ×{len(layers)}", "block",
      conf="verified" if available else "config", collapsed=len(layers),
      ev="repeated block — expand one layer below")
    # Representative block = first layer with tensors (usually 0)
    rep = rep_layer if rep_layer is not None else (layers[0] if layers else 0)
    if rep_layer is None and is_moe:
        moe_layers = sorted({t.layer for t in tensors if t.layer is not None and t.category in ("moe_router", "moe_expert")})
        if moe_layers:
            rep = moe_layers[0]
    bp = f"block:{rep}"
    N(bp, f"Layer {rep} (representative)", "block", level=2, parent="blocks", layer=rep,
      conf="verified" if available else "partial", ev="tensors present" if available else "no tensor evidence")
    E("embed", "blocks"); E("blocks", "final_norm"); E(bp, "blocks")
    fin = _group(tensors, None, "final_norm")
    N("final_norm", "Final normalization", "norm", ts=fin, conf="verified" if fin else "config")
    head = _group(tensors, None, "lm_head")
    N("lm_head", "LM head", "head", ts=head, in_dim=spec.hidden_size, out_dim=spec.vocab_size,
      conf="verified" if head else "inferred", ev="tied embeddings" if (available and not head) else "")

    pre = f"{bp}:"
    # Block internals (only if tensor evidence exists for this layer)
    if available and any(t.layer == rep for t in tensors):
        n1, n2 = _group(tensors, rep, "input_norm"), _group(tensors, rep, "post_attn_norm")
        N(pre + "norm1", "RMSNorm/LayerNorm 1", "norm", level=2, parent=bp, layer=rep, ts=n1, conf="verified" if n1 else "inferred")
        N(pre + "norm2", "RMSNorm/LayerNorm 2", "norm", level=2, parent=bp, layer=rep, ts=n2, conf="verified" if n2 else "inferred")
        q = _group(tensors, rep, "attention"); qa = _group(tensors, rep, "attention_q_a") + _group(tensors, rep, "attention_q_b")
        kva = _group(tensors, rep, "attention_kv_a"); kvb = _group(tensors, rep, "attention_kv_b")
        op = _group(tensors, rep, "attention_o"); fused = _group(tensors, rep, "attention_qkv_fused")
        qk_norm = _group(tensors, rep, "attention_qk_norm")
        if spec.attention.kind == "MLA" or qa:
            N(pre+"attn_qa", "Q latent ↓ (q_a)", "attention", level=2, parent=bp, layer=rep, ts=qa, in_dim=spec.hidden_size)
            N(pre+"attn_kva", "KV latent ↓ (kv_a, MQA)", "attention", level=2, parent=bp, layer=rep, ts=kva)
            N(pre+"attn_qb", "Q ↑ (q_b)", "attention", level=2, parent=bp, layer=rep, ts=qa)
            N(pre+"attn_kvb", "KV ↑ (kv_b)", "attention", level=2, parent=bp, layer=rep, ts=kvb)
            N(pre+"attn_o", "O projection", "attention", level=2, parent=bp, layer=rep, ts=op)
            for s, d in [(pre+"norm1", pre+"attn_qa"), (pre+"norm1", pre+"attn_kva"), (pre+"attn_qb", pre+"attn_o"), (pre+"attn_kvb", pre+"attn_o")]:
                E(s, d)
        elif fused:
            N(pre+"attn_qkv", "Fused QKV projection", "attention", level=2, parent=bp, layer=rep, ts=fused)
            N(pre+"attn_o", "O projection", "attention", level=2, parent=bp, layer=rep, ts=op)
            E(pre+"norm1", pre+"attn_qkv"); E(pre+"attn_qkv", pre+"attn_o")
        elif q:
            # per-projection nodes
            for mod, label in [("attention", "Q/K/V projections")]:
                N(pre+"attn_qkv", label, "attention", level=2, parent=bp, layer=rep, ts=q,
                  in_dim=spec.hidden_size, out_dim=_dim(q))
            N(pre+"attn_o", "O projection", "attention", level=2, parent=bp, layer=rep, ts=op)
            E(pre+"norm1", pre+"attn_qkv"); E(pre+"attn_qkv", pre+"attn_o")
        else:
            N(pre+"attn", "Attention (no tensor evidence)", "attention", level=2, parent=bp, layer=rep, conf="unknown")
            E(pre+"norm1", pre+"attn")
        # RoPE / Softmax ops (verified by architecture rules, no weight tensor)
        if spec.positional == "rope":
            N(pre+"rope", "RoPE", "op", level=2, parent=bp, layer=rep, conf="inferred", ev="positional=rope in config")
        N(pre+"softmax", "Softmax attention", "op", level=2, parent=bp, layer=rep, conf="inferred", ev="standard attention rule")
        if qk_norm:
            N(pre+"qk_norm", "Q/K norm", "norm", level=2, parent=bp, layer=rep, ts=qk_norm)
        # residual edges
        E(bp, pre+"norm1", kind="residual", conf="inferred")
        # MLP / MoE
        mlp = _group(tensors, rep, "mlp"); gate_up = _group(tensors, rep, "mlp_gate_up_fused")
        router = _group(tensors, rep, "router"); shared = _group(tensors, rep, "shared_expert")
        experts = [t for t in tensors if t.layer == rep and t.module == "moe_expert"]
        if is_moe and (router or experts):
            N(pre+"router", "MoE router (top-k)", "moe", level=2, parent=bp, layer=rep, ts=router, conf="verified" if router else "inferred")
            N(pre+"experts", f"Routed experts ×{spec.moe.num_experts if spec.moe else len({t.expert for t in experts})}", "moe",
              level=2, parent=bp, layer=rep, ts=experts[:20], conf="verified" if experts else "inferred",
              ev=f"{len({t.expert for t in experts})} expert tensors in layer {rep} (showing first 20)")
            if shared:
                N(pre+"shared", "Shared experts", "moe", level=2, parent=bp, layer=rep, ts=shared, conf="verified")
            E(pre+"norm2", pre+"router"); E(pre+"router", pre+"experts")
            if shared:
                E(pre+"norm2", pre+"shared")
        elif mlp or gate_up:
            N(pre+"mlp", "SwiGLU MLP (gate/up/down)" if len({t.module for t in mlp}) else "Fused MLP", "mlp",
              level=2, parent=bp, layer=rep, ts=mlp + gate_up, conf="verified",
              in_dim=spec.hidden_size, out_dim=_dim(mlp + gate_up))
            N(pre+"silu", "SiLU gating", "op", level=2, parent=bp, layer=rep, conf="inferred", ev=f"activation={spec.activation}")
            E(pre+"norm2", pre+"mlp"); E(pre+"mlp", pre+"silu")
        else:
            N(pre+"mlp", "Feed-forward (no tensor evidence)", "mlp", level=2, parent=bp, layer=rep, conf="unknown")
            E(pre+"norm2", pre+"mlp")
        # unknown tensors preserved as a node
        unk = [t for t in tensors if t.layer == rep and t.module == "unknown"]
        if unk:
            N(pre+"unknown", f"Unmapped tensors ×{len(unk)}", "unknown", level=2, parent=bp, layer=rep, ts=unk, conf="unknown")
        g.confidence = "verified" if available else "partial"
    else:
        g.confidence = "partial"
        if not available:
            g.metadata_note = (note or "safetensors unavailable") + " — config-only graph; block internals unavailable"
    problems = g.validate_graph()
    if problems:
        g.metadata_note = (g.metadata_note or "") + f" | validation: {problems}"
    return g

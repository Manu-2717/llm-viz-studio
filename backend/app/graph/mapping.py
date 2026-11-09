"""Tensor-name -> module hierarchy parser. Extensible via regex adapters; unknowns preserved."""
from __future__ import annotations
import re

# (regex, layer, module, category) — first match wins. Layer group named 'layer'.
_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^(?:model\.)?(?:embed_tokens|wte)\.weight$"), "embedding", "embedding"),
    (re.compile(r"^(?:lm_head|model\.lm_head)\.weight$"), "lm_head", "lm_head"),
    (re.compile(r"^model\.norm\.weight$"), "final_norm", "norm"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.input_layernorm\.weight$"), "input_norm", "norm"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.post_attention_layernorm\.weight$"), "post_attn_norm", "norm"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.self_attn\.(q|k|v|o)_proj\.(weight|bias)$"), "attention", "attention"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.self_attn\.q_a_proj\.weight$"), "attention_q_a", "attention"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.self_attn\.q_b_proj\.weight$"), "attention_q_b", "attention"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.self_attn\.(q|kv)_a(_layernorm|_proj).*"), "attention_kv_a", "attention"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.self_attn\.kv_b_proj\.weight$"), "attention_kv_b", "attention"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.self_attn\.o_proj\.weight$"), "attention_o", "attention"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.self_attn\.(q|k|v|o)_norm\.weight$"), "attention_qk_norm", "norm"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.self_attn\.qkv_proj\.weight$"), "attention_qkv_fused", "attention"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.(gate|up|down)_proj\.weight$"), "mlp", "mlp"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.gate_up_proj\.weight$"), "mlp_gate_up_fused", "mlp"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(gate|up|down)_proj\.weight$"), "expert", "moe_expert"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.shared_experts\.(gate|up|down)_proj\.weight$"), "shared_expert", "moe_shared"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.(gate|router)\.weight$"), "router", "moe_router"),
    (re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.gate_up_proj.*"), "expert_gate_up_fused", "moe_expert"),
]
_LAYER_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.")

def map_tensor(name: str):
    """Returns (layer:int|None, module, category, expert:int|None)."""
    expert = None
    layer = None
    for pat, module, category in _PATTERNS:
        m = pat.match(name)
        if m:
            gd = m.groupdict()
            layer = int(gd["layer"]) if gd.get("layer") is not None else None
            if gd.get("expert") is not None:
                expert = int(gd["expert"])
            return layer, module, category, expert
    m = _LAYER_RE.match(name)
    if m:
        layer = int(m.group("layer"))
    if "weight_scale_inv" in name:
        return layer, "quant_scale", "quant_scale", None
    return layer, "unknown", "unknown", None

def annotate(tensors) -> list:
    for t in tensors:
        layer, module, category, expert = map_tensor(t.name)
        t.layer, t.module, t.category = layer, module, category
        t.expert = expert  # type: ignore[attr-defined]
    return tensors

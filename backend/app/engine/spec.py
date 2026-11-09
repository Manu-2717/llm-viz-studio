# Vendored & modified from llmviz by Hesham Haroon (Apache-2.0, https://github.com/h9-tec/llmviz).
# Licensed under Apache-2.0; see NOTICE / THIRD_PARTY_NOTICES.md. Modifications: renamed to LLMViz Studio engine (app.engine), branding/scope ids updated.
"""Normalize a HuggingFace config.json into an ArchSpec and estimate parameter counts."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel

# model_types that apply RMSNorm to Q/K heads but don't expose it in config.json
QK_NORM_MODEL_TYPES = {"qwen3", "qwen3_moe", "qwen3_next", "olmo2", "gemma3", "gemma3_text"}
POST_NORM_MODEL_TYPES = {"olmo2"}  # norm applied to sublayer output, before the residual add
SANDWICH_NORM_MODEL_TYPES = {"gemma2", "gemma3", "gemma3_text"}


def _hybrid_note(cfg: dict, layers: int) -> str | None:
    """Summarize non-standard token mixers (linear attention, SSM/Mamba, conv) from
    whichever encoding the config uses — new architectures keep inventing new ones."""
    lt = cfg.get("layer_types")
    if lt and isinstance(lt, list):
        counts = {}
        for v in lt:
            counts[v] = counts.get(v, 0) + 1
        exotic = {
            k: n
            for k, n in counts.items()
            if k not in ("full_attention", "sliding_attention", "attention")
        }
        if exotic:
            attn = layers - sum(exotic.values())
            parts = " + ".join(f"{n} {k.replace('_', ' ')}" for k, n in exotic.items())
            return f"{parts} : {attn} attention layers"
    la = cfg.get("linear_attn_config")
    if isinstance(la, dict) and la.get("kda_layers"):
        return f"{len(la['kda_layers'])} linear-attention (KDA) : {len(la.get('full_attn_layers', []))} full attention layers"
    if cfg.get("full_attn_idxs") is not None:
        full = len(cfg["full_attn_idxs"])
        return f"{layers - full} conv-mixer : {full} full attention layers"
    if cfg.get("attn_type_list"):
        al = cfg["attn_type_list"]
        return f"{al.count(0)} lightning (linear) : {al.count(1)} softmax attention layers"
    if any(k.startswith("linear_") for k in cfg):
        return "hybrid: linear-attention mixer on most layers"
    if any(k.startswith(("mamba_", "ssm_")) for k in cfg):
        return "hybrid: Mamba/SSM mixer layers"
    return None


class AttentionKind(StrEnum):
    MHA = "MHA"
    GQA = "GQA"
    MQA = "MQA"
    MLA = "MLA"


class MLAConfig(BaseModel):
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int


class Attention(BaseModel):
    kind: AttentionKind
    num_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int | None = None
    sliding_pattern: str | None = None  # e.g. "5:1 local:global", "alternating"
    qk_norm: bool = False
    attention_bias: bool = False
    mla: MLAConfig | None = None


class MoE(BaseModel):
    num_experts: int
    experts_per_tok: int
    shared_experts: int = 0
    moe_intermediate_size: int
    dense_layers: int = 0  # leading layers that use a dense FFN instead


class ArchSpec(BaseModel):
    name: str
    model_type: str
    architecture: str
    hidden_size: int
    num_layers: int
    vocab_size: int | None
    context_length: int
    intermediate_size: int
    activation: str
    attention: Attention
    moe: MoE | None = None
    norm_type: str  # rmsnorm | layernorm
    norm_mode: str = "pre"  # pre | post (olmo2) | sandwich (gemma)
    positional: str  # rope | learned | alibi | nope
    rope_theta: float | None = None
    hybrid_note: str | None = None  # linear-attention / SSM / conv mixer summary
    parallel_block: bool = False  # falcon-style parallel attention + FFN
    tied_embeddings: bool = False
    total_params: int = 0
    active_params: int = 0
    embed_params: int = 0
    kv_cache_per_token_bytes: int = 0

    @property
    def is_moe(self) -> bool:
        return self.moe is not None


def _get(cfg: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return default


def parse_config(cfg: dict, name: str = "model") -> ArchSpec:
    # multimodal wrappers (gemma3, llama4, ...) nest the LM under text_config
    outer_arch = (cfg.get("architectures") or ["?"])[0]
    if "text_config" in cfg:
        cfg = {**cfg["text_config"], "architectures": cfg.get("architectures")}

    model_type = cfg.get("model_type", "unknown")
    hidden = _get(cfg, "hidden_size", "n_embd", "d_model", "dim", "hidden_dim")
    layers = _get(cfg, "num_hidden_layers", "n_layer", "num_layers", "n_layers")
    heads = _get(cfg, "num_attention_heads", "n_head", "num_heads", "n_heads")
    if hidden is None or layers is None:
        raise SystemExit(
            f"config for '{name}' has no recognizable hidden-size/layer fields "
            f"(model_type={model_type!r}) — not a decoder LM config this engine understands yet."
        )
    if heads is None:
        heads = max(hidden // 128, 1)  # last-resort guess; only affects the heads callout
    kv_heads = _get(cfg, "num_key_value_heads", "num_kv_heads", "n_kv_heads", default=heads)
    if cfg.get("multi_query"):
        kv_heads = 1
    head_dim = _get(cfg, "head_dim", default=hidden // heads)
    vocab = _get(cfg, "vocab_size", "padded_vocab_size")
    ctx = _get(
        cfg,
        "max_position_embeddings",
        "n_positions",
        "n_ctx",
        "max_seq_len",
        "max_sequence_length",
        "seq_length",
        "model_max_length",
        default=0,
    )
    inter = _get(
        cfg,
        "intermediate_size",
        "n_inner",
        "ffn_hidden_size",
        "ffn_dim",
        "block_ff_dim",
        default=4 * hidden,
    )
    act = _get(
        cfg,
        "hidden_act",
        "hidden_activation",
        "activation_function",
        default="silu" if cfg.get("block_use_swiglu") else "gelu",
    )

    # --- attention variant ---
    mla = None
    mla_keys = ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim")
    if all(cfg.get(k) for k in mla_keys):
        kind = AttentionKind.MLA
        mla = MLAConfig(
            q_lora_rank=cfg.get("q_lora_rank"),
            kv_lora_rank=cfg["kv_lora_rank"],
            qk_nope_head_dim=cfg["qk_nope_head_dim"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            v_head_dim=cfg["v_head_dim"],
        )
    elif kv_heads == 1:
        kind = AttentionKind.MQA
    elif kv_heads < heads:
        kind = AttentionKind.GQA
    else:
        kind = AttentionKind.MHA

    sliding = cfg.get("sliding_window") if cfg.get("use_sliding_window", True) else None
    pattern = None
    if sliding:
        if cfg.get("sliding_window_pattern"):
            n = cfg["sliding_window_pattern"]
            pattern = f"{n - 1} local : 1 global"
        elif cfg.get("layer_types"):
            lt = cfg["layer_types"]
            pattern = f"{lt.count('sliding_attention')} local / {lt.count('full_attention')} global"
        else:
            pattern = "all layers"

    attention = Attention(
        kind=kind,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        sliding_window=sliding,
        sliding_pattern=pattern,
        qk_norm=model_type in QK_NORM_MODEL_TYPES,
        attention_bias=bool(cfg.get("attention_bias", model_type == "gpt2")),
        mla=mla,
    )

    # --- MoE ---
    moe = None
    n_experts = _get(cfg, "num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")
    if n_experts and n_experts > 1:
        shared = _get(cfg, "n_shared_experts", "num_shared_experts", default=0)
        if not shared and (
            cfg.get("shared_expert_intermediate_size") or cfg.get("shared_intermediate_size")
        ):
            shared = 1
        moe = MoE(
            num_experts=n_experts,
            experts_per_tok=_get(
                cfg,
                "num_experts_per_tok",
                "experts_per_token",
                "num_experts_per_token",
                "moe_topk",
                "moe_top_k",
                "num_selected_experts",
                default=1,
            ),
            shared_experts=shared,
            moe_intermediate_size=_get(cfg, "moe_intermediate_size", default=inter),
            dense_layers=cfg.get("first_k_dense_replace", 0),
        )

    theta = _get(cfg, "rope_theta", "rotary_emb_base", "rope_embedding_base")
    if cfg.get("alibi"):
        positional = "alibi"
    elif theta is not None:
        positional = "rope"
    elif "alibi" in cfg or cfg.get("rotary"):
        positional = "rope"  # falcon-style: rotary unless alibi
        theta = 10000.0
    elif model_type == "gpt2":
        positional = "learned"
    else:
        positional = "nope"

    spec = ArchSpec(
        name=name,
        model_type=model_type,
        architecture=outer_arch,
        hidden_size=hidden,
        num_layers=layers,
        vocab_size=vocab,
        context_length=ctx,
        intermediate_size=inter,
        activation=act,
        attention=attention,
        moe=moe,
        norm_type="layernorm" if "layer_norm_epsilon" in cfg else "rmsnorm",
        norm_mode=(
            "post"
            if model_type in POST_NORM_MODEL_TYPES
            else "sandwich" if model_type in SANDWICH_NORM_MODEL_TYPES else "pre"
        ),
        positional=positional,
        rope_theta=theta,
        hybrid_note=_hybrid_note(cfg, layers),
        parallel_block=bool(cfg.get("parallel_attn")),
        # gpt2 and gemma tie embeddings by default without saying so in config.json
        tied_embeddings=bool(
            cfg.get(
                "tie_word_embeddings",
                model_type in ("gpt2", "gemma", "gemma2", "gemma3", "gemma3_text"),
            )
        ),
    )
    try:
        _estimate_params(spec)
    except (TypeError, ZeroDivisionError):  # exotic config beyond the estimator; render anyway
        spec.total_params = spec.active_params = 0
    return spec


# --- parameter estimation ---


def _attn_params(s: ArchSpec) -> int:
    a = s.attention
    h = s.hidden_size
    if a.mla is not None:
        m = a.mla
        qk_dim = m.qk_nope_head_dim + m.qk_rope_head_dim
        q = (
            h * m.q_lora_rank + m.q_lora_rank + m.q_lora_rank * a.num_heads * qk_dim
            if m.q_lora_rank
            else h * a.num_heads * qk_dim
        )
        kv = (
            h * (m.kv_lora_rank + m.qk_rope_head_dim)
            + m.kv_lora_rank
            + m.kv_lora_rank * a.num_heads * (m.qk_nope_head_dim + m.v_head_dim)
        )
        o = a.num_heads * m.v_head_dim * h
        return q + kv + o
    q_dim, kv_dim = a.num_heads * a.head_dim, a.num_kv_heads * a.head_dim
    p = h * q_dim + 2 * h * kv_dim + q_dim * h
    if a.attention_bias:
        p += q_dim + 2 * kv_dim + h
    if a.qk_norm:
        p += 2 * a.head_dim
    return p


def _ffn_params(s: ArchSpec, intermediate: int) -> int:
    h = s.hidden_size
    gated = s.activation in ("silu", "swiglu") or "gelu_pytorch_tanh" in s.activation
    n = 3 * h * intermediate if gated else 2 * h * intermediate + intermediate + h
    return n


def _estimate_params(s: ArchSpec) -> None:
    h = s.hidden_size
    vocab = s.vocab_size or 0
    embed = vocab * h + (s.context_length * h if s.positional == "learned" else 0)
    head = 0 if s.tied_embeddings else vocab * h
    norms_per_layer = 2 * h
    final_norm = h

    attn = _attn_params(s)
    total = active = 0
    for layer in range(s.num_layers):
        total += attn + norms_per_layer
        active += attn + norms_per_layer
        if s.moe and layer >= s.moe.dense_layers:
            m = s.moe
            router = h * m.num_experts
            expert = _ffn_params(s, m.moe_intermediate_size)
            # ponytail: shared-expert intermediate = moe_intermediate * n_shared (DeepSeek convention)
            shared = (
                _ffn_params(s, m.moe_intermediate_size * m.shared_experts)
                if m.shared_experts
                else 0
            )
            total += router + m.num_experts * expert + shared
            active += router + m.experts_per_tok * expert + shared
        else:
            dense = _ffn_params(s, s.intermediate_size)
            total += dense
            active += dense

    s.embed_params = embed
    s.total_params = embed + head + total + final_norm
    s.active_params = embed + head + active + final_norm

    # KV cache per token (fp16 = 2 bytes)
    a = s.attention
    if a.mla is not None:
        per_layer = a.mla.kv_lora_rank + a.mla.qk_rope_head_dim
    else:
        per_layer = 2 * a.num_kv_heads * a.head_dim
    s.kv_cache_per_token_bytes = 2 * s.num_layers * per_layer

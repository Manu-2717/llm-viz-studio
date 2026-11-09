# Vendored & modified from llmviz (Apache-2.0, h9-tec/llmviz); see THIRD_PARTY_NOTICES.md. Renamed to LLMViz Studio engine.
"""Raschka-identical architecture figure.

Grammar copied from Sebastian Raschka's LLM Architecture Gallery figures:
bottom-up flow, nested gray containers, accent-tinted repeated block, white pills,
dark attention box, dotted-leader callouts with accent numbers, curly-brace repeat,
FeedForward module inset, MoE layer inset with Router, Resource savings bullets.
"""

from __future__ import annotations

from app.engine.render.svg import SVG, fmt_count, fmt_params
from app.engine.render.tokens import accent_for, scope_id, stylesheet
from app.engine.spec import ArchSpec, AttentionKind

# tower geometry
TCX = 470  # tower flow axis
OUTER_X, OUTER_W = 150, 640
BLK_X, BLK_W = 215, 510
RAIL_X = BLK_X + BLK_W - 60  # residual rail
INSET_X, INSET_W = 900, 590  # right column


def _pill(g: SVG, cx: float, y: float, w: float, h: float, label: str, size: float = 22) -> float:
    g.rect(cx - w / 2, y, w, h, r=14, cls="pill")
    g.text(cx, y + h / 2 + size * 0.36, label, size=size, anchor="middle")
    return y + h


def _attn_lines(s: ArchSpec) -> list[list[tuple[str, str]]]:
    k = s.attention.kind
    if k == AttentionKind.MLA:
        return [[("Multi-head ", "white"), ("Latent", "accent-dark")], [("Attention", "white")]]
    if k == AttentionKind.GQA:
        return [[("Masked ", "white"), ("grouped-query", "accent-dark")], [("attention", "white")]]
    if k == AttentionKind.MQA:
        return [[("Masked ", "white"), ("multi-query", "accent-dark")], [("attention", "white")]]
    return [[("Masked multi-head", "white")], [("attention", "white")]]


_ACRONYMS = {"gpt", "gpt2", "oss", "glm", "lfm", "moe", "ai", "xl"}


def _display_name(name: str) -> str:
    import re

    words = re.split(r"[-_ ]+", name.strip())
    out = []
    for w in words:
        if re.fullmatch(r"\d+(\.\d+)?[bm]", w, re.I):
            out.append(f"({w.upper()})")
        elif w.lower() in _ACRONYMS:
            out.append(w.upper())
        elif w.islower():
            out.append(w.capitalize())
        else:
            out.append(w)
    return " ".join(out)


def _vocab_str(n: int) -> str:
    return f"{round(n / 1000)}k"


def _ctx_str(n: int) -> str:
    if n >= 16384 and n % 1024 == 0:
        return f"{n // 1024}k"
    return f"{n:,}"


def _callout(g: SVG, x: float, y: float, lines: list[list[tuple[str, str]]], anchor="start"):
    for i, runs in enumerate(lines):
        g.rich_text(x, y + i * 30, runs, size=21, anchor=anchor)


def draw_tower(g: SVG, s: ArchSpec, accent: str, x_off: float = 0.0, title: bool = True) -> float:
    """The bottom-up model tower with callouts. Returns bottom y."""
    cx = TCX + x_off
    outer_x, blk_x = OUTER_X + x_off, BLK_X + x_off
    rail_x = RAIL_X + x_off
    outer_idx = g.mark()

    if title:
        g.text(cx, 74, _display_name(s.name), size=46, cls="accent", anchor="middle", weight=800)

    # --- top: output arrow + linear output layer ---
    outer_top = 165
    g.arrow(cx, 261, cx, 128)
    y = _pill(g, cx, 261, 310, 52, "Linear output layer")
    lo_bot = y
    y = _pill(
        g, cx, y + 88, 260, 50, "Final RMSNorm" if s.norm_type == "rmsnorm" else "Final LayerNorm"
    )
    fn_bot = y
    fn_top = y - 50

    # --- repeated block container ---
    norm = "RMSNorm" if s.norm_type == "rmsnorm" else "LayerNorm"
    sandwich = s.norm_mode == "sandwich"
    post = s.norm_mode == "post"
    blk_top = y + 92
    blk_idx = g.mark()
    y = blk_top + 34

    plus2 = y + 16
    g.plus_node(cx, plus2)
    y += 52
    post2_bot = post2_top = None
    if sandwich or post:
        post2_top = y
        y = _pill(g, cx, y, 240, 44, f"{norm} 2" if post else f"Post-FFN {norm}", size=19)
        post2_bot = y
        y += 34
    pill_top = y
    g.rect(cx - 105, y, 210, 48, r=14, cls="moepill")
    g.text(cx, y + 31, "MoE" if s.moe else "Feed forward", size=22, anchor="middle")
    ffn_pill = (cx + 105, y + 24)
    pill_bot = y + 48
    y += 48
    rms2_top = rms2_bot = None
    if not post:
        rms2_top = y + 40
        y = _pill(g, cx, y + 40, 240, 48, f"{norm} 2")
        rms2_bot = y
    else:
        y += 8
        rms2_bot = y
    plus1 = y + 60
    g.plus_node(cx, plus1)
    y += 92
    post1_bot = post1_top = None
    if sandwich or post:
        post1_top = y - 8
        y = _pill(g, cx, y - 8, 240, 44, f"{norm} 1" if post else f"Post-attn {norm}", size=19)
        post1_bot = y
        y += 34

    attn_h = 96
    g.rect(cx - 155, y, 310, attn_h, r=16, cls="attnbox")
    for i, runs in enumerate(_attn_lines(s)):
        g.rich_text(cx, y + 40 + i * 30, runs, size=23, anchor="middle", weight=400)
    attn_y, attn_bot = y, y + attn_h
    y += attn_h
    if not post:
        rms1_top = y + 42
        y = _pill(g, cx, y + 42, 240, 48, f"{norm} 1")
        rms1_bot = y
    else:
        rms1_top = None
        rms1_bot = y + 12
        y += 12
    blk_bot = y + 36

    # main flow spine: one upward arrow per receiver, exactly like his figures
    PR = 15  # plus-node radius
    g.arrow(cx, fn_top, cx, lo_bot + 3)  # Final norm -> Linear output layer
    g.arrow(cx, plus2 - PR, cx, fn_bot + 3)  # block exit -> Final norm
    if sandwich or post:
        g.arrow(cx, post2_top, cx, plus2 + PR + 2)  # post/2nd norm -> (+)
        g.arrow(cx, pill_top, cx, post2_bot + 3)  # FFN -> that norm
        g.arrow(cx, post1_top, cx, plus1 + PR + 2)  # post/1st norm -> (+)
        g.arrow(cx, attn_y, cx, post1_bot + 3)  # attention -> that norm
    else:
        g.arrow(cx, pill_top, cx, plus2 + PR + 2)  # FFN/MoE -> (+)
        g.arrow(cx, attn_y, cx, plus1 + PR + 2)  # attention -> (+)
    if not post:
        g.arrow(cx, rms2_top, cx, pill_bot + 3)  # norm 2 -> FFN/MoE
        g.arrow(cx, rms1_top, cx, attn_bot + 3)  # norm 1 -> attention
        g.arrow(cx, plus1 - PR, cx, rms2_bot + 3)  # (+) -> norm 2
    else:
        g.arrow(cx, plus1 - PR, cx, pill_bot + 3)  # (+) -> FFN/MoE directly

    # residual rails: branch below each RMSNorm, up the right side, into the (+)
    for branch_y, plus_y in ((rms1_bot + 20, plus1), (rms2_bot + 22, plus2)):
        g.path(f"M {cx} {branch_y} L {rail_x} {branch_y} L {rail_x} {plus_y}")
        g.arrow(rail_x, plus_y, cx + PR + 2, plus_y)

    g.rect_at(blk_idx, blk_x, blk_top, BLK_W, blk_bot - blk_top, r=34, cls="blockc")

    # RoPE pill feeding attention, straddling the outer container's left edge
    if s.positional == "rope":
        rope_cx = outer_x - 5
        g.rect(rope_cx - 105, attn_y + attn_h / 2 - 26, 210, 52, r=14, cls="pill")
        g.text(rope_cx, attn_y + attn_h / 2 + 8, "RoPE", size=22, anchor="middle")
        g.arrow(rope_cx + 105, attn_y + attn_h / 2, cx - 157, attn_y + attn_h / 2)

    # curly brace + repeat count at the block's lower-left corner
    g.text(blk_x - 26, blk_bot + 22, "{", size=140, anchor="middle", weight=300)
    g.rich_text(blk_x - 158, blk_bot - 12, [(f"{s.num_layers} ×", "accent")], size=34)

    # --- below the block: embeddings (token embedding first, positional above it) ---
    y = blk_bot
    g.arrow(cx, y + 76, cx, rms1_bot + 3)
    emb_top = y + 76
    if s.positional == "learned":
        y = _pill(g, cx, emb_top, 360, 52, "Positional embedding layer")
        tok_top = y + 78
        g.arrow(cx, tok_top, cx, y + 4)
        y = _pill(g, cx, tok_top, 340, 52, "Token embedding layer")
    else:
        y = _pill(g, cx, emb_top, 340, 52, "Token embedding layer")
    emb_bot = y
    outer_bot = y + 44

    g.rect_at(outer_idx, outer_x, outer_top, OUTER_W, outer_bot - outer_top, r=40, cls="outer")

    # --- input, outside the model ---
    g.arrow(cx, outer_bot + 66, cx, emb_bot + 4)
    y = _pill(g, cx, outer_bot + 66, 250, 50, "Tokenized text")
    g.arrow(cx, y + 62, cx, y + 4)
    g.text(cx, y + 100, "Sample input text", size=24, cls="mono", anchor="middle")
    bottom = y + 118

    # --- callouts ---
    if s.vocab_size:
        g.rich_text(
            cx + 190,
            148,
            [("Vocabulary size of ", ""), (_vocab_str(s.vocab_size), "accent")],
            size=21,
        )
        g.leader(cx + 212, 162, cx + 140, 258)

    if s.context_length:
        _callout(
            g,
            x_off + 18,
            blk_bot + 66,
            [
                [("Supported", "")],
                [("context length", "")],
                [("of ", ""), (_ctx_str(s.context_length), "accent"), (" tokens", "")],
            ],
        )
        if s.positional == "rope":
            g.leader(x_off + 44, blk_bot + 40, x_off + 58, attn_y + attn_h / 2 + 32)
        elif s.positional == "learned":
            g.leader(x_off + 100, blk_bot + 140, cx - 184, emb_top + 30)

    a = s.attention
    head_lines = [[(f"{a.num_heads} ", "accent"), ("heads", "")]]
    if a.kind in (AttentionKind.GQA, AttentionKind.MQA):
        head_lines.append([(f"{a.num_kv_heads} ", "accent"), ("KV groups", "")])
    if a.sliding_window:
        head_lines.append(
            [("sliding window of ", ""), (fmt_count(a.sliding_window), "accent"), (" tokens", "")]
        )
    if a.qk_norm:
        head_lines.append([("QK-Norm on queries & keys", "")])
    if s.positional == "alibi":
        head_lines.append([("ALiBi positional bias", "")])
    if s.parallel_block:
        head_lines.append([("parallel attention + FFN block", "")])
    if s.hybrid_note:
        import re as _re

        runs = []
        for part in _re.split(r"(\d+)", s.hybrid_note):
            if part:
                runs.append((part, "accent" if part.isdigit() else ""))
        head_lines.append(runs)
    _callout(g, cx + 330, emb_top + 44, head_lines)
    g.leader(cx + 320, emb_top + 20, cx + 152, attn_bot - 8)
    heads_bottom = emb_top + 44 + len(head_lines) * 30

    _callout(
        g,
        cx + 210,
        emb_bot + 96,
        [[("Embedding", "")], [("dimension of ", ""), (f"{s.hidden_size:,}", "accent")]],
    )
    g.leader(cx + 245, emb_bot + 66, cx + 176, emb_bot - 20)

    if s.moe and s.moe.dense_layers:
        _callout(
            g,
            x_off + 14,
            outer_bot + 58,
            [
                [
                    ("First " if s.moe.dense_layers > 1 else "The first ", ""),
                    (str(s.moe.dense_layers) if s.moe.dense_layers > 1 else "", "accent"),
                    (" blocks use" if s.moe.dense_layers > 1 else "block uses", ""),
                ],
                [("dense FFN with", "")],
                [("hidden size ", ""), (f"{s.intermediate_size:,}", "accent")],
                [("instead of MoE", "")],
            ],
        )
        g.leader(x_off + 252, outer_bot + 44, blk_x + 66, blk_bot - 24)
    elif s.tied_embeddings:
        _callout(
            g,
            x_off + 14,
            outer_bot + 58,
            [[("Output layer reuses the", "")], [("token embedding weights", "")]],
        )
        g.leader(x_off + 296, outer_bot + 40, cx - 176, emb_bot - 20)

    return bottom, {"ffn_pill": ffn_pill, "heads_bottom": heads_bottom}


def _inset_ffn(g: SVG, s: ArchSpec, y0: float) -> float:
    """FeedForward module inset — gray fill, black dotted border. Returns bottom y."""
    gated = s.activation in ("silu", "swiglu") or "tanh" in s.activation
    act = {"silu": "SiLU", "gelu_pytorch_tanh": "GELU", "gelu_new": "GELU", "gelu": "GELU"}.get(
        s.activation, s.activation
    )
    kind = "SwiGLU" if s.activation == "silu" else ("GeGLU" if gated else act)
    x, w, h = INSET_X, INSET_W, 330
    g.text(x + w / 2, y0 - 18, f"FeedForward ({kind}) module", size=27, anchor="middle", weight=700)
    g.rect(x, y0, w, h, r=34, cls="inset")
    cxl = x + w * 0.36
    _pill(g, cxl, y0 + h - 88, 200, 52, "Linear layer")
    g.arrow(cxl, y0 + h - 88, cxl, y0 + h - 142)
    if gated:
        _pill(g, cxl, y0 + h - 194, 210, 52, f"{act} activation")
        cxr = x + w * 0.79
        _pill(g, cxr, y0 + h - 194, 180, 52, "Linear layer")
        g.arrow(cxl, y0 + h - 194, cxl, y0 + 94)
        hy = y0 + 112
        ox = cxl + 116
        g.path(f"M {cxr} {y0 + h - 194} L {cxr} {hy} L {ox + 13} {hy}")
        g.otimes_node(ox, hy)
        g.line(ox - 13, hy, cxl + 1.5, hy)
        _pill(g, cxl, y0 + 42, 200, 52, "Linear layer")
    else:
        _pill(g, cxl, y0 + h - 194, 210, 52, f"{act} activation")
        g.arrow(cxl, y0 + h - 194, cxl, y0 + 94)
        _pill(g, cxl, y0 + 42, 200, 52, "Linear layer")

    dim = s.moe.moe_intermediate_size if s.moe else s.intermediate_size
    if s.moe:
        lines = [
            [("Input expert size: ", ""), (f"{s.hidden_size:,}", "accent")],
            [("Intermediate projection size: ", ""), (f"{dim:,}", "accent")],
        ]
    else:
        lines = [
            [("Intermediate hidden", "")],
            [("layer dimension of ", ""), (f"{dim:,}", "accent")],
        ]
    _callout(g, x + 76, y0 + h + 52, lines)
    g.leader(cxl + 66, y0 + h - 62, x + w * 0.42, y0 + h + 22)
    return y0 + h + 52 + len(lines) * 30


def _inset_moe(g: SVG, s: ArchSpec, y0: float) -> float:
    """MoE layer inset — accent dotted border: Router fans to expert cards into (+)."""
    m = s.moe
    x, w, h = INSET_X, INSET_W, 470
    g.rect(x, y0, w, h, r=40, cls="inset-accent")
    g.text(x + w - 46, y0 + 70, "MoE layer", size=27, anchor="end", weight=700)
    cx = x + w / 2
    pcx, pcy, prad = cx, y0 + 104, 18  # the (+) node
    g.plus_node(pcx, pcy, r=prad)
    g.arrow(cx, pcy - prad - 2, cx, y0 + 36)  # (+) output, leaving from the circle's top edge

    lw, eh = 200, 50
    lx, rx = x + 56, x + w - 56 - lw
    ey = y0 + 196
    for px, tag, tag_cls in ((lx, "1", "tagdark"), (rx, str(m.num_experts), "tagaccent")):
        g.rect(px, ey, lw, eh, r=12, cls="moepill")
        g.text(px + lw / 2, ey + 32, "Feed forward", size=21, anchor="middle")
        tw = 36 + 12 * (len(tag) - 1)
        tx = px + lw - tw + 10
        g.rect(tx, ey + 34, tw, 36, r=8, cls=tag_cls)
        g.text(tx + tw / 2, ey + 59, tag, size=19, cls="white", anchor="middle", weight=700)
    g.text(cx, ey + 42, "· · ·", size=30, anchor="middle", weight=700)

    ry = y0 + h - 96
    g.rect(cx - 78, ry, 156, 56, r=16, cls="router")
    g.text(cx, ry + 36, "Router", size=22, anchor="middle")
    g.arrow(cx, ry + 56 + 34, cx, ry + 56 + 3)  # input into the Router's bottom edge

    def _edge_arrow(x1, y1, ccx, ccy, rr):
        """Arrow from (x1,y1) stopping exactly on the circle edge toward (ccx,ccy)."""
        import math

        ang = math.atan2(ccy - y1, ccx - x1)
        g.arrow(x1, y1, ccx - (rr + 2) * math.cos(ang), ccy - (rr + 2) * math.sin(ang))

    # Router fans out from its top edge: left card, right card, and the "..." experts
    g.arrow(cx - 40, ry, lx + lw * 0.62, ey + eh + 3)
    g.arrow(cx + 40, ry, rx + lw * 0.38, ey + eh + 3)
    g.arrow(cx, ry - 3, cx, ey + eh + 14)
    # each expert's output converges on the (+), heads landing on the circle edge
    _edge_arrow(lx + lw * 0.62, ey - 1, pcx, pcy, prad)
    _edge_arrow(rx + lw * 0.38, ey - 1, pcx, pcy, prad)
    return y0 + h


def _bullets_moe(g: SVG, s: ArchSpec, y0: float) -> float:
    m = s.moe
    x = INSET_X + 44
    g.rich_text(x, y0, [("Resource savings:", "accent")], size=23)
    experts = (
        f"{m.shared_experts} (shared) + " if m.shared_experts else ""
    ) + f"{m.experts_per_tok} experts"
    rows = [
        [("Model size is ", ""), (fmt_params(s.total_params), "accent")],
        [("but only ", ""), (experts, "accent"), (" active per token", "")],
        [("only ", ""), (fmt_params(s.active_params), "accent"), (" parameters are", "")],
        [("active per inference step", "")],
    ]
    y = y0 + 40
    for i, runs in enumerate(rows):
        if i != 3:
            g.text(x + 4, y, "•", size=22, weight=700)
        g.rich_text(x + 30, y, runs, size=21, weight=400)
        y += 33
    return y


def _kv_table(g: SVG, s: ArchSpec, y0: float) -> float:
    """KV-cache footprint at full context — the number people actually size GPUs by."""
    if not (s.kv_cache_per_token_bytes and s.context_length):
        return y0
    x = INSET_X + 44
    g.rich_text(
        x,
        y0,
        [("KV cache at ", ""), (_ctx_str(s.context_length), "accent"), (" context", "")],
        size=23,
    )
    per_tok = s.kv_cache_per_token_bytes
    full = per_tok * s.context_length
    rows = [
        ("per token", f"{per_tok / 1024:.1f} KB" if per_tok >= 1024 else f"{per_tok} B"),
        ("fp16 / bf16", f"{full / 2**30:.1f} GB"),
        ("fp8", f"{full / 2 / 2**30:.1f} GB"),
    ]
    y = y0 + 38
    for label, value in rows:
        g.text(x + 6, y, label, size=20)
        g.text(x + 330, y, value, size=20, weight=700, anchor="end")
        y += 32
    if s.attention.sliding_window:
        g.text(
            x + 6,
            y,
            f"(sliding-window layers cap at {fmt_count(s.attention.sliding_window)} tokens)",
            size=16,
        )
        y += 26
    return y


def render_model(
    s: ArchSpec, theme: str = "auto", accent: str | None = None, animate: bool = False
) -> str:
    accent = accent or accent_for(s.model_type)
    g = SVG()
    bottom, coords = draw_tower(g, s, accent)
    fx, fy = coords["ffn_pill"]

    ffn_top = 196
    y = _inset_ffn(g, s, ffn_top)
    if s.moe:
        moe_top = y + 60
        y2 = _inset_moe(g, s, moe_top)
        # accent-dotted leader from the tower's MoE pill to the MoE inset (his style)
        g.line(fx + 4, fy, INSET_X + 26, moe_top + 150, cls="leader-accent", width=4.5)
        # black dotted leader from an expert card up to the FeedForward module inset
        g.leader(INSET_X + 156, moe_top + 196, INSET_X + 140, ffn_top + 330 - 26)
        y = _bullets_moe(g, s, max(y2 + 56, coords["heads_bottom"] + 40))
    else:
        g.leader(fx + 4, fy, INSET_X - 26, ffn_top + 120)
    y = _kv_table(g, s, y + 64)

    width = INSET_X + INSET_W + 60
    height = int(max(bottom + 30, y + 40))
    css = stylesheet(accent)
    if animate:
        from app.engine.render.tokens import ANIM_CSS

        g.animate()
        css += ANIM_CSS
    return g.document(width, height, css, svg_id=scope_id(accent))

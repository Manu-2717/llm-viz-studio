"""SVG export rendered from the canonical graph (vector text, no UI chrome)."""
from __future__ import annotations
import html
from .graph import ArchGraph

_KIND_FILL = {"embedding": "#dbeafe", "norm": "#fef3c7", "attention": "#ede9fe", "mlp": "#dcfce7",
              "moe": "#fee2e2", "block": "#f3f4f6", "head": "#ffedd5", "op": "#ffffff", "unknown": "#fecaca"}

def graph_to_svg(g: ArchGraph, max_nodes: int = 60) -> str:
    nodes = [n for n in g.nodes if n.level in (1, 2)][:max_nodes]
    W, row_h, pad = 900, 64, 20
    H = pad * 2 + row_h * len(nodes) + 60
    ids = {n.id: i for i, n in enumerate(nodes)}
    pos = {}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
             "<style>text{font-family:Arial,sans-serif;font-size:15px} .t{font-size:12px;fill:#555}</style>",
             f'<text x="{pad}" y="30" font-size="20" font-weight="bold">{html.escape(g.model_id)} — architecture graph ({html.escape(g.confidence)})</text>']
    y = 60
    for i, n in enumerate(nodes):
        x = pad + (n.level - 1) * 360
        pos[n.id] = (x + 260, y + row_h // 2)
        fill = _KIND_FILL.get(n.kind, "#f3f4f6")
        parts.append(f'<rect x="{x}" y="{y}" width="260" height="{row_h-10}" rx="8" fill="{fill}" stroke="#999"/>')
        parts.append(f'<text x="{x+10}" y="{y+24}">{html.escape(n.label)}</text>')
        sub = f"{n.kind} · {n.params:,} params · {n.confidence}"
        if n.collapsed_count:
            sub += f" · ×{n.collapsed_count}"
        parts.append(f'<text class="t" x="{x+10}" y="{y+44}">{html.escape(sub)}</text>')
        y += row_h
    for e in g.edges:
        if e.src in pos and e.dst in pos:
            x1, y1 = pos[e.src]; x2, y2 = pos[e.dst]
            parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#666" stroke-width="1.5" marker-end="url(#arr)"/>')
    parts.append('<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" fill="#666"/></marker></defs>')
    parts.append("</svg>")
    return "\n".join(parts)

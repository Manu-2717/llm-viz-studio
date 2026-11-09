# Vendored & modified from llmviz (Apache-2.0, h9-tec/llmviz); see THIRD_PARTY_NOTICES.md. Renamed to LLMViz Studio engine.
"""Minimal SVG builder — a list of elements and f-strings. No dependency, full control."""

from __future__ import annotations

import math
from xml.sax.saxutils import escape


class SVG:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def rect(self, x, y, w, h, r=6, cls="", extra=""):
        self.parts.append(
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{r}" class="{cls}" {extra}/>'
        )

    def text(self, x, y, s, size=13, cls="", anchor="start", weight=None):
        w = f' font-weight="{weight}"' if weight else ""
        self.parts.append(
            f'<text x="{x:g}" y="{y:g}" font-size="{size}" class="{cls}" text-anchor="{anchor}"{w}>{escape(str(s))}</text>'
        )

    def rich_text(self, x, y, runs, size=19, anchor="start", weight=700):
        """runs: list of (text, cls). Sequentially positioned text elements instead of
        tspans — cairosvg mis-centers tspan runs, browsers don't need them either."""
        from app.engine.render.widths import text_width

        bold = bool(weight) and int(weight) >= 600
        # extra tracking at run boundaries with a space: renderers swallow edge spaces
        widths = []
        for i, (t, _) in enumerate(runs):
            pad = (
                0.14 * size
                if t.endswith(" ") or (i + 1 < len(runs) and runs[i + 1][0].startswith(" "))
                else 0
            )
            widths.append(text_width(t, size, bold) + pad)
        total = sum(widths)
        cur = x - total / 2 if anchor == "middle" else (x - total if anchor == "end" else x)
        for (t, cls), tw in zip(runs, widths, strict=True):
            self.parts.append(
                f'<text x="{cur:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" '
                f'class="{cls}" xml:space="preserve">{escape(t)}</text>'
            )
            cur += tw

    def line(self, x1, y1, x2, y2, cls="flow", width=None):
        w = f' stroke-width="{width}"' if width else ""
        self.parts.append(
            f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" class="{cls}"{w}/>'
        )

    def path(self, d, cls="flow", extra=""):
        self.parts.append(f'<path d="{d}" class="{cls}" {extra}/>')

    def arrow(self, x1, y1, x2, y2, cls="flow", head=17):
        """Straight arrow with a solid triangular head at (x2,y2), Raschka-sized."""
        ang = math.atan2(y2 - y1, x2 - x1)
        bx, by = x2 - head * math.cos(ang), y2 - head * math.sin(ang)
        px, py = head * 0.38 * -math.sin(ang), head * 0.38 * math.cos(ang)
        self.line(x1, y1, bx, by, cls=cls)
        self.parts.append(
            f'<path d="M {bx + px:g} {by + py:g} L {x2:g} {y2:g} L {bx - px:g} {by - py:g} Z" '
            f'class="flowhead"/>'
        )

    def leader(self, x1, y1, x2, y2):
        """Dotted callout leader line."""
        self.line(x1, y1, x2, y2, cls="leader")

    def circle(self, cx, cy, r, cls="", extra=""):
        self.parts.append(f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}" class="{cls}" {extra}/>')

    def plus_node(self, cx, cy, r=15):
        """The ⊕ residual-add node."""
        self.circle(cx, cy, r, cls="plusc")
        self.line(cx - r * 0.85, cy, cx + r * 0.85, cy)
        self.line(cx, cy - r * 0.85, cx, cy + r * 0.85)

    def otimes_node(self, cx, cy, r=13):
        """The ⊗ elementwise-multiply node."""
        self.circle(cx, cy, r, cls="plusc")
        d = r * 0.4
        self.line(cx - d, cy - d, cx + d, cy + d)
        self.line(cx - d, cy + d, cx + d, cy - d)

    def raw(self, s: str):
        self.parts.append(s)

    def mark(self) -> int:
        """Index to later insert an element beneath everything drawn after this point."""
        return len(self.parts)

    def rect_at(self, idx, x, y, w, h, r=6, cls="", extra=""):
        self.parts.insert(
            idx,
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{r}" class="{cls}" {extra}/>',
        )

    def animate(self, step: float = 0.012, dur: float = 0.5) -> None:
        """Staggered build-up: each element fades in, in draw order (browser-only; PNG ignores)."""
        out, i = [], 0
        for part in self.parts:
            if part.startswith(("<rect", "<text", "<line", "<path", "<circle")):
                style = f'style="opacity:0; animation: llmviz-studio-in {dur}s ease-out forwards {i * step:.3f}s"'
                part = part.replace(" ", f" {style} ", 1)
                i += 1
            out.append(part)
        self.parts = out

    def document(self, width: int, height: int, css: str, svg_id: str = "") -> str:
        body = "\n".join(self.parts)
        idattr = f' id="{svg_id}"' if svg_id else ""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg"{idattr} width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">\n'
            f"<style>{css}</style>\n"
            f'<rect width="{width}" height="{height}" class="surface"/>\n'
            f"{body}\n</svg>"
        )


def fmt_params(n: int) -> str:
    if n >= 1e12:
        v = f"{n / 1e12:.1f}".rstrip("0").rstrip(".")
        return v + "T"
    if n >= 1e9:
        v = f"{n / 1e9:.1f}".rstrip("0").rstrip(".")
        return v + "B"
    if n >= 1e6:
        return f"{n / 1e6:.0f}M"
    return f"{n:,}"


def fmt_count(n: int) -> str:
    """Raschka-style: 128k, 40k, 8192."""
    if n >= 10_000:
        k = n / 1024 if n % 1024 == 0 else n / 1000
        return f"{k:.1f}".rstrip("0").rstrip(".") + "k"
    return f"{n:,}"


def fmt_theta(theta: float) -> str:
    t = int(theta)
    if t >= 1_000_000 and t % 1_000_000 == 0:
        return f"{t // 1_000_000}M"
    if t >= 1_000 and t % 1_000 == 0:
        return f"{t // 1_000}k"
    return f"{t:,}"

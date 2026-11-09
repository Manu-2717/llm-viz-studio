# DiagramView

Independent module: backend-SVG tower renderer viewer. Owns fetching (/architecture + /diagram), zoom/pan state, toolbar, SVG/PNG downloads, info panel.

Must not import from ../graph. Shared code only via ../../shared (api, model-metadata).

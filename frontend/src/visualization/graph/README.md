# GraphView

Independent module: React Flow computational-graph explorer over the canonical graph API (/graph, /graph?layer=, /tensors, /graph/download).

Owns graph fetching, level (overview/block/tensor) state, subgraph collapse (double-click), layer selector, tensor inspector, graph exports.

Must not import from ../diagram. Shared code only via ../../shared.

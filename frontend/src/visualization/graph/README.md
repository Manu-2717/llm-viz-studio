# GraphView

Independent React Flow computational-graph explorer over the canonical graph API (`/graph`, `/graph?layer=`, `/tensors`, `/graph/download`).

Owns graph fetching, overview/block granularity, component search and focus, subgraph collapse (double-click), layer selector, tensor inspector, and graph exports.

Must not import from `../diagram`. Shared code only via `../../shared`.

// Shared API layer — used by BOTH DiagramView and GraphView.
// Neither view depends on the other's renderer; they only share model metadata + transport.
export async function fetchJSON(url: string, signal?: AbortSignal): Promise<any> {
  const r = await fetch(url, { signal })
  if (!r.ok) {
    const d = await r.json().catch(() => ({}))
    throw new Error(d.detail || `Request failed (${r.status})`)
  }
  return r.json()
}
export async function fetchText(url: string, signal?: AbortSignal): Promise<string> {
  const r = await fetch(url, { signal })
  if (!r.ok) {
    const d = await r.json().catch(() => ({}))
    throw new Error(d.detail || `Request failed (${r.status})`)
  }
  return r.text()
}
export const modelQuery = (model: string) => `model=${encodeURIComponent(model)}`
export const downloadURL = (model: string, format: 'svg' | 'png', scale = 2) =>
  `/api/v1/models/download?${modelQuery(model)}&format=${format}&scale=${scale}`
export const graphDownloadURL = (model: string, format: 'svg' | 'png', scale = 2) =>
  `/api/v1/models/graph/download?${modelQuery(model)}&format=${format}&scale=${scale}`

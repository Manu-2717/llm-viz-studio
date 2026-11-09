import { useEffect, useMemo, useState } from 'react'
import { fetchJSON, modelQuery, graphDownloadURL } from '../../shared/api/api'
import { ReactFlow, Background, Controls, type Node, type Edge } from '@xyflow/react'
import '@xyflow/react/dist/style.css'

const KIND_STYLE: Record<string, string> = {
  embedding: '#ffffff', norm: '#fafafa', attention: '#f0f0f0', mlp: '#e8e8e8',
  moe: '#e2e2e2', block: '#fafafa', head: '#f5f5f5', op: '#ffffff', unknown: '#fafafa',
}

export default function GraphView({ model }: { model: string }) {
  const [graph, setGraph] = useState<any>(null)
  const [layer, setLayer] = useState<number | null>(null)
  const [selected, setSelected] = useState<any>(null)
  const [tensors, setTensors] = useState<any[]>([])
  const [err, setErr] = useState('')
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())

  useEffect(() => {
    setGraph(null); setSelected(null); setTensors([]); setCollapsed(new Set())
    const url = layer === null
      ? `/api/v1/models/graph?${modelQuery(model)}`
      : `/api/v1/models/graph?${modelQuery(model)}&layer=${layer}`
    fetchJSON(url).then(setGraph).catch((e: any) => setErr(e.message))
  }, [model, layer])

  useEffect(() => {
    if (!selected) { setTensors([]); return }
    const l = selected.layer ?? layer
    const url = `/api/v1/models/tensors?${modelQuery(model)}${l !== null && l !== undefined ? `&layer=${l}` : ''}&limit=200`
    fetchJSON(url).then(d => setTensors(d.tensors || [])).catch(() => setTensors([]))
  }, [selected, model, layer])

  const { nodes, edges } = useMemo(() => {
    if (!graph) return { nodes: [] as Node[], edges: [] as Edge[] }
    // subgraph collapse: hide children of collapsed nodes (nodes whose id starts with collapsed id + ':')
    const all: any[] = graph.nodes || []
    const visible = all.filter((n: any) => ![...collapsed].some(c => n.id.startsWith(c + ':')))
    const ns: Node[] = visible.map((n: any, i: number) => ({
      id: n.id,
      position: { x: (n.level - 1) * 340, y: i * 90 },
      data: { label: `${collapsed.has(n.id) ? '▸' : '▾'} ${n.label}\n${n.kind} · ${Number(n.params || 0).toLocaleString()} · ${n.confidence}` },
      style: { background: KIND_STYLE[n.kind] || '#fff', border: n.confidence === 'unknown' ? '2px dashed #737373' : '1px solid #d4d4d4', borderRadius: 18, color: '#0a0a0a', fontSize: 12, width: 280, boxShadow: '0 1px 2px rgba(0,0,0,.05)' },
    }))
    const es: Edge[] = (graph.edges || []).map((e: any, i: number) => ({
      id: `e${i}`, source: e.src, target: e.dst, animated: e.kind === 'residual', label: e.kind === 'residual' ? 'residual' : '',
    })).filter((e: Edge) => ns.some(n => n.id === e.source) && ns.some(n => n.id === e.target))
    return { nodes: ns, edges: es }
  }, [graph, collapsed])

  if (err) return <div className="loading">Graph unavailable: {err}</div>
  if (!graph) return <div className="loading" role="status">Building architecture graph…</div>
  const selTensors = selected ? tensors.filter((t: any) => selected.tensors?.some((s: any) => s.name === t.name) || t.module === selected.kind || selected.id.includes(t.module)) : []
  return <div className="graph-wrap">
    <div className="graph-toolbar">
      <span>Level: {layer === null ? '1 · Model overview' : `2 · Layer ${layer} block`} · source: {graph.metadata_source} · confidence: {graph.confidence}</span>
      {layer === null
        ? <button onClick={() => setLayer(graph.layers?.[0] ?? 0)}>Expand layer {(graph.layers?.[0] ?? 0)} →</button>
        : <button onClick={() => setLayer(null)}>← Back to overview</button>}
      {graph.layers?.length > 1 && <select aria-label="Layer" value={layer ?? ''} onChange={e => setLayer(e.target.value === '' ? null : Number(e.target.value))}>
        <option value="">Overview</option>{graph.layers.slice(0, 200).map((l: number) => <option key={l} value={l}>Layer {l}</option>)}
      </select>}
      <a href={graphDownloadURL(model, 'svg')} download>Graph SVG</a>
      <a href={graphDownloadURL(model, 'png')} download>Graph PNG</a>
    </div>
    {graph.metadata_note && <p className="note">{graph.metadata_note}</p>}
    <p className="note">Double-click a node to expand/collapse its subgraph. Click a node to inspect tensors.</p>
    <div style={{ height: 520 }}>
      <ReactFlow nodes={nodes} edges={edges} onNodeClick={(_, n) => setSelected(graph.nodes.find((x: any) => x.id === n.id))} onNodeDoubleClick={(_, n) => setCollapsed(prev => { const s = new Set(prev); s.has(n.id) ? s.delete(n.id) : s.add(n.id); return s })} fitView colorMode="light">
        <Background /><Controls />
      </ReactFlow>
    </div>
    {selected && <aside className="panel tensor-panel"><h3>{selected.label}</h3>
      <p className="note">{selected.kind} · {selected.confidence} · {Number(selected.params || 0).toLocaleString()} params {selected.evidence ? `· ${selected.evidence}` : ''}</p>
      <table><thead><tr><th>Tensor</th><th>Shape</th><th>Dtype</th><th>Params</th><th>Shard</th></tr></thead>
      <tbody>{(selected.tensors?.length ? selected.tensors : selTensors).map((t: any) => <tr key={t.name}><td><code>{t.name}</code></td><td>{(t.shape || []).join('×')}</td><td>{t.dtype}</td><td>{Number(t.numel || 0).toLocaleString()}</td><td>{t.shard}</td></tr>)}</tbody></table>
      <details><summary>Raw metadata</summary><pre>{JSON.stringify(selected, null, 1).slice(0, 4000)}</pre></details>
    </aside>}
  </div>
}

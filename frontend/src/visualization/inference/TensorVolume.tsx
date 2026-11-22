import type { ChannelTensor } from '../../engine/inference/tinyRecall'

export default function TensorVolume({ tensor, position }: { tensor: ChannelTensor; position: number }) {
  const { values } = tensor
  const rows = values.length
  const columns = values[0].length
  const max = tensor.probability ? 1 : Math.max(1e-9, ...values.flat().filter(Number.isFinite).map(Math.abs))
  const cell = Math.min(30, 660 / (rows + columns))
  const amplitude = cell * 1.7
  const width = (rows + columns) * cell + 30
  const height = (rows + columns) * cell * 0.4 + amplitude * 2 + 40
  const project = (column: number, row: number, z: number) =>
    `${15 + (rows + column - row) * cell},${amplitude + 14 + (column + row) * cell * 0.4 - z}`

  const cells = values.flatMap((row, y) => row.map((value, x) => ({ x, y, value })))
    .sort((a, b) => a.x + a.y - (b.x + b.y))
  return (
    <figure className="inference-volume">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${tensor.name} shown as a three-dimensional signed value plot`}>
        {cells.map(({ x, y, value }) => {
          const finite = Number.isFinite(value)
          const z = finite ? value / max * amplitude : 0
          const start = project(x, y, 0), right = project(x + 0.84, y, 0)
          const bottom = project(x + 0.84, y + 0.84, 0), left = project(x, y + 0.84, 0)
          const a = project(x, y, z), b = project(x + 0.84, y, z)
          const c = project(x + 0.84, y + 0.84, z), d = project(x, y + 0.84, z)
          const color = value < 0 ? ['#de9d71', '#b56941', '#8c4528'] : ['#65b6c2', '#318394', '#1c6271']
          const row = tensor.rowLabels?.[y] ?? y
          const column = tensor.columnLabels?.[x] ?? x
          return (
            <g key={`${y}-${x}`}>
              <title>{`Position ${row} · Column ${column} · ${finite ? String(value) : 'masked'}`}</title>
              <polygon points={`${start} ${right} ${bottom} ${left}`}
                fill={finite ? (y === position ? '#b8e0d7' : '#e8f0f3') : '#d3d9de'}
                stroke={y === position ? '#367d74' : '#afc3ce'} strokeWidth=".8" />
              {z !== 0 && (
                <>
                  <polygon points={`${b} ${c} ${bottom} ${right}`} fill={color[2]} />
                  <polygon points={`${d} ${c} ${bottom} ${left}`} fill={color[1]} />
                  <polygon points={`${a} ${b} ${c} ${d}`} fill={color[0]} stroke={color[2]} strokeWidth=".6" />
                </>
              )}
            </g>
          )
        })}
      </svg>
      <figcaption>
        <span>{rows} positions × {columns} columns</span>
        <span>Height shows magnitude · blue positive · orange negative</span>
      </figcaption>
    </figure>
  )
}

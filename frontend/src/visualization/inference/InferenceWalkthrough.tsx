import { useEffect, useRef, useState, type FormEvent } from 'react'
import {
  BLOCK_COUNT, evaluation, EXAMPLE_PROMPT, HEADS, MODEL_WIDTH, parameterCount, traceInference, VOCABULARY,
  type ChannelTensor,
} from '../../engine/inference/tinyRecall'
import TensorVolume from './TensorVolume'
import './inference.css'

function TensorTable({ tensor, position }: { tensor: ChannelTensor; position: number }) {
  const { values, rowLabels, columnLabels, probability } = tensor
  const max = Math.max(1e-9, ...values.flat().filter(Number.isFinite).map(Math.abs))
  const columns = values[0].map((_, i) => columnLabels?.[i] ?? String(i))
  return (
    <div className="inference-matrix-scroll" tabIndex={0} aria-label="Scrollable tensor values">
      <table className="inference-matrix">
        <caption>{tensor.name} · {values.length} × {columns.length}</caption>
        <thead><tr><th scope="col">Position</th>{columns.map(column => <th key={column} scope="col">{column}</th>)}</tr></thead>
        <tbody>
          {values.map((row, i) => (
            <tr key={i} className={i === position ? 'inference-selected-row' : ''}>
              <th scope="row">{rowLabels?.[i] ?? i}</th>
              {row.map((n, j) => {
                const intensity = Math.abs(n) / (probability ? 1 : max)
                const finite = Number.isFinite(n)
                const color = n < 0 ? '171, 86, 43' : '25, 112, 135'
                return (
                  <td
                    key={j}
                    className={finite ? undefined : 'inference-masked-cell'}
                    title={finite ? String(n) : 'Masked: negative infinity'}
                    style={finite ? { backgroundColor: `rgba(${color}, ${intensity * 0.28})` } : undefined}
                  >{!finite ? '−∞' : tensor.name === 'IDs' ? n : n.toFixed(3)}</td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function InferenceWalkthrough({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [prompt, setPrompt] = useState(EXAMPLE_PROMPT)
  const [trace, setTrace] = useState(() => traceInference(EXAMPLE_PROMPT))
  const [error, setError] = useState('')
  const [stepIndex, setStepIndex] = useState(0)
  const [tensorIndex, setTensorIndex] = useState(0)
  const [showVolume, setShowVolume] = useState(false)
  const [position, setPosition] = useState(trace.tokens.length - 1)
  const step = trace.steps[stepIndex]
  const tensor = step.tensors[tensorIndex]
  const finalStep = stepIndex === trace.steps.length - 1

  useEffect(() => {
    const modal = dialog.current
    if (modal && !modal.open) modal.showModal()
  }, [])

  const goTo = (index: number) => {
    setStepIndex(index)
    setTensorIndex(0)
  }

  const run = (event: FormEvent) => {
    event.preventDefault()
    try {
      const next = traceInference(prompt)
      setTrace(next)
      setPosition(next.tokens.length - 1)
      setError('')
      goTo(0)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not run inference.')
    }
  }

  return (
    <dialog className="inference-dialog" ref={dialog} onClose={onClose} aria-labelledby="inference-title">
      <header className="inference-header">
        <div>
          <h2 id="inference-title">Live inference</h2>
          <p>Tiny Recall · {parameterCount.toLocaleString()} learned parameters · {BLOCK_COUNT} decoder blocks · {HEADS} heads · {MODEL_WIDTH} channels</p>
        </div>
        <button type="button" className="toolbar-btn" onClick={() => dialog.current?.close()} aria-label="Close live inference">Close</button>
      </header>
      <div className="inference-body">
        <form className="inference-prompt" onSubmit={run}>
          <label htmlFor="inference-prompt">Recall prompt</label>
          <div className="inference-input-row">
            <input id="inference-prompt" aria-describedby="inference-prompt-hint" aria-invalid={Boolean(error)}
              value={prompt} onChange={event => setPrompt(event.target.value)} maxLength={60} autoComplete="off" spellCheck={false} />
            <button type="submit" className="inference-primary">Run inference</button>
          </div>
          <p id="inference-prompt-hint">Set a value from 1–3 for each of A, B, and C, then ask for one key. Example · {EXAMPLE_PROMPT}</p>
          {error && <p className="inference-error" role="alert">{error}</p>}
        </form>
        <div className="inference-token-strip" role="group" aria-label="Token positions">
          {trace.tokens.map((token, i) => (
            <button type="button" key={i} aria-pressed={i === position} onClick={() => setPosition(i)}
              aria-label={`Position ${i}, token ${token}, ID ${trace.ids[i]}`}>
              <strong>{token}</strong><small>{trace.ids[i]}</small>
            </button>
          ))}
          <span className="inference-token-caption">Token IDs</span>
        </div>

        <div className="inference-workbench">
          <nav aria-label="Inference steps" className="inference-steps">
            {trace.steps.map((item, i) => (
              <button type="button" key={item.title} aria-current={i === stepIndex ? 'step' : undefined} onClick={() => goTo(i)}>
                <span>{i + 1}</span>{item.title}
              </button>
            ))}
          </nav>
          <section className="inference-stage" aria-labelledby="inference-step-title">
            <div className="inference-step-heading">
              <div>
                <h3 id="inference-step-title">{step.title}</h3>
                <p>{step.description}</p>
              </div>
              <div className="inference-step-buttons">
                <button type="button" className="toolbar-btn" disabled={stepIndex === 0} onClick={() => goTo(stepIndex - 1)}>Previous</button>
                <button type="button" className="toolbar-btn" disabled={finalStep} onClick={() => goTo(stepIndex + 1)}>Next</button>
              </div>
            </div>
            <div className="inference-tensor-controls">
              {step.tensors.length > 1 && (
                <div className="inference-tensor-select">
                  <label htmlFor="inference-tensor">Tensor</label>
                  <select id="inference-tensor" value={tensorIndex} onChange={event => setTensorIndex(Number(event.target.value))}>
                    {step.tensors.map((item, i) => <option key={item.name} value={i}>{item.name}</option>)}
                  </select>
                </div>
              )}
              <div className="inference-mode" role="group" aria-label="Tensor display">
                <button type="button" aria-pressed={!showVolume} onClick={() => setShowVolume(false)}>Values</button>
                <button type="button" aria-pressed={showVolume} onClick={() => setShowVolume(true)}>3D tensor</button>
              </div>
            </div>
            {showVolume ? <TensorVolume tensor={tensor} position={position} /> : <TensorTable tensor={tensor} position={position} />}
            {finalStep && (
              <section className="inference-prediction" aria-label="Next-token distribution">
                <h4>Predicted next token <output>{trace.prediction}</output></h4>
                <div className="inference-distribution">
                  {VOCABULARY.map((token, i) => ({ token, p: trace.finalProbabilities[i] }))
                    .sort((a, b) => b.p - a.p)
                    .map(({ token, p }) => (
                      <div key={token}>
                        <code>{token}</code>
                        <meter min={0} max={1} value={p} aria-label={`Probability of ${token}`} />
                        <span>{(p * 100).toFixed(2)}%</span>
                      </div>
                    ))}
                </div>
              </section>
            )}
          </section>
        </div>
      </div>
      <footer className="inference-footer">
        Tiny Recall is trained on a synthetic recall task and runs locally. Held-out accuracy · {evaluation.correct}/{evaluation.total} prompts.
      </footer>
    </dialog>
  )
}

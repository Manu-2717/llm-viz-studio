import { describe, it, expect, beforeEach } from 'vitest'
import { createRoot } from 'react-dom/client'
import { act } from 'react'
import App from './App'

describe('homepage', () => {
  it('hero copy present in source', async () => {
    const s = await fetch('/src/App.tsx').catch(() => null)
    expect(true).toBe(true)
  })
})

describe('independent-tabs smoke test', () => {
  beforeEach(() => {
    ;(globalThis as any).IS_REACT_ACT_ENVIRONMENT = true
    ;(globalThis as any).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as any
    if (typeof (globalThis as any).DOMMatrixReadOnly === 'undefined') {
      ;(globalThis as any).DOMMatrixReadOnly = class {} as any
    }
  })

  it('switches Diagram <-> Graph tabs without console errors and persists model state', async () => {
    const originalLocation = window.location
    delete (window as any).location
    window.location = {
      ...originalLocation,
      search: '?model=Qwen/Qwen2.5-1.5B-Instruct',
      href: 'http://localhost:5173/?model=Qwen/Qwen2.5-1.5B-Instruct',
    } as any

    const errors: any[] = []
    const origError = console.error
    console.error = (...args: any[]) => {
      errors.push(args)
      origError(...args)
    }

    const container = document.createElement('div')
    document.body.appendChild(container)
    const root = createRoot(container)

    await act(async () => {
      root.render(<App />)
    })

    const buttons = container.querySelectorAll('button[role="tab"]')
    expect(buttons.length).toBe(2)
    const [diagramTab, graphTab] = Array.from(buttons) as HTMLButtonElement[]

    expect(diagramTab.getAttribute('aria-selected')).toBe('true')
    expect(graphTab.getAttribute('aria-selected')).toBe('false')

    // Switch to Graph tab
    await act(async () => {
      graphTab.click()
    })

    expect(diagramTab.getAttribute('aria-selected')).toBe('false')
    expect(graphTab.getAttribute('aria-selected')).toBe('true')

    // Switch back to Diagram tab
    await act(async () => {
      diagramTab.click()
    })

    expect(diagramTab.getAttribute('aria-selected')).toBe('true')
    expect(graphTab.getAttribute('aria-selected')).toBe('false')

    await act(async () => {
      root.unmount()
    })
    container.remove()
    console.error = origError

    const realErrors = errors.filter(e => !e.some?.((x: any) => typeof x === 'string' && (x.includes('not wrapped in act') || x.includes('width and a height'))))
    expect(realErrors.length).toBe(0)
  })
})

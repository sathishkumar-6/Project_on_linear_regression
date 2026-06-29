import React, { useEffect, useRef } from 'react'

export default function AnimationBar({ animStep, setAnimStep, maxStep, playing, setPlaying }) {
  const rafRef = useRef(null)

  useEffect(() => {
    if (!playing) { cancelAnimationFrame(rafRef.current); return }
    let last = null
    const SPEED = 60
    const tick = (ts) => {
      if (!last || ts - last >= SPEED) {
        last = ts
        setAnimStep(s => {
          if (s >= maxStep) { setPlaying(false); return s }
          return s + 1
        })
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [playing, maxStep])

  if (maxStep === 0) return null

  return (
    <div style={{
      background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
      boxShadow: 'var(--shadow)', padding: '12px 16px',
      display: 'flex', alignItems: 'center', gap: 12,
    }}>
      <button onClick={() => { setAnimStep(0); setPlaying(false) }}
        style={{ background: 'var(--bg3)', color: 'var(--text2)', border: '1px solid var(--border)', borderRadius: 4, padding: '5px 10px', fontSize: '0.82rem' }}
        title="Reset">
        Reset
      </button>

      <button onClick={() => setPlaying(p => !p)}
        style={{
          background: playing ? 'var(--accent)' : 'var(--bg3)',
          color: playing ? '#fff' : 'var(--text)',
          border: '1px solid var(--border)', borderRadius: 4,
          padding: '5px 16px', fontSize: '0.82rem', fontWeight: 600,
        }}>
        {playing ? 'Pause' : 'Play Animation'}
      </button>

      <button onClick={() => { setAnimStep(maxStep); setPlaying(false) }}
        style={{ background: 'var(--bg3)', color: 'var(--text2)', border: '1px solid var(--border)', borderRadius: 4, padding: '5px 10px', fontSize: '0.82rem' }}
        title="Jump to final state">
        Skip to End
      </button>

      <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 10 }}>
        <input type="range" min={0} max={maxStep} value={animStep}
          onChange={e => { setPlaying(false); setAnimStep(parseInt(e.target.value)) }}
          style={{ flex: 1 }} />
        <span style={{ fontSize: '0.75rem', color: 'var(--text2)', whiteSpace: 'nowrap', fontFamily: 'var(--font-mono)' }}>
          {animStep + 1} / {maxStep + 1}
        </span>
      </div>
    </div>
  )
}

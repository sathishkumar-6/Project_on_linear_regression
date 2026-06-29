import React from 'react'

const VARIANTS = [
  { value: 'batch', label: 'Batch Gradient Descent' },
  { value: 'sgd', label: 'Stochastic GD (SGD)' },
  { value: 'mini_batch', label: 'Mini-Batch GD' },
]

export default function ControlPanel({ config, onChange, onTrain, loading, datasets }) {
  const set = (key, val) => onChange({ ...config, [key]: val })

  return (
    <aside style={{
      background: 'var(--bg)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius)',
      boxShadow: 'var(--shadow)',
      overflow: 'hidden',
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Header */}
      <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', background: 'var(--bg2)', fontSize: '0.72rem', fontWeight: 600, color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        Configuration
      </div>

      <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: 18 }}>

        {/* Dataset */}
        <Field label="Dataset">
          <select value={config.dataset} onChange={e => set('dataset', e.target.value)} style={{ width: '100%' }}>
            {Object.entries(datasets).map(([k, v]) => (
              <option key={k} value={k}>{v.label}</option>
            ))}
          </select>
        </Field>

        {/* Algorithm */}
        <Field label="Algorithm">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {VARIANTS.map(v => (
              <label key={v.value} style={{
                display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer',
                fontSize: '0.8rem',
                color: config.variant === v.value ? 'var(--accent)' : 'var(--text)',
                fontWeight: config.variant === v.value ? 600 : 400,
                padding: '6px 8px',
                borderRadius: 4,
                background: config.variant === v.value ? 'var(--accent-light)' : 'transparent',
              }}>
                <input
                  type="radio" name="variant" value={v.value}
                  checked={config.variant === v.value}
                  onChange={() => set('variant', v.value)}
                  style={{ accentColor: 'var(--accent)' }}
                />
                {v.label}
              </label>
            ))}
          </div>
        </Field>

        {/* Learning Rate */}
        <Field label={`Learning Rate (α) = ${config.alpha}`}>
          <input type="range" min="0.001" max="0.8" step="0.001" value={config.alpha}
            onChange={e => set('alpha', parseFloat(e.target.value))} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text3)', marginTop: 3 }}>
            <span>0.001</span><span>0.8</span>
          </div>
        </Field>

        {/* Iterations */}
        <Field label={`Iterations = ${config.iterations}`}>
          <input type="range" min="10" max="500" step="10" value={config.iterations}
            onChange={e => set('iterations', parseInt(e.target.value))} />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text3)', marginTop: 3 }}>
            <span>10</span><span>500</span>
          </div>
        </Field>

        {/* Mini-batch size */}
        {config.variant === 'mini_batch' && (
          <Field label={`Batch Size = ${config.batch_size}`}>
            <input type="range" min="1" max="10" step="1" value={config.batch_size}
              onChange={e => set('batch_size', parseInt(e.target.value))} />
          </Field>
        )}

        {/* Train button */}
        <button onClick={onTrain} disabled={loading} style={{
          background: loading ? 'var(--border2)' : 'var(--accent)',
          color: '#fff',
          borderRadius: 6, padding: '11px',
          fontWeight: 600, fontSize: '0.85rem',
          letterSpacing: '0.02em',
          border: 'none',
          opacity: loading ? 0.7 : 1,
        }}>
          {loading ? 'Training...' : 'Run Training'}
        </button>

        {/* Presets */}
        <Field label="Quick Presets">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {[
              { label: 'Slow convergence (α = 0.001)', alpha: 0.001, iterations: 300 },
              { label: 'Good default (α = 0.05)', alpha: 0.05, iterations: 100 },
              { label: 'Diverges (α = 0.7)', alpha: 0.7, iterations: 50 },
            ].map(p => (
              <button key={p.label} onClick={() => onChange({ ...config, alpha: p.alpha, iterations: p.iterations })}
                style={{
                  background: 'var(--bg3)', color: 'var(--text)', border: '1px solid var(--border)',
                  borderRadius: 4, padding: '7px 10px', fontSize: '0.75rem', textAlign: 'left',
                }}>
                {p.label}
              </button>
            ))}
          </div>
        </Field>
      </div>
    </aside>
  )
}

function Field({ label, children }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
      <label style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        {label}
      </label>
      {children}
    </div>
  )
}

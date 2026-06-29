import React, { useEffect, useState } from 'react'
import ControlPanel from './components/ControlPanel'
import RegressionPlot from './components/RegressionPlot'
import CostPlot from './components/CostPlot'
import LRComparison from './components/LRComparison'
import AnimationBar from './components/AnimationBar'
import PredictPanel from './components/PredictPanel'
import { fetchDatasets, trainModel } from './utils/api'

const DEFAULT_CONFIG = {
  dataset: 'house_prices',
  variant: 'batch',
  alpha: 0.05,
  iterations: 100,
  batch_size: 4,
}

export default function App() {
  const [datasets, setDatasets] = useState({})
  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('train')
  const [animStep, setAnimStep] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    fetchDatasets().then(setDatasets).catch(() => {})
  }, [])

  const handleTrain = async () => {
    setLoading(true)
    setError(null)
    setPlaying(false)
    setAnimStep(0)
    try {
      const res = await trainModel(config)
      setResult(res)
    } catch (e) {
      setError(e?.response?.data?.detail ?? 'Training failed. Is the backend running on port 8000?')
    } finally {
      setLoading(false)
    }
  }

  const maxStep = result ? result.history.length - 1 : 0

  return (
    <div style={{ minHeight: '100vh', background: '#f0f2f5' }}>
      <header style={{
        background: 'var(--bg)',
        borderBottom: '1px solid var(--border)',
        padding: '14px 28px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        boxShadow: 'var(--shadow)',
      }}>
        <div>
          <h1 style={{ fontFamily: 'var(--font-head)', fontSize: '1.15rem', fontWeight: 700, color: 'var(--accent)', letterSpacing: '-0.01em' }}>
            Linear Regression — From Scratch
          </h1>
          <p style={{ fontSize: '0.72rem', color: 'var(--text2)', marginTop: 2 }}>
            MSE cost function · Batch GD · SGD · Mini-Batch · Live visualization
          </p>
        </div>
        <div style={{ display: 'flex', gap: 4, background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 6, padding: 3 }}>
          {[{ id: 'train', label: 'Train' }, { id: 'compare', label: 'Learning Rate Compare' }].map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)} style={{
              background: activeTab === t.id ? 'var(--accent)' : 'transparent',
              color: activeTab === t.id ? '#fff' : 'var(--text2)',
              border: 'none', borderRadius: 4, padding: '6px 14px',
              fontSize: '0.78rem', fontWeight: activeTab === t.id ? 600 : 400,
            }}>{t.label}</button>
          ))}
        </div>
      </header>

      <main style={{
        maxWidth: 1300, margin: '0 auto', padding: '20px',
        display: 'grid',
        gridTemplateColumns: activeTab === 'train' ? '250px 1fr' : '1fr',
        gap: 18, alignItems: 'start',
      }}>
        {activeTab === 'train' ? (
          <>
            <ControlPanel config={config} onChange={setConfig} onTrain={handleTrain} loading={loading} datasets={datasets} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {error && (
                <div style={{ background: 'var(--red-light)', border: '1px solid var(--red)', borderRadius: 6, padding: '10px 14px', color: 'var(--red)', fontSize: '0.82rem' }}>
                  Error: {error}
                </div>
              )}
              {result && <AnimationBar animStep={animStep} setAnimStep={setAnimStep} maxStep={maxStep} playing={playing} setPlaying={setPlaying} />}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14 }}>
                <Card title="Regression Line"><RegressionPlot result={result} animStep={animStep} /></Card>
                <Card title="Cost (MSE) vs Iterations"><CostPlot result={result} animStep={animStep} /></Card>
              </div>
              <Card title="Predict — Enter a Value">
                <PredictPanel result={result} dataset={config.dataset} datasets={datasets} />
              </Card>
              {result && <Card title="Algorithm Summary"><AlgoExplainer variant={config.variant} alpha={config.alpha} history={result.history} /></Card>}
              {result && (
                <Card title="Dataset Used — All Training Points">
                  <SourceBadge source={result.source} count={result.x_orig.length} />
                  <DataTable result={result} datasets={datasets} dataset={config.dataset} />
                </Card>
              )}
            </div>
          </>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Card title="Settings">
              <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text2)', display: 'block', marginBottom: 4 }}>Dataset</label>
                  <select value={config.dataset} onChange={e => setConfig(c => ({ ...c, dataset: e.target.value }))}>
                    {Object.entries(datasets).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: '0.75rem', color: 'var(--text2)', display: 'block', marginBottom: 4 }}>Variant</label>
                  <select value={config.variant} onChange={e => setConfig(c => ({ ...c, variant: e.target.value }))}>
                    <option value="batch">Batch GD</option>
                    <option value="sgd">SGD</option>
                  </select>
                </div>
              </div>
            </Card>
            <Card title="Cost Curve — 4 Learning Rates Compared"><LRComparison dataset={config.dataset} variant={config.variant} /></Card>
            <Card title="Learning Rate Reference"><LRGuide /></Card>
          </div>
        )}
      </main>
    </div>
  )
}

function Card({ title, children }) {
  return (
    <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', boxShadow: 'var(--shadow)', overflow: 'hidden' }}>
      {title && (
        <div style={{ padding: '9px 16px', borderBottom: '1px solid var(--border)', fontSize: '0.72rem', fontWeight: 600, color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '0.06em', background: 'var(--bg2)' }}>
          {title}
        </div>
      )}
      <div style={{ padding: '16px' }}>{children}</div>
    </div>
  )
}

function AlgoExplainer({ variant, alpha, history }) {
  const finalCost = history.at(-1)?.cost
  const initCost = history[0]?.cost
  const improvement = initCost && finalCost ? (((initCost - finalCost) / initCost) * 100).toFixed(1) : null
  const desc = {
    batch: 'Batch Gradient Descent computes the gradient over the entire training set each iteration. This gives smooth, stable updates but can be slow for large datasets.',
    sgd: 'Stochastic Gradient Descent updates the weights after each individual sample. Updates are noisy but the algorithm can converge faster and escape local minima.',
    mini_batch: 'Mini-Batch GD splits the dataset into small batches and updates after each batch. It balances the stability of Batch GD with the speed of SGD — most used in practice.',
  }
  const variantLabel = { batch: 'Batch GD', sgd: 'Stochastic GD', mini_batch: 'Mini-Batch GD' }
  return (
    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
      <div style={{ flex: 1, minWidth: 220 }}>
        <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--accent)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{variantLabel[variant]}</div>
        <p style={{ fontSize: '0.82rem', color: 'var(--text)', lineHeight: 1.65 }}>{desc[variant]}</p>
      </div>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        {improvement && <Metric label="Cost Reduced" val={`${improvement}%`} bg="var(--green-light)" color="var(--green)" />}
        <Metric label="Learning Rate" val={alpha} bg="var(--accent-light)" color="var(--accent)" />
        <Metric label="History Steps" val={history.length} bg="var(--bg3)" color="var(--text2)" />
        <Metric label="Final MSE" val={finalCost?.toFixed(4) ?? '-'} bg="var(--orange-light)" color="var(--orange)" />
      </div>
    </div>
  )
}

function Metric({ label, val, bg, color }) {
  return (
    <div style={{ background: bg, borderRadius: 6, padding: '10px 16px', textAlign: 'center', minWidth: 90 }}>
      <div style={{ fontSize: '1.05rem', fontWeight: 700, color, fontFamily: 'var(--font-mono)' }}>{val}</div>
      <div style={{ fontSize: '0.63rem', color, opacity: 0.75, textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 3 }}>{label}</div>
    </div>
  )
}

function DataTable({ result, datasets, dataset }) {
  const ds = datasets[dataset]
  const { x_orig, y_orig } = result
  return (
    <div>
      <p style={{ fontSize: '0.78rem', color: 'var(--text2)', marginBottom: 10 }}>
        {x_orig.length} data points used for training. Residual = actual - predicted.
      </p>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', fontSize: '0.8rem', width: '100%' }}>
          <thead>
            <tr style={{ background: 'var(--bg3)' }}>
              {['#', ds?.x_label ?? 'X', ds?.y_label ?? 'Y', 'Predicted Y', 'Residual'].map(h => (
                <th key={h} style={thStyle}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {x_orig.map((x, i) => {
              const predicted = result.w_orig * x + result.b_orig
              const residual = y_orig[i] - predicted
              return (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)', background: i % 2 === 0 ? 'var(--bg)' : 'var(--bg2)' }}>
                  <td style={{ ...tdStyle, color: 'var(--text3)' }}>{i + 1}</td>
                  <td style={tdStyle}>{x.toLocaleString()}</td>
                  <td style={tdStyle}>{y_orig[i].toLocaleString()}</td>
                  <td style={{ ...tdStyle, color: 'var(--accent)', fontFamily: 'var(--font-mono)' }}>{predicted.toFixed(2)}</td>
                  <td style={{ ...tdStyle, color: Math.abs(residual) < 5 ? 'var(--green)' : 'var(--orange)', fontFamily: 'var(--font-mono)' }}>
                    {residual > 0 ? '+' : ''}{residual.toFixed(2)}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

const thStyle = { padding: '8px 12px', textAlign: 'left', fontWeight: 600, color: 'var(--text2)', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', borderBottom: '2px solid var(--border)' }
const tdStyle = { padding: '7px 12px', color: 'var(--text)' }

function LRGuide() {
  const rows = [
    { alpha: '0.001', behavior: 'Very slow convergence, takes many iterations to reach minimum', verdict: 'Too small', bg: 'var(--bg3)', color: 'var(--text2)' },
    { alpha: '0.01',  behavior: 'Steady descent, reliable convergence in reasonable steps',       verdict: 'Good default', bg: 'var(--green-light)', color: 'var(--green)' },
    { alpha: '0.1',   behavior: 'Fast convergence but may overshoot the minimum',                 verdict: 'Aggressive', bg: 'var(--yellow-light)', color: 'var(--yellow)' },
    { alpha: '0.5+',  behavior: 'Cost increases each step, gradient descent diverges to infinity','verdict': 'Diverges', bg: 'var(--red-light)', color: 'var(--red)' },
  ]
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
      <thead>
        <tr style={{ background: 'var(--bg3)' }}>
          {['Alpha (α)', 'What happens', 'Verdict'].map(h => <th key={h} style={thStyle}>{h}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map(r => (
          <tr key={r.alpha} style={{ borderBottom: '1px solid var(--border)' }}>
            <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{r.alpha}</td>
            <td style={{ ...tdStyle, color: 'var(--text2)' }}>{r.behavior}</td>
            <td style={tdStyle}>
              <span style={{ background: r.bg, color: r.color, borderRadius: 4, padding: '2px 8px', fontSize: '0.72rem', fontWeight: 600 }}>
                {r.verdict}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function SourceBadge({ source, count }) {
  if (!source) return null
  return (
    <div style={{
      display: 'flex', alignItems: 'flex-start', gap: 8,
      background: 'var(--accent-light)', border: '1px solid var(--accent)',
      borderRadius: 6, padding: '8px 12px', marginBottom: 14,
      fontSize: '0.72rem', color: 'var(--accent)', lineHeight: 1.6,
    }}>
      <span style={{ fontWeight: 700, whiteSpace: 'nowrap' }}>📊 {count} data points ·</span>
      <span style={{ wordBreak: 'break-word' }}>{source}</span>
    </div>
  )
}

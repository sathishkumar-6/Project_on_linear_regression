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
      setError(e?.response?.data?.detail ?? 'Training failed. Please verify the backend service is active on port 8000.')
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
          <h1 style={{ fontFamily: 'var(--font-head)', fontSize: '1.25rem', fontWeight: 800, color: 'var(--accent)', letterSpacing: '-0.02em', display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ background: 'var(--accent)', color: '#fff', padding: '2px 8px', borderRadius: 6, fontSize: '0.9rem' }}>⚡</span>
            OptiFit Studio
          </h1>
          <p style={{ fontSize: '0.74rem', color: 'var(--text2)', marginTop: 2 }}>
            Interactive Gradient Descent Optimizer · Real-time Loss Evaluation · Predictive Modeling
          </p>
        </div>
        <div style={{ display: 'flex', gap: 6, background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 6, padding: 3 }}>
          {[{ id: 'train', label: 'Model Studio' }, { id: 'compare', label: 'Learning Rate Analysis' }].map(t => (
            <button key={t.id} onClick={() => setActiveTab(t.id)} style={{
              background: activeTab === t.id ? 'var(--accent)' : 'transparent',
              color: activeTab === t.id ? '#fff' : 'var(--text2)',
              border: 'none', borderRadius: 4, padding: '7px 16px',
              fontSize: '0.78rem', fontWeight: activeTab === t.id ? 600 : 400,
            }}>{t.label}</button>
          ))}
        </div>
      </header>

      <main style={{
        maxWidth: 1320, margin: '0 auto', padding: '24px 20px',
        display: 'grid',
        gridTemplateColumns: activeTab === 'train' ? '260px 1fr' : '1fr',
        gap: 20, alignItems: 'start',
      }}>
        {activeTab === 'train' ? (
          <>
            <ControlPanel config={config} onChange={setConfig} onTrain={handleTrain} loading={loading} datasets={datasets} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {error && (
                <div style={{ background: 'var(--red-light)', border: '1px solid var(--red)', borderRadius: 6, padding: '12px 16px', color: 'var(--red)', fontSize: '0.82rem', fontWeight: 500 }}>
                  System Notice: {error}
                </div>
              )}
              {result && <AnimationBar animStep={animStep} setAnimStep={setAnimStep} maxStep={maxStep} playing={playing} setPlaying={setPlaying} />}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <Card title="Regression Line Fit"><RegressionPlot result={result} animStep={animStep} /></Card>
                <Card title="MSE Loss vs Epochs"><CostPlot result={result} animStep={animStep} /></Card>
              </div>
              <Card title="Real-Time Feature Inference">
                <PredictPanel result={result} dataset={config.dataset} datasets={datasets} />
              </Card>
              {result && <Card title="Model Performance & Algorithm Metrics"><AlgoExplainer variant={config.variant} alpha={config.alpha} history={result.history} result={result} /></Card>}
              {result && (
                <Card title="Benchmark Dataset Exploration">
                  <SourceBadge source={result.source} count={result.x_orig.length} />
                  <DataTable result={result} datasets={datasets} dataset={config.dataset} />
                </Card>
              )}
            </div>
          </>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <Card title="Hyperparameter Control">
              <div style={{ display: 'flex', gap: 20, alignItems: 'center' }}>
                <div>
                  <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text2)', display: 'block', marginBottom: 6 }}>Target Benchmark</label>
                  <select value={config.dataset} onChange={e => setConfig(c => ({ ...c, dataset: e.target.value }))}>
                    {Object.entries(datasets).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
                  </select>
                </div>
                <div>
                  <label style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text2)', display: 'block', marginBottom: 6 }}>Optimizer Strategy</label>
                  <select value={config.variant} onChange={e => setConfig(c => ({ ...c, variant: e.target.value }))}>
                    <option value="batch">Batch Gradient Descent</option>
                    <option value="sgd">Stochastic GD</option>
                  </select>
                </div>
              </div>
            </Card>
            <Card title="Loss Trajectories — Learning Rate Matrix"><LRComparison dataset={config.dataset} variant={config.variant} /></Card>
            <Card title="Hyperparameter Tuning Guide"><LRGuide /></Card>
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
        <div style={{ padding: '10px 18px', borderBottom: '1px solid var(--border)', fontSize: '0.74rem', fontWeight: 700, color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '0.06em', background: 'var(--bg2)' }}>
          {title}
        </div>
      )}
      <div style={{ padding: '18px' }}>{children}</div>
    </div>
  )
}

function AlgoExplainer({ variant, alpha, history, result }) {
  const finalCost = history.at(-1)?.cost
  const initCost = history[0]?.cost
  const improvement = initCost && finalCost ? (((initCost - finalCost) / initCost) * 100).toFixed(1) : null
  const desc = {
    batch: 'Batch Gradient Descent computes exact gradients across the entire training corpus per step, producing smooth deterministic convergence curves.',
    sgd: 'Stochastic Gradient Descent calculates gradients based on individual observations, introducing stochasticity that helps overcome local minima.',
    mini_batch: 'Mini-Batch GD samples fixed-size batches during each epoch, striking an optimal balance between vectorization performance and stochastic updates.',
  }
  const variantLabel = { batch: 'Batch GD', sgd: 'Stochastic GD', mini_batch: 'Mini-Batch GD' }
  return (
    <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', alignItems: 'flex-start' }}>
      <div style={{ flex: 1, minWidth: 240 }}>
        <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--accent)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{variantLabel[variant]} Strategy</div>
        <p style={{ fontSize: '0.84rem', color: 'var(--text)', lineHeight: 1.65 }}>{desc[variant]}</p>
      </div>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        {result?.r2_score !== undefined && <Metric label="R² Score" val={result.r2_score} bg="var(--accent-light)" color="var(--accent)" />}
        {result?.mae !== undefined && <Metric label="MAE" val={result.mae} bg="var(--purple-light, #f3e8ff)" color="var(--purple, #7c3aed)" />}
        {improvement && <Metric label="Cost Reduction" val={`${improvement}%`} bg="var(--green-light)" color="var(--green)" />}
        <Metric label="Learning Rate" val={alpha} bg="var(--bg3)" color="var(--text2)" />
        <Metric label="Final MSE" val={finalCost?.toFixed(4) ?? '-'} bg="var(--orange-light)" color="var(--orange)" />
      </div>
    </div>
  )
}

function Metric({ label, val, bg, color }) {
  return (
    <div style={{ background: bg, borderRadius: 6, padding: '10px 16px', textAlign: 'center', minWidth: 95 }}>
      <div style={{ fontSize: '1.1rem', fontWeight: 800, color, fontFamily: 'var(--font-mono)' }}>{val}</div>
      <div style={{ fontSize: '0.64rem', color, opacity: 0.8, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 3 }}>{label}</div>
    </div>
  )
}

function DataTable({ result, datasets, dataset }) {
  const ds = datasets[dataset]
  const { x_orig, y_orig } = result
  return (
    <div>
      <p style={{ fontSize: '0.78rem', color: 'var(--text2)', marginBottom: 12 }}>
        Showing {x_orig.length} observations. Residual = Observed Value - Model Prediction.
      </p>
      <div style={{ overflowX: 'auto', maxHeight: 280 }}>
        <table style={{ borderCollapse: 'collapse', fontSize: '0.8rem', width: '100%' }}>
          <thead>
            <tr style={{ background: 'var(--bg3)', position: 'sticky', top: 0 }}>
              {['Index', ds?.x_label ?? 'X Feature', ds?.y_label ?? 'Target Y', 'Fitted Model Ŷ', 'Residual Error'].map(h => (
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
                  <td style={{ ...tdStyle, color: 'var(--text3)' }}>#{i + 1}</td>
                  <td style={tdStyle}>{x.toLocaleString()}</td>
                  <td style={tdStyle}>{y_orig[i].toLocaleString()}</td>
                  <td style={{ ...tdStyle, color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{predicted.toFixed(2)}</td>
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

const thStyle = { padding: '9px 14px', textAlign: 'left', fontWeight: 700, color: 'var(--text2)', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em', borderBottom: '2px solid var(--border)' }
const tdStyle = { padding: '8px 14px', color: 'var(--text)' }

function LRGuide() {
  const rows = [
    { alpha: '0.001', behavior: 'Sub-optimal step size resulting in slow parameter updates.', verdict: 'Conservatively Slow', bg: 'var(--bg3)', color: 'var(--text2)' },
    { alpha: '0.01',  behavior: 'Optimal gradient stepping ensuring stable parameter convergence.', verdict: 'Recommended Default', bg: 'var(--green-light)', color: 'var(--green)' },
    { alpha: '0.1',   behavior: 'Aggressive learning rate with slight gradient oscillations near minimum.', verdict: 'Fast / High Risk', bg: 'var(--yellow-light)', color: 'var(--yellow)' },
    { alpha: '0.5+',  behavior: 'Loss increases exponentially across epochs; parameter divergence.', verdict: 'Divergent', bg: 'var(--red-light)', color: 'var(--red)' },
  ]
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
      <thead>
        <tr style={{ background: 'var(--bg3)' }}>
          {['Alpha Rate (α)', 'Observed Optimization Behavior', 'Assessment'].map(h => <th key={h} style={thStyle}>{h}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map(r => (
          <tr key={r.alpha} style={{ borderBottom: '1px solid var(--border)' }}>
            <td style={{ ...tdStyle, fontFamily: 'var(--font-mono)', fontWeight: 700 }}>{r.alpha}</td>
            <td style={{ ...tdStyle, color: 'var(--text2)' }}>{r.behavior}</td>
            <td style={tdStyle}>
              <span style={{ background: r.bg, color: r.color, borderRadius: 4, padding: '3px 10px', fontSize: '0.72rem', fontWeight: 700 }}>
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
      display: 'flex', alignItems: 'center', gap: 8,
      background: 'var(--accent-light)', border: '1px solid var(--accent)',
      borderRadius: 6, padding: '8px 14px', marginBottom: 14,
      fontSize: '0.74rem', color: 'var(--accent)', fontWeight: 500
    }}>
      <span style={{ fontWeight: 700 }}>📊 Benchmark Dataset:</span>
      <span>{source} ({count} records)</span>
    </div>
  )
}

import React, { useEffect, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { compareLR } from '../utils/api'

const COLORS = ['#0891b2', '#16a34a', '#ca8a04', '#dc2626']
const ALPHA_LABELS = ['a=0.001 (slow)', 'a=0.01 (steady)', 'a=0.1 (fast)', 'a=0.5 (diverges)']

export default function LRComparison({ dataset, variant }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    compareLR({ dataset, variant, iterations: 150 })
      .then(r => {
        const maxLen = Math.max(...r.runs.map(run => run.history.length))
        const merged = []
        for (let i = 0; i < maxLen; i++) {
          const point = { iter: r.runs[0].history[i]?.iter ?? i }
          r.runs.forEach((run, ri) => {
            const h = run.history[i]
            point[`cost_${ri}`] = h ? (isFinite(h.cost) ? h.cost : null) : null
          })
          merged.push(point)
        }
        setData({ merged, runs: r.runs })
      })
      .finally(() => setLoading(false))
  }, [dataset, variant])

  if (loading) {
    return (
      <div style={{ height: 250, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg3)', border: '1px dashed var(--border)', borderRadius: 6 }}>
        <span style={{ color: 'var(--text2)', fontSize: '0.82rem' }}>Loading comparison...</span>
      </div>
    )
  }
  if (!data) return null

  return (
    <div>
      <p style={{ fontSize: '0.78rem', color: 'var(--text2)', marginBottom: 14 }}>
        All four learning rates trained for 150 iterations on the same dataset. Notice how large alpha causes cost to explode.
      </p>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data.merged} margin={{ top: 5, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="iter" tick={{ fill: 'var(--text2)', fontSize: 11 }}
            label={{ value: 'Iteration', position: 'insideBottom', offset: -16, fill: 'var(--text2)', fontSize: 11 }} />
          <YAxis tick={{ fill: 'var(--text2)', fontSize: 11 }}
            tickFormatter={v => v > 10000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(1)}
            label={{ value: 'MSE', angle: -90, position: 'insideLeft', fill: 'var(--text2)', fontSize: 11 }}
            domain={[0, 'auto']} />
          <Tooltip contentStyle={{ background: '#fff', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }} />
          <Legend formatter={(val) => {
            const idx = parseInt(val.replace('cost_', ''))
            return <span style={{ color: COLORS[idx], fontSize: '0.75rem' }}>{ALPHA_LABELS[idx]}</span>
          }} />
          {data.runs.map((_, ri) => (
            <Line key={ri} type="monotone" dataKey={`cost_${ri}`}
              stroke={COLORS[ri]} strokeWidth={ri === 3 ? 1.5 : 2}
              dot={false} strokeDasharray={ri === 3 ? '5 3' : '0'}
              isAnimationActive={false} connectNulls={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 12 }}>
        {ALPHA_LABELS.map((label, i) => (
          <div key={i} style={{ background: 'var(--bg3)', border: `1px solid ${COLORS[i]}`, borderRadius: 4, padding: '4px 10px', fontSize: '0.72rem', color: COLORS[i], fontFamily: 'var(--font-mono)' }}>
            {label}
          </div>
        ))}
      </div>
    </div>
  )
}

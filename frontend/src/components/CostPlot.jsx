import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function CostPlot({ result, animStep }) {
  if (!result) {
    return (
      <div style={{ height: 270, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg3)', border: '1px dashed var(--border2)', borderRadius: 6 }}>
        <span style={{ color: 'var(--text3)', fontSize: '0.82rem' }}>Cost curve will appear here after training</span>
      </div>
    )
  }

  const { history } = result
  const visible = history.slice(0, animStep + 1)
  const isDiverged = !result.converged
  const finalCost = history.at(-1)?.cost

  return (
    <div>
      <ResponsiveContainer width="100%" height={270}>
        <LineChart data={visible} margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="iter"
            tick={{ fill: 'var(--text2)', fontSize: 11 }}
            label={{ value: 'Iteration', position: 'insideBottom', offset: -16, fill: 'var(--text2)', fontSize: 11 }} />
          <YAxis tick={{ fill: 'var(--text2)', fontSize: 11 }}
            tickFormatter={v => v > 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(2)}
            label={{ value: 'MSE', angle: -90, position: 'insideLeft', fill: 'var(--text2)', fontSize: 11 }}
            domain={['auto', 'auto']} />
          <Tooltip
            contentStyle={{ background: '#fff', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }}
            formatter={v => [v.toFixed(5), 'MSE Cost']} />
          <Line type="monotone" dataKey="cost"
            stroke={isDiverged ? 'var(--red)' : 'var(--orange)'}
            strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>

      <div style={{ marginTop: 10, fontSize: '0.78rem' }}>
        {isDiverged ? (
          <span style={{ color: 'var(--red)' }}>Cost diverged — learning rate is too high</span>
        ) : (
          <span style={{ color: 'var(--green)' }}>
            Converged — final MSE: <strong style={{ fontFamily: 'var(--font-mono)' }}>{finalCost?.toFixed(5)}</strong>
          </span>
        )}
      </div>
    </div>
  )
}

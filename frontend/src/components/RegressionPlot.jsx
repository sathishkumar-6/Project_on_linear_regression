import React from 'react'
import { ComposedChart, Scatter, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function RegressionPlot({ result, animStep }) {
  if (!result) {
    return (
      <div style={{ height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg3)', border: '1px dashed var(--border2)', borderRadius: 6 }}>
        <span style={{ color: 'var(--text3)', fontSize: '0.82rem' }}>Configure and run training to see the regression line</span>
      </div>
    )
  }

  const { x_orig, y_orig, x_min, x_max, x_label, y_label, history, w_orig, b_orig } = result

  const step = history[Math.min(animStep, history.length - 1)]
  const progress = history.length > 1 ? animStep / (history.length - 1) : 1
  const w_disp = w_orig * progress
  const b_disp = b_orig * progress

  const scatterData = x_orig.map((x, i) => ({ x, y: y_orig[i] }))
  const lineData = [{ x: x_min, y: w_disp * x_min + b_disp }, { x: x_max, y: w_disp * x_max + b_disp }]
  const finalLine = [{ x: x_min, y: w_orig * x_min + b_orig }, { x: x_max, y: w_orig * x_max + b_orig }]
  const isDiverged = !result.converged

  return (
    <div>
      {isDiverged && (
        <div style={{ background: 'var(--red-light)', border: '1px solid var(--red)', borderRadius: 4, padding: '6px 10px', fontSize: '0.75rem', color: 'var(--red)', marginBottom: 10 }}>
          Diverged — learning rate too large. Cost reached infinity. Try a smaller alpha.
        </div>
      )}
      <ResponsiveContainer width="100%" height={270}>
        <ComposedChart margin={{ top: 10, right: 20, bottom: 30, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="x" type="number" domain={['auto', 'auto']}
            tick={{ fill: 'var(--text2)', fontSize: 11 }}
            label={{ value: x_label, position: 'insideBottom', offset: -16, fill: 'var(--text2)', fontSize: 11 }} />
          <YAxis tick={{ fill: 'var(--text2)', fontSize: 11 }}
            label={{ value: y_label, angle: -90, position: 'insideLeft', fill: 'var(--text2)', fontSize: 11 }} />
          <Tooltip contentStyle={{ background: '#fff', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }} />
          <Scatter data={scatterData} fill="#2563eb" opacity={0.75} r={5} />
          <Line data={lineData} dataKey="y" type="linear" stroke="#dc2626" strokeWidth={2.5} dot={false} strokeDasharray={isDiverged ? '6 3' : '0'} />
          {animStep < history.length - 1 && (
            <Line data={finalLine} dataKey="y" type="linear" stroke="#16a34a" strokeWidth={1.5} dot={false} strokeDasharray="5 4" opacity={0.35} />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
        {[
          { label: 'w (slope)', val: w_orig.toFixed(4) },
          { label: 'b (intercept)', val: b_orig.toFixed(4) },
          { label: 'Final MSE', val: history.at(-1)?.cost.toFixed(4) ?? '-' },
        ].map(s => (
          <div key={s.label} style={{ background: 'var(--bg3)', border: '1px solid var(--border)', borderRadius: 4, padding: '5px 10px', fontSize: '0.75rem' }}>
            <span style={{ color: 'var(--text2)' }}>{s.label}: </span>
            <span style={{ color: 'var(--accent)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{s.val}</span>
          </div>
        ))}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginLeft: 4 }}>
          <LegendDot color="#2563eb" label="Data points" />
          <LegendDot color="#dc2626" label="Current line" />
          {animStep < history.length - 1 && <LegendDot color="#16a34a" label="Final line" dashed />}
        </div>
      </div>
    </div>
  )
}

function LegendDot({ color, label, dashed }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.72rem', color: 'var(--text2)' }}>
      <div style={{ width: 16, height: 3, background: dashed ? 'transparent' : color, borderTop: dashed ? `2px dashed ${color}` : 'none', borderRadius: 2 }} />
      {label}
    </div>
  )
}

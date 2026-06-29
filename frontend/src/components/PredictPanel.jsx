import React, { useState } from 'react'

const API = 'http://localhost:8000'

export default function PredictPanel({ result, dataset, datasets }) {
  const [inputVal, setInputVal] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const ds = datasets?.[dataset]
  const xLabel = ds?.x_label ?? 'X Feature'
  const yLabel = ds?.y_label ?? 'Y Target'

  const hints = {
    house_prices: 'e.g. 2500',
    student_scores: 'e.g. 8.5',
    experience_salary: 'e.g. 5.0',
  }

  const examples = {
    house_prices: [1000, 2000, 3500, 5000, 8000],
    student_scores: [3, 6, 10, 14, 18],
    experience_salary: [2, 4, 6, 8, 10],
  }

  const handlePredict = async () => {
    const x = parseFloat(inputVal)
    if (isNaN(x)) {
      setError('Please enter a valid numeric input.')
      return
    }
    setLoading(true)
    setError(null)
    setPrediction(null)
    try {
      const body = {
        dataset,
        x_value: x,
        w: result?.w_orig ?? null,
        b: result?.b_orig ?? null,
      }
      const res = await fetch(`${API}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail ?? 'Prediction failed')
      }
      const data = await res.json()
      setPrediction(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const isExtrapolation = prediction?.confidence_note?.includes('Extrapolation')

  const formatY = (val) => {
    if (dataset === 'student_scores') return `${val.toFixed(1)}%`
    return `$${val.toFixed(1)}k`
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      {!result && (
        <div style={{
          background: 'var(--accent-light)', border: '1px solid var(--accent)',
          borderRadius: 6, padding: '10px 14px', fontSize: '0.78rem', color: 'var(--accent)', fontWeight: 500
        }}>
          💡 Model Tip: Train the gradient descent optimizer above to run inference using your optimized weights.
        </div>
      )}

      <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 180 }}>
          <label style={labelStyle}>{xLabel}</label>
          <input
            type="number"
            value={inputVal}
            onChange={e => setInputVal(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handlePredict()}
            placeholder={hints[dataset] ?? 'Enter value'}
            style={{ width: '100%', boxSizing: 'border-box' }}
          />
        </div>
        <button
          onClick={handlePredict}
          disabled={loading}
          style={{
            background: loading ? 'var(--border2)' : 'var(--accent)',
            color: '#fff',
            border: 'none',
            borderRadius: 6,
            padding: '10px 22px',
            fontWeight: 600,
            fontSize: '0.85rem',
            cursor: loading ? 'not-allowed' : 'pointer',
            opacity: loading ? 0.7 : 1,
            whiteSpace: 'nowrap',
          }}
        >
          {loading ? 'Evaluating...' : 'Run Inference →'}
        </button>
      </div>

      {error && (
        <div style={{
          background: 'var(--red-light)', border: '1px solid var(--red)',
          borderRadius: 6, padding: '10px 14px', fontSize: '0.78rem', color: 'var(--red)',
        }}>
          {error}
        </div>
      )}

      {prediction && (
        <div style={{
          background: isExtrapolation ? 'var(--orange-light)' : 'var(--green-light)',
          border: `1px solid ${isExtrapolation ? 'var(--orange)' : 'var(--green)'}`,
          borderRadius: 8,
          padding: '16px 18px',
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
        }}>
          <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <div>
              <div style={{ fontSize: '0.68rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: isExtrapolation ? 'var(--orange)' : 'var(--green)', marginBottom: 4 }}>
                Input Feature — {xLabel}
              </div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: isExtrapolation ? 'var(--orange)' : 'var(--green)', fontFamily: 'var(--font-mono)' }}>
                {prediction.x_value.toLocaleString()}
              </div>
            </div>
            <div style={{ fontSize: '1.6rem', color: isExtrapolation ? 'var(--orange)' : 'var(--green)', alignSelf: 'center', opacity: 0.5 }}>→</div>
            <div>
              <div style={{ fontSize: '0.68rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: isExtrapolation ? 'var(--orange)' : 'var(--green)', marginBottom: 4 }}>
                Predicted Target — {yLabel}
              </div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: isExtrapolation ? 'var(--orange)' : 'var(--green)', fontFamily: 'var(--font-mono)' }}>
                {formatY(prediction.y_predicted)}
              </div>
            </div>
          </div>

          {result && (
            <div style={{ fontSize: '0.75rem', color: isExtrapolation ? 'var(--orange)' : 'var(--green)', fontFamily: 'var(--font-mono)', fontWeight: 600, opacity: 0.9 }}>
              ŷ = {result.w_orig.toFixed(4)} × {prediction.x_value} + ({result.b_orig.toFixed(4)}) = {prediction.y_predicted.toFixed(2)}
            </div>
          )}

          <div style={{
            fontSize: '0.75rem',
            color: isExtrapolation ? 'var(--orange)' : 'var(--green)',
            borderTop: `1px solid ${isExtrapolation ? 'var(--orange)' : 'var(--green)'}`,
            paddingTop: 8,
            fontWeight: 500,
          }}>
            {isExtrapolation ? '⚠' : '✓'} {prediction.confidence_note}
          </div>
        </div>
      )}

      <div>
        <div style={labelStyle}>Quick Benchmark Features</div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
          {(examples[dataset] ?? [1, 2, 3, 4, 5]).map(v => (
            <button
              key={v}
              onClick={() => { setInputVal(String(v)); setPrediction(null); setError(null) }}
              style={{
                background: 'var(--bg3)', color: 'var(--text)',
                border: '1px solid var(--border)', borderRadius: 4,
                padding: '6px 12px', fontSize: '0.75rem', fontWeight: 600,
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)',
              }}
            >
              {v}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

const labelStyle = {
  fontSize: '0.72rem', fontWeight: 700, color: 'var(--text2)',
  textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 6,
}

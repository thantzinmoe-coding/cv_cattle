import React, { useState, useEffect } from 'react'
import MetricsCard from '../components/MetricsCard'
import StatusBadge from '../components/StatusBadge'

const METRIC_META = {
  precision:      { label: 'Precision',        accent: false },
  recall:         { label: 'Recall',            accent: false },
  box_map50:      { label: 'Box mAP@50',        accent: true  },
  box_map50_95:   { label: 'Box mAP@50-95',     accent: false },
  pose_map50:     { label: 'Pose mAP@50',       accent: true  },
  pose_map50_95:  { label: 'Pose mAP@50-95',    accent: false },
}

export default function Evaluate() {
  const [metrics, setMetrics] = useState(null)
  const [status, setStatus] = useState('idle')
  const [error, setError] = useState(null)

  // Load existing metrics on mount
  useEffect(() => {
    fetch('/evaluate/metrics')
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setMetrics(data) })
      .catch(() => {})
  }, [])

  // Poll status while running
  useEffect(() => {
    if (status !== 'running') return
    const interval = setInterval(() => {
      fetch('/evaluate/status')
        .then(r => r.json())
        .then(d => {
          if (d.status === 'done') {
            setStatus('done')
            clearInterval(interval)
            // Fetch fresh metrics
            fetch('/evaluate/metrics').then(r => r.json()).then(setMetrics).catch(() => {})
          } else if (d.status === 'error') {
            setStatus('error')
            setError(d.error)
            clearInterval(interval)
          }
        })
    }, 2000)
    return () => clearInterval(interval)
  }, [status])

  async function runEval() {
    setStatus('running')
    setError(null)
    const res = await fetch('/evaluate/run', { method: 'POST' })
    if (!res.ok) {
      const d = await res.json()
      setError(d.detail)
      setStatus('error')
    }
  }

  return (
    <div className="p-8 space-y-6 fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white">Evaluate Model</h1>
        <p className="text-slate-400 text-sm mt-1">Measure model performance on the 10% test split.</p>
      </div>

      <div className="bg-[#1e2634] border border-[#2d3748] rounded-2xl p-6">
        <div className="flex items-center gap-4">
          <button
            onClick={runEval}
            disabled={status === 'running'}
            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-green-500 to-teal-500 text-white text-sm font-semibold hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-lg shadow-green-500/20"
          >
            {status === 'running' ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                Evaluating...
              </span>
            ) : '📊 Run Evaluation'}
          </button>
          <StatusBadge status={status} />
        </div>

        {status === 'running' && (
          <p className="text-xs text-slate-400 mt-3 animate-pulse">
            Evaluation running in background. Results will appear automatically…
          </p>
        )}

        {error && (
          <p className="text-xs text-red-400 mt-3">Error: {error}</p>
        )}
      </div>

      {/* Metrics grid */}
      {metrics ? (
        <div>
          <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Test Set Metrics</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            {Object.entries(METRIC_META).map(([key, meta]) => (
              <MetricsCard
                key={key}
                label={meta.label}
                value={metrics[key]}
                accent={meta.accent}
              />
            ))}
          </div>
        </div>
      ) : (
        <div className="rounded-2xl p-8 border border-dashed border-[#2d3748] text-center text-slate-500 text-sm">
          No evaluation data yet. Run evaluation to see metrics.
        </div>
      )}
    </div>
  )
}

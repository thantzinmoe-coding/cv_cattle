import React, { useState, useRef, useEffect } from 'react'
import StatusBadge from '../components/StatusBadge'
import LogViewer from '../components/LogViewer'

export default function Train() {
  const [epochs, setEpochs] = useState(50)
  const [imgsz, setImgsz] = useState(640)
  const [status, setStatus] = useState('idle')
  const [logs, setLogs] = useState([])
  const esRef = useRef(null)

  // Poll status on mount
  useEffect(() => {
    fetch('/train/status')
      .then(r => r.json())
      .then(d => setStatus(d.status))
      .catch(() => {})
  }, [])

  function startTraining() {
    setLogs([])
    setStatus('running')

    fetch('/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ epochs: Number(epochs), imgsz: Number(imgsz) }),
    })
      .then(r => r.json())
      .then(() => {
        // Open SSE stream
        if (esRef.current) esRef.current.close()
        const es = new EventSource('/train/logs')
        esRef.current = es

        es.onmessage = (e) => {
          const line = e.data
          setLogs(prev => [...prev, line])
          if (line.includes('[TRAINING COMPLETE]')) {
            setStatus('done')
            es.close()
          } else if (line.includes('[ERROR]')) {
            setStatus('error')
            es.close()
          }
        }
        es.onerror = () => {
          setStatus('error')
          es.close()
        }
      })
      .catch(() => setStatus('error'))
  }

  function stopTraining() {
    fetch('/train/stop', { method: 'POST' })
    esRef.current?.close()
    setStatus('idle')
  }

  return (
    <div className="p-8 space-y-6 fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white">Train Model</h1>
        <p className="text-slate-400 text-sm mt-1">Fine-tune YOLOv8n-Pose on your cattle dataset.</p>
      </div>

      <div className="bg-[#1e2634] border border-[#2d3748] rounded-2xl p-6 space-y-6">
        {/* Config */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-slate-400 font-medium mb-2 uppercase tracking-wider">Epochs</label>
            <input
              type="number"
              min={1}
              max={500}
              value={epochs}
              onChange={e => setEpochs(e.target.value)}
              disabled={status === 'running'}
              className="w-full bg-[#0f1117] border border-[#2d3748] rounded-xl px-4 py-3 text-white text-sm focus:outline-none focus:border-green-500/50 disabled:opacity-50 transition-colors"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 font-medium mb-2 uppercase tracking-wider">Image Size (px)</label>
            <input
              type="number"
              min={320}
              max={1280}
              step={32}
              value={imgsz}
              onChange={e => setImgsz(e.target.value)}
              disabled={status === 'running'}
              className="w-full bg-[#0f1117] border border-[#2d3748] rounded-xl px-4 py-3 text-white text-sm focus:outline-none focus:border-green-500/50 disabled:opacity-50 transition-colors"
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-4">
          <button
            onClick={startTraining}
            disabled={status === 'running'}
            className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-green-500 to-teal-500 text-white text-sm font-semibold hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-200 shadow-lg shadow-green-500/20"
          >
            {status === 'running' ? 'Training...' : '⚡ Start Training'}
          </button>
          {status === 'running' && (
            <button
              onClick={stopTraining}
              className="px-6 py-2.5 rounded-xl border border-red-500/30 text-red-400 text-sm font-semibold hover:bg-red-500/10 transition-all"
            >
              ■ Stop
            </button>
          )}
          <StatusBadge status={status} />
        </div>
      </div>

      {/* Log viewer */}
      <div>
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Live Training Log</h2>
        <LogViewer lines={logs} isRunning={status === 'running'} />
      </div>

      {status === 'done' && (
        <div className="rounded-2xl p-5 bg-green-500/10 border border-green-500/25 text-green-400 text-sm font-medium fade-in">
          ✓ Training complete! Model saved to <code className="font-mono text-xs bg-green-500/10 px-1.5 py-0.5 rounded">outputs/models/cattle_pose_best.pt</code>
        </div>
      )}
    </div>
  )
}

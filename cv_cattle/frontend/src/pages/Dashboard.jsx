import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [metrics, setMetrics] = useState(null)

  useEffect(() => {
    fetch('/health')
      .then(r => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'unreachable', trained_model_exists: false }))

    fetch('/evaluate/metrics')
      .then(r => r.ok ? r.json() : null)
      .then(setMetrics)
      .catch(() => setMetrics(null))
  }, [])

  const modelReady = health?.trained_model_exists

  return (
    <div className="p-8 space-y-8 fade-in">
      {/* Hero */}
      <div className="relative rounded-3xl overflow-hidden bg-gradient-to-br from-green-900/40 via-teal-900/30 to-[#161b22] border border-green-500/20 p-8">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(34,197,94,0.1),transparent_60%)]" />
        <div className="relative">
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-xs font-medium mb-4">
            <span className="w-1.5 h-1.5 rounded-full bg-green-400" />
            YOLOv8 Pose Estimation
          </span>
          <h1 className="text-4xl font-bold text-white mb-2">
            Cattle Body Pose
            <span className="bg-gradient-to-r from-green-400 to-teal-400 bg-clip-text text-transparent"> Estimation</span>
          </h1>
          <p className="text-slate-400 max-w-lg">
            Detect cattle, estimate 12 critical body keypoints, and count individual cows using ByteTrack object tracking.
          </p>
        </div>
      </div>

      {/* Status cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className={`rounded-2xl p-5 border ${modelReady ? 'bg-green-500/10 border-green-500/25' : 'bg-[#1e2634] border-[#2d3748]'}`}>
          <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Model Status</p>
          <p className={`text-lg font-semibold ${modelReady ? 'text-green-400' : 'text-yellow-400'}`}>
            {health === null ? 'Checking...' : modelReady ? '✓ Trained Model Ready' : '⚠ No Trained Model'}
          </p>
          <p className="text-xs text-slate-500 mt-1">
            {modelReady ? 'outputs/models/cattle_pose_best.pt' : 'Run training first'}
          </p>
        </div>

        <div className="rounded-2xl p-5 border bg-[#1e2634] border-[#2d3748]">
          <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Architecture</p>
          <p className="text-lg font-semibold text-white">YOLOv8n-Pose</p>
          <p className="text-xs text-slate-500 mt-1">12 keypoints · 3 dimensions</p>
        </div>

        <div className="rounded-2xl p-5 border bg-[#1e2634] border-[#2d3748]">
          <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">Keypoints</p>
          <p className="text-lg font-semibold text-white">12 Body Points</p>
          <p className="text-xs text-slate-500 mt-1">x, y, visibility per point</p>
        </div>
      </div>

      {/* Quick metrics */}
      {metrics && (
        <div>
          <h2 className="text-lg font-semibold text-white mb-4">Last Evaluation Results</h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            {Object.entries(metrics).map(([k, v]) => (
              <div key={k} className="bg-[#1e2634] border border-[#2d3748] rounded-2xl p-4">
                <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">{k.replace(/_/g,' ')}</p>
                <p className="text-2xl font-bold text-white">{(v * 100).toFixed(1)}<span className="text-sm text-slate-400">%</span></p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick actions */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {[
            { to: '/train',    icon: '⚡', title: 'Start Training',   desc: 'Fine-tune YOLOv8 on cattle dataset' },
            { to: '/predict',  icon: '🐄', title: 'Run Prediction',   desc: 'Upload image or video to detect cows' },
            { to: '/evaluate', icon: '📊', title: 'Evaluate Model',   desc: 'Measure mAP on the test split' },
          ].map(({ to, icon, title, desc }) => (
            <Link
              key={to}
              to={to}
              className="group rounded-2xl p-5 border border-[#2d3748] bg-[#1e2634] hover:border-green-500/30 hover:bg-green-500/5 transition-all duration-200 block"
            >
              <span className="text-2xl mb-3 block">{icon}</span>
              <p className="text-white font-semibold group-hover:text-green-400 transition-colors">{title}</p>
              <p className="text-xs text-slate-400 mt-1">{desc}</p>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}

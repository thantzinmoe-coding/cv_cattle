import React from 'react'

export default function MetricsCard({ label, value, unit = '', accent = false }) {
  return (
    <div className={`rounded-2xl p-5 border transition-all duration-200 hover:scale-[1.02] fade-in ${
      accent
        ? 'bg-green-500/10 border-green-500/25 glow-green'
        : 'bg-[#1e2634] border-[#2d3748]'
    }`}>
      <p className="text-xs text-slate-400 font-medium uppercase tracking-wider mb-2">{label}</p>
      <p className={`text-3xl font-bold ${accent ? 'text-green-400' : 'text-white'}`}>
        {typeof value === 'number' ? (value * 100).toFixed(1) : '—'}
        <span className="text-base font-normal text-slate-400 ml-1">{unit || '%'}</span>
      </p>
    </div>
  )
}

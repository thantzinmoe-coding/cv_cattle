import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'

const capabilities = [
  { icon: '▣', title: 'Video tracking', text: 'Track stable cow IDs, count the herd, and review movement conditions.' },
  { icon: '◉', title: 'Live camera', text: 'Monitor cattle from a device camera with continuously updated observations.' },
  { icon: '◇', title: 'Condition signals', text: 'See normal movement, low movement, high activity, or an observing state.' },
]

export default function Dashboard() {
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetch('/health')
      .then(response => response.json())
      .then(setHealth)
      .catch(() => setHealth({ status: 'unreachable', model_available: false }))
  }, [])

  const modelReady = health?.model_available ?? health?.trained_model_exists
  const apiOnline = health?.status === 'ok'

  return (
    <div className="min-h-screen p-5 md:p-8 lg:p-10 space-y-8 fade-in">
      <section className="relative overflow-hidden rounded-[28px] border border-emerald-400/20 bg-gradient-to-br from-[#173629] via-[#10251d] to-[#101814] p-7 md:p-10">
        <div className="absolute -right-20 -top-28 h-80 w-80 rounded-full bg-emerald-400/10 blur-3xl" />
        <div className="relative grid gap-8 xl:grid-cols-[1fr_auto] xl:items-center">
          <div>
            <span className="inline-flex items-center gap-2 rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1.5 text-xs font-semibold text-emerald-300">
              <span className="h-2 w-2 rounded-full bg-emerald-400" /> Live-ready cattle intelligence
            </span>
            <h1 className="mt-5 max-w-3xl text-3xl font-extrabold tracking-tight text-white md:text-5xl">
              Understand your herd while the camera is running.
            </h1>
            <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-300 md:text-base">
              Count cattle, maintain stable tracking IDs, and surface movement conditions from uploaded videos or a live camera feed.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link to="/monitor" className="rounded-xl bg-emerald-400 px-5 py-3 text-sm font-bold text-[#092116] shadow-lg shadow-emerald-950/40 transition hover:bg-emerald-300">
                Open cattle monitor →
              </Link>
              <span className="flex items-center rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-xs text-slate-300">
                Images · Videos · Live camera
              </span>
            </div>
          </div>

          <div className="grid min-w-[260px] gap-3 sm:grid-cols-3 xl:grid-cols-1">
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4 backdrop-blur">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">System API</p>
              <p className={`mt-2 text-lg font-bold ${apiOnline ? 'text-emerald-300' : 'text-rose-300'}`}>{health === null ? 'Checking…' : apiOnline ? 'Online' : 'Unavailable'}</p>
            </div>
            <div className="rounded-2xl border border-white/10 bg-black/20 p-4 backdrop-blur">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Detection model</p>
              <p className={`mt-2 text-lg font-bold ${modelReady ? 'text-emerald-300' : 'text-amber-300'}`}>{health === null ? 'Checking…' : modelReady ? 'Ready' : 'Not installed'}</p>
            </div>
          </div>
        </div>
      </section>

      <section>
        <div className="mb-4">
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-400">Monitoring workflow</p>
          <h2 className="mt-1 text-2xl font-bold text-white">One place for daily herd observation</h2>
        </div>
        <div className="grid gap-4 md:grid-cols-3">
          {capabilities.map(item => (
            <div key={item.title} className="rounded-2xl border border-white/8 bg-[#121b17] p-5 transition hover:-translate-y-0.5 hover:border-emerald-400/25">
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-emerald-400/10 text-xl text-emerald-300">{item.icon}</span>
              <h3 className="mt-4 font-bold text-white">{item.title}</h3>
              <p className="mt-2 text-sm leading-6 text-slate-400">{item.text}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[1.3fr_1fr]">
        <div className="rounded-2xl border border-white/8 bg-[#121b17] p-6">
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-400">Condition guide</p>
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            {[
              ['Normal movement', 'Recent movement is within the expected tracking range.', 'bg-emerald-400'],
              ['Low movement', 'The cow remained mostly stationary and may need observation.', 'bg-amber-400'],
              ['High activity', 'Movement is faster than the normal tracking range.', 'bg-sky-400'],
              ['Observing', 'More frames are needed before showing a movement condition.', 'bg-slate-400'],
            ].map(([title, text, color]) => (
              <div key={title} className="flex gap-3 rounded-xl bg-white/[0.035] p-3">
                <span className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${color}`} />
                <div><p className="text-sm font-semibold text-white">{title}</p><p className="mt-1 text-xs leading-5 text-slate-500">{text}</p></div>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-2xl border border-amber-400/15 bg-amber-400/[0.055] p-6">
          <p className="text-xs font-bold uppercase tracking-[0.18em] text-amber-300">Important</p>
          <h2 className="mt-2 text-lg font-bold text-white">Movement observation only</h2>
          <p className="mt-3 text-sm leading-6 text-slate-400">The system reports visible movement patterns and detection confidence. It cannot diagnose lameness, illness, pain, pregnancy, or other medical conditions.</p>
          <p className="mt-3 text-sm font-semibold text-amber-200">Contact a veterinarian when a cow shows persistent abnormal behavior.</p>
        </div>
      </section>
    </div>
  )
}

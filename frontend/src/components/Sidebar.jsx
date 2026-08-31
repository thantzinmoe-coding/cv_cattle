import React from 'react'
import { NavLink } from 'react-router-dom'

const nav = [
  { to: '/', label: 'Overview', icon: '⌂' },
  { to: '/predict', label: 'Cattle Monitor', icon: '◉' },
]

export default function Sidebar() {
  return (
    <aside className="w-20 lg:w-72 min-h-screen bg-[#101814] border-r border-emerald-950 flex flex-col shrink-0 sticky top-0 h-screen">
      {/* Brand */}
      <div className="px-4 lg:px-6 py-6 border-b border-emerald-950">
        <div className="flex items-center gap-3">
          <div className="w-11 h-11 shrink-0 rounded-2xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center text-xl font-bold shadow-lg shadow-emerald-950/50">
            🐄
          </div>
          <div className="hidden lg:block">
            <p className="text-base font-bold text-white leading-tight">HerdWatch</p>
            <p className="text-xs text-emerald-300/60">Cattle monitoring</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-5 space-y-2">
        {nav.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center justify-center lg:justify-start gap-3 px-3 lg:px-4 py-3 rounded-xl text-sm font-semibold transition-all duration-200 ${
                isActive
                  ? 'bg-emerald-400/12 text-emerald-300 border border-emerald-400/20 shadow-sm'
                  : 'text-slate-400 hover:text-white hover:bg-white/5 border border-transparent'
              }`
            }
          >
            <span className="text-lg">{icon}</span>
            <span className="hidden lg:inline">{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="hidden lg:block px-6 py-5 border-t border-emerald-950">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          Monitoring workspace
        </div>
        <p className="text-[11px] text-slate-600 mt-2">Movement observations are not a veterinary diagnosis.</p>
      </div>
    </aside>
  )
}

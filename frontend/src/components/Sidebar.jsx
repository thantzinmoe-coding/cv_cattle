import React from 'react'
import { NavLink } from 'react-router-dom'

const nav = [
  { to: '/', label: 'Dashboard', icon: '⬡' },
  { to: '/train', label: 'Train', icon: '⚡' },
  { to: '/predict', label: 'Predict', icon: '🐄' },
  { to: '/evaluate', label: 'Evaluate', icon: '📊' },
]

export default function Sidebar() {
  return (
    <aside className="w-64 min-h-screen bg-[#161b22] border-r border-[#2d3748] flex flex-col shrink-0">
      {/* Brand */}
      <div className="px-6 py-6 border-b border-[#2d3748]">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-green-500 to-teal-600 flex items-center justify-center text-lg font-bold shadow-lg">
            🐄
          </div>
          <div>
            <p className="text-sm font-700 text-white leading-tight">CattlePose</p>
            <p className="text-xs text-slate-400">YOLOv8 Estimation</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {nav.map(({ to, label, icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-green-500/10 text-green-400 border border-green-500/20'
                  : 'text-slate-400 hover:text-white hover:bg-white/5'
              }`
            }
          >
            <span className="text-base">{icon}</span>
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-6 py-4 border-t border-[#2d3748]">
        <p className="text-xs text-slate-500">v1.0 · Pose Estimation</p>
      </div>
    </aside>
  )
}

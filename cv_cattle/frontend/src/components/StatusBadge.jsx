import React from 'react'

const variants = {
  idle:     { dot: 'bg-slate-400',  text: 'text-slate-400',  bg: 'bg-slate-400/10',  border: 'border-slate-400/20',  label: 'Idle' },
  running:  { dot: 'bg-green-400 pulse-ring', text: 'text-green-400', bg: 'bg-green-400/10', border: 'border-green-400/20', label: 'Running' },
  done:     { dot: 'bg-teal-400',   text: 'text-teal-400',   bg: 'bg-teal-400/10',   border: 'border-teal-400/20',   label: 'Done' },
  error:    { dot: 'bg-red-400',    text: 'text-red-400',    bg: 'bg-red-400/10',    border: 'border-red-400/20',    label: 'Error' },
}

export default function StatusBadge({ status = 'idle', label }) {
  const v = variants[status] ?? variants.idle
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${v.bg} ${v.border} ${v.text}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${v.dot}`} />
      {label ?? v.label}
    </span>
  )
}

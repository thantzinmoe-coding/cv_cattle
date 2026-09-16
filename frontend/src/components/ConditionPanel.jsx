import React from 'react'

const styles = {
  normal: { dot: 'bg-emerald-400', badge: 'border-emerald-400/25 bg-emerald-400/10 text-emerald-300' },
  watch: { dot: 'bg-amber-400', badge: 'border-amber-400/25 bg-amber-400/10 text-amber-300' },
  active: { dot: 'bg-sky-400', badge: 'border-sky-400/25 bg-sky-400/10 text-sky-300' },
  neutral: { dot: 'bg-slate-400', badge: 'border-slate-400/25 bg-slate-400/10 text-slate-300' },
}

function normalizeConditions(conditions) {
  if (Array.isArray(conditions)) return conditions
  return Object.entries(conditions || {}).map(([cowId, value]) => ({ cow_id: cowId, ...value }))
}

export default function ConditionPanel({ conditions, lameness = {}, reportedCount, active = false }) {
  const allLameCows = normalizeConditions(lameness).filter(item => item.status === 'POSSIBLE LAMENESS')
  const lameCowIds = new Set(allLameCows.map(item => String(item.cow_id)))
  const allItems = normalizeConditions(conditions).map(item => (
    lameCowIds.has(String(item.cow_id))
      ? {
          ...item,
          status: 'POSSIBLE LAMENESS',
          severity: 'watch',
          description: 'The gait classifier flagged this track for review; the basic speed measurement cannot clear the flag.',
        }
      : item
  ))
  const limit = Number.isFinite(Number(reportedCount)) && Number(reportedCount) > 0
    ? Number(reportedCount)
    : allItems.length
  const rankedIds = [...allItems]
    .sort((left, right) => {
      const lamenessDifference = Number(lameCowIds.has(String(right.cow_id))) - Number(lameCowIds.has(String(left.cow_id)))
      return lamenessDifference || (right.frames_observed || 0) - (left.frames_observed || 0)
    })
    .slice(0, limit)
    .map(item => String(item.cow_id))
  const visibleIds = new Set(rankedIds)
  const items = allItems.filter(item => visibleIds.has(String(item.cow_id)))
  const lameCows = allLameCows.filter(item => visibleIds.has(String(item.cow_id)))
  const attentionCount = items.filter(item => item.severity === 'watch').length

  return (
    <section className="rounded-2xl border border-white/8 bg-[#121b17] overflow-hidden">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/8 px-5 py-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.16em] text-emerald-400">Cow conditions</p>
          <h2 className="mt-1 text-lg font-bold text-white">Movement observations</h2>
        </div>
        <div className="flex gap-2">
          <span className="rounded-full bg-white/5 px-3 py-1.5 text-xs font-semibold text-slate-300">{items.length} tracked</span>
          {attentionCount > 0 && <span className="rounded-full bg-amber-400/10 px-3 py-1.5 text-xs font-semibold text-amber-300">{attentionCount} to watch</span>}
        </div>
      </header>

      <div className="p-4 space-y-4">
        {lameCows.length > 0 && (
          <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4">
            <h3 className="font-bold text-rose-300 flex items-center gap-2">
              <span className="animate-pulse">🚨</span> Possible lameness flag
            </h3>
            <div className="mt-3 space-y-2">
              {lameCows.map(lame => (
                <div key={lame.cow_id} className="flex items-center gap-3 rounded-lg bg-black/20 p-2.5 text-sm">
                  {lame.thumbnail_url ? (
                    <img src={lame.thumbnail_url} alt={`Cow ${lame.cow_id} crop`} className="h-12 w-12 rounded object-cover border border-rose-500/50" />
                  ) : (
                    <div className="h-12 w-12 rounded bg-rose-500/10 grid place-items-center text-xl">🐄</div>
                  )}
                  <div className="flex-1">
                    <span className="block font-bold text-white">Cow #{lame.cow_id}</span>
                    <span className="block text-xs text-rose-200 mt-0.5">Model score: {Math.round(lame.confidence * 100)}%</span>
                  </div>
                </div>
              ))}
            </div>
            <p className="mt-3 text-xs text-rose-200/60">The movement pattern model flagged this track for review. Confirm persistent concerns with a veterinarian.</p>
          </div>
        )}
        {items.length === 0 ? (
          <div className="rounded-xl border border-dashed border-white/10 px-5 py-8 text-center">
            <div className="mx-auto grid h-10 w-10 place-items-center rounded-full bg-white/5 text-slate-400">◇</div>
            <p className="mt-3 text-sm font-semibold text-slate-300">{active ? 'Building movement history…' : 'No tracked conditions yet'}</p>
            <p className="mt-1 text-xs leading-5 text-slate-500">A cow must remain tracked for several frames before a condition appears.</p>
          </div>
        ) : (
          <div className="grid gap-3 xl:grid-cols-2">
            {items.map(item => {
              const style = styles[item.severity] || styles.neutral
              const confidence = Math.round((item.detection_confidence || 0) * 100)
              return (
                <article key={item.cow_id} className="rounded-xl border border-white/8 bg-white/[0.025] p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <span className={`h-2.5 w-2.5 rounded-full ${style.dot}`} />
                      <div><p className="text-xs text-slate-500">Tracked animal</p><h3 className="font-bold text-white">Cow #{item.cow_id}</h3></div>
                    </div>
                    <span className={`rounded-full border px-2.5 py-1 text-[11px] font-bold ${style.badge}`}>{item.status}</span>
                  </div>
                  <p className="mt-3 text-xs leading-5 text-slate-400">{item.description}</p>
                  <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                    <div className="rounded-lg bg-black/20 p-2.5"><p className="text-slate-600">Detection</p><p className="mt-1 font-bold text-slate-200">{confidence}%</p></div>
                    <div className="rounded-lg bg-black/20 p-2.5"><p className="text-slate-600">Observed</p><p className="mt-1 font-bold text-slate-200">{item.frames_observed || 0} frames</p></div>
                  </div>
                </article>
              )
            })}
          </div>
        )}
        <p className="mt-4 rounded-lg border border-amber-400/10 bg-amber-400/[0.045] px-3 py-2 text-[11px] leading-5 text-amber-100/65">
          These are visual movement signals, not medical diagnoses. Review persistent low movement or unusual activity with a qualified veterinarian.
        </p>
      </div>
    </section>
  )
}

import React from 'react'

export default function KnowledgeSharing() {
    return (
        <section className="mt-6 rounded-2xl border border-emerald-500/20 bg-emerald-400/[0.03] overflow-hidden">
            <header className="border-b border-emerald-500/20 bg-emerald-400/[0.05] px-5 py-4">
                <h2 className="text-lg font-bold text-emerald-300 flex items-center gap-2">
                    <span>📚</span> The Impact of Abnormal Health Indicators
                </h2>
                <p className="mt-1 text-xs text-emerald-100/60 leading-relaxed">
                    Understanding the potential consequences of movement irregularities can help in proactive herd management and reduce severe economic and welfare impacts.
                </p>
            </header>

            <div className="grid gap-4 p-5 md:grid-cols-3">
                <article className="rounded-xl border border-white/5 bg-[#121b17] p-4 text-sm shadow-sm transition hover:border-emerald-500/30">
                    <div className="flex items-center gap-2 font-bold text-rose-300 mb-2">
                        <span className="text-base">⚠️</span> Lameness & Arched Backs
                    </div>
                    <ul className="list-disc list-outside pl-4 space-y-1.5 text-slate-300 text-xs">
                        <li><strong>Reduced Yield:</strong> Can cause up to a 20% drop in daily milk production.</li>
                        <li><strong>Weight Loss:</strong> Lame cows eat and drink less as standing becomes painful.</li>
                        <li><strong>Reproduction Drop:</strong> Associated with delayed estrus and lower conception rates.</li>
                    </ul>
                </article>

                <article className="rounded-xl border border-white/5 bg-[#121b17] p-4 text-sm shadow-sm transition hover:border-emerald-500/30">
                    <div className="flex items-center gap-2 font-bold text-amber-300 mb-2">
                        <span className="text-base">📉</span> Lethargy & Low Movement
                    </div>
                    <ul className="list-disc list-outside pl-4 space-y-1.5 text-slate-300 text-xs">
                        <li><strong>Early Illness Marker:</strong> Sudden drops in mobility often precede clinical signs of Bovine Respiratory Disease (BRD) or mastitis by several days.</li>
                        <li><strong>Heat Stress:</strong> Sluggishness combined with grouping can indicate dangerous levels of heat exhaustion.</li>
                    </ul>
                </article>

                <article className="rounded-xl border border-white/5 bg-[#121b17] p-4 text-sm shadow-sm transition hover:border-emerald-500/30">
                    <div className="flex items-center gap-2 font-bold text-sky-300 mb-2">
                        <span className="text-base">⚡</span> Erratic or High Agitation
                    </div>
                    <ul className="list-disc list-outside pl-4 space-y-1.5 text-slate-300 text-xs">
                        <li><strong>Parasite Stress:</strong> Excessive movement, kicking, or tail swishing often points to severe fly or parasite irritation.</li>
                        <li><strong>Predator/Environmental Threat:</strong> Herd-wide agitation usually reflects immediate environmental stress or predator proximity.</li>
                    </ul>
                </article>
            </div>
        </section>
    )
}

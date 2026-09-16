import React, { useEffect, useState } from 'react'

export default function VetLogs() {
    const [jobs, setJobs] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch('/predict/jobs')
            .then(res => res.json())
            .then(data => {
                const lamenessJobs = data
                    .map(job => ({
                        ...job,
                        lameness_results: Object.fromEntries(
                            Object.entries(job.lameness_results || {})
                                .filter(([, result]) => result.status === 'POSSIBLE LAMENESS')
                        ),
                    }))
                    .filter(job => Object.keys(job.lameness_results).length > 0)
                setJobs(lamenessJobs)
            })
            .catch(console.error)
            .finally(() => setLoading(false))
    }, [])

    return (
        <div className="min-h-screen p-5 md:p-8 lg:p-10 fade-in text-white">
            <header className="mb-10">
                <div>
                    <p className="text-xs font-bold uppercase tracking-[0.2em] text-rose-400">Health Interventions</p>
                    <h1 className="mt-2 text-3xl font-extrabold md:text-4xl">Vet Incident Log</h1>
                    <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">A historical timeline of all cattle flagged with suspected lameness or severe health irregularities by the tracking AI.</p>
                </div>
            </header>

            {loading ? (
                <div className="flex items-center gap-3 text-slate-400">
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-400 border-t-transparent" />
                    Loading historical health logs...
                </div>
            ) : jobs.length === 0 ? (
                <div className="rounded-3xl border border-white/5 bg-[#121b17] p-16 text-center shadow-sm">
                    <span className="text-5xl">🌿</span>
                    <h2 className="text-xl font-bold mt-4 text-white">No Incidents Found</h2>
                    <p className="text-slate-400 mt-2 max-w-sm mx-auto text-sm">The herd looks healthy! No severe tracking anomalies or lameness events have been saved to the metrics history.</p>
                </div>
            ) : (
                <div className="space-y-6">
                    {jobs.map(job => {
                        const date = new Date((job.created_at || 0) * 1000)
                        const lamenessArray = Object.entries(job.lameness_results).map(([cowId, data]) => ({ cow_id: cowId, ...data }))

                        return (
                            <article key={job.job_id} className="rounded-3xl border border-rose-500/10 bg-[#121b17] overflow-hidden">
                                <div className="border-b border-rose-500/10 bg-rose-500/[0.03] px-6 py-4 flex items-center justify-between">
                                    <div>
                                        <h3 className="font-bold text-white text-lg">Incident Report #{job.job_id.slice(0, 8)}</h3>
                                        <p className="text-xs font-medium text-slate-500 mt-1">{date.toLocaleString()} · Source: {job.source ? job.source.split(/[\\/]/).pop() : 'Direct Video'}</p>
                                    </div>
                                    <span className="rounded-full bg-rose-500/10 border border-rose-500/20 px-3 py-1.5 text-[11px] font-bold text-rose-300 uppercase tracking-wider">
                                        {lamenessArray.length} Flagged
                                    </span>
                                </div>

                                <div className="p-6">
                                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                                        {lamenessArray.map(lame => (
                                            <div key={lame.cow_id} className="flex gap-4 rounded-2xl bg-black/30 p-4 border border-white/5">
                                                {lame.thumbnail_url ? (
                                                    <img src={lame.thumbnail_url} alt={`Cow ${lame.cow_id} crop`} className="h-20 w-20 shrink-0 rounded-xl object-cover border border-rose-500/30 shadow-md" />
                                                ) : (
                                                    <div className="h-20 w-20 shrink-0 rounded-xl bg-rose-500/10 flex items-center justify-center text-3xl">🐄</div>
                                                )}
                                                <div className="flex flex-col flex-1 justify-center">
                                                    <span className="block font-extrabold text-white text-lg">Cow #{lame.cow_id}</span>
                                                    <span className="block text-xs font-semibold text-rose-300 mt-1 ring-1 ring-rose-300/30 bg-rose-300/10 px-2 py-0.5 rounded-full w-max">
                                                        {lame.status}
                                                    </span>
                                                    <span className="block text-xs text-slate-400 mt-2">Confidence: {Math.round(lame.confidence * 100)}%</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>

                                    {job.output_video && (
                                        <div className="mt-6 flex items-center justify-end">
                                            <a href={`/outputs/predictions/${job.output_video}`} target="_blank" rel="noreferrer" className="text-xs font-bold text-emerald-400 hover:text-emerald-300 transition underline underline-offset-4">
                                                View Source Video &rarr;
                                            </a>
                                        </div>
                                    )}
                                </div>
                            </article>
                        )
                    })}
                </div>
            )}
        </div>
    )
}

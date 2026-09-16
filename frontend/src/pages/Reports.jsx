import React, { useEffect, useState } from 'react'
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer
} from 'recharts'

export default function Reports() {
    const [jobs, setJobs] = useState([])
    const [loading, setLoading] = useState(true)

    useEffect(() => {
        fetch('/predict/jobs')
            .then(res => res.json())
            .then(data => {
                setJobs(data)
                setLoading(false)
            })
            .catch(err => {
                console.error(err)
                setLoading(false)
            })
    }, [])

    const downloadCSV = () => {
        if (!jobs.length) return
        const headers = ['Date', 'Job ID', 'File/Source', 'Detected Count', 'Peak Visible', 'Forward', 'Reverse', 'Mode']
        const rows = jobs.map(job => {
            const date = new Date(job.created_at * 1000).toLocaleString()
            const source = job.source ? job.source.split('\\').pop().split('/').pop() : 'Live Camera'
            const mode = job.count_mode || 'unknown'
            const count = job.cow_count ?? job.cows_detected ?? job.unique_cows_detected ?? 0
            const peak = job.peak_visible_count ?? count
            const fw = job.forward_count ?? 0
            const rv = job.reverse_count ?? 0
            return `"${date}","${job.job_id}","${source}","${count}","${peak}","${fw}","${rv}","${mode}"`
        })
        const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows].join('\n')
        const encodedUri = encodeURI(csvContent)
        const link = document.createElement("a")
        link.setAttribute("href", encodedUri)
        link.setAttribute("download", "herd_analytics_report.csv")
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
    }

    // Aggregate past 7 days based on the jobs data
    const chartData = React.useMemo(() => {
        if (!jobs.length) return []
        // Initialize an array of the last 7 days
        const days = []
        for (let i = 6; i >= 0; i--) {
            const d = new Date()
            d.setDate(d.getDate() - i)
            days.push({
                dateStr: d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }),
                dateValue: d.toDateString(),
                count: 0
            })
        }

        // Assign job counts to corresponding days
        jobs.forEach(job => {
            // created_at is seconds
            const jobDate = new Date(job.created_at * 1000)
            const dateValue = jobDate.toDateString()
            const index = days.findIndex(d => d.dateValue === dateValue)
            if (index !== -1) {
                days[index].count += (job.cow_count ?? job.cows_detected ?? job.unique_cows_detected ?? 0)
            }
        })

        return days
    }, [jobs])

    return (
        <div className="min-h-screen p-5 md:p-8 lg:p-10 fade-in space-y-8">
            <header className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-400">Herd Analytics</p>
                    <h1 className="mt-2 text-3xl font-extrabold text-white md:text-4xl">History & Reports</h1>
                    <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
                        Visualize cattle visibility and crossing records over time, and learn how to interpret automated behavior patterns.
                    </p>
                </div>
                <button onClick={downloadCSV} className="rounded-xl bg-emerald-400 px-5 py-3 text-sm font-bold text-[#092116] shadow-lg shadow-emerald-950/40 transition hover:bg-emerald-300">
                    📥 Download CSV Report
                </button>
            </header>

            {/* Analytics Chart */}
            <section className="rounded-2xl border border-white/8 bg-[#121b17] p-6 lg:p-8">
                <h2 className="text-lg font-bold text-white mb-6">Activity (Last 7 Days)</h2>

                {loading ? (
                    <div className="h-64 grid place-items-center text-slate-500">Loading historical data...</div>
                ) : (
                    <div className="h-72 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#2a3b32" vertical={false} />
                                <XAxis
                                    dataKey="dateStr"
                                    stroke="#64748b"
                                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                                    axisLine={false}
                                    tickLine={false}
                                    dy={10}
                                />
                                <YAxis
                                    stroke="#64748b"
                                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                                    axisLine={false}
                                    tickLine={false}
                                />
                                <Tooltip
                                    cursor={{ fill: 'rgba(52, 211, 153, 0.05)' }}
                                    contentStyle={{ backgroundColor: '#0a0f0d', border: '1px solid #1f2937', borderRadius: '12px' }}
                                />
                                <Bar
                                    dataKey="count"
                                    fill="#34d399"
                                    radius={[4, 4, 0, 0]}
                                    name="Cattle Counted"
                                    animationDuration={1500}
                                />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                )}
            </section>

            {/* Vocabulary / Knowledge Base */}
            <section>
                <div className="mb-6">
                    <p className="text-xs font-bold uppercase tracking-[0.18em] text-emerald-400">Veterinary Knowledge Base</p>
                    <h2 className="mt-1 text-2xl font-bold text-white">Interpreting Abnormal Movement</h2>
                    <p className="mt-3 text-sm leading-6 text-slate-400 max-w-3xl">
                        Computer vision analyzes track speed, orientation, and isolation to suggest condition flags. Use this guide to understand how movement markers generated by the AI correlate with potential real-world health issues.
                    </p>
                </div>

                <div className="grid gap-6 md:grid-cols-3">

                    {/* Low Movement */}
                    <div className="rounded-2xl border border-amber-400/15 bg-amber-400/[0.04] p-6 transition hover:bg-amber-400/[0.06]">
                        <div className="flex items-center gap-3">
                            <span className="grid h-10 w-10 place-items-center rounded-full bg-amber-400/10 text-xl font-bold text-amber-300">!</span>
                            <h3 className="font-bold text-white">Low Movement</h3>
                        </div>
                        <p className="mt-4 text-sm leading-6 text-slate-400">
                            The AI triggers this when a cow moves significantly slower than the herd average or remains isolated.
                        </p>
                        <div className="mt-5 space-y-2 border-t border-white/5 pt-4">
                            <p className="text-xs font-semibold text-slate-300">Potential Causes:</p>
                            <ul className="text-sm text-amber-200/80 space-y-1 list-disc pl-4">
                                <li><strong className="text-amber-200">Lameness (Foot rot, digital dermatitis)</strong></li>
                                <li>Systemic illness / fever</li>
                                <li>Impending calving isolation</li>
                                <li>Heat stress exhaustion</li>
                            </ul>
                        </div>
                    </div>

                    {/* High Activity */}
                    <div className="rounded-2xl border border-sky-400/15 bg-sky-400/[0.04] p-6 transition hover:bg-sky-400/[0.06]">
                        <div className="flex items-center gap-3">
                            <span className="grid h-10 w-10 place-items-center rounded-full bg-sky-400/10 text-xl font-bold text-sky-300">⚡</span>
                            <h3 className="font-bold text-white">High Activity</h3>
                        </div>
                        <p className="mt-4 text-sm leading-6 text-slate-400">
                            Triggered when tracking IDs cross the field of view rapidly or exhibit abnormal sudden directional shifts.
                        </p>
                        <div className="mt-5 space-y-2 border-t border-white/5 pt-4">
                            <p className="text-xs font-semibold text-slate-300">Potential Causes:</p>
                            <ul className="text-sm text-sky-200/80 space-y-1 list-disc pl-4">
                                <li><strong className="text-sky-200">Estrus (standing heat)</strong></li>
                                <li>Predator panic / agitation</li>
                                <li>Biting insect swarms (fly strike)</li>
                                <li>Handler-induced stress</li>
                            </ul>
                        </div>
                    </div>

                    {/* Stoppages & Pacing */}
                    <div className="rounded-2xl border border-rose-400/10 bg-rose-400/[0.03] p-6 transition hover:bg-rose-400/[0.05]">
                        <div className="flex items-center gap-3">
                            <span className="grid h-10 w-10 place-items-center rounded-full bg-rose-400/10 text-xl font-bold text-rose-300">↻</span>
                            <h3 className="font-bold text-white">Line Pacing</h3>
                        </div>
                        <p className="mt-4 text-sm leading-6 text-slate-400">
                            The AI observes frequent forward/reverse crossings over the counting line without progressing.
                        </p>
                        <div className="mt-5 space-y-2 border-t border-white/5 pt-4">
                            <p className="text-xs font-semibold text-slate-300">Potential Causes:</p>
                            <ul className="text-sm text-rose-200/80 space-y-1 list-disc pl-4">
                                <li><strong className="text-rose-200">Water / Feed scarcity</strong></li>
                                <li>Fence line pressure</li>
                                <li>Separation anxiety (weaning)</li>
                                <li>Environmental bottlenecks</li>
                            </ul>
                        </div>
                    </div>

                </div>
            </section>

            <footer className="mt-8 pt-8 border-t border-white/5 flex items-center justify-between">
                <p className="text-xs text-slate-500">Always consult a qualified veterinarian when diagnosing clinical signs.</p>
            </footer>
        </div>
    )
}

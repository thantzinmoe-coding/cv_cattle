import React from 'react'

export default function Wiki() {
    return (
        <div className="min-h-screen p-5 md:p-8 lg:p-10 fade-in text-white">
            <header className="mb-10">
                <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-400">Knowledge Base</p>
                <h1 className="mt-2 text-3xl font-extrabold md:text-4xl">Cattle Health & Management Hub</h1>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">Library of common cattle ailments, symptoms, and proactive prevention strategies.</p>
            </header>

            <div className="grid gap-6 xl:grid-cols-2">
                <article className="rounded-2xl border border-white/10 bg-[#121b17] p-6 lg:p-8 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <span className="grid h-12 w-12 place-items-center rounded-2xl bg-rose-500/10 text-2xl text-rose-400">🦶</span>
                        <div>
                            <h2 className="text-xl font-bold text-white">Digital Dermatitis (Foot Rot)</h2>
                            <p className="text-xs font-semibold text-rose-300 uppercase tracking-widest mt-1">Causes Lameness</p>
                        </div>
                    </div>

                    <div className="space-y-4 text-sm text-slate-400 leading-relaxed">
                        <p><strong>What it is:</strong> A highly contagious bacterial infection of the hoof, often leading to severe lameness, reduced feed intake, and drop in milk yield. Usually presents as arched backs or severe head-bobbing.</p>
                        <p><strong>Prevention & Management:</strong>
                            <ul className="list-disc pl-5 mt-2 space-y-1">
                                <li>Maintain clean and dry walking alleys.</li>
                                <li>Implement routine copper sulfate or formalin footbaths.</li>
                                <li>Schedule professional hoof trimming proactively every 6 months.</li>
                            </ul>
                        </p>
                    </div>
                </article>

                <article className="rounded-2xl border border-white/10 bg-[#121b17] p-6 lg:p-8 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <span className="grid h-12 w-12 place-items-center rounded-2xl bg-sky-500/10 text-2xl text-sky-400">🫁</span>
                        <div>
                            <h2 className="text-xl font-bold text-white">Bovine Respiratory Disease (BRD)</h2>
                            <p className="text-xs font-semibold text-sky-300 uppercase tracking-widest mt-1">Causes Lethargy</p>
                        </div>
                    </div>

                    <div className="space-y-4 text-sm text-slate-400 leading-relaxed">
                        <p><strong>What it is:</strong> Often called "shipping fever," it is a severe respiratory complex affecting the lungs of cattle, particularly common in calves after transport or environmental stress.</p>
                        <p><strong>Prevention & Management:</strong>
                            <ul className="list-disc pl-5 mt-2 space-y-1">
                                <li>Ensure adequate colostrum intake at birth.</li>
                                <li>Minimize stress during weaning and transport.</li>
                                <li>Follow strict vaccination protocols before commingling cattle.</li>
                            </ul>
                        </p>
                    </div>
                </article>

                <article className="rounded-2xl border border-white/10 bg-[#121b17] p-6 lg:p-8 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <span className="grid h-12 w-12 place-items-center rounded-2xl bg-amber-500/10 text-2xl text-amber-400">☀️</span>
                        <div>
                            <h2 className="text-xl font-bold text-white">Heat Stress</h2>
                            <p className="text-xs font-semibold text-amber-300 uppercase tracking-widest mt-1">Causes Low Movement</p>
                        </div>
                    </div>

                    <div className="space-y-4 text-sm text-slate-400 leading-relaxed">
                        <p><strong>What it is:</strong> When temperatures and humidity exceed a cattle's thermoneutral zone, leading to rapid breathing, panting, and failure to dissipate metabolic heat.</p>
                        <p><strong>Prevention & Management:</strong>
                            <ul className="list-disc pl-5 mt-2 space-y-1">
                                <li>Provide ample shade coverage in pastures.</li>
                                <li>Install misters and large industrial fans in the barns.</li>
                                <li>Ensure unlimited access to clean, cool drinking water.</li>
                            </ul>
                        </p>
                    </div>
                </article>

                <article className="rounded-2xl border border-white/10 bg-[#121b17] p-6 lg:p-8 shadow-sm">
                    <div className="flex items-center gap-3 mb-6">
                        <span className="grid h-12 w-12 place-items-center rounded-2xl bg-indigo-500/10 text-2xl text-indigo-400">🪰</span>
                        <div>
                            <h2 className="text-xl font-bold text-white">Fly & Parasite Irritation</h2>
                            <p className="text-xs font-semibold text-indigo-300 uppercase tracking-widest mt-1">Causes Agitation</p>
                        </div>
                    </div>

                    <div className="space-y-4 text-sm text-slate-400 leading-relaxed">
                        <p><strong>What it is:</strong> Severe infestations of horn flies, face flies, or internal parasites that cause intense itching, restlessness, and energy loss.</p>
                        <p><strong>Prevention & Management:</strong>
                            <ul className="list-disc pl-5 mt-2 space-y-1">
                                <li>Use pour-on or injectible parasiticides.</li>
                                <li>Employ fly tags and dust bags in high-traffic corridors.</li>
                                <li>Maintain strict pasture rotation and manage manure piles.</li>
                            </ul>
                        </p>
                    </div>
                </article>
            </div>
        </div>
    )
}

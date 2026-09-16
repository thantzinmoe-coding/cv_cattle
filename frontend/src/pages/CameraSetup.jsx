import React from 'react'

export default function CameraSetup() {
    return (
        <div className="min-h-screen p-5 md:p-8 lg:p-10 fade-in text-white">
            <header className="mb-10">
                <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-400">Optimization</p>
                <h1 className="mt-2 text-3xl font-extrabold md:text-4xl">Camera Placement & Setup Guide</h1>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">Configure your barn and field cameras to maximize the accuracy of the YOLO AI tracking algorithms.</p>
            </header>

            <div className="space-y-10">
                <section className="rounded-3xl border border-white/10 bg-white/[0.02] p-8 lg:p-10">
                    <div className="grid lg:grid-cols-[1fr_2fr] gap-10 items-center">
                        <div className="aspect-square bg-[#0b120f] border border-white/5 rounded-2xl flex items-center justify-center p-8 relative overflow-hidden">
                            <div className="absolute inset-x-0 top-1/3 border-b-2 border-emerald-400/50 border-dashed" />
                            <div className="text-6xl z-10">🐄</div>
                            <div className="absolute top-4 right-4 bg-emerald-400/20 text-emerald-300 text-xs px-2 py-1 rounded font-bold">IDEAL</div>
                        </div>
                        <div>
                            <h2 className="text-2xl font-bold mb-4">1. The Side-Profile Angle (Best for Lameness)</h2>
                            <p className="text-slate-400 text-sm leading-relaxed mb-6">
                                To accurately detect lameness via arched backs and head-bobbing, the camera must see the side profile of the cow. Avoid pointing the camera straight down from the ceiling or directly head-on.
                            </p>
                            <ul className="space-y-3">
                                <li className="flex items-start gap-3">
                                    <span className="text-emerald-400 mt-0.5">✓</span>
                                    <div>
                                        <strong className="block text-sm text-slate-200">Height: 2.5 to 3 meters (8 - 10 ft)</strong>
                                        <span className="text-xs text-slate-500">Mount the camera slightly above cow height, angled slightly downward to avoid overlapping bodies.</span>
                                    </div>
                                </li>
                                <li className="flex items-start gap-3">
                                    <span className="text-emerald-400 mt-0.5">✓</span>
                                    <div>
                                        <strong className="block text-sm text-slate-200">Angle: 45° to 90° from the travel path</strong>
                                        <span className="text-xs text-slate-500">The cow should walk across the screen horizontally rather than walking directly toward the lens.</span>
                                    </div>
                                </li>
                            </ul>
                        </div>
                    </div>
                </section>

                <section className="rounded-3xl border border-white/10 bg-white/[0.02] p-8 lg:p-10">
                    <div className="grid lg:grid-cols-[2fr_1fr] gap-10 items-center">
                        <div>
                            <h2 className="text-2xl font-bold mb-4">2. The Top-Down Angle (Best for Counting)</h2>
                            <p className="text-slate-400 text-sm leading-relaxed mb-6">
                                If your primary goal is counting cattle entering or leaving the milking parlor, a top-down approach eliminates occlusion (cows hiding behind other cows).
                            </p>
                            <ul className="space-y-3">
                                <li className="flex items-start gap-3">
                                    <span className="text-emerald-400 mt-0.5">✓</span>
                                    <div>
                                        <strong className="block text-sm text-slate-200">Height: 4+ meters (12 - 15 ft)</strong>
                                        <span className="text-xs text-slate-500">Mount as high as possible in the choke-point.</span>
                                    </div>
                                </li>
                                <li className="flex items-start gap-3">
                                    <span className="text-rose-400 mt-0.5">✗</span>
                                    <div>
                                        <strong className="block text-sm text-slate-200">Drawback: Lameness Detection Failing</strong>
                                        <span className="text-xs text-slate-500">Top-down angles mask the spine arch and head movements. Use this angle strictly for Line-Crossing counts.</span>
                                    </div>
                                </li>
                            </ul>
                        </div>
                        <div className="aspect-square bg-[#0b120f] border border-white/5 rounded-2xl flex items-center justify-center p-8 relative overflow-hidden order-first lg:order-last">
                            <div className="absolute inset-y-0 left-1/2 border-l-2 border-emerald-400/50 border-dashed" />
                            <div className="text-5xl z-10 grid grid-cols-2 gap-4"><span>🐄</span><span>🐄</span><span>🐄</span><span>🐄</span></div>
                        </div>
                    </div>
                </section>

                <section className="rounded-3xl border border-white/10 bg-white/[0.02] p-8 lg:p-10">
                    <h2 className="text-2xl font-bold mb-4">3. Lighting Requirements</h2>
                    <div className="grid md:grid-cols-2 gap-6 mt-6">
                        <div className="p-5 bg-black/20 rounded-xl border border-white/5">
                            <strong className="text-slate-200 block mb-2 text-sm">Indoor Barns</strong>
                            <p className="text-xs text-slate-400 leading-relaxed">Equip the barn with consistent LED lighting. Avoid positioning cameras pointing directly at bright windows or doors, as the backlight will cast the cattle into deep shadows.</p>
                        </div>
                        <div className="p-5 bg-black/20 rounded-xl border border-white/5">
                            <strong className="text-slate-200 block mb-2 text-sm">Outdoor Pastures</strong>
                            <p className="text-xs text-slate-400 leading-relaxed">If monitoring at night, infrared (IR) cameras are supported by the AI tracking engine, but ensure the IR range covers the entire Region of Interest (ROI) evenly to prevent ghosting.</p>
                        </div>
                    </div>
                </section>

            </div>
        </div>
    )
}

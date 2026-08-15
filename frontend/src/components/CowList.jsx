import React from 'react';

export default function CowList({ lamenessResults, detectedIds }) {
  if (!detectedIds || detectedIds.length === 0) return null;

  // Aggregate the fragmented track sequences into a single holistic assessment
  const statuses = Object.values(lamenessResults).map(res => res.status);
  const isLame = statuses.includes("POSSIBLE LAMENESS");
  const overallStatus = isLame ? "POSSIBLE LAMENESS" : "NORMAL";

  // Average the confidence
  const confidences = Object.values(lamenessResults).map(res => res.confidence);
  const avgConfidence = confidences.length > 0
    ? confidences.reduce((a, b) => a + b) / confidences.length
    : 0;

  return (
    <div className="bg-white/5 backdrop-blur-xl border border-white/10 rounded-2xl p-6 shadow-xl w-full">
      <div className="flex items-center justify-between mb-6 pb-4 border-b border-white/10">
        <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-indigo-400">
          Target Analysis
        </h2>
      </div>

      <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-5 hover:border-[#8b949e] transition-colors relative overflow-hidden group">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-white font-bold shadow-lg">
              🐄
            </div>
            <div>
              <p className="text-gray-400 text-sm">Subject</p>
              <h3 className="text-white font-semibold">Main Cattle</h3>
            </div>
          </div>
          <div className={`px-4 py-1.5 rounded-full border border-opacity-30 ${overallStatus === 'NORMAL' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500' : 'bg-red-500/20 text-red-400 border-red-500'}`}>
            <span className="text-sm font-bold tracking-wide">{overallStatus}</span>
          </div>
        </div>

        <div className="w-full h-px bg-white/10 my-3"></div>

        <div className="space-y-3">
          <div className="flex justify-between text-sm items-center">
            <span className="text-gray-400">Gait Confidence</span>
            <div className="flex items-center gap-2">
              <div className="w-24 h-1.5 bg-white/10 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-1000 ${overallStatus === 'NORMAL' ? 'bg-gradient-to-r from-emerald-500 to-green-400' : 'bg-gradient-to-r from-red-500 to-orange-400'}`}
                  style={{ width: `${avgConfidence * 100}%` }}
                />
              </div>
              <span className="text-white font-medium">{(avgConfidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

import React, { useEffect, useRef } from 'react'

export default function LogViewer({ lines = [], isRunning = false }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [lines])

  return (
    <div className="bg-[#0d1117] border border-[#2d3748] rounded-2xl overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-[#2d3748] bg-[#161b22]">
        <div className="flex gap-1.5">
          <span className="w-3 h-3 rounded-full bg-red-500/70" />
          <span className="w-3 h-3 rounded-full bg-yellow-500/70" />
          <span className="w-3 h-3 rounded-full bg-green-500/70" />
        </div>
        <span className="text-xs text-slate-400 ml-2 font-mono">training_log.txt</span>
        {isRunning && (
          <span className="ml-auto text-xs text-green-400 animate-pulse">● Live</span>
        )}
      </div>

      {/* Log body */}
      <div className="h-80 overflow-y-auto p-4 log-output">
        {lines.length === 0 ? (
          <span className="text-slate-600">Waiting for training output...</span>
        ) : (
          lines.map((line, i) => {
            const isEpoch = /^Epoch\s+\d+/.test(line)
            const isComplete = line.includes('[TRAINING COMPLETE]')
            const isError = line.includes('[ERROR]')
            return (
              <div
                key={i}
                className={`${
                  isComplete ? 'text-green-400 font-semibold' :
                  isError    ? 'text-red-400' :
                  isEpoch    ? 'text-teal-300' :
                               'text-slate-300'
                }`}
              >
                {line}
              </div>
            )
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

import React, { useState, useCallback } from 'react'

export default function Predict() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)

  const ACCEPTED = ['image/jpeg', 'image/png', 'image/bmp', 'video/mp4', 'video/avi', 'video/quicktime']

  function handleFile(f) {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    if (f.type.startsWith('image/')) {
      setPreview(URL.createObjectURL(f))
    } else {
      setPreview(null)
    }
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) handleFile(f)
  }, [])

  async function runPrediction() {
    if (!file) return
    setLoading(true)
    setResult(null)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/predict/image', { method: 'POST', body: formData })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Prediction failed')
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-8 space-y-6 fade-in">
      <div>
        <h1 className="text-2xl font-bold text-white">Predict & Count</h1>
        <p className="text-slate-400 text-sm mt-1">Upload an image or video to detect cattle and estimate keypoints.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Upload panel */}
        <div className="space-y-4">
          <div
            onDragOver={e => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 p-8 text-center cursor-pointer ${
              dragging
                ? 'border-green-400 bg-green-500/10'
                : 'border-[#2d3748] hover:border-green-500/40 hover:bg-green-500/5'
            }`}
            onClick={() => document.getElementById('fileInput').click()}
          >
            <input
              id="fileInput"
              type="file"
              accept="image/jpeg,image/png,image/bmp,video/mp4,video/avi,video/quicktime,.jpg,.jpeg,.png,.bmp,.mp4,.avi,.mov"
              className="hidden"
              onChange={e => handleFile(e.target.files[0])}
            />
            <div className="text-4xl mb-3">🐄</div>
            <p className="text-white font-medium">Drop your file here</p>
            <p className="text-slate-400 text-sm mt-1">or click to browse</p>
            <p className="text-xs text-slate-500 mt-2">Supports: JPG, PNG, BMP, MP4, AVI, MOV</p>
          </div>

          {/* Preview */}
          {preview && (
            <div className="rounded-2xl overflow-hidden border border-[#2d3748] fade-in">
              <p className="text-xs text-slate-400 px-4 py-2 bg-[#161b22] border-b border-[#2d3748]">Preview — {file?.name}</p>
              <img src={preview} alt="preview" className="w-full object-contain max-h-64 bg-[#0f1117]" />
            </div>
          )}

          {file && !preview && (
            <div className="rounded-2xl p-4 bg-[#1e2634] border border-[#2d3748] text-sm text-slate-300 fade-in">
              🎬 <span className="font-medium">{file.name}</span>
              <span className="text-slate-500 ml-2">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
            </div>
          )}

          <button
            onClick={runPrediction}
            disabled={!file || loading}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-green-500 to-teal-500 text-white font-semibold text-sm hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-lg shadow-green-500/20"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"/>
                </svg>
                Running prediction...
              </span>
            ) : '🔍 Run Prediction'}
          </button>
        </div>

        {/* Results panel */}
        <div className="space-y-4">
          {result && (
            <>
              {/* Cow count */}
              <div className="rounded-2xl p-6 bg-green-500/10 border border-green-500/25 glow-green text-center fade-in">
                <p className="text-xs text-green-300 uppercase tracking-wider mb-1">Cows Detected</p>
                <p className="text-6xl font-bold text-green-400">{result.cow_count ?? '?'}</p>
                <p className="text-sm text-green-300 mt-1">unique cattle</p>
              </div>

              {/* Result image */}
              {result.result_image_url && (
                <div className="rounded-2xl overflow-hidden border border-[#2d3748] fade-in">
                  <p className="text-xs text-slate-400 px-4 py-2 bg-[#161b22] border-b border-[#2d3748]">Annotated Output</p>
                  <img
                    src={result.result_image_url}
                    alt="prediction result"
                    className="w-full object-contain max-h-72 bg-[#0f1117]"
                  />
                </div>
              )}
            </>
          )}

          {error && (
            <div className="rounded-2xl p-4 bg-red-500/10 border border-red-500/25 text-red-400 text-sm fade-in">
              ✗ {error}
            </div>
          )}

          {!result && !error && !loading && (
            <div className="rounded-2xl p-8 border border-dashed border-[#2d3748] text-center text-slate-500 text-sm">
              Results will appear here after prediction.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

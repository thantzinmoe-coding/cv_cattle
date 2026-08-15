import React, { useState, useCallback } from 'react'
import CowList from './../components/CowList'

export default function Predict() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [activeTab, setActiveTab] = useState('lameness') // 'lameness' | 'counting'

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

  async function runWebcamPrediction() {
    setLoading(true)
    setResult(null)
    setError(null)
    try {
      const res = await fetch('/predict/webcam', { method: 'POST' })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Webcam failed')
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
            className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 p-8 text-center cursor-pointer ${dragging
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

          <div className="flex flex-col gap-3">
            <button
              onClick={runPrediction}
              disabled={!file || loading}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 text-white font-semibold flex items-center justify-center gap-2 hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-all shadow-lg"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                  PROCESSING
                </span>
              ) : (
                '🔍 RUN FILE PREDICTION'
              )}
            </button>

            <button
              onClick={runWebcamPrediction}
              disabled={loading}
              className="w-full py-3 rounded-xl bg-[#161b22] border border-[#30363d] text-white font-semibold flex items-center justify-center gap-2 hover:border-[#8b949e] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                  STREAMING...
                </span>
              ) : (
                '📷 LIVE WEBCAM MONITOR'
              )}
            </button>
          </div>
        </div>

        {/* Results panel */}
        <div className="space-y-4">
          {result && (
            <>
              {/* Result output */}
              {result.result_image_url && (
                <div className="rounded-2xl overflow-hidden border border-[#2d3748] fade-in">
                  <p className="text-xs text-slate-400 px-4 py-2 bg-[#161b22] border-b border-[#2d3748]">Annotated Output</p>
                  {result.result_image_url.endsWith('.mp4') || result.result_image_url.endsWith('.webm') ? (
                    <video
                      src={result.result_image_url}
                      controls
                      autoPlay
                      muted
                      loop
                      className="w-full object-contain max-h-[500px] bg-[#0f1117]"
                    />
                  ) : (
                    <img
                      src={result.result_image_url}
                      alt="prediction result"
                      className="w-full object-contain max-h-[500px] bg-[#0f1117]"
                    />
                  )}
                </div>
              )}

              {/* Cow List */}
              {result.lameness_results && Object.keys(result.lameness_results).length > 0 && (
                <div className="mt-6 fade-in">
                  <CowList
                    lamenessResults={result.lameness_results}
                    detectedIds={result.detected_ids}
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

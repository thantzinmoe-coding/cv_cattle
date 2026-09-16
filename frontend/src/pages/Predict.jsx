import React, { useCallback, useEffect, useRef, useState } from 'react'
import ConditionPanel from '../components/ConditionPanel'
import KnowledgeSharing from '../components/KnowledgeSharing'

const FRAME_INTERVAL_MS = 200
const DEFAULT_COUNTING_CONFIG = {
  lineOrientation: 'vertical',
  linePosition: 0.5,
  confidence: 0.3,
  roi: { x1: 0, y1: 0, x2: 1, y2: 1 },
}

export default function Predict() {
  const [mode, setMode] = useState('upload')
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragging, setDragging] = useState(false)
  const [jobs, setJobs] = useState([])
  const { lineOrientation, linePosition, confidence, roi } = DEFAULT_COUNTING_CONFIG

  const [camActive, setCamActive] = useState(false)
  const [camError, setCamError] = useState(null)
  const [camCowCount, setCamCowCount] = useState(null)
  const [camVisibleCount, setCamVisibleCount] = useState(null)
  const [camForwardCount, setCamForwardCount] = useState(0)
  const [camReverseCount, setCamReverseCount] = useState(0)
  const [camAnnotated, setCamAnnotated] = useState(null)
  const [camConditions, setCamConditions] = useState([])

  const fileInputRef = useRef(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const timerRef = useRef(null)
  const frameInFlightRef = useRef(false)

  function stopCamera() {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    if (wsRef.current) {
      const socket = wsRef.current
      wsRef.current = null
      socket.close()
    }
    frameInFlightRef.current = false
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }
    setCamActive(false)
  }

  useEffect(() => () => stopCamera(), [])
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])



  useEffect(() => {
    fetch('/predict/jobs')
      .then(res => res.json())
      .then(setJobs)
      .catch(console.error)
  }, [])

  function loadJob(job) {
    chooseMode('upload')
    setFile(null)
    setPreview(null)
    setError(null)

    const output_name = job.output_video || job.output_image
    setResult({
      job_id: job.job_id,
      result_image_url: output_name ? `/outputs/predictions/${output_name}` : null,
      cow_count: job.cow_count ?? job.cows_detected ?? job.unique_cows_detected ?? 0,
      count_mode: job.count_mode,
      crossing_count: job.crossing_count ?? 0,
      forward_count: job.forward_count ?? 0,
      reverse_count: job.reverse_count ?? 0,
      peak_visible_count: job.peak_visible_count ?? job.cow_count ?? job.cows_detected ?? 0,
      stable_track_count: job.stable_track_count ?? job.detected_ids?.length ?? 0,
      original_filename: job.original_filename || (job.source ? job.source.split('\\').pop().split('/').pop() : 'Previous Job'),
      condition_results: job.condition_results || {},
      lameness_results: job.lameness_results || {},
    })
  }

  function chooseMode(nextMode) {
    if (nextMode !== 'live') stopCamera()
    setMode(nextMode)
  }

  function handleFile(nextFile) {
    if (!nextFile) return
    setFile(nextFile)
    setResult(null)
    setError(null)
    setPreview(URL.createObjectURL(nextFile))
  }

  const onDrop = useCallback(event => {
    event.preventDefault()
    setDragging(false)
    handleFile(event.dataTransfer.files[0])
  }, [])

  async function runPrediction() {
    if (!file) return
    setLoading(true)
    setResult(null)
    setError(null)
    const formData = new FormData()
    formData.append('file', file)
    formData.append('line_orientation', lineOrientation)
    formData.append('line_position', String(linePosition))
    formData.append('confidence', String(confidence))
    formData.append('roi_x1', String(roi.x1))
    formData.append('roi_y1', String(roi.y1))
    formData.append('roi_x2', String(roi.x2))
    formData.append('roi_y2', String(roi.y2))
    try {
      const response = await fetch('/predict/image', { method: 'POST', body: formData })
      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Analysis failed')
      setResult(data)
      fetch('/predict/jobs').then(response => response.json()).then(setJobs).catch(() => {})
    } catch (requestError) {
      setError(requestError.message)
    } finally {
      setLoading(false)
    }
  }

  async function startCamera() {
    setCamError(null)
    setCamCowCount(null)
    setCamVisibleCount(null)
    setCamForwardCount(0)
    setCamReverseCount(0)
    setCamAnnotated(null)
    setCamConditions([])

    let stream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
        audio: false,
      })
    } catch (cameraError) {
      setCamError(`Camera access denied or unavailable: ${cameraError.message}`)
      return
    }

    try {
      videoRef.current.srcObject = stream
      await videoRef.current.play()
    } catch (cameraError) {
      stream.getTracks().forEach(track => track.stop())
      if (videoRef.current) videoRef.current.srcObject = null
      setCamError(`Camera could not start: ${cameraError.message}`)
      return
    }
    setCamActive(true)
    frameInFlightRef.current = false

    const wsProtocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const query = new URLSearchParams({
      orientation: lineOrientation,
      position: String(linePosition),
      confidence: String(confidence),
      roi_x1: String(roi.x1), roi_y1: String(roi.y1),
      roi_x2: String(roi.x2), roi_y2: String(roi.y2),
    })
    const socket = new WebSocket(`${wsProtocol}://${window.location.host}/predict/ws/webcam?${query}`)
    wsRef.current = socket

    socket.onmessage = event => {
      frameInFlightRef.current = false
      let data
      try {
        data = JSON.parse(event.data)
      } catch {
        setCamError('The live analysis server returned an invalid response.')
        stopCamera()
        return
      }
      if (data.error) {
        setCamError(data.error)
        stopCamera()
        return
      }
      setCamCowCount(data.cow_count)
      setCamVisibleCount(data.visible_count)
      setCamForwardCount(data.forward_count || 0)
      setCamReverseCount(data.reverse_count || 0)
      setCamConditions(data.conditions || [])
      setCamAnnotated(`data:image/jpeg;base64,${data.annotated_frame_b64}`)
    }

    socket.onerror = () => {
      setCamError('Live analysis connection failed. Confirm that the monitoring server is running.')
      stopCamera()
    }

    socket.onclose = () => {
      if (wsRef.current === socket) {
        setCamError(current => current || 'The live analysis connection closed unexpectedly.')
        stopCamera()
      }
    }

    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    timerRef.current = setInterval(() => {
      if (
        !videoRef.current
        || socket.readyState !== WebSocket.OPEN
        || frameInFlightRef.current
        || socket.bufferedAmount > 0
      ) return
      frameInFlightRef.current = true
      canvas.width = videoRef.current.videoWidth || 1280
      canvas.height = videoRef.current.videoHeight || 720
      context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
      canvas.toBlob(blob => {
        if (blob && socket.readyState === WebSocket.OPEN) {
          try {
            socket.send(blob)
          } catch {
            frameInFlightRef.current = false
          }
        } else {
          frameInFlightRef.current = false
        }
      }, 'image/jpeg', 0.82)
    }, FRAME_INTERVAL_MS)
  }

  const isVideo = Boolean(file && (file.type.startsWith('video/') || /\.(mp4|avi|mov|mkv)$/i.test(file.name)))
  const cowCount = result?.cow_count ?? null

  return (
    <div className="min-h-screen p-5 md:p-8 lg:p-10 fade-in">
      <header className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.2em] text-emerald-400">HerdWatch vision</p>
          <h1 className="mt-2 text-3xl font-extrabold text-white md:text-4xl">Cattle Monitor</h1>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">Track cattle, count visible animals, and review movement conditions from recorded footage or a live camera.</p>
        </div>
        <div className="rounded-xl border border-amber-400/15 bg-amber-400/[0.055] px-4 py-3 text-xs leading-5 text-amber-100/70">
          Conditions are visual observations—not veterinary diagnoses.
        </div>
      </header>

      <div className="mt-7 inline-flex rounded-2xl border border-white/8 bg-[#101814] p-1.5">
        <button onClick={() => chooseMode('upload')} className={`rounded-xl px-5 py-2.5 text-sm font-bold transition ${mode === 'upload' ? 'bg-emerald-400 text-[#092116] shadow' : 'text-slate-400 hover:text-white'}`}>
          ▣ Upload footage
        </button>
        <button onClick={() => chooseMode('live')} className={`rounded-xl px-5 py-2.5 text-sm font-bold transition ${mode === 'live' ? 'bg-emerald-400 text-[#092116] shadow' : 'text-slate-400 hover:text-white'}`}>
          ◉ Live camera
        </button>
      </div>

      {mode === 'upload' ? (
        <div className="mt-6 space-y-6">
          <div className="grid gap-6 xl:grid-cols-[0.8fr_1.2fr]">
            <section className="space-y-4">
              <div
                onDragOver={event => { event.preventDefault(); setDragging(true) }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`group cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition ${dragging ? 'border-emerald-300 bg-emerald-400/10' : 'border-white/10 bg-[#121b17] hover:border-emerald-400/35 hover:bg-emerald-400/[0.035]'}`}
              >
                <input ref={fileInputRef} type="file" accept="image/jpeg,image/png,image/bmp,video/mp4,video/avi,video/quicktime,.jpg,.jpeg,.png,.bmp,.mp4,.avi,.mov,.mkv" className="hidden" onChange={event => handleFile(event.target.files[0])} />
                <span className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-emerald-400/10 text-2xl text-emerald-300 transition group-hover:scale-105">＋</span>
                <p className="mt-4 font-bold text-white">Choose a cattle image or video</p>
                <p className="mt-1 text-sm text-slate-500">Drop a file here or click to browse</p>
                <p className="mt-3 text-xs text-slate-600">JPG, PNG, BMP, MP4, AVI, MOV or MKV</p>
              </div>

              {file && (
                <div className="rounded-2xl border border-white/8 bg-[#121b17] p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="min-w-0"><p className="truncate text-sm font-semibold text-white">{file.name}</p><p className="mt-1 text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(1)} MB · {isVideo ? 'Video tracking' : 'Image detection'}</p></div>
                    <span className="rounded-full bg-emerald-400/10 px-2.5 py-1 text-[11px] font-bold text-emerald-300">READY</span>
                  </div>
                </div>
              )}

              <button onClick={runPrediction} disabled={!file || loading} className="flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-400 py-3.5 text-sm font-extrabold text-[#092116] shadow-lg shadow-emerald-950/30 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:opacity-35">
                {loading ? <><span className="h-4 w-4 animate-spin rounded-full border-2 border-[#092116]/30 border-t-[#092116]" /> ANALYZING FOOTAGE…</> : 'Analyze cattle footage →'}
              </button>

              {result && (
                <div className="grid grid-cols-2 gap-3 rounded-2xl border border-white/8 bg-[#121b17] p-4">
                  <div><p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-600">{result.count_mode === 'line_crossing' ? 'Cattle detected' : 'Visible in ROI'}</p><p className="mt-1 text-2xl font-extrabold text-emerald-300">{cowCount ?? 0}</p></div>
                  <div><p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-600">Peak visible</p><p className="mt-1 text-2xl font-extrabold text-white">{result.peak_visible_count ?? cowCount ?? 0}</p></div>
                  {result.count_mode === 'line_crossing' && <><div><p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-600">Line crossings</p><p className="mt-1 text-xl font-bold text-white">{result.crossing_count ?? 0}</p></div><div><p className="text-[10px] font-bold uppercase tracking-[0.14em] text-slate-600">Forward / reverse</p><p className="mt-1 text-xl font-bold text-white">{result.forward_count ?? 0} / {result.reverse_count ?? 0}</p></div></>}
                </div>
              )}

              {error && <div className="rounded-xl border border-rose-400/20 bg-rose-400/10 p-4 text-sm text-rose-300">{error}</div>}

              {jobs.length > 0 && !result && !file && (
                <div className="mt-6 rounded-2xl border border-white/8 bg-[#121b17] p-4">
                  <p className="mb-3 text-xs font-bold uppercase tracking-[0.14em] text-slate-400">Previous Jobs</p>
                  <div className="custom-scrollbar grid max-h-48 gap-2 overflow-y-auto pr-2">
                    {jobs.map(job => (
                      <button key={job.job_id} onClick={() => loadJob(job)} className="flex items-center justify-between rounded-xl bg-[#1a251f] px-3 py-2.5 text-left transition hover:bg-[#223129]">
                        <span className="max-w-[200px] truncate text-sm text-white">{job.source ? job.source.split('\\').pop().split('/').pop() : job.job_id}</span>
                        <span className="whitespace-nowrap text-xs font-bold text-emerald-400">{job.cow_count ?? job.cows_detected ?? job.unique_cows_detected ?? 0} cows</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </section>

            <section className="overflow-hidden rounded-2xl border border-white/8 bg-[#080d0b] min-h-[360px]">
              <div className="flex items-center justify-between border-b border-white/8 bg-[#121b17] px-4 py-3">
                <div><p className="text-sm font-bold text-white">Visual analysis</p><p className="text-xs text-slate-500">{result ? 'Annotated model output' : 'Source preview'}</p></div>
                {result && <span className="rounded-full bg-emerald-400/10 px-3 py-1 text-xs font-bold text-emerald-300">{cowCount ?? 0} cattle detected</span>}
              </div>
              <div className="grid min-h-[310px] place-items-center">
                {result?.result_image_url ? (
                  result.result_image_url.match(/\.(mp4|webm)$/i)
                    ? <video src={result.result_image_url} controls autoPlay muted loop className="max-h-[540px] w-full object-contain" />
                    : <img src={result.result_image_url} alt="Annotated cattle analysis" className="max-h-[540px] w-full object-contain" />
                ) : preview ? (
                  isVideo
                    ? <video src={preview} controls muted className="max-h-[540px] w-full object-contain" />
                    : <img src={preview} alt="Selected cattle footage" className="max-h-[540px] w-full object-contain" />
                ) : (
                  <div className="px-6 text-center"><span className="text-4xl text-slate-700">▣</span><p className="mt-3 text-sm text-slate-600">Your footage will appear here.</p></div>
                )}
              </div>
            </section>
          </div>

          <ConditionPanel conditions={result?.condition_results} lameness={result?.lameness_results} reportedCount={cowCount} active={loading} />
          <KnowledgeSharing />
        </div>
      ) : (
        <div className="mt-6 grid gap-6 2xl:grid-cols-[1.35fr_0.85fr]">
          <section className="overflow-hidden rounded-2xl border border-white/8 bg-[#121b17]">
            <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/8 px-5 py-4">
              <div><p className="text-sm font-bold text-white">Live field camera</p><p className="mt-1 text-xs text-slate-500">Frames are analyzed five times per second.</p></div>
              {!camActive ? (
                <button onClick={startCamera} className="rounded-xl bg-emerald-400 px-4 py-2.5 text-sm font-bold text-[#092116] transition hover:bg-emerald-300">Start camera</button>
              ) : (
                <button onClick={stopCamera} className="rounded-xl bg-rose-500 px-4 py-2.5 text-sm font-bold text-white transition hover:bg-rose-400">Stop camera</button>
              )}
            </header>

            <canvas ref={canvasRef} className="hidden" />
            <div className="grid lg:grid-cols-2">
              <div className="relative grid min-h-[300px] place-items-center bg-[#080d0b] lg:border-r lg:border-white/8">
                <video ref={videoRef} autoPlay muted playsInline className={`max-h-[480px] w-full object-contain ${camActive ? '' : 'hidden'}`} />
                {!camActive && <div className="text-center"><span className="text-4xl text-slate-700">◉</span><p className="mt-3 text-sm text-slate-600">Camera is inactive</p></div>}
                <span className="absolute left-3 top-3 rounded-lg bg-black/70 px-2.5 py-1 text-[11px] font-bold text-slate-300">SOURCE</span>
              </div>
              <div className="relative grid min-h-[300px] place-items-center bg-[#080d0b]">
                {camAnnotated ? <img src={camAnnotated} alt="Live annotated cattle tracking" className="max-h-[480px] w-full object-contain" /> : <div className="text-center"><span className="text-4xl text-slate-700">◇</span><p className="mt-3 text-sm text-slate-600">{camActive ? 'Waiting for analysis…' : 'AI output will appear here'}</p></div>}
                <span className="absolute left-3 top-3 rounded-lg bg-black/70 px-2.5 py-1 text-[11px] font-bold text-emerald-300">AI TRACKING</span>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-4 border-t border-white/8 bg-[#0e1612] px-5 py-4">
              <div><p className="text-[11px] font-bold uppercase tracking-[0.16em] text-slate-600">Session crossings</p><p className="mt-1 text-3xl font-extrabold text-emerald-300">{camCowCount ?? '—'}</p></div>
              <div className="h-10 w-px bg-white/8" />
              <div><p className="text-[11px] font-bold uppercase tracking-[0.16em] text-slate-600">Visible in ROI</p><p className="mt-1 text-3xl font-extrabold text-white">{camVisibleCount ?? '—'}</p></div>
              <div className="h-10 w-px bg-white/8" />
              <div><p className="text-[11px] font-bold uppercase tracking-[0.16em] text-slate-600">Forward / reverse</p><p className="mt-1 text-xl font-extrabold text-white">{camForwardCount} / {camReverseCount}</p></div>
              <div className="h-10 w-px bg-white/8" />
              <div><p className="text-[11px] font-bold uppercase tracking-[0.16em] text-slate-600">Tracked conditions</p><p className="mt-1 text-3xl font-extrabold text-white">{camConditions.length}</p></div>
              {camActive && <span className="ml-auto inline-flex items-center gap-2 rounded-full bg-emerald-400/10 px-3 py-1.5 text-xs font-bold text-emerald-300"><span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" /> LIVE</span>}
            </div>
            {camError && <div className="border-t border-rose-400/20 bg-rose-400/10 px-5 py-3 text-sm text-rose-300">{camError}</div>}
          </section>

          <ConditionPanel conditions={camConditions} lameness={{}} active={camActive} />
          <KnowledgeSharing />
        </div>
      )}
    </div>
  )
}

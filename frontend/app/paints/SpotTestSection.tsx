'use client'

import { useState, useRef, useEffect } from 'react'
import { API_BASE_URL } from '@/lib/config'

interface Paint {
  id: string
  name: string
  hex_approx?: string
}

interface SpotTestSectionProps {
  selectedGroup: string
  paints: Paint[]
}

type InputMode = 'percent' | 'weight'

interface RecipeRow {
  paintId: string
  value: number
}

function rgbToHex(rgb: number[]): string {
  if (!rgb || rgb.length < 3) return '#000000'
  const [r, g, b] = rgb.map((c) => Math.max(0, Math.min(255, Math.round(c))))
  return '#' + [r, g, b].map((x) => x.toString(16).padStart(2, '0')).join('')
}

function buildComponents(rows: RecipeRow[], mode: InputMode): Array<{ paint_id: string; ratio: number }> {
  const valid = rows.filter((r) => (r.paintId || '').trim() && r.value > 0)
  if (valid.length === 0) return []
  if (mode === 'percent') {
    const total = valid.reduce((s, r) => s + r.value, 0)
    if (total <= 0) return []
    return valid.map((r) => ({ paint_id: r.paintId, ratio: r.value / 100 }))
  }
  const total = valid.reduce((s, r) => s + r.value, 0)
  if (total <= 0) return []
  return valid.map((r) => ({ paint_id: r.paintId, ratio: r.value / total }))
}

export function SpotTestSection({ selectedGroup, paints }: SpotTestSectionProps) {
  const [inputMode, setInputMode] = useState<InputMode>('percent')
  const [rows, setRows] = useState<RecipeRow[]>([{ paintId: '', value: 0 }])
  const [predictedHex, setPredictedHex] = useState<string | null>(null)
  const [predictedLoading, setPredictedLoading] = useState(false)
  const [step, setStep] = useState<'recipe' | 'upload' | 'sample' | 'confirm_commit' | 'done'>('recipe')
  const [uploading, setUploading] = useState(false)
  const [imageId, setImageId] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null)
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null)
  const [selectedRegion, setSelectedRegion] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
  const [imageDisplaySize, setImageDisplaySize] = useState<{ width: number; height: number } | null>(null)
  const [sampling, setSampling] = useState(false)
  const [focusPaintId, setFocusPaintId] = useState('')
  const [result, setResult] = useState<{
    delta_e: number | null
    measured_rgb?: number[]
    feedback_updated?: boolean
    paints_updated?: string[]
  } | null>(null)
  const [committing, setCommitting] = useState(false)
  const [biasList, setBiasList] = useState<Record<string, number[]> | null>(null)
  const [biasLoading, setBiasLoading] = useState(false)
  const [resetting, setResetting] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)

  const loadBiasList = () => {
    setBiasLoading(true)
    fetch(`${API_BASE_URL}/api/paint/feedback-bias?group=${encodeURIComponent(selectedGroup)}`)
      .then((res) => res.json())
      .then((data) => setBiasList(data.biases ?? {}))
      .catch(() => setBiasList({}))
      .finally(() => setBiasLoading(false))
  }

  const handleRemoveBias = (paintId: string) => {
    setResetting(paintId)
    fetch(`${API_BASE_URL}/api/paint/feedback-bias/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ group: selectedGroup, paint_id: paintId }),
    })
      .then(() => loadBiasList())
      .finally(() => setResetting(null))
  }

  const handleResetAllBias = () => {
    if (!confirm('Remove all spot-test corrections for this library? Future recipe generation will no longer use them.')) return
    setResetting('all')
    fetch(`${API_BASE_URL}/api/paint/feedback-bias/reset`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ group: selectedGroup }),
    })
      .then(() => loadBiasList())
      .finally(() => setResetting(null))
  }

  const totalPercent = rows.reduce((s, r) => s + r.value, 0)
  const totalWeight = inputMode === 'weight' ? rows.reduce((s, r) => s + r.value, 0) : 0
  const components = buildComponents(rows, inputMode)
  const canPredict = components.length > 0 && (inputMode === 'weight' ? totalWeight > 0 : Math.abs(totalPercent - 100) < 0.01)
  const pigmentIds = components.filter((c) => !['white', 'titanium white', 'zinc white'].includes(c.paint_id.toLowerCase())).map((c) => c.paint_id)

  useEffect(() => {
    if (!canPredict || !selectedGroup) {
      setPredictedHex(null)
      return
    }
    setPredictedLoading(true)
    const body = {
      library_group: selectedGroup,
      target_hex: '#808080',
      components: components.map((c) => ({ paint_id: c.paint_id, ratio: c.ratio })),
    }
    fetch(`${API_BASE_URL}/api/paint/recipes/predict-mix`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.predicted_hex) setPredictedHex(data.predicted_hex)
        else setPredictedHex(null)
      })
      .catch(() => setPredictedHex(null))
      .finally(() => setPredictedLoading(false))
  }, [canPredict, selectedGroup, JSON.stringify(components)])

  function displayToImage(displayX: number, displayY: number): { x: number; y: number } {
    if (!imageRef.current) return { x: 0, y: 0 }
    const rect = imageRef.current.getBoundingClientRect()
    const scaleX = imageRef.current.naturalWidth / rect.width
    const scaleY = imageRef.current.naturalHeight / rect.height
    return { x: Math.round(displayX * scaleX), y: Math.round(displayY * scaleY) }
  }

  const addRow = () => setRows([...rows, { paintId: '', value: 0 }])
  const removeRow = (i: number) => setRows(rows.filter((_, idx) => idx !== i))
  const setRow = (i: number, paintId: string, value: number) => {
    const next = [...rows]
    next[i] = { paintId, value }
    setRows(next)
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !predictedHex) return
    setError(null)
    setUploading(true)
    try {
      const form = new FormData()
      form.append('image', file)
      form.append('library_group', selectedGroup)
      const res = await fetch(`${API_BASE_URL}/api/paint/spot-test/upload`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(await res.text().catch(() => `HTTP ${res.status}`))
      const data = await res.json()
      setImageId(data.image_id)
      const raw = data.preview_url
      setPreviewUrl(raw ? (raw.startsWith('http') ? raw : `${API_BASE_URL}${raw}`) : null)
      setStep('sample')
      setSelectedRegion(null)
      setResult(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setUploading(false)
    }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imageRef.current || !previewUrl) return
    const rect = imageRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) return
    setDragStart({ x, y })
    setDragCurrent({ x, y })
    setSelectedRegion(null)
  }
  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!dragStart) return
    if (!imageRef.current) return
    const rect = imageRef.current.getBoundingClientRect()
    setDragCurrent({ x: e.clientX - rect.left, y: e.clientY - rect.top })
  }
  const handleMouseUp = () => {
    if (!imageRef.current || !dragStart || !dragCurrent) {
      setDragStart(null)
      setDragCurrent(null)
      return
    }
    const rect = imageRef.current.getBoundingClientRect()
    const x1 = Math.max(0, Math.min(rect.width, Math.min(dragStart.x, dragCurrent.x)))
    const x2 = Math.max(0, Math.min(rect.width, Math.max(dragStart.x, dragCurrent.x)))
    const y1 = Math.max(0, Math.min(rect.height, Math.min(dragStart.y, dragCurrent.y)))
    const y2 = Math.max(0, Math.min(rect.height, Math.max(dragStart.y, dragCurrent.y)))
    if (x2 - x1 >= 5 && y2 - y1 >= 5) {
      const p1 = displayToImage(x1, y1)
      const p2 = displayToImage(x2, y2)
      setSelectedRegion({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y })
    }
    setDragStart(null)
    setDragCurrent(null)
  }
  const handleMouseLeave = () => { setDragStart(null); setDragCurrent(null) }

  const buildSampleForm = (applyFeedback: boolean) => {
    const form = new FormData()
    form.append('library_group', selectedGroup)
    form.append('image_id', imageId!)
    form.append('x1', String(selectedRegion!.x1))
    form.append('y1', String(selectedRegion!.y1))
    form.append('x2', String(selectedRegion!.x2))
    form.append('y2', String(selectedRegion!.y2))
    form.append('target_hex', predictedHex!)
    form.append('recipe', JSON.stringify(components.map((c) => ({ paint_id: c.paint_id, ratio: c.ratio }))))
    form.append('apply_feedback', applyFeedback ? 'true' : 'false')
    if (focusPaintId.trim()) form.append('focus_paint_id', focusPaintId.trim())
    return form
  }

  const handleSample = async () => {
    if (!imageId || !selectedRegion || !predictedHex) return
    setError(null)
    setSampling(true)
    try {
      const res = await fetch(`${API_BASE_URL}/api/paint/spot-test/sample`, { method: 'POST', body: buildSampleForm(false) })
      if (!res.ok) throw new Error(await res.text().catch(() => `HTTP ${res.status}`))
      const data = await res.json()
      setResult({
        delta_e: data.delta_e ?? null,
        measured_rgb: data.measured_rgb ?? undefined,
      })
      setStep('confirm_commit')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSampling(false)
    }
  }

  const handleCommit = async (commit: boolean) => {
    if (!commit) {
      setStep('sample')
      setResult(null)
      return
    }
    if (!imageId || !selectedRegion || !predictedHex) return
    setError(null)
    setCommitting(true)
    try {
      const res = await fetch(`${API_BASE_URL}/api/paint/spot-test/sample`, { method: 'POST', body: buildSampleForm(true) })
      if (!res.ok) throw new Error(await res.text().catch(() => `HTTP ${res.status}`))
      const data = await res.json()
      setResult({
        delta_e: data.delta_e ?? null,
        measured_rgb: data.measured_rgb ?? result?.measured_rgb,
        feedback_updated: data.feedback_updated === true,
        paints_updated: Array.isArray(data.paints_updated) ? data.paints_updated : [],
      })
      setStep('done')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setCommitting(false)
    }
  }

  const resetToRecipe = () => {
    setStep('recipe')
    setImageId(null)
    setPreviewUrl(null)
    setSelectedRegion(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="mt-8 p-4 bg-gray-800 rounded">
      <h2 className="text-2xl font-bold mb-2">Spot test</h2>
      <p className="text-sm text-gray-400 mb-4">
        Use an existing swatch photo to correct the recipe model. Enter the recipe (by percentage or weight), then upload the image and draw a rectangle over the mixed paint area. If the result is poor, suggest the problem paint so the correction applies to that paint in all future recipes.
      </p>

      {step === 'recipe' && (
        <>
          <div className="flex gap-4 mb-3">
            <label className="flex items-center gap-2">
              <input type="radio" checked={inputMode === 'percent'} onChange={() => setInputMode('percent')} />
              By percentage
            </label>
            <label className="flex items-center gap-2">
              <input type="radio" checked={inputMode === 'weight'} onChange={() => setInputMode('weight')} />
              By weight (g)
            </label>
          </div>
          <div className="space-y-2 mb-3">
            {rows.map((row, i) => (
              <div key={i} className="flex items-center gap-2 flex-wrap">
                <select
                  value={row.paintId}
                  onChange={(e) => setRow(i, e.target.value, row.value)}
                  className="px-2 py-1.5 bg-gray-700 rounded border border-gray-600 min-w-[140px]"
                >
                  <option value="">Select paint</option>
                  {paints.map((p) => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
                <input
                  type="number"
                  min={0}
                  step={inputMode === 'percent' ? 1 : 0.1}
                  value={row.value || ''}
                  onChange={(e) => setRow(i, row.paintId, parseFloat(e.target.value) || 0)}
                  className="w-20 px-2 py-1.5 bg-gray-700 rounded border border-gray-600"
                  placeholder={inputMode === 'percent' ? '%' : 'g'}
                />
                {inputMode === 'percent' && <span className="text-gray-400">%</span>}
                {inputMode === 'weight' && <span className="text-gray-400">g</span>}
                <button type="button" onClick={() => removeRow(i)} className="text-red-400 hover:text-red-300">Remove</button>
              </div>
            ))}
            <button type="button" onClick={addRow} className="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm">+ Add paint</button>
          </div>
          {inputMode === 'percent' && <p className="text-sm text-gray-400 mb-2">Total: {totalPercent.toFixed(1)}% {totalPercent === 100 ? '✓' : '(must be 100%)'}</p>}
          {inputMode === 'weight' && totalWeight > 0 && (
            <p className="text-sm text-gray-400 mb-2">Total: {totalWeight.toFixed(2)} g → {rows.map((r) => (r.value / totalWeight * 100).toFixed(1)).join('%, ')}%</p>
          )}
          {predictedLoading && <p className="text-sm text-gray-400">Computing predicted color…</p>}
          {predictedHex && canPredict && (
            <div className="flex items-center gap-4 mt-3 flex-wrap">
              <div className="flex items-center gap-2">
                <div className="w-12 h-12 rounded border-2 border-gray-600" style={{ backgroundColor: predictedHex }} title={predictedHex} />
                <span className="text-sm text-gray-400">Predicted mix: {predictedHex.toUpperCase()}</span>
              </div>
              <button
                type="button"
                onClick={() => setStep('upload')}
                className="px-4 py-2 bg-amber-600 hover:bg-amber-500 rounded text-white"
              >
                Continue to spot test (upload image)
              </button>
            </div>
          )}
        </>
      )}

      {step === 'upload' && predictedHex && (
        <div className="space-y-3">
          <p className="text-sm text-gray-300">Upload a photo of the swatch for this recipe.</p>
          <input type="file" accept="image/*" onChange={handleFileChange} disabled={uploading} className="block text-sm text-gray-300 file:mr-2 file:py-2 file:px-3 file:rounded file:border-0 file:bg-gray-600 file:text-gray-200" />
          {uploading && <p className="text-sm text-gray-400">Uploading…</p>}
          <button type="button" onClick={resetToRecipe} className="px-3 py-1.5 bg-gray-600 hover:bg-gray-500 rounded text-sm">← Back to recipe</button>
        </div>
      )}

      {step === 'sample' && previewUrl && (
        <div className="space-y-3">
          <p className="text-sm text-gray-300">Draw a rectangle around the mixed paint area (click and drag). All pixels in the box are averaged.</p>
          <div
            className="relative inline-block cursor-crosshair border-2 border-gray-600"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
          >
            <img
              ref={imageRef}
              src={previewUrl}
              alt="Swatch"
              className="max-w-full block pointer-events-none select-none rounded"
              draggable={false}
              crossOrigin="anonymous"
              onLoad={() => imageRef.current && setImageDisplaySize({ width: imageRef.current.getBoundingClientRect().width, height: imageRef.current.getBoundingClientRect().height })}
            />
            {imageDisplaySize && imageRef.current && (
              <svg className="absolute top-0 left-0 pointer-events-none" width={imageDisplaySize.width} height={imageDisplaySize.height} style={{ display: 'block' }}>
                {selectedRegion && (() => {
                  const nw = imageRef.current!.naturalWidth
                  const nh = imageRef.current!.naturalHeight
                  const sx = imageDisplaySize.width / nw
                  const sy = imageDisplaySize.height / nh
                  const x = Math.min(selectedRegion.x1, selectedRegion.x2) * sx
                  const y = Math.min(selectedRegion.y1, selectedRegion.y2) * sy
                  const w = Math.abs(selectedRegion.x2 - selectedRegion.x1) * sx
                  const h = Math.abs(selectedRegion.y2 - selectedRegion.y1) * sy
                  return <rect x={x} y={y} width={w} height={h} fill="rgba(34,197,94,0.2)" stroke="rgb(34,197,94)" strokeWidth={2} />
                })()}
                {dragStart && dragCurrent && (
                  <rect x={Math.min(dragStart.x, dragCurrent.x)} y={Math.min(dragStart.y, dragCurrent.y)} width={Math.abs(dragCurrent.x - dragStart.x)} height={Math.abs(dragCurrent.y - dragStart.y)} fill="rgba(255,255,255,0.2)" stroke="white" strokeWidth={2} strokeDasharray="4" />
                )}
              </svg>
            )}
            {sampling && <div className="absolute inset-0 flex items-center justify-center bg-black/40 rounded pointer-events-none"><span className="text-white">Sampling…</span></div>}
          </div>
          {selectedRegion && (
            <button type="button" onClick={handleSample} disabled={sampling} className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-white disabled:opacity-50">
              {sampling ? 'Sampling…' : 'Sample area & compare'}
            </button>
          )}
          <button type="button" onClick={() => { setStep('upload'); setSelectedRegion(null); setPreviewUrl(null); setImageId(null) }} className="ml-2 px-3 py-1.5 bg-gray-600 hover:bg-gray-500 rounded text-sm">← Back</button>
        </div>
      )}

      {step === 'confirm_commit' && result && predictedHex && (
        <div className="space-y-4">
          <p className="text-sm text-gray-300 font-semibold">Compare expected vs actual, then choose whether to commit the correction.</p>
          <div className="flex items-end gap-6 flex-wrap">
            <div className="text-center">
              <div className="w-24 h-24 rounded border-2 border-gray-500" style={{ backgroundColor: predictedHex }} title={predictedHex} />
              <div className="text-xs text-gray-400 mt-1">Expected</div>
            </div>
            <div className="text-center">
              <div
                className="w-24 h-24 rounded border-2 border-gray-500"
                style={{ backgroundColor: result.measured_rgb ? rgbToHex(result.measured_rgb) : '#000000' }}
                title={result.measured_rgb ? rgbToHex(result.measured_rgb) : ''}
              />
              <div className="text-xs text-gray-400 mt-1">Actual (from photo)</div>
            </div>
            <div className="flex items-center">
              <span className="text-lg font-bold text-gray-200">ΔE = {result.delta_e != null ? result.delta_e.toFixed(2) : '—'}</span>
            </div>
          </div>
          {pigmentIds.length > 1 && (
            <div className="space-y-1">
              <label className="text-sm text-gray-400 block">If the problem is one paint (e.g. too much of it), choose it so the correction applies only to that paint:</label>
              <select value={focusPaintId} onChange={(e) => setFocusPaintId(e.target.value)} className="block w-full max-w-xs px-2 py-1.5 bg-gray-700 rounded border border-gray-600">
                <option value="">All pigments (share by ratio)</option>
                {pigmentIds.map((id) => (
                  <option key={id} value={id}>{id}</option>
                ))}
              </select>
            </div>
          )}
          <p className="text-sm text-gray-400">Commit this correction? Future recipes will use it for the selected paint(s).</p>
          <div className="flex gap-3">
            <button type="button" onClick={() => handleCommit(true)} disabled={committing} className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-white disabled:opacity-50">
              {committing ? 'Committing…' : 'Yes'}
            </button>
            <button type="button" onClick={() => handleCommit(false)} disabled={committing} className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded disabled:opacity-50">
              No
            </button>
          </div>
        </div>
      )}

      {step === 'done' && result && (
        <div className="space-y-2">
          {result.delta_e != null && <p className="text-sm text-gray-300">Measured vs predicted: <strong>ΔE = {result.delta_e.toFixed(2)}</strong></p>}
          {result.feedback_updated && result.paints_updated && result.paints_updated.length > 0 && (
            <p className="text-sm text-green-400">Correction applied to: {result.paints_updated.join(', ')}. Future recipes will use this correction.</p>
          )}
          <button type="button" onClick={resetToRecipe} className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded">Do another spot test</button>
        </div>
      )}

      {error && <p className="text-sm text-red-400 mt-2">{error}</p>}

      <div className="mt-6 pt-4 border-t border-gray-600">
        <h3 className="text-lg font-bold mb-2">Remove a previous spot-test correction</h3>
        <p className="text-sm text-gray-400 mb-2">
          Corrections that were committed earlier can be removed so they no longer affect recipe generation.
        </p>
        <button type="button" onClick={loadBiasList} disabled={biasLoading} className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm disabled:opacity-50 mb-3">
          {biasLoading ? 'Loading…' : 'Load corrections for this library'}
        </button>
        {biasList && Object.keys(biasList).length === 0 && <p className="text-sm text-gray-500">No spot-test corrections for this library.</p>}
        {biasList && Object.keys(biasList).length > 0 && (
          <ul className="space-y-2">
            {Object.keys(biasList).map((paintId) => (
              <li key={paintId} className="flex items-center gap-3">
                <span className="text-sm text-gray-300">{paintId}</span>
                <button type="button" onClick={() => handleRemoveBias(paintId)} disabled={resetting !== null} className="px-2 py-1 bg-red-700 hover:bg-red-600 rounded text-xs disabled:opacity-50">
                  {resetting === paintId ? 'Removing…' : 'Remove'}
                </button>
              </li>
            ))}
            <li>
              <button type="button" onClick={handleResetAllBias} disabled={resetting !== null} className="px-3 py-1.5 bg-red-800 hover:bg-red-700 rounded text-sm disabled:opacity-50">
                {resetting === 'all' ? 'Resetting…' : 'Reset all corrections'}
              </button>
            </li>
          </ul>
        )}
      </div>
    </div>
  )
}

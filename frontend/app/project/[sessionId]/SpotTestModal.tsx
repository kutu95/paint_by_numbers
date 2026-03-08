'use client'

import { useState, useRef } from 'react'
import { API_BASE_URL } from '@/lib/config'

/** Build recipe components for API: [{ paint_id, ratio }, ...] with ratio 0..1 */
export function recipeToComponents(recipe: any): Array<{ paint_id: string; ratio: number }> {
  if (!recipe) return []
  const out: Array<{ paint_id: string; ratio: number }> = []
  if (recipe.ingredients && Array.isArray(recipe.ingredients)) {
    for (const ing of recipe.ingredients) {
      const id = ing.paint_id ?? ing.paint_name
      const pct = ing.percentage
      if (id != null && typeof pct === 'number' && pct >= 0) {
        out.push({ paint_id: String(id), ratio: pct / 100 })
      }
    }
    if (out.length > 0) return out
  }
  const whiteRatio = recipe.white_ratio
  if (typeof whiteRatio === 'number' && whiteRatio > 0) {
    out.push({ paint_id: 'white', ratio: whiteRatio })
  }
  if (recipe.pigment_id != null && typeof recipe.pigment_ratio === 'number') {
    out.push({ paint_id: String(recipe.pigment_id), ratio: recipe.pigment_ratio })
  }
  if (recipe.pigment1_id != null && typeof recipe.pigment1_ratio === 'number') {
    out.push({ paint_id: String(recipe.pigment1_id), ratio: recipe.pigment1_ratio })
  }
  if (recipe.pigment2_id != null && typeof recipe.pigment2_ratio === 'number') {
    out.push({ paint_id: String(recipe.pigment2_id), ratio: recipe.pigment2_ratio })
  }
  if (Array.isArray(recipe.pigment_ids) && Array.isArray(recipe.pigment_ratios)) {
    recipe.pigment_ids.forEach((id: string, i: number) => {
      const r = recipe.pigment_ratios[i]
      if (id != null && typeof r === 'number') {
        out.push({ paint_id: String(id), ratio: r })
      }
    })
  }
  return out
}

export interface SpotTestModalProps {
  open: boolean
  onClose: () => void
  sessionId: string
  paletteIndex: number
  targetHex: string
  recipe: any
  libraryGroup: string
}

export function SpotTestModal({
  open,
  onClose,
  sessionId,
  paletteIndex,
  targetHex,
  recipe,
  libraryGroup,
}: SpotTestModalProps) {
  const [step, setStep] = useState<'upload' | 'click' | 'confirm' | 'done'>('upload')
  const [uploading, setUploading] = useState(false)
  const [sampling, setSampling] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [imageId, setImageId] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<{
    delta_e: number | null
    feedback_updated: boolean
    paints_updated: string[]
  } | null>(null)
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null)
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null)
  const [selectedRegion, setSelectedRegion] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null)
  const [imageDisplaySize, setImageDisplaySize] = useState<{ width: number; height: number } | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const containerRef = useRef<HTMLDivElement | null>(null)

  function displayToImage(displayX: number, displayY: number): { x: number; y: number } {
    if (!imageRef.current) return { x: 0, y: 0 }
    const rect = imageRef.current.getBoundingClientRect()
    const scaleX = imageRef.current.naturalWidth / rect.width
    const scaleY = imageRef.current.naturalHeight / rect.height
    return { x: Math.round(displayX * scaleX), y: Math.round(displayY * scaleY) }
  }

  if (!open) return null

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    setError(null)
    setUploading(true)
    try {
      const form = new FormData()
      form.append('image', file)
      form.append('session_id', sessionId)
      form.append('palette_index', String(paletteIndex))
      const res = await fetch(`${API_BASE_URL}/api/paint/verify/upload`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) throw new Error(await res.text().catch(() => `HTTP ${res.status}`))
      const data = await res.json()
      setImageId(data.image_id)
      const raw = data.preview_url
      const url = raw
        ? (raw.startsWith('http') ? raw : `${API_BASE_URL}${raw}`)
        : null
      setPreviewUrl(url)
      setStep('click')
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
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    setDragCurrent({ x, y })
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
    const minSize = 5
    if (x2 - x1 >= minSize && y2 - y1 >= minSize) {
      const p1 = displayToImage(x1, y1)
      const p2 = displayToImage(x2, y2)
      setSelectedRegion({ x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y })
    }
    setDragStart(null)
    setDragCurrent(null)
  }

  const handleMouseLeave = () => {
    setDragStart(null)
    setDragCurrent(null)
  }

  const handleSampleRegion = async () => {
    if (!imageId || !selectedRegion) return
    setError(null)
    setSampling(true)
    try {
      const form = new FormData()
      form.append('session_id', sessionId)
      form.append('palette_index', String(paletteIndex))
      form.append('image_id', imageId)
      form.append('x1', String(selectedRegion.x1))
      form.append('y1', String(selectedRegion.y1))
      form.append('x2', String(selectedRegion.x2))
      form.append('y2', String(selectedRegion.y2))
      form.append('library_group', libraryGroup)
      form.append('target_hex', targetHex)
      form.append('recipe', JSON.stringify(recipeToComponents(recipe)))
      form.append('apply_feedback', 'true')
      const res = await fetch(`${API_BASE_URL}/api/paint/verify/sample`, {
        method: 'POST',
        body: form,
      })
      if (!res.ok) throw new Error(await res.text().catch(() => `HTTP ${res.status}`))
      const data = await res.json()
      setResult({
        delta_e: data.delta_e ?? null,
        feedback_updated: data.feedback_updated === true,
        paints_updated: Array.isArray(data.paints_updated) ? data.paints_updated : [],
      })
      setStep('done')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSampling(false)
    }
  }

  const handleClose = () => {
    setStep('upload')
    setImageId(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    setSelectedRegion(null)
    setDragStart(null)
    setDragCurrent(null)
    setImageDisplaySize(null)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4" onClick={handleClose}>
      <div
        className="bg-gray-800 rounded-lg shadow-xl max-w-lg w-full max-h-[90vh] overflow-y-auto border border-gray-600"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-4 flex justify-between items-center border-b border-gray-600">
          <h3 className="text-lg font-bold">Verify mix (spot test)</h3>
          <button
            type="button"
            onClick={handleClose}
            className="text-gray-400 hover:text-white px-2"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="p-4 space-y-4">
          {step === 'upload' && (
            <>
              <p className="text-sm text-gray-300">
                Upload a photo of the swatch you mixed for <strong>Palette Color {paletteIndex}</strong>. Then draw a rectangle around the mixed paint area; we’ll average that region (like calibration) and compare it to the target.
              </p>
              <label className="block">
                <span className="text-sm text-gray-400 block mb-1">Swatch photo</span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  disabled={uploading}
                  className="block w-full text-sm text-gray-300 file:mr-2 file:py-2 file:px-3 file:rounded file:border-0 file:bg-gray-600 file:text-gray-200"
                />
              </label>
              {uploading && <p className="text-sm text-gray-400">Uploading…</p>}
            </>
          )}
          {step === 'click' && previewUrl && (
            <>
              <p className="text-sm text-gray-300">
                Draw a <strong>rectangle</strong> around the mixed paint area (click and drag). All pixels in the box are averaged, same as the calibration flow.
              </p>
              <div
                ref={containerRef}
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
                  onLoad={() => {
                    if (imageRef.current) {
                      const r = imageRef.current.getBoundingClientRect()
                      setImageDisplaySize({ width: r.width, height: r.height })
                    }
                  }}
                />
                {imageDisplaySize && imageRef.current && (
                  <svg
                    className="absolute top-0 left-0 pointer-events-none"
                    width={imageDisplaySize.width}
                    height={imageDisplaySize.height}
                    style={{ display: 'block' }}
                  >
                    {selectedRegion && (() => {
                      const nw = imageRef.current!.naturalWidth
                      const nh = imageRef.current!.naturalHeight
                      const scaleX = imageDisplaySize.width / nw
                      const scaleY = imageDisplaySize.height / nh
                      const x = Math.min(selectedRegion.x1, selectedRegion.x2) * scaleX
                      const y = Math.min(selectedRegion.y1, selectedRegion.y2) * scaleY
                      const w = Math.abs(selectedRegion.x2 - selectedRegion.x1) * scaleX
                      const h = Math.abs(selectedRegion.y2 - selectedRegion.y1) * scaleY
                      return (
                        <rect
                          x={x}
                          y={y}
                          width={w}
                          height={h}
                          fill="rgba(34,197,94,0.2)"
                          stroke="rgb(34,197,94)"
                          strokeWidth={2}
                        />
                      )
                    })()}
                    {dragStart && dragCurrent && (
                      <rect
                        x={Math.min(dragStart.x, dragCurrent.x)}
                        y={Math.min(dragStart.y, dragCurrent.y)}
                        width={Math.abs(dragCurrent.x - dragStart.x)}
                        height={Math.abs(dragCurrent.y - dragStart.y)}
                        fill="rgba(255,255,255,0.2)"
                        stroke="white"
                        strokeWidth={2}
                        strokeDasharray="4"
                      />
                    )}
                  </svg>
                )}
                {sampling && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/40 rounded pointer-events-none">
                    <span className="text-white">Sampling…</span>
                  </div>
                )}
              </div>
              {selectedRegion && (
                <button
                  type="button"
                  onClick={() => setStep('confirm')}
                  className="mt-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-white"
                >
                  Sample area & compare
                </button>
              )}
            </>
          )}
          {step === 'confirm' && selectedRegion && (
            <>
              <p className="text-sm text-gray-300">
                The recipe model will be updated for this combination; future recipes will use this correction.
              </p>
              <p className="text-sm text-gray-400">
                Continue?
              </p>
              <div className="flex gap-3">
                <button
                  type="button"
                  onClick={handleSampleRegion}
                  disabled={sampling}
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded disabled:opacity-50 text-white"
                >
                  {sampling ? 'Sampling…' : 'Yes'}
                </button>
                <button
                  type="button"
                  onClick={() => setStep('click')}
                  disabled={sampling}
                  className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded text-white disabled:opacity-50"
                >
                  Cancel
                </button>
              </div>
            </>
          )}
          {step === 'done' && result && (
            <>
              <p className="text-sm text-gray-300">
                {result.delta_e != null ? (
                  <>Measured colour vs target: <strong>ΔE = {result.delta_e.toFixed(2)}</strong></>
                ) : (
                  'Sample completed.'
                )}
              </p>
              {result.feedback_updated && result.paints_updated.length > 0 && (
                <p className="text-sm text-green-400">
                  Recipe model updated for: {result.paints_updated.join(', ')}. Future recipes will use this correction.
                </p>
              )}
              <button
                type="button"
                onClick={handleClose}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded"
              >
                Okay
              </button>
            </>
          )}
          {error && (
            <p className="text-sm text-red-400">{error}</p>
          )}
        </div>
      </div>
    </div>
  )
}

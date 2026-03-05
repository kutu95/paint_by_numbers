'use client'

import { useState, useRef, useEffect } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import { API_BASE_URL } from '@/lib/config'

interface CalibrationSample {
  ratio: number
  rgb: number[]
  lab: number[]
}

export default function CalibratePage() {
  const params = useParams()
  const router = useRouter()
  const searchParams = useSearchParams()
  const paintId = params.paintId as string
  const group = (searchParams.get('group') || (typeof window !== 'undefined' ? localStorage.getItem('lastSelectedPaintLibrary') : null) || 'default')
  const [ratios, setRatios] = useState<number[]>([1, 0.5, 0.25, 0.125, 0.0625, 0.03125])
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageId, setImageId] = useState<string | null>(null)
  const [paintRegions, setPaintRegions] = useState<Array<{ x1: number; y1: number; x2: number; y2: number }>>([])
  const [referenceRegions, setReferenceRegions] = useState<Array<{ x1: number; y1: number; x2: number; y2: number }>>([])
  const [samples, setSamples] = useState<CalibrationSample[]>([])
  const [uploading, setUploading] = useState(false)
  const [sampling, setSampling] = useState(false)
  const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null)
  const [dragCurrent, setDragCurrent] = useState<{ x: number; y: number } | null>(null)
  const [imageDisplaySize, setImageDisplaySize] = useState<{ width: number; height: number } | null>(null)
  const imageRef = useRef<HTMLImageElement>(null)

  const REFERENCE_LABELS = ['White', 'Mid-grey', 'Black'] as const
  const allRegionsDone = paintRegions.length === ratios.length && referenceRegions.length === 3

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    const formData = new FormData()
    formData.append('image', file)
    formData.append('paint_id', paintId)
    formData.append('group', group)

    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/calibration/upload`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setImageId(data.image_id)
      setImageUrl(`${API_BASE_URL}${data.preview_url}`)
      setPaintRegions([])
      setReferenceRegions([])
      setSamples([])
      setImageDisplaySize(null)
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to upload image')
    } finally {
      setUploading(false)
    }
  }

  function displayToImage(displayX: number, displayY: number): { x: number; y: number } {
    if (!imageRef.current) return { x: 0, y: 0 }
    const rect = imageRef.current.getBoundingClientRect()
    const scaleX = imageRef.current.naturalWidth / rect.width
    const scaleY = imageRef.current.naturalHeight / rect.height
    return { x: Math.round(displayX * scaleX), y: Math.round(displayY * scaleY) }
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!imageRef.current || !imageId) return
    if (paintRegions.length >= ratios.length && referenceRegions.length >= 3) return
    const rect = imageRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) return
    setDragStart({ x, y })
    setDragCurrent({ x, y })
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
    if (x2 - x1 < minSize || y2 - y1 < minSize) {
      setDragStart(null)
      setDragCurrent(null)
      return
    }
    const p1 = displayToImage(x1, y1)
    const p2 = displayToImage(x2, y2)
    const region = { x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y }
    if (paintRegions.length < ratios.length) {
      setPaintRegions([...paintRegions, region])
    } else {
      setReferenceRegions([...referenceRegions, region])
    }
    setDragStart(null)
    setDragCurrent(null)
  }

  const handleMouseLeave = () => {
    setDragStart(null)
    setDragCurrent(null)
  }

  const handleSample = async () => {
    if (!imageId || !allRegionsDone) {
      alert('Please draw a rectangle over each paint swatch and then over the white, mid-grey, and black reference patches')
      return
    }

    setSampling(true)
    const formData = new FormData()
    formData.append('image_id', imageId)
    formData.append('paint_id', paintId)
    formData.append('regions', JSON.stringify(paintRegions))
    formData.append('ratios', JSON.stringify(ratios))
    formData.append('reference_regions', JSON.stringify(referenceRegions))
    formData.append('group', group)

    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/calibration/sample`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setSamples(data.samples)
      alert('Calibration saved successfully!')
      router.push(`/paints?group=${encodeURIComponent(group)}`)
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to sample colors')
    } finally {
      setSampling(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-4xl font-bold">Calibrate Paint: {paintId}</h1>
          <button
            onClick={() => router.push(`/paints?group=${encodeURIComponent(group)}`)}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
          >
            ← Back
          </button>
        </div>

        <div className="space-y-6">
          <div className="p-6 bg-gray-800 rounded">
            <h2 className="text-2xl font-bold mb-4">Instructions</h2>
            <ol className="list-decimal list-inside space-y-2 text-gray-300">
              <li>Mix your paint with white at these ratios: {ratios.map(r => `${(r * 100).toFixed(1)}%`).join(', ')}</li>
              <li>Paint small squares for each ratio on your target surface</li>
              <li>Include a reference strip: white, mid-grey, black (these will be sampled too)</li>
              <li>Take a photo straight-on with good lighting</li>
              <li>Upload the photo, then draw a rectangle inside each paint swatch (darkest to lightest)</li>
              <li>Then draw a rectangle inside the white, mid-grey, and black reference patches in that order</li>
              <li>All pixels inside each rectangle are averaged for a stable sample</li>
            </ol>
          </div>

          <div className="p-6 bg-gray-800 rounded">
            <h2 className="text-2xl font-bold mb-4">Ratios</h2>
            <div className="space-y-2">
              {ratios.map((ratio, idx) => (
                <div key={idx} className="flex items-center gap-4">
                  <span className="w-32">Swatch {idx + 1}:</span>
                  <input
                    type="number"
                    step="0.001"
                    value={ratio}
                    onChange={(e) => {
                      const newRatios = [...ratios]
                      newRatios[idx] = parseFloat(e.target.value)
                      setRatios(newRatios)
                    }}
                    className="px-3 py-1 bg-gray-700 rounded text-white w-32"
                  />
                  <span className="text-gray-400">({(ratio * 100).toFixed(1)}% pigment)</span>
                </div>
              ))}
            </div>
          </div>

          <div className="p-6 bg-gray-800 rounded">
            <h2 className="text-2xl font-bold mb-4">Upload Calibration Photo</h2>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              disabled={uploading}
              className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
            />
            {uploading && <p className="mt-2 text-gray-400">Uploading...</p>}
          </div>

          {imageUrl && (
            <div className="p-6 bg-gray-800 rounded">
              <h2 className="text-2xl font-bold mb-4">
                {paintRegions.length < ratios.length
                  ? `Draw a rectangle inside each paint swatch (${paintRegions.length} / ${ratios.length})`
                  : referenceRegions.length < 3
                    ? `Draw reference strip: ${REFERENCE_LABELS[referenceRegions.length]} (${referenceRegions.length + 1} / 3)`
                    : `All done (${ratios.length} paint + 3 reference)`}
              </h2>
              <p className="text-gray-400 mb-4">
                {paintRegions.length < ratios.length
                  ? 'Click and drag to draw a rectangle inside each paint swatch, from darkest (100%) to lightest.'
                  : 'Draw a rectangle inside the white, mid-grey, and black patches. All pixels in the box are averaged.'}
              </p>
              <div
                className="relative inline-block cursor-crosshair border-2 border-gray-600"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseLeave}
              >
                <img
                  ref={imageRef}
                  src={imageUrl}
                  alt="Calibration photo"
                  className="max-w-full block pointer-events-none select-none"
                  draggable={false}
                  onLoad={() => {
                    if (imageRef.current) {
                      const r = imageRef.current.getBoundingClientRect()
                      setImageDisplaySize({ width: r.width, height: r.height })
                    }
                  }}
                />
                {/* Overlay: existing regions + drag preview */}
                {imageDisplaySize && imageRef.current && (
                  <svg
                    className="absolute top-0 left-0 pointer-events-none"
                    width={imageDisplaySize.width}
                    height={imageDisplaySize.height}
                    style={{ display: 'block' }}
                  >
                    {(() => {
                      const nw = imageRef.current!.naturalWidth
                      const nh = imageRef.current!.naturalHeight
                      const scaleX = imageDisplaySize.width / nw
                      const scaleY = imageDisplaySize.height / nh
                      const toDisplay = (ix: number, iy: number) => ({ x: ix * scaleX, y: iy * scaleY })
                      return (
                        <>
                          {paintRegions.map((reg, idx) => {
                            const a = toDisplay(reg.x1, reg.y1)
                            const b = toDisplay(reg.x2, reg.y2)
                            return (
                              <rect
                                key={`p-${idx}`}
                                x={Math.min(a.x, b.x)}
                                y={Math.min(a.y, b.y)}
                                width={Math.abs(b.x - a.x)}
                                height={Math.abs(b.y - a.y)}
                                fill="rgba(34,197,94,0.15)"
                                stroke="rgb(34,197,94)"
                                strokeWidth={2}
                              />
                            )
                          })}
                          {referenceRegions.map((reg, idx) => {
                            const a = toDisplay(reg.x1, reg.y1)
                            const b = toDisplay(reg.x2, reg.y2)
                            return (
                              <rect
                                key={`r-${idx}`}
                                x={Math.min(a.x, b.x)}
                                y={Math.min(a.y, b.y)}
                                width={Math.abs(b.x - a.x)}
                                height={Math.abs(b.y - a.y)}
                                fill="rgba(59,130,246,0.15)"
                                stroke="rgb(59,130,246)"
                                strokeWidth={2}
                              />
                            )
                          })}
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
                        </>
                      )
                    })()}
                  </svg>
                )}
              </div>
              {(paintRegions.length > 0 || referenceRegions.length > 0) && (
                <div className="mt-4">
                  <h3 className="font-bold mb-2">Selected regions</h3>
                  <div className="space-y-1 text-sm text-gray-300">
                    {paintRegions.map((reg, idx) => (
                      <div key={`p-${idx}`}>
                        Paint {idx + 1}: {(ratios[idx] * 100).toFixed(1)}% – box from ({reg.x1},{reg.y1}) to ({reg.x2},{reg.y2})
                      </div>
                    ))}
                    {referenceRegions.map((_, idx) => (
                      <div key={`r-${idx}`}>{REFERENCE_LABELS[idx]} – selected</div>
                    ))}
                  </div>
                </div>
              )}
              {allRegionsDone && (
                <button
                  onClick={handleSample}
                  disabled={sampling}
                  className="mt-4 px-6 py-3 bg-green-600 hover:bg-green-700 rounded disabled:opacity-50"
                >
                  {sampling ? 'Sampling...' : 'Sample Colors & Save Calibration'}
                </button>
              )}
            </div>
          )}

          {samples.length > 0 && (
            <div className="p-6 bg-gray-800 rounded">
              <h2 className="text-2xl font-bold mb-4">Calibration Results</h2>
              <div className="space-y-2">
                {samples.map((sample, idx) => (
                  <div key={idx} className="flex items-center gap-4 p-2 bg-gray-700 rounded">
                    <div
                      className="w-12 h-12 rounded border border-gray-600"
                      style={{ backgroundColor: `rgb(${sample.rgb.join(',')})` }}
                    />
                    <div>
                      <div>Ratio: {(sample.ratio * 100).toFixed(1)}%</div>
                      <div className="text-xs text-gray-400">
                        RGB: {sample.rgb.join(', ')} | Lab: {sample.lab.map(v => v.toFixed(1)).join(', ')}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

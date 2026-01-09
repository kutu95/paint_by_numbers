'use client'

import { useState, useRef, useEffect } from 'react'
import { useParams, useRouter } from 'next/navigation'
import { API_BASE_URL } from '@/lib/config'

interface CalibrationSample {
  ratio: number
  rgb: number[]
  lab: number[]
}

export default function CalibratePage() {
  const params = useParams()
  const router = useRouter()
  const paintId = params.paintId as string
  const [ratios, setRatios] = useState<number[]>([0.5, 0.25, 0.125, 0.0625, 0.03125])
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [imageId, setImageId] = useState<string | null>(null)
  const [clickedPoints, setClickedPoints] = useState<Array<{ x: number; y: number; ratio: number }>>([])
  const [samples, setSamples] = useState<CalibrationSample[]>([])
  const [uploading, setUploading] = useState(false)
  const [sampling, setSampling] = useState(false)
  const imageRef = useRef<HTMLImageElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    const formData = new FormData()
    formData.append('image', file)
    formData.append('paint_id', paintId)

    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/calibration/upload`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setImageId(data.image_id)
      setImageUrl(`${API_BASE_URL}${data.preview_url}`)
      setClickedPoints([])
      setSamples([])
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to upload image')
    } finally {
      setUploading(false)
    }
  }

  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!imageRef.current || !imageId) return

    const rect = imageRef.current.getBoundingClientRect()
    const x = Math.round(e.clientX - rect.left)
    const y = Math.round(e.clientY - rect.top)

    // Scale coordinates to actual image size
    const scaleX = imageRef.current.naturalWidth / rect.width
    const scaleY = imageRef.current.naturalHeight / rect.height
    const actualX = Math.round(x * scaleX)
    const actualY = Math.round(y * scaleY)

    const nextIndex = clickedPoints.length
    if (nextIndex >= ratios.length) {
      alert('All swatches have been clicked')
      return
    }

    const newPoint = { x: actualX, y: actualY, ratio: ratios[nextIndex] }
    setClickedPoints([...clickedPoints, newPoint])
  }

  const handleSample = async () => {
    if (!imageId || clickedPoints.length !== ratios.length) {
      alert('Please click all swatches in order')
      return
    }

    setSampling(true)
    const formData = new FormData()
    formData.append('image_id', imageId)
    formData.append('paint_id', paintId)
    formData.append('points', JSON.stringify(clickedPoints.map(p => ({ x: p.x, y: p.y }))))
    formData.append('ratios', JSON.stringify(ratios))

    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/calibration/sample`, {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      setSamples(data.samples)
      alert('Calibration saved successfully!')
      router.push('/paints')
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
            onClick={() => router.push('/paints')}
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
              <li>Include a reference strip: white, mid-grey, black</li>
              <li>Take a photo straight-on with good lighting</li>
              <li>Upload the photo and click each swatch in order (darkest to lightest)</li>
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
                Click Swatches in Order ({clickedPoints.length} / {ratios.length})
              </h2>
              <p className="text-gray-400 mb-4">
                Click each swatch from darkest (highest ratio) to lightest (lowest ratio)
              </p>
              <div className="relative inline-block">
                <img
                  ref={imageRef}
                  src={imageUrl}
                  alt="Calibration photo"
                  onClick={handleImageClick}
                  className="max-w-full cursor-crosshair border-2 border-gray-600"
                />
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 pointer-events-none"
                  style={{ display: 'none' }}
                />
              </div>
              {clickedPoints.length > 0 && (
                <div className="mt-4">
                  <h3 className="font-bold mb-2">Clicked Points:</h3>
                  <div className="space-y-1 text-sm">
                    {clickedPoints.map((point, idx) => (
                      <div key={idx}>
                        Point {idx + 1}: ({point.x}, {point.y}) - Ratio: {(point.ratio * 100).toFixed(1)}%
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {clickedPoints.length === ratios.length && (
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


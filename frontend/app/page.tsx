'use client'

import { useState, useRef, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { API_BASE_URL } from '@/lib/config'

interface PaletteColor {
  index: number
  hex: string
  coverage: number
}

interface Layer {
  layer_index: number
  palette_index: number
  mask_url: string
  outline_thin_url: string
  outline_thick_url: string
  outline_glow_url: string
}

interface SessionResponse {
  session_id: string
  width: number
  height: number
  palette: PaletteColor[]
  order: number[]
  quantized_preview_url: string
  layers: Layer[]
}

export default function Home() {
  const [image, setImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [nColors, setNColors] = useState(16)
  const [overpaintMm, setOverpaintMm] = useState(5)
  const [orderMode, setOrderMode] = useState<'largest' | 'smallest' | 'manual'>('largest')
  const [maxSide, setMaxSide] = useState(1920)
  const [processing, setProcessing] = useState(false)
  const [sessionData, setSessionData] = useState<SessionResponse | null>(null)
  const [manualOrder, setManualOrder] = useState<number[]>([])
  const [recipes, setRecipes] = useState<any[]>([])
  const [loadingRecipes, setLoadingRecipes] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const router = useRouter()

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setImage(file)
      const reader = new FileReader()
      reader.onload = (e) => {
        const previewData = e.target?.result as string
        setPreview(previewData)
        // Save preview to localStorage so it persists across navigation
        localStorage.setItem('current_image_preview', previewData)
        // Also save the file name for reference
        localStorage.setItem('current_image_name', file.name)
      }
      reader.readAsDataURL(file)
    }
  }

  // Restore image preview from localStorage on mount
  useEffect(() => {
    const savedPreview = localStorage.getItem('current_image_preview')
    const savedFileName = localStorage.getItem('current_image_name')
    if (savedPreview) {
      setPreview(savedPreview)
      // Convert data URL back to File object
      try {
        fetch(savedPreview)
          .then(res => res.blob())
          .then(blob => {
            const file = new File([blob], savedFileName || 'image.jpg', { type: blob.type })
            setImage(file)
          })
          .catch(err => {
            console.error('Failed to restore image file:', err)
            // Preview will still be shown, user can re-upload if needed
          })
      } catch (err) {
        console.error('Failed to restore image file:', err)
      }
    }
  }, [])

  const handleGenerate = async () => {
    if (!image) {
      if (preview) {
        alert('Please re-select the image file to generate layers. The preview is shown but the file needs to be selected again.')
      }
      return
    }

    setProcessing(true)
    const formData = new FormData()
    formData.append('image', image)
    formData.append('n_colors', nColors.toString())
    formData.append('overpaint_mm', overpaintMm.toString())
    formData.append('order_mode', orderMode)
    formData.append('max_side', maxSide.toString())

    try {
      const response = await fetch(`${API_BASE_URL}/api/sessions`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        // Try to get error message from response
        let errorMessage = 'Processing failed'
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorMessage
        } catch {
          errorMessage = `Server error: ${response.status} ${response.statusText}`
        }
        throw new Error(errorMessage)
      }

      const data: SessionResponse = await response.json()
      setSessionData(data)
      // Save to localStorage for projection viewer
      localStorage.setItem(`session_${data.session_id}`, JSON.stringify(data))
      if (orderMode === 'manual') {
        setManualOrder([...data.order])
      }
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to process image'
      alert(errorMessage)
    } finally {
      setProcessing(false)
    }
  }

  const moveLayer = (index: number, direction: 'up' | 'down') => {
    if (!sessionData || orderMode !== 'manual') return
    const newOrder = [...manualOrder]
    const newIndex = direction === 'up' ? index - 1 : index + 1
    if (newIndex < 0 || newIndex >= newOrder.length) return
    ;[newOrder[index], newOrder[newIndex]] = [newOrder[newIndex], newOrder[index]]
    setManualOrder(newOrder)
  }

  const handleBack = () => {
    // Clear results but keep the uploaded image
    setSessionData(null)
    setManualOrder([])
  }

  const handleStartProjection = () => {
    if (sessionData) {
      router.push(`/project/${sessionData.session_id}`)
    }
  }

  // Helper function to convert hex to RGB
  const hexToRgb = (hex: string): [number, number, number] => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
    return result
      ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
      : [0, 0, 0]
  }

  // Helper function to get error level based on ΔE value
  const getErrorLevel = (error: number): { level: string; color: string } | null => {
    if (error < 1) return { level: 'Excellent', color: 'green' }
    if (error < 3) return { level: 'Good', color: 'green' }
    if (error < 6) return { level: 'Acceptable', color: 'yellow' }
    return { level: 'Poor', color: 'red' }
  }

  // Helper function to format recipe display
  const formatRecipe = (recipeData: any): string => {
    if (!recipeData.recipe) {
      return recipeData.error || 'No recipe available'
    }

    const recipe = recipeData.recipe
    if (recipeData.type === 'one_pigment') {
      const whitePercent = (recipe.white_ratio * 100).toFixed(1)
      const pigmentPercent = (recipe.pigment_ratio * 100).toFixed(1)
      return `White ${whitePercent}% + ${recipe.pigment_id} ${pigmentPercent}%`
    } else if (recipeData.type === 'two_pigment') {
      const whitePercent = (recipe.white_ratio * 100).toFixed(1)
      const p1Percent = (recipe.pigment1_ratio * 100).toFixed(1)
      const p2Percent = (recipe.pigment2_ratio * 100).toFixed(1)
      return `White ${whitePercent}% + ${recipe.pigment1_id} ${p1Percent}% + ${recipe.pigment2_id} ${p2Percent}%`
    }
    return 'Unknown recipe type'
  }

  // Handle generating recipes from palette
  const handleGenerateRecipes = async () => {
    if (!sessionData) return

    setLoadingRecipes(true)
    try {
      // Convert palette from hex to RGB format expected by API
      const paletteForApi = sessionData.palette.map((color) => ({
        index: color.index,
        rgb: hexToRgb(color.hex),
      }))

      const formData = new FormData()
      formData.append('palette', JSON.stringify(paletteForApi))

      const response = await fetch(`${API_BASE_URL}/api/paint/recipes/from-palette`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to generate recipes')
      }

      const data = await response.json()
      setRecipes(data.recipes || [])
    } catch (error) {
      console.error('Error generating recipes:', error)
      alert('Failed to generate recipes. Make sure you have calibrated at least one paint.')
    } finally {
      setLoadingRecipes(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold mb-8">LayerPainter</h1>

        {!sessionData ? (
          <div className="space-y-6">
            <div>
              <label className="block mb-2">Upload Image</label>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                ref={fileInputRef}
                className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
              />
              {preview && (
                <img src={preview} alt="Preview" className="mt-4 max-w-md rounded" />
              )}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block mb-2">Number of Colors (2-100)</label>
                <input
                  type="number"
                  min="2"
                  max="100"
                  value={nColors}
                  onChange={(e) => setNColors(parseInt(e.target.value))}
                  className="w-full px-3 py-2 bg-gray-800 rounded text-white"
                />
              </div>

              <div>
                <label className="block mb-2">Overpaint (mm)</label>
                <input
                  type="number"
                  min="0"
                  max="50"
                  step="0.5"
                  value={overpaintMm}
                  onChange={(e) => setOverpaintMm(parseFloat(e.target.value))}
                  className="w-full px-3 py-2 bg-gray-800 rounded text-white"
                />
              </div>

              <div>
                <label className="block mb-2">Order Mode</label>
                <select
                  value={orderMode}
                  onChange={(e) => setOrderMode(e.target.value as any)}
                  className="w-full px-3 py-2 bg-gray-800 rounded text-white"
                >
                  <option value="largest">Largest Coverage First</option>
                  <option value="smallest">Smallest Coverage First</option>
                  <option value="manual">Manual</option>
                </select>
              </div>

              <div>
                <label className="block mb-2">Max Resolution</label>
                <select
                  value={maxSide}
                  onChange={(e) => setMaxSide(parseInt(e.target.value))}
                  className="w-full px-3 py-2 bg-gray-800 rounded text-white"
                >
                  <option value="1920">1920px</option>
                  <option value="2400">2400px</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={!image || processing}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {processing ? 'Processing...' : 'Generate Layers'}
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="flex items-center justify-between mb-4">
              <button
                onClick={handleBack}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-2"
              >
                <span>←</span> Back to Settings
              </button>
            </div>
            <div>
              <h2 className="text-2xl font-bold mb-4">Quantized Preview</h2>
              <img
                src={`${API_BASE_URL}${sessionData.quantized_preview_url}`}
                alt="Quantized"
                className="max-w-full rounded"
              />
            </div>

            <div>
              <h2 className="text-2xl font-bold mb-4">Palette</h2>
              <div className="grid grid-cols-8 gap-2">
                {sessionData.palette.map((color) => (
                  <div key={color.index} className="text-center">
                    <div
                      className="w-16 h-16 rounded border border-gray-600"
                      style={{ backgroundColor: color.hex }}
                    />
                    <div className="text-xs mt-1">{color.coverage.toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <h2 className="text-2xl font-bold mb-4">Layers</h2>
              <div className="space-y-2">
                {(orderMode === 'manual' ? manualOrder : sessionData.order).map((paletteIdx, layerIdx) => {
                  const layer = sessionData.layers.find((l) => l.palette_index === paletteIdx)
                  const color = sessionData.palette.find((p) => p.index === paletteIdx)
                  if (!layer || !color) return null

                  return (
                    <div
                      key={layerIdx}
                      className="flex items-center gap-4 p-4 bg-gray-800 rounded"
                    >
                      <div className="text-lg font-mono">{layerIdx}</div>
                      <div
                        className="w-16 h-16 rounded border border-gray-600"
                        style={{ backgroundColor: color.hex }}
                      />
                      <img
                        src={`${API_BASE_URL}${layer.mask_url}`}
                        alt={`Layer ${layerIdx}`}
                        className="w-16 h-16 object-contain bg-gray-700 rounded"
                      />
                      {orderMode === 'manual' && (
                        <div className="flex gap-2">
                          <button
                            onClick={() => moveLayer(layerIdx, 'up')}
                            disabled={layerIdx === 0}
                            className="px-3 py-1 bg-gray-700 rounded disabled:opacity-50"
                          >
                            ↑
                          </button>
                          <button
                            onClick={() => moveLayer(layerIdx, 'down')}
                            disabled={layerIdx === manualOrder.length - 1}
                            className="px-3 py-1 bg-gray-700 rounded disabled:opacity-50"
                          >
                            ↓
                          </button>
                        </div>
                      )}
                      <div className="flex-1 text-sm text-gray-400">
                        Palette {paletteIdx} - {color.coverage.toFixed(1)}% coverage
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            <div className="flex gap-4">
              <button
                onClick={handleStartProjection}
                className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded"
              >
                Start Projection
              </button>
              <button
                onClick={() => router.push('/paints')}
                className="px-6 py-3 bg-purple-600 hover:bg-purple-700 rounded"
              >
                Manage Paints
              </button>
            </div>

            {/* Paint Recipes Panel */}
            <div className="mt-8 p-6 bg-gray-800 rounded">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold">Paint Mixing Recipes</h2>
                <button
                  onClick={handleGenerateRecipes}
                  disabled={loadingRecipes}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50"
                >
                  {loadingRecipes ? 'Generating...' : 'Generate Recipes'}
                </button>
              </div>
              <p className="text-gray-400 mb-4 text-sm">
                Generate mixing recipes for each palette color using your calibrated paints.
                Make sure you have calibrated at least one paint in the Paint Library.
              </p>

              {recipes.length > 0 && (
                <div className="space-y-3">
                  {recipes.map((recipeData) => {
                    const color = sessionData.palette.find(p => p.index === recipeData.palette_index)
                    if (!color) return null

                    const recipe = recipeData.recipe
                    const errorInfo = recipe ? getErrorLevel(recipe.error) : null

                    return (
                      <div
                        key={recipeData.palette_index}
                        className="flex items-center gap-4 p-4 bg-gray-700 rounded"
                      >
                        <div
                          className="w-16 h-16 rounded border border-gray-600 flex-shrink-0"
                          style={{ backgroundColor: color.hex }}
                        />
                        <div className="flex-1">
                          <div className="font-bold">Palette Color {recipeData.palette_index}</div>
                          <div className="text-sm text-gray-300">
                            {formatRecipe(recipeData)}
                          </div>
                          {recipe && (
                            <div className="text-xs text-gray-400 mt-1 flex items-center gap-2">
                              <span>Error: {recipe.error.toFixed(2)} ΔE</span>
                              {errorInfo && (
                                <span
                                  className="px-2 py-0.5 rounded text-xs"
                                  style={{
                                    backgroundColor:
                                      errorInfo.color === 'green'
                                        ? '#16a34a'
                                        : errorInfo.color === 'yellow'
                                        ? '#ca8a04'
                                        : '#dc2626',
                                  }}
                                >
                                  {errorInfo.level}
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}

              {recipes.length === 0 && !loadingRecipes && (
                <div className="text-center py-8 text-gray-400">
                  Click "Generate Recipes" to create mixing recipes for each palette color.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}


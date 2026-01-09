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
  // Load settings from localStorage or use defaults
  const loadSettings = () => {
    if (typeof window === 'undefined') return null
    try {
      const saved = localStorage.getItem('layerpainter_settings')
      if (saved) {
        const parsed = JSON.parse(saved)
        return {
          nColors: parsed.nColors ?? 16,
          overpaintMm: parsed.overpaintMm ?? 5,
          orderMode: parsed.orderMode ?? 'largest',
          maxSide: parsed.maxSide ?? 1920,
          saturationBoost: parsed.saturationBoost ?? 1.0,
          detailLevel: parsed.detailLevel ?? 0.5,
        }
      }
    } catch (e) {
      console.error('Failed to load settings from localStorage:', e)
    }
    return null
  }

  const savedSettings = loadSettings()
  const [nColors, setNColors] = useState(savedSettings?.nColors ?? 16)
  const [overpaintMm, setOverpaintMm] = useState(savedSettings?.overpaintMm ?? 5)
  const [orderMode, setOrderMode] = useState<'largest' | 'smallest' | 'manual' | 'lightest'>(
    savedSettings?.orderMode ?? 'largest'
  )
  const [maxSide, setMaxSide] = useState(savedSettings?.maxSide ?? 1920)
  const [saturationBoost, setSaturationBoost] = useState(savedSettings?.saturationBoost ?? 1.0)
  const [detailLevel, setDetailLevel] = useState(savedSettings?.detailLevel ?? 0.5)
  const [processing, setProcessing] = useState(false)
  const [sessionData, setSessionData] = useState<SessionResponse | null>(null)
  const [manualOrder, setManualOrder] = useState<number[]>([])
  const [recipes, setRecipes] = useState<any[]>([])
  const [loadingRecipes, setLoadingRecipes] = useState(false)
  const [selectedColor, setSelectedColor] = useState<PaletteColor | null>(null)
  const [mounted, setMounted] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const router = useRouter()

  // Helper function to convert hex to RGB object for modal display
  const hexToRgbObject = (hex: string) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
    return result
      ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16),
        }
      : null
  }

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

  // Save settings to localStorage whenever they change
  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      const settings = {
        nColors,
        overpaintMm,
        orderMode,
        maxSide,
        saturationBoost,
        detailLevel,
      }
      localStorage.setItem('layerpainter_settings', JSON.stringify(settings))
    } catch (e) {
      console.error('Failed to save settings to localStorage:', e)
    }
  }, [nColors, overpaintMm, orderMode, maxSide, saturationBoost, detailLevel])

  // Set mounted flag after component mounts (client-side only)
  useEffect(() => {
    setMounted(true)
  }, [])

  // Handle ESC key to close modal
  useEffect(() => {
    if (!mounted || typeof window === 'undefined') return
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && selectedColor) {
        setSelectedColor(null)
      }
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [selectedColor, mounted])

  // Restore image preview and session data from localStorage on mount
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

    // Restore session data if coming back from projection viewer
    const currentSessionId = localStorage.getItem('current_session_id')
    if (currentSessionId) {
      const savedSession = localStorage.getItem(`session_${currentSessionId}`)
      if (savedSession) {
        try {
          const data: SessionResponse = JSON.parse(savedSession)
          setSessionData(data)
          // Restore manual order if needed (check saved order mode or assume manual if order exists)
          // Manual order restoration will happen when orderMode is checked
        } catch (e) {
          console.error('Failed to restore session data:', e)
        }
      }
    }
  }, [])

  // Restore manual order when sessionData is restored and orderMode is manual
  useEffect(() => {
    if (sessionData && orderMode === 'manual') {
      setManualOrder([...sessionData.order])
    }
  }, [sessionData, orderMode])

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
    formData.append('saturation_boost', saturationBoost.toString())
    formData.append('detail_level', detailLevel.toString())

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
      // Save session ID so we can restore when coming back
      localStorage.setItem('current_session_id', sessionData.session_id)
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
    const isUncalibrated = recipe.uncalibrated === true
    const warning = isUncalibrated ? ' (Estimated - not calibrated) ' : ''
    
    if (recipeData.type === 'one_pigment') {
      const whitePercent = (recipe.white_ratio * 100).toFixed(1)
      const pigmentPercent = (recipe.pigment_ratio * 100).toFixed(1)
      return `${warning}White ${whitePercent}% + ${recipe.pigment_id} ${pigmentPercent}%`
    } else if (recipeData.type === 'two_pigment') {
      const whitePercent = (recipe.white_ratio * 100).toFixed(1)
      const p1Percent = (recipe.pigment1_ratio * 100).toFixed(1)
      const p2Percent = (recipe.pigment2_ratio * 100).toFixed(1)
      return `${warning}White ${whitePercent}% + ${recipe.pigment1_id} ${p1Percent}% + ${recipe.pigment2_id} ${p2Percent}%`
    } else if (recipeData.type === 'three_pigment' || recipeData.type === 'four_pigment' || recipeData.type === 'multi_pigment') {
      const whitePercent = (recipe.white_ratio * 100).toFixed(1)
      const pigmentParts = recipe.pigment_ids.map((id: string, idx: number) => {
        const ratio = recipe.pigment_ratios[idx]
        return `${id} ${(ratio * 100).toFixed(1)}%`
      }).join(' + ')
      return `${warning}White ${whitePercent}% + ${pigmentParts}`
    }
    return 'Unknown recipe type'
  }

  // Handle generating recipes from palette
  const [selectedLibraryGroup, setSelectedLibraryGroup] = useState<string>('default')
  const [libraryGroups, setLibraryGroups] = useState<Array<{group: string, name: string, paint_count: number, calibrated_count: number}>>([])
  const [libraryGroupsLoaded, setLibraryGroupsLoaded] = useState(false)

  // Load library groups on mount (client-side only)
  useEffect(() => {
    if (typeof window === 'undefined') return
    loadLibraryGroups()
  }, [])

  // Auto-generate recipes when library group changes and we have session data (client-side only)
  useEffect(() => {
    if (typeof window === 'undefined' || !mounted) return
    if (sessionData && selectedLibraryGroup && libraryGroups.length > 0 && !loadingRecipes && libraryGroupsLoaded) {
      // Use a ref or flag to prevent duplicate calls
      const recipeKey = `${sessionData.session_id}_${selectedLibraryGroup}`
      const lastGenerated = sessionStorage.getItem(`recipes_${recipeKey}`)
      
      if (!lastGenerated) {
        // Delay slightly to avoid hydration issues
        const timer = setTimeout(() => {
          handleGenerateRecipes()
          sessionStorage.setItem(`recipes_${recipeKey}`, 'true')
        }, 100)
        return () => clearTimeout(timer)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedLibraryGroup, libraryGroupsLoaded, mounted]) // Only regenerate when group changes

  const loadLibraryGroups = async () => {
    if (typeof window === 'undefined') return
    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/library/groups`)
      const data = await response.json()
      const groups = data.groups || []
      setLibraryGroups(groups)
      setLibraryGroupsLoaded(true)
      if (groups.length > 0) {
        // Try to find a group with calibrated paints, otherwise use default
        const calibratedGroup = groups.find((g: any) => g.calibrated_count > 0)
        setSelectedLibraryGroup(calibratedGroup ? calibratedGroup.group : groups[0].group)
      }
    } catch (error) {
      console.error('Failed to load library groups:', error)
      setLibraryGroupsLoaded(true) // Set to true even on error to prevent infinite waiting
    }
  }

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
      formData.append('library_group', selectedLibraryGroup)

      const response = await fetch(`${API_BASE_URL}/api/paint/recipes/from-palette`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to generate recipes')
      }

      const data = await response.json()
      const recipes = data.recipes || []
      
      if (recipes.length === 0) {
        alert('No recipes were generated. Make sure you have paints in the selected library.')
        return
      }
      
      // Check if any recipes were successfully generated
      const successfulRecipes = recipes.filter((r: any) => r.recipe !== null)
      if (successfulRecipes.length === 0) {
        alert('Could not generate recipes. Make sure you have paints with approximate colors in the selected library.')
        return
      }
      
      setRecipes(recipes)
      
      // Show info about uncalibrated recipes
      const uncalibratedCount = successfulRecipes.filter((r: any) => r.recipe?.uncalibrated).length
      if (uncalibratedCount > 0) {
        console.log(`${uncalibratedCount} recipe(s) use estimated colors (paints not calibrated)`)
      }
    } catch (error) {
      console.error('Error generating recipes:', error)
      alert('Failed to generate recipes. Check the console for details.')
    } finally {
      setLoadingRecipes(false)
    }
  }

  // Handle loading Matisse paint library and generating recipes
  const handleLoadMatisseAndGenerate = async () => {
    if (!sessionData) return
    if (!confirm('This will add Derivan Matisse paints to your library (if not already present) and generate recipes. Continue?')) return

    const matissePaints = [
      { name: 'Titanium White', hex_approx: '#F5F5F5', notes: 'Series 1' },
      { name: 'Red Oxide', hex_approx: '#A0522D', notes: 'Series 1' },
      { name: 'Phthalo Blue', hex_approx: '#003D82', notes: 'Series 2' },
      { name: 'Carbon Black', hex_approx: '#1A1A1A', notes: 'Series 1' },
      { name: 'Yellow Oxide', hex_approx: '#DAA520', notes: 'Series 1' },
      { name: 'Australian Olive Green', hex_approx: '#6B8E23', notes: 'Series 2' },
    ]

    try {
      setLoadingRecipes(true)
      
      let added = 0
      let skipped = 0
      
      // Add paints to library
      for (const paint of matissePaints) {
        const formData = new FormData()
        formData.append('name', paint.name)
        formData.append('hex_approx', paint.hex_approx)
        formData.append('notes', paint.notes)
        formData.append('group', selectedLibraryGroup)

        try {
          const response = await fetch(`${API_BASE_URL}/api/paint/library`, {
            method: 'POST',
            body: formData,
          })
          
          if (response.ok) {
            added++
          } else if (response.status === 400) {
            // Paint already exists - this is fine, skip it silently
            skipped++
            // Don't log this - it's expected behavior
          } else {
            // Other error - only log if it's not a 400
            const errorText = await response.text().catch(() => '')
            console.error(`Failed to add ${paint.name}:`, response.status, errorText)
          }
        } catch (error) {
          // Network error - only log actual network failures
          if (error instanceof TypeError && error.message.includes('fetch')) {
            console.error(`Network error adding ${paint.name}:`, error)
          } else {
            // Might be a 400 response that failed to parse, which is fine
            skipped++
          }
        }
      }
      
      // Log summary (only if something was added)
      if (added > 0) {
        console.log(`Matisse paints: ${added} added, ${skipped} already existed`)
      } else if (skipped === matissePaints.length) {
        console.log(`All Matisse paints already exist in library group "${selectedLibraryGroup}"`)
      }
      
      // Wait a bit for paints to be saved (if any were added)
      if (added > 0) {
        await new Promise(resolve => setTimeout(resolve, 500))
      }
      
      // Now generate recipes using the selected library group
      const paletteForApi = sessionData.palette.map((color) => ({
        index: color.index,
        rgb: hexToRgb(color.hex),
      }))

      const recipeFormData = new FormData()
      recipeFormData.append('palette', JSON.stringify(paletteForApi))
      recipeFormData.append('library_group', selectedLibraryGroup)

      const recipeResponse = await fetch(`${API_BASE_URL}/api/paint/recipes/from-palette`, {
        method: 'POST',
        body: recipeFormData,
      })

      if (!recipeResponse.ok) {
        const errorText = await recipeResponse.text().catch(() => '')
        throw new Error(`Failed to generate recipes: ${recipeResponse.status} ${errorText}`)
      }

      const recipeData = await recipeResponse.json()
      const recipes = recipeData.recipes || []
      
      if (recipes.length === 0) {
        alert('No recipes were generated. Make sure you have paints in the selected library group.')
        return
      }
      
      // Check if any recipes were successfully generated
      const successfulRecipes = recipes.filter((r: any) => r.recipe !== null)
      if (successfulRecipes.length === 0) {
        alert(`Could not generate recipes. The selected library group "${selectedLibraryGroup}" may not have paints with approximate colors set.`)
        return
      }
      
      setRecipes(recipes)
      
      // Show info about uncalibrated recipes
      const uncalibratedCount = successfulRecipes.filter((r: any) => r.recipe?.uncalibrated).length
      const message = added > 0 
        ? `Matisse paints added! ${successfulRecipes.length} recipe(s) generated.`
        : `Using existing Matisse paints. ${successfulRecipes.length} recipe(s) generated.`
      
      if (uncalibratedCount > 0) {
        alert(`${message} ${uncalibratedCount} recipe(s) use estimated colors (paints not calibrated). For accurate recipes, calibrate these paints in the Paint Library.`)
      } else {
        alert(`${message} Note: For accurate recipes, you should calibrate these paints in the Paint Library.`)
      }
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to add Matisse paints or generate recipes. Make sure paints are calibrated.')
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
                  <option value="lightest">Lightest Colours First</option>
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

              <div className="col-span-2">
                <label className="block mb-2">
                  Color Vibrancy Boost: {(saturationBoost * 100).toFixed(0)}%
                  <span className="text-xs text-gray-400 ml-2">
                    (100% = no change, 150% = more vibrant, 500% = maximum vibrancy)
                  </span>
                </label>
                <input
                  type="range"
                  min="0.5"
                  max="5.0"
                  step="0.05"
                  value={saturationBoost}
                  onChange={(e) => setSaturationBoost(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>50% (Less Vibrant)</span>
                  <span>100% (Normal)</span>
                  <span>500% (Maximum)</span>
                </div>
              </div>

              <div className="col-span-2">
                <label className="block mb-2">
                  Detail Level: {(detailLevel * 100).toFixed(0)}%
                  <span className="text-xs text-gray-400 ml-2">
                    (Higher = more detail preserved, Lower = cleaner/simpler)
                  </span>
                </label>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.01"
                  value={detailLevel}
                  onChange={(e) => setDetailLevel(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-green-600"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>0% (Simple)</span>
                  <span>50% (Balanced)</span>
                  <span>100% (Maximum Detail)</span>
                </div>
              </div>
            </div>

            <button
              onClick={handleGenerate}
              disabled={!image || processing}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {processing && (
                <svg
                  className="animate-spin h-5 w-5 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
              )}
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

            {/* Step 1: Preview Image */}
            <div>
              <h2 className="text-2xl font-bold mb-4">Quantized Preview</h2>
              <img
                src={`${API_BASE_URL}${sessionData.quantized_preview_url}`}
                alt="Quantized"
                className="max-w-full rounded"
              />
            </div>

            {/* Step 2: Paint Group Selection */}
            <div className="p-6 bg-gray-800 rounded">
              <h2 className="text-2xl font-bold mb-4">Select Paint Library</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-semibold mb-2">Paint Library Group:</label>
                  <select
                    value={selectedLibraryGroup}
                    onChange={(e) => setSelectedLibraryGroup(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600"
                  >
                    {libraryGroups.map((group) => (
                      <option key={group.group} value={group.group}>
                        {group.name} ({group.paint_count} paints, {group.calibrated_count} calibrated)
                      </option>
                    ))}
                  </select>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => router.push('/paints')}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm"
                  >
                    Manage Paint Libraries
                  </button>
                  <button
                    onClick={() => {
                      // Navigate to paint library with the selected group
                      router.push(`/paints?group=${selectedLibraryGroup}`)
                    }}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                  >
                    Calibrate Paints in {libraryGroups.find(g => g.group === selectedLibraryGroup)?.name || 'Library'}
                  </button>
                </div>
                {libraryGroupsLoaded && libraryGroups.find(g => g.group === selectedLibraryGroup)?.calibrated_count === 0 && (
                  <p className="text-yellow-400 text-sm">
                    ⚠️ No calibrated paints in this library. Recipes will use estimated colors. 
                    <button 
                      onClick={() => router.push(`/paints?group=${selectedLibraryGroup}`)}
                      className="underline ml-1"
                    >
                      Calibrate paints for accurate recipes.
                    </button>
                  </p>
                )}
                {loadingRecipes && (
                  <p className="text-gray-400 text-sm">Generating recipes...</p>
                )}
              </div>
            </div>

            {/* Step 3: Palette with Recipes */}
            <div>
              <h2 className="text-2xl font-bold mb-4">Palette & Recipes</h2>
              <div className="grid grid-cols-8 gap-2 mb-6">
                {sessionData.palette.map((color) => (
                  <div key={color.index} className="text-center">
                    <div
                      className="w-16 h-16 rounded border border-gray-600 relative flex items-center justify-center cursor-pointer hover:opacity-90 transition-opacity"
                      style={{ backgroundColor: color.hex }}
                      onClick={() => setSelectedColor(color)}
                    >
                      <span
                        className="text-white font-bold text-lg drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]"
                        style={{
                          textShadow: '0 1px 2px rgba(0,0,0,0.8), 0 0 4px rgba(0,0,0,0.5)',
                        }}
                      >
                        {color.index}
                      </span>
                    </div>
                    <div className="text-xs mt-1">{color.coverage.toFixed(1)}%</div>
                  </div>
                ))}
              </div>

              {/* Recipes Display */}
              {recipes.length > 0 ? (
                <div className="space-y-3 mt-6">
                  <h3 className="text-xl font-bold mb-3">Mixing Recipes</h3>
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
                            <div className="text-xs text-gray-400 mt-1 flex items-center gap-2 flex-wrap">
                              {recipe.uncalibrated && (
                                <span className="px-2 py-0.5 rounded text-xs bg-yellow-600/30 text-yellow-300 border border-yellow-500/50">
                                  ⚠️ Estimated (not calibrated)
                                </span>
                              )}
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
                          {!recipe && (
                            <div className="text-xs text-red-400 mt-1">
                              {recipeData.error || 'No recipe available'}
                            </div>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              ) : (
                <div className="p-4 bg-gray-700 rounded text-gray-400 mt-6">
                  {loadingRecipes ? (
                    'Generating recipes...'
                  ) : (
                    'Recipes will be generated automatically when you select a paint library group above.'
                  )}
                </div>
              )}
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
                <div className="flex gap-2">
                  <button
                    onClick={handleGenerateRecipes}
                    disabled={loadingRecipes || !sessionData}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50"
                  >
                    {loadingRecipes ? 'Generating...' : 'Generate Recipes'}
                  </button>
                </div>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-semibold mb-2">Select Paint Library:</label>
                <select
                  value={selectedLibraryGroup}
                  onChange={(e) => setSelectedLibraryGroup(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 mb-2"
                >
                  {libraryGroups.map((group) => (
                    <option key={group.group} value={group.group}>
                      {group.name} ({group.paint_count} paints, {group.calibrated_count} calibrated)
                    </option>
                  ))}
                </select>
                <p className="text-gray-400 text-xs">
                  Recipes will be generated using paints from the selected library. 
                  {libraryGroups.find(g => g.group === selectedLibraryGroup)?.calibrated_count === 0 && 
                    ' No calibrated paints in this library - recipes will use estimated colors.'}
                </p>
              </div>
              <p className="text-gray-400 mb-4 text-sm">
                Generate mixing recipes for each palette color using your calibrated paints.
                Make sure you have calibrated at least one paint in the Paint Library, or use the Matisse Paint Library option below.
              </p>
              
              <div className="mb-4 p-4 bg-gray-700 rounded">
                <h3 className="font-bold mb-2">Quick Setup: Derivan Matisse Paint Library</h3>
                <p className="text-sm text-gray-300 mb-3">
                  Use our predefined Matisse paint library (Titanium White, Red Oxide, Phthalo Blue, Carbon Black, Yellow Oxide, Australian Olive Green).
                  These will be added to your paint library and recipes will be generated automatically.
                </p>
                <p className="text-xs text-gray-400 mb-3">
                  Note: For accurate recipes, calibrate these paints in the Paint Library. Uncalibrated paints will use estimated colors.
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={handleLoadMatisseAndGenerate}
                    disabled={loadingRecipes || !sessionData}
                    className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm disabled:opacity-50"
                  >
                    {loadingRecipes ? 'Adding Paints & Generating...' : 'Use Matisse Paint Library & Generate Recipes'}
                  </button>
                  <button
                    onClick={() => router.push('/paints')}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm"
                  >
                    Manage Paints
                  </button>
                </div>
              </div>

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
                            <div className="text-xs text-gray-400 mt-1 flex items-center gap-2 flex-wrap">
                              {recipe.uncalibrated && (
                                <span className="px-2 py-0.5 rounded text-xs bg-yellow-600/30 text-yellow-300 border border-yellow-500/50">
                                  ⚠️ Estimated (not calibrated)
                                </span>
                              )}
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

        {/* Color Modal - only render after mount to avoid hydration issues */}
        {mounted && selectedColor && (
          <div
            className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
            onClick={() => setSelectedColor(null)}
          >
            <div
              className="bg-gray-800 rounded-lg p-8 max-w-md w-full mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold">Palette Color {selectedColor.index}</h3>
                <button
                  onClick={() => setSelectedColor(null)}
                  className="text-gray-400 hover:text-white text-2xl font-bold w-8 h-8 flex items-center justify-center rounded hover:bg-gray-700"
                >
                  ×
                </button>
              </div>

              {/* Large color swatch */}
              <div
                className="w-full aspect-square rounded-lg border-4 border-gray-600 mb-6 shadow-2xl"
                style={{ backgroundColor: selectedColor.hex }}
              />

              {/* Color information */}
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                  <span className="text-gray-300 font-semibold">Hex:</span>
                  <span className="text-white font-mono">{selectedColor.hex.toUpperCase()}</span>
                </div>
                {hexToRgbObject(selectedColor.hex) && (
                  <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                    <span className="text-gray-300 font-semibold">RGB:</span>
                    <span className="text-white font-mono">
                      R: {hexToRgbObject(selectedColor.hex)!.r} | G: {hexToRgbObject(selectedColor.hex)!.g} | B: {hexToRgbObject(selectedColor.hex)!.b}
                    </span>
                  </div>
                )}
                <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                  <span className="text-gray-300 font-semibold">Coverage:</span>
                  <span className="text-white">{selectedColor.coverage.toFixed(1)}%</span>
                </div>
              </div>

              <div className="mt-6 text-sm text-gray-400 text-center">
                Click outside or press ESC to close
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}


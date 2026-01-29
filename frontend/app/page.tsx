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
  is_gradient?: boolean
  is_glaze?: boolean
  gradient_region_id?: string
  gradient_step_index?: number
  hex?: string
  rgb?: number[]
  is_finished?: boolean
  source_palette_indices?: number[]
}

interface GradientRegion {
  id: string
  bounding_box: [number, number, number, number]
  steps_n: number
  direction: string
  transition_mode: string
  transition_width_px: number
  stops: Array<{
    index: number
    hex_color: string
    rgb: number[]
  }>
}

interface SessionResponse {
  session_id: string
  width: number
  height: number
  palette: PaletteColor[]
  order: number[]
  quantized_preview_url: string
  layers: Layer[]
  gradient_regions?: GradientRegion[]
}

export default function Home() {
  const [image, setImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  
  // Always use default values for initial state to prevent hydration mismatches
  // Load from localStorage in useEffect after mount
  const [nColors, setNColors] = useState(16)
  const [overpaintMm, setOverpaintMm] = useState(5)
  const [orderMode, setOrderMode] = useState<'largest' | 'smallest' | 'manual' | 'lightest'>('largest')
  const [maxSide, setMaxSide] = useState(1920)
  const [saturationBoost, setSaturationBoost] = useState(1.0)
  const [detailLevel, setDetailLevel] = useState(0.5)
  const [enableGradients, setEnableGradients] = useState(true)
  const [gradientStepsN, setGradientStepsN] = useState(9)
  const [gradientTransitionMode, setGradientTransitionMode] = useState<'off' | 'dither' | 'feather-preview'>('dither')
  const [gradientTransitionWidth, setGradientTransitionWidth] = useState(25)
  const [enableGlaze, setEnableGlaze] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [sessionData, setSessionData] = useState<SessionResponse | null>(null)
  const [manualOrder, setManualOrder] = useState<number[]>([])
  const [recipes, setRecipes] = useState<any[]>([])
  const [loadingRecipes, setLoadingRecipes] = useState(false)
  const [selectedColor, setSelectedColor] = useState<PaletteColor | null>(null)
  const [selectedLayerColor, setSelectedLayerColor] = useState<{
    hex: string
    paletteIndex?: number
    coverage?: number
    isGradient: boolean
    gradientStepIndex?: number
    layerIndex: number
  } | null>(null)
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
        
        // Compress image before saving to localStorage to avoid quota errors
        const img = new Image()
        img.onload = () => {
          // Create canvas to compress image
          const canvas = document.createElement('canvas')
          const maxWidth = 800 // Max width for compressed preview
          const maxHeight = 600 // Max height for compressed preview
          
          let width = img.width
          let height = img.height
          
          // Calculate new dimensions while maintaining aspect ratio
          if (width > maxWidth || height > maxHeight) {
            const ratio = Math.min(maxWidth / width, maxHeight / height)
            width = width * ratio
            height = height * ratio
          }
          
          canvas.width = width
          canvas.height = height
          const ctx = canvas.getContext('2d')
          
          if (ctx) {
            ctx.drawImage(img, 0, 0, width, height)
            // Convert to compressed JPEG (quality 0.7)
            const compressedDataUrl = canvas.toDataURL('image/jpeg', 0.7)
            
            // Try to save compressed preview to localStorage
            try {
              localStorage.setItem('current_image_preview', compressedDataUrl)
              localStorage.setItem('current_image_name', file.name)
            } catch (err) {
              // If still too large, just don't save it
              console.warn('Image too large for localStorage, preview will not persist across navigation')
              // Remove any existing preview to free up space
              try {
                localStorage.removeItem('current_image_preview')
                localStorage.removeItem('current_image_name')
              } catch (removeErr) {
                // Ignore removal errors
              }
            }
          }
        }
        img.src = previewData
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
        enableGradients,
        gradientStepsN,
        gradientTransitionMode,
        gradientTransitionWidth,
        enableGlaze,
      }
      localStorage.setItem('layerpainter_settings', JSON.stringify(settings))
    } catch (e) {
      console.error('Failed to save settings to localStorage:', e)
    }
  }, [nColors, overpaintMm, orderMode, maxSide, saturationBoost, detailLevel, enableGradients, gradientStepsN, gradientTransitionMode, gradientTransitionWidth, enableGlaze])

  // Set mounted flag and load settings from localStorage after component mounts (client-side only)
  useEffect(() => {
    setMounted(true)
    
    // Load settings from localStorage after mount to prevent hydration mismatches
    if (typeof window !== 'undefined') {
      try {
        const saved = localStorage.getItem('layerpainter_settings')
        if (saved) {
          const parsed = JSON.parse(saved)
          if (parsed.nColors !== undefined) setNColors(parsed.nColors)
          if (parsed.overpaintMm !== undefined) setOverpaintMm(parsed.overpaintMm)
          if (parsed.orderMode !== undefined) setOrderMode(parsed.orderMode)
          if (parsed.maxSide !== undefined) setMaxSide(parsed.maxSide)
          if (parsed.saturationBoost !== undefined) setSaturationBoost(parsed.saturationBoost)
          if (parsed.detailLevel !== undefined) setDetailLevel(parsed.detailLevel)
          if (parsed.enableGradients !== undefined) setEnableGradients(parsed.enableGradients)
          if (parsed.gradientStepsN !== undefined) setGradientStepsN(parsed.gradientStepsN)
          if (parsed.gradientTransitionMode !== undefined) setGradientTransitionMode(parsed.gradientTransitionMode)
          if (parsed.gradientTransitionWidth !== undefined) setGradientTransitionWidth(parsed.gradientTransitionWidth)
          if (parsed.enableGlaze !== undefined) setEnableGlaze(parsed.enableGlaze)
        }
      } catch (e) {
        console.error('Failed to load settings from localStorage:', e)
      }
    }
  }, [])

  // Handle ESC key to close modals
  useEffect(() => {
    if (!mounted || typeof window === 'undefined') return
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (selectedColor) setSelectedColor(null)
        if (selectedLayerColor) setSelectedLayerColor(null)
      }
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [selectedColor, selectedLayerColor, mounted])

  // Restore image preview and session data from localStorage on mount (client-side only)
  useEffect(() => {
    if (typeof window === 'undefined') return
    
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
    formData.append('enable_gradients', enableGradients.toString())
    formData.append('gradient_steps_n', gradientStepsN.toString())
    formData.append('gradient_transition_mode', gradientTransitionMode)
    formData.append('gradient_transition_width', gradientTransitionWidth.toString())
    formData.append('enable_glaze', enableGlaze.toString())

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
    
    // Handle ChatGPT-generated recipes (structured format)
    if (recipeData.type === 'chatgpt' || recipe.type === 'chatgpt') {
      // New structured format with ingredients
      if (recipe.ingredients && Array.isArray(recipe.ingredients) && recipe.ingredients.length > 0) {
        const ingredientParts = recipe.ingredients.map((ing: any) => {
          if (!ing || !ing.paint_name) {
            console.warn('Invalid ingredient in recipe:', ing)
            return null
          }
          const percentage = ing.percentage !== undefined ? ing.percentage : 0
          if (ing.grams !== undefined) {
            return `${ing.paint_name} ${percentage.toFixed(2)}% (${ing.grams.toFixed(2)}g)`
          }
          return `${ing.paint_name} ${percentage.toFixed(2)}%`
        }).filter((part: string | null) => part !== null)
        if (ingredientParts.length > 0) {
          return ingredientParts.join(' + ')
        }
      }
      // Log if we have recipe but no valid ingredients
      if (recipe.ingredients) {
        console.warn('Recipe has ingredients array but no valid ingredients:', recipe)
      }
      // Fallback to old instructions format for backward compatibility
      return recipe.instructions || 'Recipe instructions from ChatGPT'
    }
    
    // Legacy recipe formats (for backwards compatibility)
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

  // Recipes are now only generated when user clicks the "Generate Recipes" button
  // (Removed auto-generation on page load)

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

  const handleGenerateRecipes = async (forceRegenerate: boolean = false) => {
    if (!sessionData) return

    setLoadingRecipes(true)
    try {
      // Send palette with hex values (backend expects hex for ChatGPT)
      const paletteForApi = sessionData.palette.map((color) => ({
        index: color.index,
        hex: color.hex,  // Backend expects hex for ChatGPT API
      }))

      const formData = new FormData()
      formData.append('palette', JSON.stringify(paletteForApi))
      formData.append('library_group', selectedLibraryGroup)
      if (forceRegenerate) {
        formData.append('force_regenerate', 'true')
      }

      const response = await fetch(`${API_BASE_URL}/api/paint/recipes/from-palette`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to generate recipes')
      }

      const data = await response.json()
      const recipes = data.recipes || []
      
      // Debug logging
      console.log('Received recipes from API:', recipes)
      if (recipes.length > 0) {
        console.log('First recipe structure:', recipes[0])
        if (recipes[0].recipe) {
          console.log('First recipe data:', recipes[0].recipe)
          console.log('First recipe ingredients:', recipes[0].recipe.ingredients)
        }
      }
      
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

              {/* Gradient-Aware Quantization Settings */}
              <div className="col-span-2 border-t border-gray-700 pt-4 mt-4">
                <h3 className="text-lg font-semibold mb-4">Gradient-Aware Quantization</h3>
                <p className="text-sm text-gray-400 mb-4">
                  Automatically detects smooth gradients (sky, water) and generates multi-step ramps instead of flat color bands.
                </p>
                
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <input
                      type="checkbox"
                      id="enableGradients"
                      checked={enableGradients}
                      onChange={(e) => setEnableGradients(e.target.checked)}
                      className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="enableGradients" className="text-sm">
                      Enable gradient detection and ramp generation
                    </label>
                  </div>

                  {enableGradients && (
                    <>
                      <div>
                        <label className="block mb-2">
                          Gradient Steps: {gradientStepsN}
                          <span className="text-xs text-gray-400 ml-2">
                            (Number of steps in gradient ramps, 5-15)
                          </span>
                        </label>
                        <input
                          type="range"
                          min="5"
                          max="15"
                          step="1"
                          value={gradientStepsN}
                          onChange={(e) => setGradientStepsN(parseInt(e.target.value))}
                          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
                        />
                        <div className="flex justify-between text-xs text-gray-400 mt-1">
                          <span>5 (Fewer steps)</span>
                          <span>9 (Default)</span>
                          <span>15 (More steps)</span>
                        </div>
                      </div>

                      <div>
                        <label className="block mb-2">Transition Mode</label>
                        <select
                          value={gradientTransitionMode}
                          onChange={(e) => setGradientTransitionMode(e.target.value as any)}
                          className="w-full px-3 py-2 bg-gray-800 rounded text-white"
                        >
                          <option value="off">Off (Hard edges)</option>
                          <option value="dither">Dither (Smooth transitions)</option>
                          <option value="feather-preview">Feather Preview (Preview only)</option>
                        </select>
                      </div>

                      {gradientTransitionMode !== 'off' && (
                        <div>
                          <label className="block mb-2">
                            Transition Width: {gradientTransitionWidth}px
                            <span className="text-xs text-gray-400 ml-2">
                              (Width of transition bands between steps, 5-60px)
                            </span>
                          </label>
                          <input
                            type="range"
                            min="5"
                            max="60"
                            step="5"
                            value={gradientTransitionWidth}
                            onChange={(e) => setGradientTransitionWidth(parseInt(e.target.value))}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
                          />
                          <div className="flex justify-between text-xs text-gray-400 mt-1">
                            <span>5px (Narrow)</span>
                            <span>25px (Default)</span>
                            <span>60px (Wide)</span>
                          </div>
                        </div>
                      )}

                      <div className="flex items-center gap-3">
                        <input
                          type="checkbox"
                          id="enableGlaze"
                          checked={enableGlaze}
                          onChange={(e) => setEnableGlaze(e.target.checked)}
                          className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                        />
                        <label htmlFor="enableGlaze" className="text-sm">
                          Glaze pass (add a unifying thin layer per gradient region — paint last)
                        </label>
                      </div>
                    </>
                  )}
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
                  {mounted && libraryGroupsLoaded && libraryGroups.length > 0 ? (
                    <>
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
                      <div className="flex gap-3 mt-3">
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
                      {libraryGroups.find(g => g.group === selectedLibraryGroup)?.calibrated_count === 0 && (
                        <p className="text-yellow-400 text-sm mt-2">
                          ⚠️ No calibrated paints in this library. Recipes will use estimated colors. 
                          <button 
                            onClick={() => router.push(`/paints?group=${selectedLibraryGroup}`)}
                            className="underline ml-1"
                          >
                            Calibrate paints for accurate recipes.
                          </button>
                        </p>
                      )}
                    </>
                  ) : (
                    <div className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 text-gray-400">
                      Loading library groups...
                    </div>
                  )}
                </div>
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
                    const errorInfo = (recipe && recipeData.type !== 'chatgpt' && recipe.error !== undefined) ? getErrorLevel(recipe.error) : null

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
                          {recipe && (recipeData.type === 'chatgpt' || recipe.type === 'chatgpt') && recipe.ingredients && (
                            <div className="text-xs text-gray-400 mt-2 space-y-1">
                              {recipe.mixing_strategy && <div><strong>Strategy:</strong> {recipe.mixing_strategy}</div>}
                              {recipe.expected_result && <div><strong>Expected:</strong> {recipe.expected_result}</div>}
                              {recipe.adjustment_ladder && <div><strong>Adjustments:</strong> {recipe.adjustment_ladder}</div>}
                              {recipe.tips && <div><strong>Tips:</strong> {recipe.tips}</div>}
                            </div>
                          )}
                          {recipe && recipeData.type !== 'chatgpt' && recipe.type !== 'chatgpt' && (
                            <div className="text-xs text-gray-400 mt-1 flex items-center gap-2 flex-wrap">
                              {recipe.uncalibrated && (
                                <span className="px-2 py-0.5 rounded text-xs bg-yellow-600/30 text-yellow-300 border border-yellow-500/50">
                                  ⚠️ Estimated (not calibrated)
                                </span>
                              )}
                              {recipe.error !== undefined && (
                                <span>Error: {recipe.error.toFixed(2)} ΔE</span>
                              )}
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
                {sessionData.layers
                  .filter(l => !l.is_finished) // Exclude finished layer from main list
                  .map((layer) => {
                    const isGradient = layer.is_gradient || false
                    let colorHex = '#000000'
                    let displayText = ''
                    
                    if (isGradient) {
                      // Gradient layer - use the gradient step's own color
                      colorHex = layer.hex || '#808080'
                      const isGlaze = layer.is_glaze || false
                      if (isGlaze) {
                        displayText = 'Glaze (paint last, very thin)'
                      } else {
                        const stepNum = (layer.gradient_step_index ?? 0) >= 0 && (layer.gradient_step_index ?? 0) < 100
                          ? (layer.gradient_step_index ?? 0) + 1
                          : 0
                        let paletteInfo = ''
                        if (layer.source_palette_indices && layer.source_palette_indices.length > 0) {
                          if (layer.source_palette_indices.length === 1) {
                            paletteInfo = ` (replaces Palette ${layer.source_palette_indices[0]})`
                          } else {
                            paletteInfo = ` (replaces Palettes ${layer.source_palette_indices.join(', ')})`
                          }
                        } else if (layer.palette_index !== undefined && layer.palette_index >= 0) {
                          paletteInfo = ` (replaces Palette ${layer.palette_index})`
                        }
                        displayText = `Gradient Step ${stepNum}${paletteInfo}`
                      }
                    } else {
                      // Regular quantized layer
                      const color = sessionData.palette.find((p) => p.index === layer.palette_index)
                      if (!color) return null
                      colorHex = color.hex
                      displayText = `Palette ${layer.palette_index} - ${color.coverage.toFixed(1)}% coverage`
                    }

                    return (
                      <div
                        key={layer.layer_index}
                        className={`flex items-center gap-4 p-4 rounded ${isGradient ? 'bg-purple-900/30 border border-purple-700' : 'bg-gray-800'}`}
                      >
                        <div className="text-lg font-mono">{layer.layer_index + 1}</div>
                        <div
                          className="w-16 h-16 rounded border border-gray-600 cursor-pointer hover:opacity-90 transition-opacity hover:ring-2 hover:ring-white"
                          style={{ backgroundColor: colorHex }}
                          onClick={() => {
                            if (isGradient) {
                              setSelectedLayerColor({
                                hex: colorHex,
                                paletteIndex: layer.palette_index >= 0 ? layer.palette_index : undefined,
                                isGradient: true,
                                gradientStepIndex: layer.gradient_step_index,
                                layerIndex: layer.layer_index
                              })
                            } else {
                              const color = sessionData.palette.find((p) => p.index === layer.palette_index)
                              if (color) {
                                setSelectedLayerColor({
                                  hex: colorHex,
                                  paletteIndex: layer.palette_index,
                                  coverage: color.coverage,
                                  isGradient: false,
                                  layerIndex: layer.layer_index
                                })
                              }
                            }
                          }}
                          title="Click to view color info"
                        />
                        <img
                          src={`${API_BASE_URL}${layer.mask_url}`}
                          alt={`Layer ${layer.layer_index + 1}`}
                          className="w-16 h-16 object-contain bg-gray-700 rounded"
                        />
                        {orderMode === 'manual' && !isGradient && (
                          <div className="flex gap-2">
                            <button
                              onClick={() => {
                                const orderIdx = manualOrder.indexOf(layer.palette_index)
                                if (orderIdx >= 0) moveLayer(orderIdx, 'up')
                              }}
                              className="px-3 py-1 bg-gray-700 rounded disabled:opacity-50"
                            >
                              ↑
                            </button>
                            <button
                              onClick={() => {
                                const orderIdx = manualOrder.indexOf(layer.palette_index)
                                if (orderIdx >= 0) moveLayer(orderIdx, 'down')
                              }}
                              className="px-3 py-1 bg-gray-700 rounded disabled:opacity-50"
                            >
                              ↓
                            </button>
                          </div>
                        )}
                        <div className="flex-1 text-sm text-gray-400">
                          {displayText}
                          {isGradient && (
                            <span className="ml-2 text-xs text-purple-400">(Gradient)</span>
                          )}
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
                    onClick={() => handleGenerateRecipes(false)}
                    disabled={loadingRecipes || !sessionData}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50"
                  >
                    {loadingRecipes ? 'Generating...' : 'Generate Recipes'}
                  </button>
                  <button
                    onClick={() => handleGenerateRecipes(true)}
                    disabled={loadingRecipes || !sessionData}
                    className="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded disabled:opacity-50"
                    title="Force regenerate recipes from ChatGPT (ignores cache)"
                  >
                    {loadingRecipes ? 'Regenerating...' : 'Force Regenerate'}
                  </button>
                </div>
              </div>
              <div className="mb-4">
                <label className="block text-sm font-semibold mb-2">Select Paint Library:</label>
                {mounted && libraryGroupsLoaded && libraryGroups.length > 0 ? (
                  <>
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
                  </>
                ) : (
                  <div className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 mb-2 text-gray-400">
                    Loading library groups...
                  </div>
                )}
              </div>
              <p className="text-gray-400 mb-4 text-sm">
                Generate mixing recipes for each palette color using your calibrated paints.
                Make sure you have calibrated at least one paint in the Paint Library for accurate recipes.
              </p>
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

        {/* Layer Color Info Modal */}
        {mounted && selectedLayerColor && (
          <div
            className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50"
            onClick={() => setSelectedLayerColor(null)}
          >
            <div
              className="bg-gray-800 rounded-lg p-8 max-w-md w-full mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-6">
                <h3 className="text-2xl font-bold">
                  {selectedLayerColor.isGradient 
                    ? `Gradient Step ${(selectedLayerColor.gradientStepIndex ?? 0) + 1}`
                    : `Palette Color ${selectedLayerColor.paletteIndex}`}
                </h3>
                <button
                  onClick={() => setSelectedLayerColor(null)}
                  className="text-gray-400 hover:text-white text-2xl font-bold w-8 h-8 flex items-center justify-center rounded hover:bg-gray-700"
                >
                  ×
                </button>
              </div>

              {/* Color swatch (50% size) */}
              <div className="flex justify-center mb-6">
                <div
                  className="w-1/2 aspect-square rounded-lg border-4 border-gray-600 shadow-2xl"
                  style={{ backgroundColor: selectedLayerColor.hex }}
                />
              </div>

              {/* Color information */}
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                  <span className="text-gray-300 font-semibold">Hex:</span>
                  <span className="text-white font-mono">{selectedLayerColor.hex.toUpperCase()}</span>
                </div>
                {hexToRgbObject(selectedLayerColor.hex) && (
                  <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                    <span className="text-gray-300 font-semibold">RGB:</span>
                    <span className="text-white font-mono">
                      R: {hexToRgbObject(selectedLayerColor.hex)!.r} | G: {hexToRgbObject(selectedLayerColor.hex)!.g} | B: {hexToRgbObject(selectedLayerColor.hex)!.b}
                    </span>
                  </div>
                )}
                {selectedLayerColor.paletteIndex !== undefined && (
                  <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                    <span className="text-gray-300 font-semibold">Palette Number:</span>
                    <span className="text-white">{selectedLayerColor.paletteIndex}</span>
                  </div>
                )}
                {selectedLayerColor.coverage !== undefined && (
                  <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                    <span className="text-gray-300 font-semibold">Coverage:</span>
                    <span className="text-white">{selectedLayerColor.coverage.toFixed(1)}%</span>
                  </div>
                )}
                {selectedLayerColor.isGradient && (
                  <div className="flex items-center justify-between p-3 bg-purple-900/30 rounded border border-purple-700">
                    <span className="text-purple-300 font-semibold">Type:</span>
                    <span className="text-purple-300">Gradient Step</span>
                  </div>
                )}
                {selectedLayerColor.paletteIndex !== undefined && (
                  <div className="flex items-center justify-between p-3 bg-gray-700 rounded">
                    <span className="text-gray-300 font-semibold">Palette Index:</span>
                    <span className="text-white">{selectedLayerColor.paletteIndex}</span>
                  </div>
                )}
              </div>

              {/* Recipe information */}
              {selectedLayerColor.paletteIndex !== undefined && recipes.length > 0 && (() => {
                const recipeData = recipes.find((r: any) => r.palette_index === selectedLayerColor.paletteIndex)
                if (!recipeData) return null
                
                const recipe = recipeData.recipe
                const errorInfo = recipe && recipeData.type !== 'chatgpt' && recipe.error !== undefined 
                  ? getErrorLevel(recipe.error) 
                  : null
                
                return (
                  <div className="mt-6 pt-6 border-t border-gray-700">
                    <h4 className="text-lg font-bold mb-3">Mixing Recipe</h4>
                    <div className="text-sm text-gray-300 mb-3">
                      {formatRecipe(recipeData)}
                    </div>
                    
                    {/* Recipe details for ChatGPT recipes */}
                    {recipe && (recipeData.type === 'chatgpt' || recipe.type === 'chatgpt') && recipe.ingredients && (
                      <div className="text-xs text-gray-400 mt-2 space-y-1">
                        {recipe.mixing_strategy && (
                          <div><strong>Strategy:</strong> {recipe.mixing_strategy}</div>
                        )}
                        {recipe.expected_result && (
                          <div><strong>Expected:</strong> {recipe.expected_result}</div>
                        )}
                        {recipe.adjustment_ladder && (
                          <div><strong>Adjustments:</strong> {recipe.adjustment_ladder}</div>
                        )}
                        {recipe.tips && (
                          <div><strong>Tips:</strong> {recipe.tips}</div>
                        )}
                      </div>
                    )}
                    
                    {/* Recipe metadata for non-ChatGPT recipes */}
                    {recipe && recipeData.type !== 'chatgpt' && recipe.type !== 'chatgpt' && (
                      <div className="text-xs text-gray-400 mt-2 flex items-center gap-2 flex-wrap">
                        {recipe.uncalibrated && (
                          <span className="px-2 py-0.5 rounded text-xs bg-yellow-600/30 text-yellow-300 border border-yellow-500/50">
                            ⚠️ Estimated (not calibrated)
                          </span>
                        )}
                        {errorInfo && (
                          <span 
                            className="px-2 py-0.5 rounded text-xs"
                            style={{
                              backgroundColor: errorInfo.color === 'green' ? '#16a34a' : 
                                            errorInfo.color === 'yellow' ? '#ca8a04' : '#dc2626'
                            }}
                          >
                            Error: {recipe.error.toFixed(2)} ΔE - {errorInfo.level}
                          </span>
                        )}
                      </div>
                    )}
                    
                    {!recipe && (
                      <div className="text-xs text-red-400 mt-2">
                        {recipeData.error || 'No recipe available'}
                      </div>
                    )}
                  </div>
                )
              })()}

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


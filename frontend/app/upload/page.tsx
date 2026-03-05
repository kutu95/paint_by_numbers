'use client'

import { useState, useRef, useEffect } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { API_BASE_URL } from '@/lib/config'
import { saveProject, getProjectBySessionId, removeProject, syncProjectsFromServer } from '@/lib/projects'

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
  original_url?: string
  layers: Layer[]
  gradient_regions?: GradientRegion[]
}

interface OptimizedPaletteRecipe {
  ingredients: Array<{
    paint_id: string
    paint_name: string
    percentage: number
  }>
  error?: number | null
  type?: string
  error_text?: string
}

interface OptimizedPaletteColor {
  index: number
  target_hex: string
  coverage: number
  lab: [number, number, number]
  recipe: OptimizedPaletteRecipe
}

interface PaletteOptimizationResult {
  optimal_palette_size: number
  average_delta_e: number
  maximum_delta_e: number
  target_delta_e: number
  max_palette_size: number
  library_group: string
  prefer_simpler: boolean
  downsample: {
    width: number
    height: number
  }
  palette: OptimizedPaletteColor[]
  paint_order: number[]
  met_target: boolean
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
  const [canvasWidthCm, setCanvasWidthCm] = useState(50)
  const [canvasHeightCm, setCanvasHeightCm] = useState(40)
  const [saturationBoost, setSaturationBoost] = useState(1.0)
  const [detailLevel, setDetailLevel] = useState(0.5)
  const [enableGradients, setEnableGradients] = useState(false)
  const [gradientStepsN, setGradientStepsN] = useState(9)
  const [gradientTransitionMode, setGradientTransitionMode] = useState<'off' | 'dither' | 'feather-preview'>('dither')
  const [gradientTransitionWidth, setGradientTransitionWidth] = useState(25)
  const [enableGlaze, setEnableGlaze] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [optimizingPalette, setOptimizingPalette] = useState(false)
  const [targetErrorDeltaE, setTargetErrorDeltaE] = useState(5)
  const [maxPaletteSize, setMaxPaletteSize] = useState(16)
  const [preferSimplerMixes, setPreferSimplerMixes] = useState(false)
  const [paletteOptimization, setPaletteOptimization] = useState<PaletteOptimizationResult | null>(null)
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
  const [projectName, setProjectName] = useState('')
  const [editSessionOriginalUrl, setEditSessionOriginalUrl] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const router = useRouter()
  const searchParams = useSearchParams()
  const isNewProject = searchParams.get('new') === '1'
  const editSessionId = searchParams.get('edit')
  const returnToHome = searchParams.get('returnTo') === 'home'

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
      reader.onload = (event) => {
        const originalDataUrl = event.target?.result as string
        
        // Compress image before saving to localStorage to avoid quota errors,
        // and always rotate portrait images to landscape for preview.
        const img = new Image()
        img.onload = () => {
          const maxWidth = 800 // Max width for compressed preview
          const maxHeight = 600 // Max height for compressed preview
          const isPortrait = img.height > img.width
          
          // Determine canvas size based on orientation, ensuring final preview is landscape
          let canvasWidth: number
          let canvasHeight: number
          if (isPortrait) {
            const ratio = Math.min(maxWidth / img.height, maxHeight / img.width, 1)
            canvasWidth = img.height * ratio
            canvasHeight = img.width * ratio
          } else {
            const ratio = Math.min(maxWidth / img.width, maxHeight / img.height, 1)
            canvasWidth = img.width * ratio
            canvasHeight = img.height * ratio
          }
          
          const canvas = document.createElement('canvas')
          canvas.width = canvasWidth
          canvas.height = canvasHeight
          const ctx = canvas.getContext('2d')
          
          if (ctx) {
            if (isPortrait) {
              // Rotate 90 degrees counter-clockwise around canvas center
              ctx.save()
              ctx.translate(canvasWidth / 2, canvasHeight / 2)
              ctx.rotate(-Math.PI / 2)
              const scale = Math.min(canvasWidth / img.height, canvasHeight / img.width)
              const drawWidth = img.width * scale
              const drawHeight = img.height * scale
              ctx.drawImage(img, -drawWidth / 2, -drawHeight / 2, drawWidth, drawHeight)
              ctx.restore()
            } else {
              ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight)
            }
            
            // Convert to compressed JPEG (quality 0.7)
            const compressedDataUrl = canvas.toDataURL('image/jpeg', 0.7)
            setPreview(compressedDataUrl)
            
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
        img.src = originalDataUrl
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
        canvasWidthCm,
        canvasHeightCm,
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
  }, [nColors, overpaintMm, orderMode, maxSide, canvasWidthCm, canvasHeightCm, saturationBoost, detailLevel, enableGradients, gradientStepsN, gradientTransitionMode, gradientTransitionWidth, enableGlaze])

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
          if (typeof parsed.canvasWidthCm === 'number' && parsed.canvasWidthCm >= 0) setCanvasWidthCm(parsed.canvasWidthCm)
          if (typeof parsed.canvasHeightCm === 'number' && parsed.canvasHeightCm >= 0) setCanvasHeightCm(parsed.canvasHeightCm)
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

  // Restore image preview and session data from localStorage on mount (client-side only).
  // Skip restoration when starting a new project (?new=1) or editing a project (?edit=sessionId).
  useEffect(() => {
    if (typeof window === 'undefined') return
    if (searchParams.get('new') === '1') return
    if (searchParams.get('edit')) return

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
  }, [searchParams])

  // When editing a project (?edit=sessionId), pre-fill form from stored project and session.
  // Also restore sessionData from localStorage so that after generating and switching tabs, returning shows results + original image.
  useEffect(() => {
    if (!editSessionId || typeof window === 'undefined') return
    void (async () => {
      await syncProjectsFromServer()
      const p = getProjectBySessionId(editSessionId)
      if (p) {
        setProjectName(p.name)
        setCanvasWidthCm(p.canvasWidthCm)
        setCanvasHeightCm(p.canvasHeightCm)
        setSaturationBoost(p.saturationBoost)
        setDetailLevel(p.detailLevel)
        setSelectedLibraryGroup(p.libraryGroup)
      }
    })()
    const savedSession = localStorage.getItem(`session_${editSessionId}`)
    if (savedSession) {
      try {
        const parsed = JSON.parse(savedSession) as { original_url?: string }
        setEditSessionOriginalUrl(parsed.original_url ?? null)
      } catch {
        setEditSessionOriginalUrl(null)
      }
    } else {
      setEditSessionOriginalUrl(null)
      fetch(`${API_BASE_URL}/api/sessions/${editSessionId}/info`)
        .then((res) => (res.ok ? res.json() : null))
        .then((info: { original_url?: string; has_stored_image?: boolean } | null) => {
          if (info?.original_url && info?.has_stored_image) setEditSessionOriginalUrl(info.original_url)
        })
        .catch(() => {})
    }
  }, [editSessionId])

  // Restore manual order when sessionData is restored and orderMode is manual
  useEffect(() => {
    if (sessionData && orderMode === 'manual') {
      setManualOrder([...sessionData.order])
    }
  }, [sessionData, orderMode])

  const handleGenerate = async () => {
    const useStoredImage = editSessionId && !image && editSessionOriginalUrl
    if (!image && !useStoredImage) {
      if (preview) {
        alert('Please re-select the image file to generate layers. The preview is shown but the file needs to be selected again.')
      } else if (editSessionId) {
        alert('No image selected and no stored image found for this project. Upload a new image or open the project from the same browser where it was created.')
      }
      return
    }

    setProcessing(true)
    try {
      let response: Response
      if (useStoredImage) {
        const formData = new FormData()
        formData.append('n_colors', nColors.toString())
        formData.append('overpaint_mm', overpaintMm.toString())
        formData.append('order_mode', orderMode)
        formData.append('max_side', maxSide.toString())
        formData.append('canvas_width_cm', canvasWidthCm.toString())
        formData.append('canvas_height_cm', canvasHeightCm.toString())
        formData.append('saturation_boost', saturationBoost.toString())
        formData.append('detail_level', detailLevel.toString())
        response = await fetch(`${API_BASE_URL}/api/sessions/${editSessionId}/reprocess`, {
          method: 'POST',
          body: formData,
        })
      } else {
        const formData = new FormData()
        formData.append('image', image!)
        formData.append('n_colors', nColors.toString())
        formData.append('overpaint_mm', overpaintMm.toString())
        formData.append('order_mode', orderMode)
        formData.append('max_side', maxSide.toString())
        formData.append('canvas_width_cm', canvasWidthCm.toString())
        formData.append('canvas_height_cm', canvasHeightCm.toString())
        formData.append('saturation_boost', saturationBoost.toString())
        formData.append('detail_level', detailLevel.toString())
        formData.append('enable_gradients', enableGradients.toString())
        formData.append('gradient_steps_n', gradientStepsN.toString())
        formData.append('gradient_transition_mode', gradientTransitionMode)
        formData.append('gradient_transition_width', gradientTransitionWidth.toString())
        formData.append('enable_glaze', enableGlaze.toString())
        response = await fetch(`${API_BASE_URL}/api/sessions`, {
          method: 'POST',
          body: formData,
        })
      }

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
      // Persist as a named project when created via New project or when editing
      if (editSessionId) {
        const existingProject = getProjectBySessionId(editSessionId)
        if (!useStoredImage) removeProject(editSessionId)
        saveProject({
          sessionId: data.session_id,
          name: projectName.trim() || existingProject?.name || 'Untitled',
          imageFileName: useStoredImage ? (existingProject?.imageFileName ?? 'image') : (image?.name || 'image'),
          libraryGroup: selectedLibraryGroup,
          canvasWidthCm,
          canvasHeightCm,
          saturationBoost,
          detailLevel,
          createdAt: existingProject?.createdAt ?? Date.now(),
          nColors,
          overpaintMm,
          orderMode,
          maxSide,
        })
        if (returnToHome && typeof window !== 'undefined' && window.top !== window) {
          window.top!.location.href = `/?tab=projection&session=${data.session_id}`
        } else {
          router.push(returnToHome ? `/?tab=projection&session=${data.session_id}` : `/project/${data.session_id}`)
        }
        return
      }
      if (isNewProject && projectName.trim()) {
        saveProject({
          sessionId: data.session_id,
          name: projectName.trim(),
          imageFileName: image?.name || 'image',
          libraryGroup: selectedLibraryGroup,
          canvasWidthCm,
          canvasHeightCm,
          saturationBoost,
          detailLevel,
          createdAt: Date.now(),
          nColors,
          overpaintMm,
          orderMode,
          maxSide,
        })
        if (returnToHome) {
          if (typeof window !== 'undefined' && window.top !== window) {
            window.top!.location.href = `/?tab=projection&session=${data.session_id}`
          } else {
            router.push(`/?tab=projection&session=${data.session_id}`)
          }
          return
        }
      }
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to process image'
      alert(errorMessage)
    } finally {
      setProcessing(false)
    }
  }

  const handleComputeOptimalPalette = async () => {
    const useStoredImage = editSessionId && !image && editSessionOriginalUrl
    if (!image && !useStoredImage) {
      alert('Please select an image first (or open an editable project with a stored image).')
      return
    }

    setOptimizingPalette(true)
    try {
      const formData = new FormData()
      formData.append('target_delta_e', targetErrorDeltaE.toString())
      formData.append('max_palette_size', maxPaletteSize.toString())
      formData.append('library_group', selectedLibraryGroup)
      formData.append('prefer_simpler', preferSimplerMixes ? 'true' : 'false')
      if (image) {
        formData.append('image', image)
      } else if (editSessionId) {
        formData.append('session_id', editSessionId)
      }

      const response = await fetch(`${API_BASE_URL}/api/paint/optimize-palette`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || `HTTP ${response.status}`)
      }
      const data: PaletteOptimizationResult = await response.json()
      setPaletteOptimization(data)
      setNColors(data.optimal_palette_size)
    } catch (error) {
      console.error('Failed to compute optimal palette:', error)
      alert('Failed to compute optimal palette. Check console for details.')
    } finally {
      setOptimizingPalette(false)
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
      localStorage.setItem('current_session_id', sessionData.session_id)
      if (returnToHome && typeof window !== 'undefined' && window.top !== window) {
        window.top!.location.href = `/?tab=projection&session=${sessionData.session_id}`
      } else {
        router.push(returnToHome ? `/?tab=projection&session=${sessionData.session_id}` : `/project/${sessionData.session_id}`)
      }
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
    
    // Structured ingredient recipes (ChatGPT or deterministic)
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

    if (recipe.instructions) {
      // Log if we have recipe but no valid ingredients
      if (recipe.ingredients) {
        console.warn('Recipe has ingredients array but no valid ingredients:', recipe)
      }
      return recipe.instructions
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
    loadLibraryGroups(searchParams.get('edit'))
  }, [searchParams])

  // Recipes are now only generated when user clicks the "Generate Recipes" button
  // (Removed auto-generation on page load)

  const loadLibraryGroups = async (editSessionIdParam?: string | null) => {
    if (typeof window === 'undefined') return
    const url = `${API_BASE_URL}/api/paint/library/groups`
    try {
      let data: any = null
      let lastError: unknown = null
      for (const delayMs of [0, 400]) {
        if (delayMs > 0) {
          await new Promise((resolve) => setTimeout(resolve, delayMs))
        }
        try {
          const response = await fetch(url, { cache: 'no-store' })
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`)
          }
          data = await response.json()
          break
        } catch (error) {
          lastError = error
        }
      }
      if (!data) {
        throw lastError || new Error('No response data')
      }
      const groups = data.groups || []
      setLibraryGroups(groups)
      setLibraryGroupsLoaded(true)
      if (groups.length > 0 && !editSessionIdParam) {
        // When not editing, pick default library; when editing, edit effect already set from project
        const calibratedGroup = groups.find((g: any) => g.calibrated_count > 0)
        setSelectedLibraryGroup(calibratedGroup ? calibratedGroup.group : groups[0].group)
      }
    } catch (error) {
      console.error(`Failed to load library groups from ${url}:`, error)
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
        <div className="flex items-center gap-4 mb-6">
          {editSessionId ? (
            <Link
              href={returnToHome ? `/?tab=file&session=${editSessionId}` : `/project/${editSessionId}`}
              className="inline-flex items-center text-gray-400 hover:text-white"
            >
              ← Back to project
            </Link>
          ) : (
            <Link href="/" className="inline-flex items-center text-gray-400 hover:text-white">
              ← Back to menu
            </Link>
          )}
        </div>
        <h1 className="text-4xl font-bold mb-8">
          {editSessionId ? 'Edit image & settings' : 'LayerPainter'}
        </h1>

        {!sessionData ? (
          <div className="space-y-6">
            {(isNewProject || editSessionId) && (
              <div>
                <label className="block mb-2">Project name</label>
                <input
                  type="text"
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  placeholder="e.g. Sunset landscape"
                  className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-600 text-white placeholder-gray-500"
                />
                <p className="text-xs text-gray-500 mt-1">
                  {editSessionId ? 'Update the project name if you like.' : 'Give this project a title so you can find it in recent projects.'}
                </p>
              </div>
            )}

            <div>
              {editSessionId && editSessionOriginalUrl && (
                <div className="mb-4">
                  <label className="block mb-2">Current image (stored on server)</label>
                  <img
                    src={`${API_BASE_URL}${editSessionOriginalUrl}`}
                    alt="Current project image"
                    className="max-w-md rounded border border-gray-600"
                  />
                  <p className="text-xs text-gray-500 mt-1">Change settings below and click Generate to reprocess, or upload a new image to replace.</p>
                </div>
              )}
              <label className="block mb-2">
                {editSessionId && editSessionOriginalUrl ? 'Replace image (optional)' : 'Upload Image'}
              </label>
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

              <div>
                <label className="block mb-2">Canvas width (cm)</label>
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={canvasWidthCm}
                  onChange={(e) => setCanvasWidthCm(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 bg-gray-800 rounded text-white"
                />
                <p className="text-xs text-gray-500 mt-1">Override default from Settings. Used for recipe weights.</p>
              </div>

              <div>
                <label className="block mb-2">Canvas height (cm)</label>
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={canvasHeightCm}
                  onChange={(e) => setCanvasHeightCm(parseFloat(e.target.value) || 0)}
                  className="w-full px-3 py-2 bg-gray-800 rounded text-white"
                />
                <p className="text-xs text-gray-500 mt-1">Override default from Settings. Used for recipe weights.</p>
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

              <div className="col-span-2">
                <label className="block mb-2">Paint library</label>
                {mounted && libraryGroupsLoaded && libraryGroups.length > 0 ? (
                  <>
                    <select
                      value={selectedLibraryGroup}
                      onChange={(e) => setSelectedLibraryGroup(e.target.value)}
                      className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-600 text-white"
                    >
                      {libraryGroups.map((group) => (
                        <option key={group.group} value={group.group}>
                          {group.name} ({group.paint_count} paints, {group.calibrated_count} calibrated)
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-gray-500 mt-1">
                      Used for recipe weights and mixing recipes. You can calibrate paints in Manage Paint Libraries.
                    </p>
                  </>
                ) : (
                  <div className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-600 text-gray-500 text-sm">
                    Loading library groups...
                  </div>
                )}
              </div>

              <div className="col-span-2 border border-gray-700 rounded-lg p-4 bg-gray-800/50">
                <h3 className="text-lg font-semibold mb-3">Palette Optimisation</h3>

                <div className="space-y-4">
                  <div>
                    <label className="block mb-2">
                      Target error (ΔE): {targetErrorDeltaE.toFixed(0)}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="15"
                      step="1"
                      value={targetErrorDeltaE}
                      onChange={(e) => setTargetErrorDeltaE(parseInt(e.target.value, 10))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>1</span>
                      <span>15</span>
                    </div>
                  </div>

                  <div>
                    <label className="block mb-2">
                      Maximum palette size: {maxPaletteSize}
                    </label>
                    <input
                      type="range"
                      min="4"
                      max="24"
                      step="1"
                      value={maxPaletteSize}
                      onChange={(e) => setMaxPaletteSize(parseInt(e.target.value, 10))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-blue-600"
                    />
                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                      <span>4</span>
                      <span>24</span>
                    </div>
                  </div>

                  <label className="flex items-center gap-3 text-sm">
                    <input
                      type="checkbox"
                      checked={preferSimplerMixes}
                      onChange={(e) => setPreferSimplerMixes(e.target.checked)}
                      className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                    />
                    Prefer simpler mixes
                  </label>

                  <button
                    onClick={handleComputeOptimalPalette}
                    disabled={optimizingPalette || (!image && !(editSessionId && editSessionOriginalUrl))}
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {optimizingPalette && (
                      <svg
                        className="animate-spin h-4 w-4 text-white"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                      >
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path
                          className="opacity-75"
                          fill="currentColor"
                          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                        ></path>
                      </svg>
                    )}
                    {optimizingPalette ? 'Computing…' : 'Compute Optimal Palette'}
                  </button>

                  {paletteOptimization && (
                    <div className="mt-3 p-3 rounded border border-gray-700 bg-gray-900/60 space-y-2">
                      <div className="text-sm">
                        Optimal palette size: <span className="font-semibold">{paletteOptimization.optimal_palette_size} colours</span>
                      </div>
                      <div className="text-sm">
                        Average ΔE: <span className="font-semibold">{paletteOptimization.average_delta_e.toFixed(2)}</span>
                      </div>
                      <div className="text-sm">
                        Maximum ΔE: <span className="font-semibold">{paletteOptimization.maximum_delta_e.toFixed(2)}</span>
                      </div>
                      <div className="text-xs text-gray-400">
                        Number of Colors has been set to {paletteOptimization.optimal_palette_size}. Generate layers to apply.
                      </div>
                      <div className="grid grid-cols-8 gap-2 pt-1">
                        {paletteOptimization.palette.map((color) => (
                          <div key={color.index} className="flex flex-col items-center">
                            <div
                              className="w-7 h-7 rounded border border-gray-600"
                              style={{ backgroundColor: color.target_hex }}
                              title={`Color ${color.index}: ${color.target_hex}`}
                            />
                            <span className="text-[10px] text-gray-400 mt-1">{color.index}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Gradient-Aware Quantization - disabled and hidden for now */}
              {false && (
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
              )}
            </div>

            <button
              onClick={handleGenerate}
              disabled={Boolean(processing || (isNewProject && (!image || !projectName.trim())) || (!editSessionId && !image) || (editSessionId && !image && !editSessionOriginalUrl))}
              title={editSessionId && editSessionOriginalUrl ? 'Use stored image and current settings, or upload a new image to replace.' : undefined}
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
            <div className="p-6 bg-gray-800 rounded-lg max-w-md">
              <p className="text-lg font-medium">Layers generated.</p>
              <p className="text-gray-400 text-sm mb-2">The original image is stored on the server and will be reused if you change settings and regenerate.</p>
              <p className="text-gray-400 text-sm mb-4">View palette, recipes, and layers on the Projection tab.</p>
              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => {
                    if (returnToHome && typeof window !== 'undefined' && window.top !== window) {
                      window.top!.location.href = `/?tab=projection&session=${sessionData.session_id}`
                    } else {
                      router.push(returnToHome ? `/?tab=projection&session=${sessionData.session_id}` : `/project/${sessionData.session_id}`)
                    }
                  }}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded"
                >
                  Open Projection tab
                </button>
                <button
                  onClick={handleBack}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                >
                  Back to settings
                </button>
              </div>
            </div>
            {sessionData.original_url && (
              <div>
                <h2 className="text-xl font-bold mb-2">Original image (stored on server)</h2>
                <img
                  src={`${API_BASE_URL}${sessionData.original_url}`}
                  alt="Original"
                  className="max-w-md rounded border border-gray-600"
                />
              </div>
            )}
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
                const errorInfo = recipe && recipe.error !== undefined 
                  ? getErrorLevel(recipe.error) 
                  : null
                
                return (
                  <div className="mt-6 pt-6 border-t border-gray-700">
                    <h4 className="text-lg font-bold mb-3">Mixing Recipe</h4>
                    <div className="text-sm text-gray-300 mb-3">
                      {formatRecipe(recipeData)}
                    </div>
                    
                    {/* Recipe metadata */}
                    {recipe && (
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

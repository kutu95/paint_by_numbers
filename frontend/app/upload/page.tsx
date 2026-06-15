'use client'

import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import { API_BASE_URL } from '@/lib/config'
import { projectAssetUrl } from '@/lib/projectAssets'
import { fetchProjectInfo, fetchProjectSession } from '@/lib/projectSession'
import {
  saveProject,
  getProjectBySessionId,
  removeProject,
  syncProjectsFromServer,
  resolveProjectLibraryGroup,
  type Project,
} from '@/lib/projects'
import { canvasCmForImageOrientation } from '@/lib/canvasOrientation'
import {
  IMAGE_STYLE_PRESETS,
  normalizeStylePreset,
  presetShowsSimplifyControls,
  presetShowsFigureDetailControls,
  presetUsesLegacyEasyPainting,
  presetForcesFigureDetail,
  type ImageStylePreset,
} from '@/lib/imageStylePresets'
import { PriorityRegionEditor } from '@/components/PriorityRegionEditor'

interface PaletteColor {
  index: number
  hex: string
  coverage: number
  skin?: boolean
  must_include?: boolean
  rgb?: number[]
}

function normalizeMustIncludeHexList(colors: unknown): string[] {
  if (!Array.isArray(colors)) return []
  return colors
    .map((c) => {
      const s = String(c).trim().toUpperCase()
      return s.startsWith('#') ? s : `#${s}`
    })
    .filter((h) => /^#[0-9A-F]{6}$/.test(h))
}

function mustIncludeColorsFromPalette(
  palette: Array<{ hex?: string; must_include?: boolean }> | undefined
): string[] {
  if (!Array.isArray(palette)) return []
  return normalizeMustIncludeHexList(
    palette.filter((c) => c.must_include && c.hex).map((c) => c.hex as string)
  )
}

function isMustIncludeSwatch(color: PaletteColor, mustIncludeSet: ReadonlySet<string>): boolean {
  if (color.must_include) return true
  const hex = color.hex.trim().toUpperCase()
  return mustIncludeSet.has(hex.startsWith('#') ? hex : `#${hex}`)
}

function hexToRgbTuple(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]
}

/** Force must-include picks to appear as exact swatches in the palette strip. */
function mergeMustIncludeIntoPalette(
  palette: PaletteColor[],
  mustIncludeColors: string[]
): PaletteColor[] {
  const mustList = normalizeMustIncludeHexList(mustIncludeColors)
  if (mustList.length === 0) return palette
  const result = palette.map((c) => ({ ...c }))
  for (let i = 0; i < mustList.length; i++) {
    const hex = mustList[i]
    const matchIdx = result.findIndex((c) => c.hex.trim().toUpperCase() === hex)
    const slotIdx = matchIdx >= 0 ? matchIdx : i < result.length ? i : -1
    if (slotIdx < 0) continue
    const rgb = hexToRgbTuple(hex)
    result[slotIdx] = {
      ...result[slotIdx],
      hex,
      must_include: true,
      rgb,
    }
  }
  return result
}

function PreviewPaletteSwatch({
  color,
  mustIncludeSet,
}: {
  color: PaletteColor
  mustIncludeSet: ReadonlySet<string>
}) {
  const isMustInclude = isMustIncludeSwatch(color, mustIncludeSet)
  const ring = isMustInclude
    ? 'ring-2 ring-violet-400 rounded'
    : color.skin
      ? 'ring-2 ring-amber-400 rounded'
      : ''
  const hex = color.hex.toUpperCase()
  const label = isMustInclude
    ? `Must include ${color.index}, ${hex}, ${color.coverage.toFixed(1)}% coverage`
    : color.skin
      ? `Skin tone ${color.index}, ${hex}, ${color.coverage.toFixed(1)}% coverage`
      : `Colour ${color.index}, ${hex}, ${color.coverage.toFixed(1)}% coverage`

  return (
    <div className={`relative min-w-0 group ${ring}`}>
      <div
        className="w-full aspect-square rounded border border-gray-600 cursor-default"
        style={{ backgroundColor: color.hex }}
        aria-label={label}
      />
      <div
        className="absolute bottom-[calc(100%+8px)] left-1/2 -translate-x-1/2 z-50 pointer-events-none opacity-0 scale-95 group-hover:opacity-100 group-hover:scale-100 transition-[opacity,transform] duration-150"
        role="tooltip"
      >
        <div className="rounded-lg border border-gray-500 bg-gray-900 shadow-xl p-2.5 flex flex-col items-center gap-2">
          <div
            className="w-16 h-16 rounded-md border-2 border-gray-500 shadow-inner"
            style={{ backgroundColor: color.hex }}
          />
          <p className="font-mono text-sm text-gray-100 whitespace-nowrap">{hex}</p>
        </div>
      </div>
    </div>
  )
}

function PreviewPaletteSwatches({
  palette,
  mustIncludeColors = [],
}: {
  palette: PaletteColor[]
  mustIncludeColors?: string[]
}) {
  const mustIncludeSet = useMemo(
    () => new Set(normalizeMustIncludeHexList(mustIncludeColors)),
    [mustIncludeColors]
  )
  const sorted = useMemo(
    () => [...palette].sort((a, b) => a.index - b.index),
    [palette]
  )
  return (
    <div className="grid w-full gap-1.5 grid-cols-[repeat(auto-fill,minmax(1.75rem,1fr))] overflow-visible">
      {sorted.map((color) => (
        <PreviewPaletteSwatch key={color.index} color={color} mustIncludeSet={mustIncludeSet} />
      ))}
    </div>
  )
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
  artifacts_version?: number
  layers: Layer[]
  gradient_regions?: GradientRegion[]
  canvas_width_cm?: number
  canvas_height_cm?: number
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
  const [uploadObjectUrl, setUploadObjectUrl] = useState<string | null>(null)
  
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
  const [favorSkinTones, setFavorSkinTones] = useState(true)
  const [skinToneStrength, setSkinToneStrength] = useState(0.65)
  const [stylePreset, setStylePreset] = useState<ImageStylePreset>('none')
  const [easyPainting, setEasyPainting] = useState(false)
  const [easySimplify, setEasySimplify] = useState(0.65)
  const [easyFaceDetail, setEasyFaceDetail] = useState(false)
  const [detailEyes, setDetailEyes] = useState(true)
  const [detailFace, setDetailFace] = useState(true)
  const [detailBodyOutline, setDetailBodyOutline] = useState(false)
  const [priorityRegionMaskBlob, setPriorityRegionMaskBlob] = useState<Blob | null>(null)
  const [priorityRegionMaskVersion, setPriorityRegionMaskVersion] = useState(0)
  const [priorityRegionCleared, setPriorityRegionCleared] = useState(false)
  const [priorityRegionStrength, setPriorityRegionStrength] = useState(0.7)
  const [priorityBrushSize, setPriorityBrushSize] = useState(28)
  const [mustIncludeColors, setMustIncludeColors] = useState<string[]>([])
  const mustIncludeKey = mustIncludeColors.join('|')
  const [editPriorityRegionUrl, setEditPriorityRegionUrl] = useState<string | null>(null)
  const [stylePreviewUrl, setStylePreviewUrl] = useState<string | null>(null)
  const [stylePreviewPalette, setStylePreviewPalette] = useState<PaletteColor[] | null>(null)
  const [stylePreviewLoading, setStylePreviewLoading] = useState(false)
  const [stylePreviewError, setStylePreviewError] = useState<string | null>(null)
  const stylePreviewObjectUrlRef = useRef<string | null>(null)
  const [processing, setProcessing] = useState(false)
  const [optimizingPalette, setOptimizingPalette] = useState(false)
  const [paletteOptimizeStatus, setPaletteOptimizeStatus] = useState<string | null>(null)
  const paletteOptimizeAbortRef = useRef<AbortController | null>(null)
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

  const appendPriorityRegionToForm = (formData: FormData) => {
    formData.append('priority_region_strength', priorityRegionStrength.toString())
    if (priorityRegionMaskBlob) {
      formData.append('priority_region_mask', priorityRegionMaskBlob, 'priority_region.png')
      formData.append('clear_priority_region', 'false')
    } else if (priorityRegionCleared) {
      formData.append('clear_priority_region', 'true')
    }
    formData.append('must_include_colors', JSON.stringify(mustIncludeColors))
  }
  const [editArtifactsVersion, setEditArtifactsVersion] = useState<number | null>(null)
  /** Server still has input.jpg (etc.) — can reprocess without picking a new file. */
  const [editSessionCanReprocess, setEditSessionCanReprocess] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const settingsHydratedRef = useRef(false)
  const favorSkinTonesTouchedRef = useRef(false)
  const [projectSettingsHydrated, setProjectSettingsHydrated] = useState(false)
  const router = useRouter()
  const searchParams = useSearchParams()
  const isNewProject = searchParams.get('new') === '1'
  const editSessionId = searchParams.get('edit')

  const persistProjectImageSettings = useCallback(
    async (
      patch: Partial<Pick<Project, 'favorSkinTones' | 'skinToneStrength' | 'mustIncludeColors'>>
    ) => {
      if (!editSessionId) return
      const existing = getProjectBySessionId(editSessionId)
      if (!existing) return
      await saveProject({ ...existing, ...patch }, { awaitServer: true })
    },
    [editSessionId]
  )

  const updateMustIncludeColors = useCallback(
    (colors: string[]) => {
      const normalized = normalizeMustIncludeHexList(colors)
      setMustIncludeColors(normalized)
      if (editSessionId) {
        void persistProjectImageSettings({
          mustIncludeColors: normalized,
          favorSkinTones,
          skinToneStrength,
        })
      }
    },
    [editSessionId, favorSkinTones, skinToneStrength, persistProjectImageSettings]
  )

  useEffect(() => {
    if (!image) {
      setUploadObjectUrl(null)
      return
    }
    const url = URL.createObjectURL(image)
    setUploadObjectUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [image])

  const originalImageSrc = useMemo(() => {
    if (preview) return preview
    if (uploadObjectUrl) return uploadObjectUrl
    if (editSessionOriginalUrl) {
      return projectAssetUrl(
        editSessionOriginalUrl,
        sessionData?.artifacts_version ?? editArtifactsVersion
      )
    }
    return null
  }, [preview, uploadObjectUrl, editSessionOriginalUrl, sessionData?.artifacts_version, editArtifactsVersion])

  const canShowImageComparison = Boolean(
    originalImageSrc || image || (editSessionId && editSessionCanReprocess)
  )
  const returnToHome = searchParams.get('returnTo') === 'home'

  useEffect(() => {
    const max = Math.max(0, nColors - 1)
    setMustIncludeColors((prev) => (prev.length > max ? prev.slice(0, max) : prev))
  }, [nColors])

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
      setMustIncludeColors([])

      const finishImageLoad = (imgW: number, imgH: number, drawToPreview: (ctx: CanvasRenderingContext2D, cw: number, ch: number) => void) => {
        const nextCanvas = canvasCmForImageOrientation(imgW, imgH, canvasWidthCm, canvasHeightCm)
        if (nextCanvas.widthCm !== canvasWidthCm || nextCanvas.heightCm !== canvasHeightCm) {
          setCanvasWidthCm(nextCanvas.widthCm)
          setCanvasHeightCm(nextCanvas.heightCm)
        }

        const maxWidth = 800
        const maxHeight = 600
        const ratio = Math.min(maxWidth / imgW, maxHeight / imgH, 1)
        const canvasWidth = imgW * ratio
        const canvasHeight = imgH * ratio

        const canvas = document.createElement('canvas')
        canvas.width = canvasWidth
        canvas.height = canvasHeight
        const ctx = canvas.getContext('2d')
        if (ctx) {
          drawToPreview(ctx, canvasWidth, canvasHeight)
          const compressedDataUrl = canvas.toDataURL('image/jpeg', 0.7)
          setPreview(compressedDataUrl)
          try {
            localStorage.setItem('current_image_preview', compressedDataUrl)
            localStorage.setItem('current_image_name', file.name)
          } catch {
            console.warn('Image too large for localStorage, preview will not persist across navigation')
            try {
              localStorage.removeItem('current_image_preview')
              localStorage.removeItem('current_image_name')
            } catch {
              /* ignore */
            }
          }
        }
      }

      const loadViaDataUrl = () => {
        const reader = new FileReader()
        reader.onload = (event) => {
          const originalDataUrl = event.target?.result as string
          const img = new Image()
          img.onload = () => {
            finishImageLoad(img.width, img.height, (ctx, cw, ch) => {
              ctx.drawImage(img, 0, 0, cw, ch)
            })
          }
          img.src = originalDataUrl
        }
        reader.readAsDataURL(file)
      }

      if (typeof createImageBitmap !== 'undefined') {
        void createImageBitmap(file, { imageOrientation: 'from-image' as ImageOrientation })
          .then((bitmap) => {
            finishImageLoad(bitmap.width, bitmap.height, (ctx, cw, ch) => {
              ctx.drawImage(bitmap, 0, 0, cw, ch)
              bitmap.close()
            })
          })
          .catch(() => {
            loadViaDataUrl()
          })
      } else {
        loadViaDataUrl()
      }
    }
  }

  // Save settings to localStorage whenever they change (not while editing a project — project manifest is source of truth).
  useEffect(() => {
    if (typeof window === 'undefined' || !settingsHydratedRef.current) return
    if (editSessionId) return
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
        favorSkinTones,
        skinToneStrength,
        stylePreset,
        easyPainting,
        easySimplify,
        easyFaceDetail,
        detailEyes,
        detailFace,
        detailBodyOutline,
        mustIncludeColors,
      }
      localStorage.setItem('layerpainter_settings', JSON.stringify(settings))
    } catch (e) {
      console.error('Failed to save settings to localStorage:', e)
    }
  }, [nColors, overpaintMm, orderMode, maxSide, canvasWidthCm, canvasHeightCm, saturationBoost, detailLevel, favorSkinTones, skinToneStrength, stylePreset, easyPainting, easySimplify, easyFaceDetail, detailEyes, detailFace, detailBodyOutline, mustIncludeKey, editSessionId])

  // Set mounted flag and load settings from localStorage after component mounts (client-side only)
  useEffect(() => {
    setMounted(true)

    // When editing a project, image settings come from the project manifest (see edit effect below).
    if (searchParams.get('edit')) {
      settingsHydratedRef.current = true
      return
    }

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
          if (parsed.favorSkinTones !== undefined) setFavorSkinTones(!!parsed.favorSkinTones)
          if (typeof parsed.skinToneStrength === 'number') setSkinToneStrength(parsed.skinToneStrength)
          if (typeof parsed.stylePreset === 'string') {
            setStylePreset(normalizeStylePreset(parsed.stylePreset))
          } else if (parsed.easyPainting) {
            setStylePreset('none')
          }
          if (parsed.easyPainting !== undefined) setEasyPainting(!!parsed.easyPainting)
          if (typeof parsed.easySimplify === 'number') setEasySimplify(parsed.easySimplify)
          if (parsed.easyFaceDetail !== undefined) setEasyFaceDetail(parsed.easyFaceDetail)
          if (parsed.detailEyes !== undefined) setDetailEyes(parsed.detailEyes)
          if (parsed.detailFace !== undefined) setDetailFace(parsed.detailFace)
          if (parsed.detailBodyOutline !== undefined) setDetailBodyOutline(parsed.detailBodyOutline)
          if (Array.isArray(parsed.mustIncludeColors)) {
            setMustIncludeColors(normalizeMustIncludeHexList(parsed.mustIncludeColors))
          }
        }
      } catch (e) {
        console.error('Failed to load settings from localStorage:', e)
      }
    }
    settingsHydratedRef.current = true
  }, [searchParams])

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
  // Restore saved layers from localStorage (iframe remount / switching back to Image tab skips the generic restore effect when ?edit= is set).
  // Hydrate original_url from localStorage and/or server so Image tab + projection G key always have a full-colour original when the file exists.
  useEffect(() => {
    if (!editSessionId || typeof window === 'undefined') return
    favorSkinTonesTouchedRef.current = false
    setProjectSettingsHydrated(false)
    let cancelled = false
    void (async () => {
      const [_, info, session] = await Promise.all([
        syncProjectsFromServer(),
        fetchProjectInfo(editSessionId),
        fetchProjectSession(editSessionId),
      ])
      if (cancelled) return
      const p = getProjectBySessionId(editSessionId)
      if (!favorSkinTonesTouchedRef.current) {
        if (typeof info?.favor_skin_tones === 'boolean') {
          setFavorSkinTones(info.favor_skin_tones)
        } else if (p?.favorSkinTones !== undefined) {
          setFavorSkinTones(!!p.favorSkinTones)
        }
      }
      if (typeof info?.skin_tone_strength === 'number') {
        setSkinToneStrength(info.skin_tone_strength)
      } else if (typeof p?.skinToneStrength === 'number') {
        setSkinToneStrength(p.skinToneStrength)
      }
      if (p) {
        setProjectName(p.name)
        setCanvasWidthCm(p.canvasWidthCm)
        setCanvasHeightCm(p.canvasHeightCm)
        setSaturationBoost(p.saturationBoost)
        setDetailLevel(p.detailLevel)
        if (typeof p.stylePreset === 'string') {
          setStylePreset(normalizeStylePreset(p.stylePreset))
        } else if (p.easyPainting) {
          setStylePreset('none')
        }
        if (p.easyPainting !== undefined) setEasyPainting(!!p.easyPainting)
        if (typeof p.easySimplify === 'number') setEasySimplify(p.easySimplify)
        if (p.easyFaceDetail !== undefined) setEasyFaceDetail(p.easyFaceDetail)
        if (p.detailEyes !== undefined) setDetailEyes(p.detailEyes)
        if (p.detailFace !== undefined) setDetailFace(p.detailFace)
        if (p.detailBodyOutline !== undefined) setDetailBodyOutline(p.detailBodyOutline)
        if (typeof p.priorityRegionStrength === 'number') setPriorityRegionStrength(p.priorityRegionStrength)
      }
      const mustIncludeFromManifest = Array.isArray(info?.must_include_colors)
        ? normalizeMustIncludeHexList(info.must_include_colors)
        : normalizeMustIncludeHexList(p?.mustIncludeColors)
      let mustInclude = mustIncludeFromManifest
      if (mustInclude.length === 0) {
        mustInclude = mustIncludeColorsFromPalette(session?.palette)
      }
      setMustIncludeColors(mustInclude)
      if (mustInclude.length > 0 && mustIncludeFromManifest.length === 0) {
        const existing = getProjectBySessionId(editSessionId)
        if (existing) {
          void saveProject({ ...existing, mustIncludeColors: mustInclude }, { awaitServer: true })
        }
      }
      if (!cancelled) {
        setProjectSettingsHydrated(true)
      }
    })()

    setEditSessionCanReprocess(false)
    void (async () => {
      const [info, session] = await Promise.all([
        fetchProjectInfo(editSessionId),
        fetchProjectSession(editSessionId),
      ])
      if (cancelled) return
      const p = getProjectBySessionId(editSessionId)
      if (info) {
        setEditSessionCanReprocess(Boolean(info.has_stored_image))
        if (info.original_url) setEditSessionOriginalUrl(info.original_url)
        if (info.priority_region_url) {
          setEditPriorityRegionUrl(projectAssetUrl(info.priority_region_url))
        } else {
          setEditPriorityRegionUrl(null)
        }
      }
      if (session && Array.isArray(session.layers) && session.layers.length > 0) {
        setSessionData(session as SessionResponse)
        if (session.original_url) setEditSessionOriginalUrl(session.original_url)
        if (typeof session.artifacts_version === 'number') {
          setEditArtifactsVersion(session.artifacts_version)
        }
        if (typeof session.width === 'number' && typeof session.height === 'number') {
          const next = canvasCmForImageOrientation(
            session.width,
            session.height,
            p?.canvasWidthCm ?? canvasWidthCm,
            p?.canvasHeightCm ?? canvasHeightCm
          )
          setCanvasWidthCm(next.widthCm)
          setCanvasHeightCm(next.heightCm)
        }
      } else if (info?.original_url) {
        const img = new Image()
        img.onload = () => {
          const next = canvasCmForImageOrientation(
            img.width,
            img.height,
            p?.canvasWidthCm ?? canvasWidthCm,
            p?.canvasHeightCm ?? canvasHeightCm
          )
          setCanvasWidthCm(next.widthCm)
          setCanvasHeightCm(next.heightCm)
        }
        img.src = `${API_BASE_URL}${info.original_url}`
      }
    })()

    return () => {
      cancelled = true
      setProjectSettingsHydrated(false)
    }
  }, [editSessionId])

  // Persist slider / must-include changes after hydration (checkbox saves immediately on change).
  useEffect(() => {
    if (!editSessionId || !projectSettingsHydrated) return
    const timer = window.setTimeout(() => {
      void persistProjectImageSettings({ skinToneStrength })
    }, 300)
    return () => clearTimeout(timer)
  }, [editSessionId, projectSettingsHydrated, skinToneStrength, persistProjectImageSettings])

  // Reflect must-include picks in the palette strip immediately (preview request may still be in flight).
  useEffect(() => {
    setStylePreviewPalette((prev) =>
      prev && mustIncludeColors.length > 0
        ? mergeMustIncludeIntoPalette(prev, mustIncludeColors)
        : prev
    )
  }, [mustIncludeKey, mustIncludeColors.length])

  useEffect(() => {
    const canPreview = Boolean(image) || Boolean(editSessionId && editSessionCanReprocess)
    if (!canPreview) {
      if (stylePreviewObjectUrlRef.current) {
        URL.revokeObjectURL(stylePreviewObjectUrlRef.current)
        stylePreviewObjectUrlRef.current = null
      }
      setStylePreviewUrl(null)
      setStylePreviewPalette(null)
      setStylePreviewError('Upload an image (or use a project with a stored image) to preview.')
      setStylePreviewLoading(false)
      return
    }

    const controller = new AbortController()
    const debounceMs = mustIncludeColors.length > 0 ? 180 : 650
    const timer = window.setTimeout(() => {
      void (async () => {
        setStylePreviewLoading(true)
        setStylePreviewError(null)
        try {
          const formData = new FormData()
          if (image) {
            formData.append('image', image)
          } else if (editSessionId) {
            formData.append('project_id', editSessionId)
          }
          formData.append('n_colors', nColors.toString())
          formData.append('max_side', maxSide.toString())
          formData.append('saturation_boost', saturationBoost.toString())
          formData.append('style_preset', stylePreset)
          formData.append('easy_painting', easyPainting ? 'true' : 'false')
          formData.append('easy_simplify', easySimplify.toString())
          formData.append('easy_face_detail', easyFaceDetail ? 'true' : 'false')
          formData.append('detail_eyes', detailEyes ? 'true' : 'false')
          formData.append('detail_face', detailFace ? 'true' : 'false')
          formData.append('detail_body_outline', detailBodyOutline ? 'true' : 'false')
          formData.append('favor_skin_tones', favorSkinTones ? 'true' : 'false')
          formData.append('skin_tone_strength', skinToneStrength.toString())
          formData.append('include_palette', 'true')
          formData.append('detail_level', detailLevel.toString())
          formData.append('priority_region_strength', priorityRegionStrength.toString())
          if (priorityRegionMaskBlob) {
            formData.append('priority_region_mask', priorityRegionMaskBlob, 'priority_region.png')
            formData.append('clear_priority_region', 'false')
          } else if (priorityRegionCleared) {
            formData.append('clear_priority_region', 'true')
          }
          formData.append('must_include_colors', JSON.stringify(mustIncludeColors))
          const response = await fetch(`${API_BASE_URL}/api/preview/quantize`, {
            method: 'POST',
            body: formData,
            signal: controller.signal,
          })
          if (!response.ok) {
            let detail = `HTTP ${response.status}`
            try {
              const err = await response.json()
              detail = (err as { detail?: string }).detail || detail
            } catch {
              detail = (await response.text()) || detail
            }
            throw new Error(detail)
          }
          const data = (await response.json()) as {
            jpeg_base64: string
            palette: PaletteColor[]
          }
          const binary = atob(data.jpeg_base64)
          const bytes = new Uint8Array(binary.length)
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
          const blob = new Blob([bytes], { type: 'image/jpeg' })
          const url = URL.createObjectURL(blob)
          if (stylePreviewObjectUrlRef.current) {
            URL.revokeObjectURL(stylePreviewObjectUrlRef.current)
          }
          stylePreviewObjectUrlRef.current = url
          setStylePreviewUrl(url)
          setStylePreviewPalette(
            Array.isArray(data.palette)
              ? mergeMustIncludeIntoPalette(data.palette, mustIncludeColors)
              : null
          )
        } catch (e) {
          if (controller.signal.aborted) return
          setStylePreviewUrl(null)
          setStylePreviewPalette(null)
          setStylePreviewError(e instanceof Error ? e.message : 'Preview failed')
        } finally {
          if (!controller.signal.aborted) setStylePreviewLoading(false)
        }
      })()
    }, debounceMs)

    return () => {
      clearTimeout(timer)
      controller.abort()
    }
  }, [
    stylePreset,
    easyPainting,
    easySimplify,
    easyFaceDetail,
    detailEyes,
    detailFace,
    detailBodyOutline,
    image,
    editSessionId,
    editSessionCanReprocess,
    nColors,
    maxSide,
    saturationBoost,
    favorSkinTones,
    skinToneStrength,
    priorityRegionMaskVersion,
    priorityRegionStrength,
    priorityRegionCleared,
    priorityRegionMaskBlob,
    mustIncludeKey,
    detailLevel,
  ])

  useEffect(() => {
    return () => {
      if (stylePreviewObjectUrlRef.current) {
        URL.revokeObjectURL(stylePreviewObjectUrlRef.current)
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
    const useStoredImage = Boolean(editSessionId && !image && editSessionCanReprocess)
    if (!image && !useStoredImage) {
      if (preview) {
        alert('Please re-select the image file to generate layers. The preview is shown but the file needs to be selected again.')
      } else if (editSessionId) {
        alert('No image selected and no stored upload found for this project on the server. Upload a new image or open the project from the same browser where it was created.')
      }
      return
    }

    setProcessing(true)
    const libraryGroup = resolveProjectLibraryGroup(editSessionId)
    try {
      const formData = new FormData()
      if (image) formData.append('image', image)
      formData.append('n_colors', nColors.toString())
      formData.append('overpaint_mm', overpaintMm.toString())
      formData.append('order_mode', orderMode)
      formData.append('max_side', maxSide.toString())
      formData.append('canvas_width_cm', canvasWidthCm.toString())
      formData.append('canvas_height_cm', canvasHeightCm.toString())
      formData.append('saturation_boost', saturationBoost.toString())
      formData.append('detail_level', detailLevel.toString())
      formData.append('style_preset', stylePreset)
      formData.append('easy_painting', easyPainting ? 'true' : 'false')
      formData.append('easy_simplify', easySimplify.toString())
      formData.append('easy_face_detail', easyFaceDetail ? 'true' : 'false')
      formData.append('detail_eyes', detailEyes ? 'true' : 'false')
      formData.append('detail_face', detailFace ? 'true' : 'false')
      formData.append('detail_body_outline', detailBodyOutline ? 'true' : 'false')
      formData.append('favor_skin_tones', favorSkinTones ? 'true' : 'false')
      formData.append('skin_tone_strength', skinToneStrength.toString())
      appendPriorityRegionToForm(formData)
      formData.append('name', projectName.trim() || 'Untitled')
      formData.append('library_group', libraryGroup)

      let response: Response
      if (editSessionId) {
        // Existing project: single generate endpoint (optional image replaces stored source).
        response = await fetch(`${API_BASE_URL}/api/projects/${editSessionId}/generate`, {
          method: 'POST',
          body: formData,
        })
      } else {
        if (!image) {
          alert('Please select an image to create a new project.')
          return
        }
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
      if (Array.isArray(data.palette) && data.palette.length > 0) {
        setStylePreviewPalette(mergeMustIncludeIntoPalette(data.palette, mustIncludeColors))
      }
      if (data.quantized_preview_url) {
        if (stylePreviewObjectUrlRef.current?.startsWith('blob:')) {
          URL.revokeObjectURL(stylePreviewObjectUrlRef.current)
        }
        stylePreviewObjectUrlRef.current = null
        setStylePreviewUrl(
          projectAssetUrl(data.quantized_preview_url, data.artifacts_version)
        )
      }
      if (typeof window !== 'undefined' && window.parent !== window) {
        window.parent.postMessage(
          { type: 'layerpainter-session-updated', sessionId: data.session_id },
          window.location.origin
        )
      }
      try {
        localStorage.setItem('layerpainter_current_session_id', data.session_id)
      } catch {
        /* ignore */
      }
      if (data.original_url) setEditSessionOriginalUrl(data.original_url)
      if (typeof data.artifacts_version === 'number') setEditArtifactsVersion(data.artifacts_version)
      setEditSessionCanReprocess(true)
      if (orderMode === 'manual') {
        setManualOrder([...data.order])
      }
      // Persist as a named project when created via New project or when editing
      if (editSessionId) {
        const existingProject = getProjectBySessionId(editSessionId)
        saveProject({
          sessionId: data.session_id,
          name: projectName.trim() || existingProject?.name || 'Untitled',
          imageFileName: useStoredImage ? (existingProject?.imageFileName ?? 'image') : (image?.name || 'image'),
          libraryGroup: existingProject?.libraryGroup ?? libraryGroup,
          canvasWidthCm,
          canvasHeightCm,
          saturationBoost,
          detailLevel,
          favorSkinTones,
          skinToneStrength,
          easyPainting,
          easySimplify,
          easyFaceDetail,
          stylePreset,
          detailEyes,
          detailFace,
          detailBodyOutline,
          priorityRegionStrength,
          hasPriorityRegion: Boolean(priorityRegionMaskBlob) || Boolean(editPriorityRegionUrl),
          mustIncludeColors,
          createdAt: existingProject?.createdAt ?? Date.now(),
          nColors,
          overpaintMm,
          orderMode,
          maxSide,
        })
        return
      }
      if (isNewProject && projectName.trim()) {
        saveProject({
          sessionId: data.session_id,
          name: projectName.trim(),
          imageFileName: image?.name || 'image',
          libraryGroup: 'default',
          canvasWidthCm,
          canvasHeightCm,
          saturationBoost,
          detailLevel,
          favorSkinTones,
          skinToneStrength,
          easyPainting,
          easySimplify,
          easyFaceDetail,
          stylePreset,
          detailEyes,
          detailFace,
          detailBodyOutline,
          priorityRegionStrength,
          hasPriorityRegion: Boolean(priorityRegionMaskBlob),
          mustIncludeColors,
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

  useEffect(() => {
    return () => {
      paletteOptimizeAbortRef.current?.abort()
    }
  }, [])

  const handleComputeOptimalPalette = async () => {
    const useStoredImage = Boolean(editSessionId && !image && editSessionCanReprocess)
    if (!image && !useStoredImage) {
      alert('Please select an image first (or open an editable project with a stored image).')
      return
    }

    paletteOptimizeAbortRef.current?.abort()
    const controller = new AbortController()
    paletteOptimizeAbortRef.current = controller
    const timeoutId = window.setTimeout(() => controller.abort(), 120_000)

    setOptimizingPalette(true)
    setPaletteOptimizeStatus('Scanning palette sizes…')
    try {
      const formData = new FormData()
      formData.append('target_delta_e', targetErrorDeltaE.toString())
      formData.append('max_palette_size', maxPaletteSize.toString())
      formData.append('library_group', resolveProjectLibraryGroup(editSessionId))
      formData.append('prefer_simpler', preferSimplerMixes ? 'true' : 'false')
      if (image) {
        formData.append('image', image)
      } else if (editSessionId) {
        formData.append('session_id', editSessionId)
      }

      setPaletteOptimizeStatus('Finding colours and paint recipes (may take ~30s)…')
      const response = await fetch(`${API_BASE_URL}/api/paint/optimize-palette`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      if (!response.ok) {
        let detail = `HTTP ${response.status}`
        try {
          const err = await response.json()
          detail = (err as { detail?: string }).detail || detail
        } catch {
          detail = (await response.text()) || detail
        }
        throw new Error(detail)
      }
      const data: PaletteOptimizationResult = await response.json()
      setPaletteOptimization(data)
      setNColors(data.optimal_palette_size)
      setPaletteOptimizeStatus(null)
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        alert('Palette optimisation timed out or was cancelled. Try a lower max palette size, or use Generate Layers without optimising first.')
      } else {
        console.error('Failed to compute optimal palette:', error)
        const msg = error instanceof Error ? error.message : 'Unknown error'
        alert(`Failed to compute optimal palette: ${msg}`)
      }
      setPaletteOptimizeStatus(null)
    } finally {
      window.clearTimeout(timeoutId)
      setOptimizingPalette(false)
      if (paletteOptimizeAbortRef.current === controller) {
        paletteOptimizeAbortRef.current = null
      }
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
      formData.append('library_group', resolveProjectLibraryGroup(sessionData.session_id))
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
        <h1 className="text-4xl font-bold mb-8">
          {editSessionId ? 'Edit image & settings' : 'LayerPainter'}
        </h1>

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
              {editSessionId && editSessionOriginalUrl && !image && (
                <p className="text-xs text-gray-500 mt-2">
                  Using the stored project image. Upload a new file to replace it, or adjust settings below.
                </p>
              )}
            </div>

            {canShowImageComparison && (
              <div className="border border-gray-700 rounded-lg p-4 bg-gray-900/40">
                <h3 className="text-lg font-semibold text-gray-200 mb-1">Preview</h3>
                <p className="text-sm text-gray-500 mb-4">
                  Mark a priority region (brush/lasso) or use <strong className="text-gray-400">Pick colour</strong> to
                  force colours into the palette. Toggle <strong className="text-gray-400">Mask Dim/Off</strong> to
                  inspect the photo. Settings below update the processed preview (right).
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="flex flex-col min-w-0">
                    <p className="text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
                      Original · priority region
                    </p>
                    <PriorityRegionEditor
                      imageSrc={originalImageSrc}
                      initialMaskUrl={
                        priorityRegionCleared || priorityRegionMaskBlob
                          ? null
                          : editPriorityRegionUrl
                      }
                      brushSize={priorityBrushSize}
                      onBrushSizeChange={setPriorityBrushSize}
                      onMaskChange={(blob) => {
                        setPriorityRegionMaskBlob(blob)
                        setPriorityRegionCleared(blob === null)
                        setPriorityRegionMaskVersion((v) => v + 1)
                      }}
                      detailInRegion={priorityRegionStrength}
                      onDetailInRegionChange={setPriorityRegionStrength}
                      viewportHeightPx={280}
                      mustIncludeColors={mustIncludeColors}
                      onMustIncludeColorsChange={updateMustIncludeColors}
                      maxMustIncludeColors={Math.max(0, nColors - 1)}
                    />
                  </div>
                  <div className="flex flex-col">
                    <p className="text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
                      Processed ({nColors} colours)
                    </p>
                    <div className="flex flex-1 items-center justify-center rounded-lg border border-gray-600 bg-black min-h-[280px] max-h-[420px] overflow-hidden">
                      {stylePreviewLoading && (
                        <p className="text-sm text-gray-500 animate-pulse p-6">Building preview…</p>
                      )}
                      {!stylePreviewLoading && stylePreviewError && (
                        <p className="text-sm text-amber-400 p-6 text-center">{stylePreviewError}</p>
                      )}
                      {!stylePreviewLoading && stylePreviewUrl && (
                        <img
                          src={stylePreviewUrl}
                          alt="Processed preview"
                          className="w-full h-full max-h-[420px] object-contain"
                        />
                      )}
                      {!stylePreviewLoading && !stylePreviewUrl && !stylePreviewError && (
                        <p className="text-sm text-gray-500 p-6 text-center">
                          Adjust settings to update the processed preview.
                        </p>
                      )}
                    </div>
                  </div>
                </div>
                {stylePreviewPalette && stylePreviewPalette.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-gray-700 overflow-visible">
                    <p className="text-xs font-medium text-gray-400 mb-2 uppercase tracking-wide">
                      Palette ({stylePreviewPalette.length} colours)
                      {mustIncludeColors.length > 0 && (
                        <span className="normal-case font-normal text-gray-500 ml-2">
                          violet ring = must include
                        </span>
                      )}
                      {favorSkinTones && stylePreviewPalette.some((c) => c.skin) && (
                        <span className="normal-case font-normal text-gray-500 ml-2">
                          amber ring = skin tone
                        </span>
                      )}
                    </p>
                    <PreviewPaletteSwatches
                      palette={stylePreviewPalette}
                      mustIncludeColors={mustIncludeColors}
                    />
                  </div>
                )}
              </div>
            )}

            <section className="border border-gray-700 rounded-lg p-4 bg-gray-900/40 space-y-5">
              <h3 className="text-lg font-semibold text-gray-200">Image settings</h3>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block mb-2 text-sm font-medium">Number of colours (2–100)</label>
                  <input
                    type="number"
                    min="2"
                    max="100"
                    value={nColors}
                    onChange={(e) => {
                      const n = parseInt(e.target.value, 10)
                      setNColors(Number.isNaN(n) ? 16 : Math.min(100, Math.max(2, n)))
                    }}
                    className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-600 text-white"
                  />
                </div>

                <div>
                  <label className="block mb-2 text-sm font-medium">Style preset</label>
                  <select
                    value={stylePreset}
                    onChange={(e) => {
                      const next = normalizeStylePreset(e.target.value)
                      setStylePreset(next)
                      if (presetForcesFigureDetail(next)) {
                        setEasyFaceDetail(true)
                        setDetailEyes(true)
                        setDetailFace(true)
                        setDetailBodyOutline(true)
                      }
                    }}
                    className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-600 text-white"
                  >
                    {IMAGE_STYLE_PRESETS.map((p) => (
                      <option key={p.id} value={p.id}>
                        {p.label}
                      </option>
                    ))}
                  </select>
                  <p className="text-xs text-gray-500 mt-1">
                    {IMAGE_STYLE_PRESETS.find((p) => p.id === stylePreset)?.description}
                  </p>
                </div>
              </div>

              <div>
                <label className="block mb-2 text-sm font-medium">
                  Detail level: {(detailLevel * 100).toFixed(0)}%
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
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Simpler</span>
                  <span>More detail</span>
                </div>
              </div>

              <div className="rounded-lg border border-amber-900/40 p-3 bg-amber-950/15 space-y-3">
                <label className="flex items-start gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={favorSkinTones}
                    onChange={(e) => {
                      const checked = e.target.checked
                      favorSkinTonesTouchedRef.current = true
                      setFavorSkinTones(checked)
                      void persistProjectImageSettings({
                        favorSkinTones: checked,
                        skinToneStrength,
                      })
                    }}
                    className="mt-0.5 w-4 h-4 text-amber-500 bg-gray-700 border-gray-600 rounded"
                  />
                  <span>
                    <span className="font-medium text-amber-100">Favor skin tones</span>
                    <span className="block text-xs text-gray-500 mt-0.5">
                      Reserves warm skin colours in the palette (amber ring in the palette strip).
                    </span>
                  </span>
                </label>
                {favorSkinTones && (
                  <div>
                    <label className="block text-sm mb-2">
                      Skin priority: {(skinToneStrength * 100).toFixed(0)}%
                    </label>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      value={skinToneStrength}
                      onChange={(e) => setSkinToneStrength(parseFloat(e.target.value))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                    />
                  </div>
                )}
              </div>
            </section>

            <details className="rounded-lg border border-gray-700 bg-gray-800/30 group">
              <summary className="cursor-pointer select-none px-4 py-3 font-semibold text-gray-200 list-none flex items-center justify-between gap-2 [&::-webkit-details-marker]:hidden">
                <span>Advanced settings</span>
                <span className="text-gray-500 text-sm font-normal group-open:hidden">Canvas, output, optimisation…</span>
                <span className="text-gray-500 text-sm hidden group-open:inline">▼</span>
              </summary>
              <div className="px-4 pb-4 pt-2 space-y-5 border-t border-gray-700">
                <p className="text-xs text-gray-500">
                  Optional tweaks. Most projects only need the settings above and a priority region.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block mb-2 text-sm">Canvas width (cm)</label>
                    <input
                      type="number"
                      min="0"
                      step="0.1"
                      value={canvasWidthCm}
                      onChange={(e) => setCanvasWidthCm(parseFloat(e.target.value) || 0)}
                      className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-600 text-white"
                    />
                  </div>
                  <div>
                    <label className="block mb-2 text-sm">Canvas height (cm)</label>
                    <input
                      type="number"
                      min="0"
                      step="0.1"
                      value={canvasHeightCm}
                      onChange={(e) => setCanvasHeightCm(parseFloat(e.target.value) || 0)}
                      className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-600 text-white"
                    />
                    <p className="text-xs text-gray-500 mt-1">Swapped for portrait images automatically.</p>
                  </div>
                  <div>
                    <label className="block mb-2 text-sm">Overpaint (mm)</label>
                    <input
                      type="number"
                      min="0"
                      max="50"
                      step="0.5"
                      value={overpaintMm}
                      onChange={(e) => setOverpaintMm(parseFloat(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-600 text-white"
                    />
                  </div>
                  <div>
                    <label className="block mb-2 text-sm">Paint order</label>
                    <select
                      value={orderMode}
                      onChange={(e) => setOrderMode(e.target.value as any)}
                      className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-600 text-white"
                    >
                      <option value="largest">Largest coverage first</option>
                      <option value="smallest">Smallest coverage first</option>
                      <option value="lightest">Lightest colours first</option>
                      <option value="manual">Manual</option>
                    </select>
                  </div>
                  <div>
                    <label className="block mb-2 text-sm">Max resolution</label>
                    <select
                      value={maxSide}
                      onChange={(e) => setMaxSide(parseInt(e.target.value))}
                      className="w-full px-3 py-2 bg-gray-800 rounded border border-gray-600 text-white"
                    >
                      <option value="1920">1920px</option>
                      <option value="2400">2400px</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block mb-2 text-sm">
                    Colour vibrancy: {(saturationBoost * 100).toFixed(0)}%
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
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>50%</span>
                    <span>100% normal</span>
                    <span>500%</span>
                  </div>
                </div>

                {(presetUsesLegacyEasyPainting(stylePreset) ||
                  presetShowsFigureDetailControls(stylePreset, easyPainting) ||
                  presetShowsSimplifyControls(stylePreset, easyPainting)) && (
                  <div className="rounded-lg border border-gray-600 p-3 bg-gray-900/50 space-y-3">
                    <p className="text-sm font-medium text-gray-300">Style tweaks</p>
                    {presetUsesLegacyEasyPainting(stylePreset) && (
                      <label className="flex items-start gap-2 text-sm cursor-pointer">
                        <input
                          type="checkbox"
                          checked={easyPainting}
                          onChange={(e) => setEasyPainting(e.target.checked)}
                          className="mt-0.5 w-4 h-4 text-emerald-500 bg-gray-700 border-gray-600 rounded"
                        />
                        <span>
                          Easy painting mode
                          <span className="block text-xs text-gray-500">Classic: soften background, keep figure detail.</span>
                        </span>
                      </label>
                    )}
                    {presetShowsFigureDetailControls(stylePreset, easyPainting) && (
                      <div className="space-y-2">
                        <p className="text-xs text-amber-400/90">
                          Auto-detect face detail can cause halos — prefer a priority region instead.
                        </p>
                        <label className="flex items-start gap-2 text-sm cursor-pointer">
                          <input
                            type="checkbox"
                            checked={easyFaceDetail}
                            disabled={presetForcesFigureDetail(stylePreset)}
                            onChange={(e) => setEasyFaceDetail(e.target.checked)}
                            className="mt-0.5 w-4 h-4 text-emerald-500 bg-gray-700 border-gray-600 rounded"
                          />
                          <span>Preserve detail (auto-detect eyes/face)</span>
                        </label>
                        {easyFaceDetail && (
                          <div className="ml-6 space-y-1.5 text-sm">
                            <label className="flex items-center gap-2 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={detailEyes}
                                disabled={presetForcesFigureDetail(stylePreset)}
                                onChange={(e) => setDetailEyes(e.target.checked)}
                                className="w-4 h-4 text-emerald-500 bg-gray-700 border-gray-600 rounded"
                              />
                              Eyes
                            </label>
                            <label className="flex items-center gap-2 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={detailFace}
                                disabled={presetForcesFigureDetail(stylePreset)}
                                onChange={(e) => setDetailFace(e.target.checked)}
                                className="w-4 h-4 text-emerald-500 bg-gray-700 border-gray-600 rounded"
                              />
                              Face
                            </label>
                            <label className="flex items-center gap-2 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={detailBodyOutline}
                                disabled={presetForcesFigureDetail(stylePreset)}
                                onChange={(e) => setDetailBodyOutline(e.target.checked)}
                                className="w-4 h-4 text-emerald-500 bg-gray-700 border-gray-600 rounded"
                              />
                              Body outline
                            </label>
                          </div>
                        )}
                      </div>
                    )}
                    {presetShowsSimplifyControls(stylePreset, easyPainting) && (
                      <div>
                        <label className="block text-sm mb-2">
                          Background simplification: {(easySimplify * 100).toFixed(0)}%
                        </label>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.01}
                          value={easySimplify}
                          onChange={(e) => setEasySimplify(parseFloat(e.target.value))}
                          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                        />
                      </div>
                    )}
                  </div>
                )}

                <div className="rounded-lg border border-gray-600 p-3 bg-gray-900/50 space-y-4">
                  <div>
                    <h4 className="text-sm font-semibold text-gray-200">Palette optimisation (paint library)</h4>
                    <p className="text-xs text-gray-500 mt-1">
                      Suggests a colour count from your paints and ΔE target (uses the paint library chosen on the
                      Projection tab). Uses a simplified pass — tune with the preview above for final look.
                    </p>
                  </div>
                  <div>
                    <label className="block mb-2 text-sm">
                      Target error (ΔE): {targetErrorDeltaE.toFixed(0)}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="15"
                      step="1"
                      value={targetErrorDeltaE}
                      onChange={(e) => setTargetErrorDeltaE(parseInt(e.target.value, 10))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                    />
                  </div>
                  <div>
                    <label className="block mb-2 text-sm">Maximum palette size: {maxPaletteSize}</label>
                    <input
                      type="range"
                      min="4"
                      max="24"
                      step="1"
                      value={maxPaletteSize}
                      onChange={(e) => setMaxPaletteSize(parseInt(e.target.value, 10))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
                    />
                  </div>
                  <label className="flex items-center gap-3 text-sm">
                    <input
                      type="checkbox"
                      checked={preferSimplerMixes}
                      onChange={(e) => setPreferSimplerMixes(e.target.checked)}
                      className="w-4 h-4 text-indigo-500 bg-gray-700 border-gray-600 rounded"
                    />
                    Prefer simpler mixes
                  </label>
                  <button
                    type="button"
                    onClick={handleComputeOptimalPalette}
                    disabled={optimizingPalette || (!image && !(editSessionId && editSessionCanReprocess))}
                    className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 rounded disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 text-sm"
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
                    {optimizingPalette ? 'Computing…' : 'Compute optimal palette'}
                  </button>
                  {paletteOptimizeStatus && (
                    <p className="text-xs text-gray-400">{paletteOptimizeStatus}</p>
                  )}
                  {paletteOptimization && (
                    <div className="p-3 rounded border border-gray-700 bg-gray-900/60 space-y-2 text-sm">
                      <p>
                        Suggested: <span className="font-semibold">{paletteOptimization.optimal_palette_size} colours</span>
                        {' '}(avg ΔE {paletteOptimization.average_delta_e.toFixed(2)})
                      </p>
                      <p className="text-xs text-gray-400">
                        Number of colours updated — generate layers to apply the full pipeline.
                      </p>
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        {paletteOptimization.palette.map((color) => (
                          <div
                            key={color.index}
                            className="w-6 h-6 rounded border border-gray-600"
                            style={{ backgroundColor: color.target_hex }}
                            title={color.target_hex}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </details>

            <button
              onClick={handleGenerate}
              disabled={Boolean(processing || (isNewProject && (!image || !projectName.trim())) || (!editSessionId && !image) || (editSessionId && !image && !editSessionCanReprocess))}
              title={editSessionId && editSessionCanReprocess ? 'Use stored image and current settings, or upload a new image to replace.' : undefined}
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
              {processing ? 'Processing...' : sessionData ? 'Regenerate Layers' : 'Generate Layers'}
            </button>
          </div>

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

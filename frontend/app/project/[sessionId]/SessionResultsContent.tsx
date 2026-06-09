'use client'

import { useState, useEffect, useRef } from 'react'
import Link from 'next/link'
import type { SessionData } from './types'
import { API_BASE_URL } from '@/lib/config'
import { projectAssetUrl } from '@/lib/projectAssets'
import { getProjectBySessionId, saveProject, syncProjectsFromServer } from '@/lib/projects'
import { VirtualPaintMixer } from './VirtualPaintMixer'
import { SpotTestModal } from './SpotTestModal'

type LayerWithSource = SessionData['layers'][0] & { source_palette_indices?: number[] }

function getErrorLevel(error: number): { level: string; color: string } | null {
  if (error < 1) return { level: 'Excellent', color: 'green' }
  if (error < 3) return { level: 'Good', color: 'green' }
  if (error < 6) return { level: 'Acceptable', color: 'yellow' }
  return { level: 'Poor', color: 'red' }
}

function hexToRgbObject(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
  return result
    ? { r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16) }
    : null
}

function hexToRgbTuple(hex: string): [number, number, number] | null {
  const o = hexToRgbObject(hex)
  return o ? [o.r, o.g, o.b] : null
}

function formatRecipe(recipeData: any, totalWeightGrams: number | null = null): string {
  if (!recipeData.recipe) return recipeData.error || 'No recipe available'
  const recipe = recipeData.recipe
  const total = totalWeightGrams ?? 0
  const g = (pct: number) => (total > 0 ? ` (${((pct / 100) * total).toFixed(2)} g)` : '')

  if (recipe.ingredients && Array.isArray(recipe.ingredients) && recipe.ingredients.length > 0) {
    const parts = recipe.ingredients
      .map((ing: any) => {
        if (ing?.paint_name == null) return null
        const pct = ing.percentage != null ? Number(ing.percentage) : 0
        const gramsText = g(pct)
        return `${ing.paint_name} ${pct.toFixed(2)}%${gramsText}`
      })
      .filter(Boolean)
    if (parts.length > 0) return parts.join(' + ')
  }
  if (recipe.instructions) return recipe.instructions
  const warning = recipe.uncalibrated ? ' (Estimated - not calibrated) ' : ''
  if (recipeData.type === 'one_pigment') {
    const w = (recipe.white_ratio * 100)
    const p = (recipe.pigment_ratio * 100)
    return `${warning}White ${w.toFixed(1)}%${g(w)} + ${recipe.pigment_id} ${p.toFixed(1)}%${g(p)}`
  }
  if (recipeData.type === 'two_pigment') {
    const w = (recipe.white_ratio * 100)
    const p1 = (recipe.pigment1_ratio * 100)
    const p2 = (recipe.pigment2_ratio * 100)
    return `${warning}White ${w.toFixed(1)}%${g(w)} + ${recipe.pigment1_id} ${p1.toFixed(1)}%${g(p1)} + ${recipe.pigment2_id} ${p2.toFixed(1)}%${g(p2)}`
  }
  if (['three_pigment', 'four_pigment', 'multi_pigment'].includes(recipeData.type) && recipe.pigment_ids?.length) {
    const whitePct = (recipe.white_ratio * 100)
    const whitePart = `White ${whitePct.toFixed(1)}%${g(whitePct)}`
    const pigmentParts = recipe.pigment_ids.map((id: string, idx: number) => {
      const ratio = (recipe.pigment_ratios || [])[idx] ?? 0
      const pct = ratio * 100
      return `${id} ${pct.toFixed(1)}%${g(pct)}`
    })
    return `${warning}${whitePart} + ${pigmentParts.join(' + ')}`
  }
  return 'Unknown recipe type'
}

export interface SessionResultsContentProps {
  sessionId: string
  sessionData: SessionData
}

export function SessionResultsContent({ sessionId, sessionData }: SessionResultsContentProps) {
  const [recipes, setRecipes] = useState<any[]>([])
  const [loadingRecipes, setLoadingRecipes] = useState(false)
  const [activeRecipeJobId, setActiveRecipeJobId] = useState<string | null>(null)
  const [cancellingRecipes, setCancellingRecipes] = useState(false)
  const [recipeActionLabel, setRecipeActionLabel] = useState('Generate Recipes')
  const [recipeProgressIndex, setRecipeProgressIndex] = useState<number | null>(null)
  const [recipeProgressTotal, setRecipeProgressTotal] = useState<number>(0)
  const [recipeProgressStatus, setRecipeProgressStatus] = useState<string>('idle')
  const [selectedColor, setSelectedColor] = useState<{ index: number; hex: string; coverage: number } | null>(null)
  const [selectedLayerColor, setSelectedLayerColor] = useState<{
    hex: string
    paletteIndex?: number
    coverage?: number
    isGradient: boolean
    gradientStepIndex?: number
    layerIndex: number
  } | null>(null)
  const [libraryGroups, setLibraryGroups] = useState<Array<{ group: string; name: string; paint_count: number; calibrated_count: number; coverage_mg_per_cm2?: number | null }>>([])
  const [libraryGroupsLoaded, setLibraryGroupsLoaded] = useState(false)
  const [selectedLibraryGroup, setSelectedLibraryGroup] = useState('default')
  const [mounted, setMounted] = useState(false)
  const [, setProjectSyncTick] = useState(0)
  const [showRecipeColours, setShowRecipeColours] = useState(false)
  const [imageView, setImageView] = useState<'preview' | 'original'>('preview')
  const recipePreviewCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const [spotTestPaletteIndex, setSpotTestPaletteIndex] = useState<number | null>(null)

  const project = typeof window !== 'undefined' ? getProjectBySessionId(sessionId) : null
  const recipeMargin = typeof window !== 'undefined' ? (parseFloat(localStorage.getItem('layerpainter_recipe_margin') || '1.33') || 1.33) : 1.33
  const selectedGroupInfo = libraryGroups.find((g) => g.group === selectedLibraryGroup)
  // Library coverage is stored as g/cm² (e.g. 0.008). Weight in grams = (coverage%/100) × area_cm² × coverage_g_per_cm² × margin.
  const libraryCoverageGPerCm2 = selectedGroupInfo?.coverage_mg_per_cm2 != null && selectedGroupInfo.coverage_mg_per_cm2 > 0 ? selectedGroupInfo.coverage_mg_per_cm2 : null
  const effectiveCanvasWidthCm =
    (project?.canvasWidthCm != null && project.canvasWidthCm > 0)
      ? project.canvasWidthCm
      : ((sessionData.canvas_width_cm != null && sessionData.canvas_width_cm > 0) ? sessionData.canvas_width_cm : 0)
  const effectiveCanvasHeightCm =
    (project?.canvasHeightCm != null && project.canvasHeightCm > 0)
      ? project.canvasHeightCm
      : ((sessionData.canvas_height_cm != null && sessionData.canvas_height_cm > 0) ? sessionData.canvas_height_cm : 0)

  function getTotalWeightGrams(paletteIndex: number): number | null {
    if (libraryCoverageGPerCm2 == null || libraryCoverageGPerCm2 <= 0) return null
    const paletteColor = sessionData.palette.find((p) => p.index === paletteIndex)
    if (!paletteColor) return null
    const areaCm2 = effectiveCanvasWidthCm * effectiveCanvasHeightCm
    if (areaCm2 <= 0) return null
    return (paletteColor.coverage / 100) * areaCm2 * libraryCoverageGPerCm2 * recipeMargin
  }

  function mergeRecipesByPaletteIndex(base: any[], updates: any[]): any[] {
    const byIndex = new Map<number, any>()
    for (const item of base || []) {
      const idx = Number(item?.palette_index)
      if (Number.isFinite(idx)) byIndex.set(idx, item)
    }
    for (const item of updates || []) {
      const idx = Number(item?.palette_index)
      if (Number.isFinite(idx)) byIndex.set(idx, item)
    }
    return Array.from(byIndex.values()).sort((a, b) => Number(a.palette_index) - Number(b.palette_index))
  }

  function getMissingWeightInputs(): string[] {
    const missing: string[] = []

    const width = effectiveCanvasWidthCm
    const height = effectiveCanvasHeightCm
    if (!(width > 0 && height > 0)) {
      missing.push('Canvas size')
    }

    const marginRaw = typeof window !== 'undefined' ? localStorage.getItem('layerpainter_recipe_margin') : null
    const margin = marginRaw != null ? Number(marginRaw) : NaN
    if (!(margin > 0)) {
      missing.push('Paint mix margin')
    }

    if (!(libraryCoverageGPerCm2 != null && libraryCoverageGPerCm2 > 0)) {
      missing.push('Library coverage (g/cm²)')
    }

    return missing
  }

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    void (async () => {
      await syncProjectsFromServer()
      setProjectSyncTick((v) => v + 1)
    })()
  }, [sessionId])

  // On load: fetch any existing cached recipes for the current palette so we show them without waiting for 'Generate recipes'.
  useEffect(() => {
    if (typeof window === 'undefined') return
    const palette = sessionData?.palette
    if (!palette?.length || !selectedLibraryGroup || !libraryGroupsLoaded) return
    const url = `${API_BASE_URL}/api/paint/recipes/cached`
    let cancelled = false
    const load = async () => {
      try {
        const body = {
          palette: palette.map((c) => ({
            index: c.index,
            hex: c.hex,
            target_grams: getTotalWeightGrams(c.index),
          })),
          library_group: selectedLibraryGroup,
        }
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          cache: 'no-store',
        })
        if (cancelled) return
        if (!res.ok) return
        const data = await res.json()
        if (cancelled) return
        const list = data?.recipes
        if (Array.isArray(list) && list.length > 0) {
          setRecipes(list)
        }
      } catch (_) {
        // Ignore: cache lookup is best-effort on load
      }
    }
    void load()
    return () => { cancelled = true }
  }, [sessionId, sessionData?.palette, selectedLibraryGroup, libraryGroupsLoaded])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const url = `${API_BASE_URL}/api/paint/library/groups`
    const load = async () => {
      try {
        let data: any = null
        let lastError: unknown = null
        for (const delayMs of [0, 400]) {
          if (delayMs > 0) {
            await new Promise((resolve) => setTimeout(resolve, delayMs))
          }
          try {
            const res = await fetch(url, { cache: 'no-store' })
            if (!res.ok) {
              throw new Error(`HTTP ${res.status}`)
            }
            data = await res.json()
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
        if (groups.length > 0) {
          const lib = project?.libraryGroup && groups.some((g: any) => g.group === project.libraryGroup)
            ? project.libraryGroup
            : groups.find((g: any) => g.calibrated_count > 0)?.group ?? groups[0].group
          setSelectedLibraryGroup(lib)
        }
      } catch (error) {
        console.error(`Failed to load library groups from ${url}:`, error)
        setLibraryGroupsLoaded(true)
      }
    }
    void load()
  }, [sessionId, project?.libraryGroup])

  const handleGenerateRecipes = async (
    forceRegenerate: boolean = false,
    useAiSecondPass: boolean = false,
    qualityMode: 'balanced' | 'high' | 'fast' | 'server_fast' = 'balanced',
    paletteOverride: Array<{ index: number; hex: string; target_grams: number | null }> | null = null,
    mergeIntoExisting: boolean = false,
    actionLabel: string = 'Generate Recipes',
  ) => {
    const palettePayload = paletteOverride ?? sessionData.palette.map((c) => ({
      index: c.index,
      hex: c.hex,
      target_grams: getTotalWeightGrams(c.index),
    }))

    const missing = getMissingWeightInputs()
    if (missing.length > 0) {
      const message =
        `Missing info for absolute weight calculation:\n\n` +
        `${missing.map((item) => `- ${item}`).join('\n')}\n\n` +
        `Continue anyway?`
      const shouldContinue = window.confirm(message)
      if (!shouldContinue) return
    }

    setLoadingRecipes(true)
    setCancellingRecipes(false)
    setRecipeActionLabel(actionLabel)
    setRecipeProgressIndex(0)
    setRecipeProgressTotal(palettePayload.length)
    setRecipeProgressStatus('starting')
    try {
      const formData = new FormData()
      formData.append('palette', JSON.stringify(palettePayload))
      formData.append('library_group', selectedLibraryGroup)
      formData.append('use_ai_second_pass', useAiSecondPass ? 'true' : 'false')
      formData.append('quality_mode', qualityMode)
      if (forceRegenerate) formData.append('force_regenerate', 'true')

      const startRes = await fetch(`${API_BASE_URL}/api/paint/recipes/jobs`, {
        method: 'POST',
        body: formData,
      })
      if (!startRes.ok) {
        const body = await startRes.text().catch(() => '')
        throw new Error(`Failed to start recipe job: HTTP ${startRes.status} ${body}`.trim())
      }
      const { job_id } = await startRes.json()
      if (!job_id) throw new Error('No job_id returned')
      setActiveRecipeJobId(job_id)

      setRecipeProgressStatus('running')
      const pollIntervalMs = 2000
      for (;;) {
        await new Promise((r) => setTimeout(r, pollIntervalMs))
        const pollRes = await fetch(`${API_BASE_URL}/api/paint/recipes/jobs/${encodeURIComponent(job_id)}`, {
          cache: 'no-store',
        })
        if (!pollRes.ok) throw new Error(`Job poll failed: HTTP ${pollRes.status}`)
        const job = await pollRes.json()
        if (typeof job.total === 'number' && Number(job.total) > 0) {
          setRecipeProgressTotal(Number(job.total))
        }
        if (Array.isArray(job.partial_recipes) && job.partial_recipes.length > 0) {
          setRecipes((prev) => mergeRecipesByPaletteIndex(prev, job.partial_recipes))
        }
        if (typeof job.completed === 'number') {
          const completed = Math.max(0, Math.min(sessionData.palette.length, Number(job.completed)))
          setRecipeProgressIndex(completed)
        }
        if (typeof job.progress_status === 'string') {
          setRecipeProgressStatus(job.progress_status)
        } else if (typeof job.status === 'string') {
          setRecipeProgressStatus(job.status)
        }
        if (job.status === 'completed' && Array.isArray(job.recipes)) {
          setRecipes((prev) => mergeRecipesByPaletteIndex(prev, job.recipes))
          setRecipeProgressStatus('completed')
          break
        }
        if (job.status === 'cancelled') {
          if (Array.isArray(job.recipes) && job.recipes.length > 0) {
            setRecipes((prev) => mergeRecipesByPaletteIndex(prev, job.recipes))
          }
          setRecipeProgressStatus('idle')
          break
        }
        if (job.status === 'failed') {
          throw new Error(job.error || 'Recipe generation failed')
        }
      }
    } catch (e) {
      console.error(e)
      alert(e instanceof Error ? e.message : 'Failed to generate recipes')
    } finally {
      setLoadingRecipes(false)
      setCancellingRecipes(false)
      setActiveRecipeJobId(null)
      setRecipeActionLabel('Generate Recipes')
      setRecipeProgressIndex(null)
      setRecipeProgressTotal(0)
      setRecipeProgressStatus('idle')
    }
  }

  const handleRefineWeakColours = async () => {
    if (recipes.length === 0) {
      alert('Generate recipes first, then refine weak colours.')
      return
    }
    const threshold = 2.5
    const weakIndices = recipes
      .filter((r: any) => r?.recipe && typeof r.recipe.error === 'number' && Number(r.recipe.error) > threshold)
      .map((r: any) => Number(r.palette_index))
      .filter((idx: number) => Number.isFinite(idx))

    if (weakIndices.length === 0) {
      alert(`No weak colours found (all recipes are ≤ ${threshold.toFixed(1)} ΔE).`)
      return
    }

    const palettePayload = sessionData.palette
      .filter((c) => weakIndices.includes(c.index))
      .map((c) => ({
        index: c.index,
        hex: c.hex,
        target_grams: getTotalWeightGrams(c.index),
      }))

    await handleGenerateRecipes(
      true,
      false,
      'high',
      palettePayload,
      true,
      `Refining weak colours (${palettePayload.length})`,
    )
  }

  const handleCancelRecipeGeneration = async () => {
    if (!activeRecipeJobId) return
    setCancellingRecipes(true)
    try {
      await fetch(`${API_BASE_URL}/api/paint/recipes/jobs/${encodeURIComponent(activeRecipeJobId)}/cancel`, {
        method: 'POST',
      })
    } catch (e) {
      console.error(e)
    } finally {
      setCancellingRecipes(false)
    }
  }

  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSelectedColor(null)
        setSelectedLayerColor(null)
      }
    }
    window.addEventListener('keydown', handleEsc)
    return () => window.removeEventListener('keydown', handleEsc)
  }, [])

  const canShowRecipeColours =
    sessionData.palette.length > 0 &&
    sessionData.palette.every((c) => {
      const r = recipes.find((rec: any) => rec.palette_index === c.index)
      return r?.recipe && typeof (r.recipe as any).predicted_hex === 'string'
    })

  useEffect(() => {
    if (!showRecipeColours || !canShowRecipeColours || !sessionData.quantized_preview_url || !recipePreviewCanvasRef.current) return
    const canvas = recipePreviewCanvasRef.current
    const palette = sessionData.palette
    const indexToRecipeRgb: Map<number, [number, number, number]> = new Map()
    for (const c of palette) {
      const rec = recipes.find((r: any) => r.palette_index === c.index)
      const hex = (rec?.recipe as any)?.predicted_hex
      if (typeof hex === 'string') {
        const t = hexToRgbTuple(hex)
        if (t) indexToRecipeRgb.set(c.index, t)
      }
    }
    if (indexToRecipeRgb.size !== palette.length) return
    const hexToIndex = new Map<string, number>()
    for (const c of palette) {
      const h = c.hex.toUpperCase().replace(/^#/, '')
      hexToIndex.set(h, c.index)
    }
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      if (!recipePreviewCanvasRef.current) return
      const cvs = recipePreviewCanvasRef.current
      cvs.width = img.naturalWidth
      cvs.height = img.naturalHeight
      const ctx = cvs.getContext('2d')
      if (!ctx) return
      ctx.drawImage(img, 0, 0)
      try {
        const id = ctx.getImageData(0, 0, cvs.width, cvs.height)
        const data = id.data
        for (let i = 0; i < data.length; i += 4) {
          const r = data[i]
          const g = data[i + 1]
          const b = data[i + 2]
          const hex = [r, g, b].map((x) => x.toString(16).padStart(2, '0')).join('')
          const idx = hexToIndex.get(hex.toUpperCase())
          if (idx !== undefined) {
            const repl = indexToRecipeRgb.get(idx)
            if (repl) {
              data[i] = repl[0]
              data[i + 1] = repl[1]
              data[i + 2] = repl[2]
            }
          }
        }
        ctx.putImageData(id, 0, 0)
      } catch {
        // Canvas tainted (e.g. CORS) or getImageData failed; leave canvas as drawn image
      }
    }
    img.onerror = () => {}
    // Use same-origin URL so the request goes through Next.js rewrites and avoids CORS (e.g. localhost vs 127.0.0.1)
    const imageOrigin = typeof window !== 'undefined' ? window.location.origin : API_BASE_URL
    img.src = projectAssetUrl(sessionData.quantized_preview_url, sessionData.artifacts_version)
  }, [showRecipeColours, canShowRecipeColours, sessionData.quantized_preview_url, sessionData.artifacts_version, sessionData.palette, recipes])

  const layers = sessionData.layers as LayerWithSource[]
  const assetVersion = sessionData.artifacts_version
  const hasOriginal = Boolean(sessionData.original_url)
  const hasPreview = Boolean(sessionData.quantized_preview_url)
  const canFlipImage = hasOriginal && hasPreview
  const showingOriginal = imageView === 'original' && hasOriginal
  const showingPreview = !showingOriginal && hasPreview

  return (
    <div className="space-y-6">
      {(hasOriginal || hasPreview) && (
        <div>
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <h2 className="text-2xl font-bold">
              {showingOriginal ? 'Original' : 'Preview'}
            </h2>
            {canFlipImage && (
              <div
                className="inline-flex rounded-lg border border-gray-600 overflow-hidden text-sm shrink-0"
                role="group"
                aria-label="Image view"
              >
                <button
                  type="button"
                  onClick={() => setImageView('preview')}
                  className={`px-3 py-1.5 font-medium transition-colors ${
                    imageView === 'preview'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  Preview
                </button>
                <button
                  type="button"
                  onClick={() => setImageView('original')}
                  className={`px-3 py-1.5 font-medium transition-colors border-l border-gray-600 ${
                    imageView === 'original'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  Original
                </button>
              </div>
            )}
          </div>

          {showingPreview && (
            <label className="flex items-center gap-2 mb-3 text-sm text-gray-300">
              <input
                type="checkbox"
                checked={showRecipeColours}
                onChange={(e) => setShowRecipeColours(e.target.checked)}
                disabled={!canShowRecipeColours}
                className="rounded border-gray-500 bg-gray-700"
              />
              <span>Use recipe colours</span>
              {!canShowRecipeColours && (
                <span className="text-gray-500 text-xs">(Generate recipes for all palette colours to enable)</span>
              )}
            </label>
          )}

          <div className="rounded-lg border border-gray-600 bg-black/40 flex items-center justify-center overflow-hidden min-h-[12rem]">
            {showingOriginal && (
              <img
                key={`original-${sessionId}-${assetVersion ?? 0}`}
                src={projectAssetUrl(sessionData.original_url!, assetVersion)}
                alt="Original"
                className="max-w-full max-h-[min(70vh,640px)] object-contain"
              />
            )}
            {showingPreview && !showRecipeColours && (
              <img
                key={`quantized-${sessionId}-${assetVersion ?? 0}`}
                src={projectAssetUrl(sessionData.quantized_preview_url!, assetVersion)}
                alt="Quantized preview"
                className="max-w-full max-h-[min(70vh,640px)] object-contain"
              />
            )}
            {showingPreview && showRecipeColours && (
              <canvas
                ref={recipePreviewCanvasRef}
                className="max-w-full block"
                style={{ maxWidth: '100%', maxHeight: 'min(70vh, 640px)', height: 'auto' }}
                aria-label="Quantized preview with recipe colours"
              />
            )}
            {!showingOriginal && !showingPreview && hasOriginal && (
              <img
                key={`original-fallback-${sessionId}-${assetVersion ?? 0}`}
                src={projectAssetUrl(sessionData.original_url!, assetVersion)}
                alt="Original"
                className="max-w-full max-h-[min(70vh,640px)] object-contain"
              />
            )}
          </div>
        </div>
      )}

      <div className="p-6 bg-gray-800 rounded">
        <h2 className="text-2xl font-bold mb-4">Palette & Recipes</h2>
        <div className="grid grid-cols-8 gap-2 mb-4">
          {sessionData.palette.map((color) => (
            <div key={color.index} className="text-center">
              <div
                className="w-16 h-16 rounded border border-gray-600 flex items-center justify-center cursor-pointer hover:opacity-90 transition-opacity"
                style={{ backgroundColor: color.hex }}
                onClick={() => setSelectedColor(color)}
              >
                <span className="text-white font-bold text-lg drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]">{color.index}</span>
              </div>
              <div className="text-xs mt-1">{color.coverage.toFixed(1)}%</div>
              <div className="text-xs text-gray-400 font-mono">{color.hex.toUpperCase()}</div>
              {(() => {
                const w = getTotalWeightGrams(color.index)
                return w != null ? <div className="text-xs text-gray-500">{w.toFixed(2)} g</div> : null
              })()}
            </div>
          ))}
        </div>

        <div className="mb-4">
          <VirtualPaintMixer sessionData={sessionData} selectedLibraryGroup={selectedLibraryGroup} recipes={recipes} />
        </div>

        <div className="mb-4">
          <label className="block text-sm font-semibold mb-2">Paint library</label>
          {libraryGroupsLoaded && libraryGroups.length > 0 ? (
            <select
              value={selectedLibraryGroup}
              onChange={(e) => {
                const next = e.target.value
                setSelectedLibraryGroup(next)
                const existing = getProjectBySessionId(sessionId)
                if (existing) {
                  saveProject({ ...existing, libraryGroup: next })
                }
              }}
              className="w-full max-w-xs px-3 py-2 bg-gray-700 rounded border border-gray-600"
            >
              {libraryGroups.map((g) => (
                <option key={g.group} value={g.group}>
                  {g.name} ({g.paint_count} paints, {g.calibrated_count} calibrated)
                </option>
              ))}
            </select>
          ) : (
            <div className="text-gray-400 text-sm">Loading libraries…</div>
          )}
        </div>

        <div className="flex flex-col gap-2 mb-4">
          <div className="flex gap-2">
            <button
              onClick={() => handleGenerateRecipes(false, false, 'balanced')}
              disabled={loadingRecipes}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50"
            >
              {loadingRecipes ? 'Generating…' : 'Generate Recipes'}
            </button>
            <button
              onClick={() => handleGenerateRecipes(true, false, 'balanced')}
              disabled={loadingRecipes}
              className="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded disabled:opacity-50"
            >
              Force Regenerate
            </button>
            <button
              onClick={() => handleGenerateRecipes(true, true, 'balanced')}
              disabled={loadingRecipes}
              className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded disabled:opacity-50"
            >
              Refine with AI
            </button>
            <button
              onClick={handleRefineWeakColours}
              disabled={loadingRecipes || recipes.length === 0}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded disabled:opacity-50"
            >
              Refine Weak Colours
            </button>
            {loadingRecipes && activeRecipeJobId && (
              <button
                onClick={handleCancelRecipeGeneration}
                disabled={cancellingRecipes}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded disabled:opacity-50"
              >
                {cancellingRecipes ? 'Cancelling…' : 'Cancel'}
              </button>
            )}
          </div>
          {loadingRecipes && (
            <div className="flex items-center gap-2 text-sm text-gray-300">
              <div
                className="h-4 w-4 border-2 border-gray-500 border-t-transparent rounded-full animate-spin"
                aria-hidden="true"
              />
              <span>
                {recipeProgressStatus === 'starting'
                  ? `${recipeActionLabel}: starting…`
                  : recipeProgressStatus === 'cancelled'
                    ? 'Cancelled.'
                  : recipeProgressStatus === 'running'
                    ? `${recipeActionLabel}… (${Math.max(0, Math.min(recipeProgressTotal || sessionData.palette.length, recipeProgressIndex ?? 0))}/${recipeProgressTotal || sessionData.palette.length})`
                    : recipeProgressStatus === 'completed'
                      ? 'Done.'
                      : `Preparing…`}
              </span>
            </div>
          )}
        </div>

        {recipes.length > 0 && (
          <div className="space-y-3 mt-4">
            <h3 className="text-xl font-bold">Mixing Recipes</h3>
            {recipes.map((recipeData: any) => {
              const color = sessionData.palette.find((p) => p.index === recipeData.palette_index)
              if (!color) return null
              const recipe = recipeData.recipe
              const errorInfo = recipe && recipe.error != null ? getErrorLevel(recipe.error) : null
              const predictedHex = typeof recipe?.predicted_hex === 'string' ? recipe.predicted_hex : null
              const totalWeight = getTotalWeightGrams(recipeData.palette_index)
              return (
                <div key={recipeData.palette_index} className="flex items-center gap-4 p-4 bg-gray-700 rounded">
                  <div className="flex gap-2 flex-shrink-0">
                    <div className="text-center">
                      <div className="w-16 h-16 rounded border border-gray-600" style={{ backgroundColor: color.hex }} />
                      <div className="text-[10px] text-gray-400 mt-1">Target</div>
                    </div>
                    <div className="text-center">
                      {predictedHex ? (
                        <div
                          className="w-16 h-16 rounded border border-gray-600"
                          style={{ backgroundColor: predictedHex }}
                          title={predictedHex}
                        />
                      ) : (
                        <div
                          className="w-16 h-16 rounded border border-red-500 bg-transparent flex items-center justify-center text-red-400 text-[10px]"
                          title="Missing predicted color"
                        >
                          N/A
                        </div>
                      )}
                      <div className="text-[10px] text-gray-400 mt-1">Expected Mix</div>
                    </div>
                  </div>
                  <div className="flex-1">
                    <div className="font-bold">Palette Color {recipeData.palette_index}</div>
                    <div className="text-xs text-gray-400">
                      Total paint: {totalWeight != null ? `${totalWeight.toFixed(2)} g` : '—'}
                    </div>
                    <div className="text-xs text-gray-400">
                      Target hex: {color.hex.toUpperCase()}
                    </div>
                    <div className="text-sm text-gray-300">{formatRecipe(recipeData, getTotalWeightGrams(recipeData.palette_index) ?? null)}</div>
                    {recipe && errorInfo && (
                      <span className="inline-block mt-1 px-2 py-0.5 rounded text-xs" style={{ backgroundColor: errorInfo.color === 'green' ? '#16a34a' : errorInfo.color === 'yellow' ? '#ca8a04' : '#dc2626' }}>
                        {recipe.error?.toFixed(2)} ΔE – {errorInfo.level}
                      </span>
                    )}
                    {recipe && (
                      <button
                        type="button"
                        onClick={() => setSpotTestPaletteIndex(recipeData.palette_index)}
                        className="mt-2 px-3 py-1.5 text-sm bg-amber-600 hover:bg-amber-500 rounded text-white"
                      >
                        Verify mix
                      </button>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {spotTestPaletteIndex != null && (() => {
          const recipeData = recipes.find((r: any) => r.palette_index === spotTestPaletteIndex)
          const color = sessionData.palette.find((p) => p.index === spotTestPaletteIndex)
          if (!recipeData?.recipe || !color) return null
          return (
            <SpotTestModal
              open={true}
              onClose={() => setSpotTestPaletteIndex(null)}
              sessionId={sessionId}
              paletteIndex={spotTestPaletteIndex}
              targetHex={color.hex}
              recipe={recipeData.recipe}
              libraryGroup={selectedLibraryGroup}
            />
          )
        })()}
      </div>

      <div>
        <h2 className="text-2xl font-bold mb-4">Layers</h2>
        <div className="space-y-2">
          {layers.filter((l) => !l.is_finished).map((layer) => {
            const isGradient = layer.is_gradient ?? false
            let colorHex = '#000000'
            let displayText = ''
            if (isGradient) {
              colorHex = layer.hex ?? '#808080'
              const stepNum = ((layer.gradient_step_index ?? 0) >= 0 ? (layer.gradient_step_index ?? 0) + 1 : 0)
              const src = layer.source_palette_indices
              displayText = src?.length === 1 ? `Gradient Step ${stepNum} (replaces Palette ${src[0]})` : src?.length ? `Gradient Step ${stepNum} (replaces Palettes ${src.join(', ')})` : `Gradient Step ${stepNum}`
            } else {
              const color = sessionData.palette.find((p) => p.index === layer.palette_index)
              if (!color) return null
              colorHex = color.hex
              const weightG = getTotalWeightGrams(layer.palette_index)
              const weightStr = weightG != null ? `${weightG.toFixed(2)} g` : '—'
              displayText = `Palette ${layer.palette_index} – ${color.coverage.toFixed(1)}% · ${weightStr}`
            }
            return (
              <div key={layer.layer_index} className={`flex items-center gap-4 p-4 rounded ${isGradient ? 'bg-purple-900/30 border border-purple-700' : 'bg-gray-800'}`}>
                <div className="text-lg font-mono">{layer.layer_index + 1}</div>
                <div
                  className="w-16 h-16 rounded border border-gray-600 cursor-pointer hover:opacity-90"
                  style={{ backgroundColor: colorHex }}
                  onClick={() =>
                    setSelectedLayerColor({
                      hex: colorHex,
                      paletteIndex: layer.palette_index >= 0 ? layer.palette_index : undefined,
                      coverage: sessionData.palette.find((p) => p.index === layer.palette_index)?.coverage,
                      isGradient: !!isGradient,
                      gradientStepIndex: layer.gradient_step_index,
                      layerIndex: layer.layer_index,
                    })
                  }
                />
                <img
                  key={`layer-${layer.layer_index}-${assetVersion ?? 0}`}
                  src={projectAssetUrl(
                    layer.mask_pure_url ?? `/api/projects/${sessionId}/artifacts/layer_${layer.layer_index}_pure_mask.png`,
                    assetVersion
                  )}
                  alt={`Layer ${layer.layer_index + 1}`}
                  className="w-16 h-16 object-contain bg-gray-700 rounded"
                />
                <div className="flex-1 text-sm text-gray-400">{displayText}</div>
              </div>
            )
          })}
        </div>
      </div>

      {mounted && selectedColor && (
        <div className="fixed inset-0 bg-black/75 flex items-center justify-center z-50" onClick={() => setSelectedColor(null)}>
          <div className="bg-gray-800 rounded-lg p-8 max-w-md w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between mb-6">
              <h3 className="text-2xl font-bold">Palette Color {selectedColor.index}</h3>
              <button onClick={() => setSelectedColor(null)} className="text-gray-400 hover:text-white text-2xl font-bold w-8 h-8 rounded hover:bg-gray-700">×</button>
            </div>
            <div className="w-full aspect-square rounded-lg border-4 border-gray-600 mb-6" style={{ backgroundColor: selectedColor.hex }} />
            <div className="space-y-3">
              <div className="flex justify-between p-3 bg-gray-700 rounded">
                <span className="text-gray-300 font-semibold">Hex:</span>
                <span className="text-white font-mono">{selectedColor.hex.toUpperCase()}</span>
              </div>
              {hexToRgbObject(selectedColor.hex) && (
                <div className="flex justify-between p-3 bg-gray-700 rounded">
                  <span className="text-gray-300 font-semibold">RGB:</span>
                  <span className="text-white font-mono">R: {hexToRgbObject(selectedColor.hex)!.r} G: {hexToRgbObject(selectedColor.hex)!.g} B: {hexToRgbObject(selectedColor.hex)!.b}</span>
                </div>
              )}
              <div className="flex justify-between p-3 bg-gray-700 rounded">
                <span className="text-gray-300 font-semibold">Coverage:</span>
                <span className="text-white">{selectedColor.coverage.toFixed(1)}%</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {mounted && selectedLayerColor && (
        <div className="fixed inset-0 bg-black/75 flex items-center justify-center z-50" onClick={() => setSelectedLayerColor(null)}>
          <div className="bg-gray-800 rounded-lg p-8 max-w-md w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <div className="flex justify-between mb-6">
              <h3 className="text-2xl font-bold">
                {selectedLayerColor.isGradient ? `Gradient Step ${(selectedLayerColor.gradientStepIndex ?? 0) + 1}` : `Palette Color ${selectedLayerColor.paletteIndex}`}
              </h3>
              <button onClick={() => setSelectedLayerColor(null)} className="text-gray-400 hover:text-white text-2xl font-bold w-8 h-8 rounded hover:bg-gray-700">×</button>
            </div>
            <div className="flex justify-center mb-6">
              <div className="w-1/2 aspect-square rounded-lg border-4 border-gray-600" style={{ backgroundColor: selectedLayerColor.hex }} />
            </div>
            <div className="space-y-3">
              <div className="flex justify-between p-3 bg-gray-700 rounded">
                <span className="text-gray-300 font-semibold">Hex:</span>
                <span className="text-white font-mono">{selectedLayerColor.hex.toUpperCase()}</span>
              </div>
              {selectedLayerColor.paletteIndex != null && (
                <div className="flex justify-between p-3 bg-gray-700 rounded">
                  <span className="text-gray-300 font-semibold">Coverage:</span>
                  <span className="text-white">{selectedLayerColor.coverage?.toFixed(1) ?? '—'}%</span>
                </div>
              )}
            </div>
            {selectedLayerColor.paletteIndex != null && recipes.length > 0 && (() => {
              const recipeData = recipes.find((r: any) => r.palette_index === selectedLayerColor!.paletteIndex)
              if (!recipeData) return null
              return (
                <div className="mt-6 pt-6 border-t border-gray-700">
                  <h4 className="text-lg font-bold mb-3">Mixing Recipe</h4>
                  {(() => {
                    const totalWeight = getTotalWeightGrams(selectedLayerColor.paletteIndex!)
                    const paletteColor = sessionData.palette.find((p) => p.index === selectedLayerColor.paletteIndex!)
                    return (
                      <div className="mb-1 space-y-0.5">
                        <div className="text-xs text-gray-400">
                          Total paint: {totalWeight != null ? `${totalWeight.toFixed(2)} g` : '—'}
                        </div>
                        <div className="text-xs text-gray-400">
                          Target hex: {paletteColor?.hex?.toUpperCase() ?? '—'}
                        </div>
                      </div>
                    )
                  })()}
                  <div className="text-sm text-gray-300">{formatRecipe(recipeData, getTotalWeightGrams(selectedLayerColor.paletteIndex!) ?? null)}</div>
                </div>
              )
            })()}
          </div>
        </div>
      )}
    </div>
  )
}

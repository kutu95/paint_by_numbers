'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import type { SessionData } from './types'
import { API_BASE_URL } from '@/lib/config'
import { getProjectBySessionId } from '@/lib/projects'

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

function formatRecipe(recipeData: any): string {
  if (!recipeData.recipe) return recipeData.error || 'No recipe available'
  const recipe = recipeData.recipe
  if (recipeData.type === 'chatgpt' || recipe.type === 'chatgpt') {
    if (recipe.ingredients && Array.isArray(recipe.ingredients) && recipe.ingredients.length > 0) {
      const parts = recipe.ingredients
        .map((ing: any) => ing?.paint_name != null && (ing.percentage != null ? `${ing.paint_name} ${Number(ing.percentage).toFixed(2)}%` : ing.paint_name))
        .filter(Boolean)
      if (parts.length > 0) return parts.join(' + ')
    }
    return recipe.instructions || 'Recipe instructions from ChatGPT'
  }
  const warning = recipe.uncalibrated ? ' (Estimated - not calibrated) ' : ''
  if (recipeData.type === 'one_pigment') return `${warning}White ${(recipe.white_ratio * 100).toFixed(1)}% + ${recipe.pigment_id} ${(recipe.pigment_ratio * 100).toFixed(1)}%`
  if (recipeData.type === 'two_pigment') return `${warning}White ${(recipe.white_ratio * 100).toFixed(1)}% + ${recipe.pigment1_id} ${(recipe.pigment1_ratio * 100).toFixed(1)}% + ${recipe.pigment2_id} ${(recipe.pigment2_ratio * 100).toFixed(1)}%`
  if (['three_pigment', 'four_pigment', 'multi_pigment'].includes(recipeData.type) && recipe.pigment_ids?.length) {
    const parts = recipe.pigment_ids.map((id: string, idx: number) => `${id} ${((recipe.pigment_ratios || [])[idx] * 100).toFixed(1)}%`)
    return `${warning}White ${(recipe.white_ratio * 100).toFixed(1)}% + ${parts.join(' + ')}`
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

  const project = typeof window !== 'undefined' ? getProjectBySessionId(sessionId) : null
  const recipeMargin = typeof window !== 'undefined' ? (parseFloat(localStorage.getItem('layerpainter_recipe_margin') || '1.33') || 1.33) : 1.33
  const selectedGroupInfo = libraryGroups.find((g) => g.group === selectedLibraryGroup)
  // Library coverage is stored as g/cm² (e.g. 0.008). Weight in grams = (coverage%/100) × area_cm² × coverage_g_per_cm² × margin.
  const libraryCoverageGPerCm2 = selectedGroupInfo?.coverage_mg_per_cm2 != null && selectedGroupInfo.coverage_mg_per_cm2 > 0 ? selectedGroupInfo.coverage_mg_per_cm2 : null

  function getTotalWeightGrams(paletteIndex: number): number | null {
    if (!project || libraryCoverageGPerCm2 == null || libraryCoverageGPerCm2 <= 0) return null
    const paletteColor = sessionData.palette.find((p) => p.index === paletteIndex)
    if (!paletteColor) return null
    const areaCm2 = project.canvasWidthCm * project.canvasHeightCm
    if (areaCm2 <= 0) return null
    return (paletteColor.coverage / 100) * areaCm2 * libraryCoverageGPerCm2 * recipeMargin
  }

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined') return
    fetch(`${API_BASE_URL}/api/paint/library/groups`)
      .then((res) => res.json())
      .then((data) => {
        const groups = data.groups || []
        setLibraryGroups(groups)
        setLibraryGroupsLoaded(true)
        if (groups.length > 0) {
          const lib = project?.libraryGroup && groups.some((g: any) => g.group === project.libraryGroup)
            ? project.libraryGroup
            : groups.find((g: any) => g.calibrated_count > 0)?.group ?? groups[0].group
          setSelectedLibraryGroup(lib)
        }
      })
      .catch(() => setLibraryGroupsLoaded(true))
  }, [sessionId, project?.libraryGroup])

  const handleGenerateRecipes = async (forceRegenerate: boolean = false) => {
    setLoadingRecipes(true)
    try {
      const formData = new FormData()
      formData.append('palette', JSON.stringify(sessionData.palette.map((c) => ({ index: c.index, hex: c.hex }))))
      formData.append('library_group', selectedLibraryGroup)
      if (forceRegenerate) formData.append('force_regenerate', 'true')
      const response = await fetch(`${API_BASE_URL}/api/paint/recipes/from-palette`, { method: 'POST', body: formData })
      if (!response.ok) throw new Error('Failed to generate recipes')
      const data = await response.json()
      setRecipes(data.recipes || [])
    } catch (e) {
      console.error(e)
      alert('Failed to generate recipes')
    } finally {
      setLoadingRecipes(false)
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

  const layers = sessionData.layers as LayerWithSource[]

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-3">
        <Link href={`/upload?edit=${sessionId}&returnTo=home`} className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded">
          Edit image & settings
        </Link>
        <Link href="/paints" className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded">
          Manage Paints
        </Link>
      </div>

      {sessionData.original_url && (
        <div>
          <h2 className="text-xl font-bold mb-2">Original image</h2>
          <img src={`${API_BASE_URL}${sessionData.original_url}`} alt="Original" className="max-w-md rounded border border-gray-600" />
        </div>
      )}

      {sessionData.quantized_preview_url && (
        <div>
          <h2 className="text-2xl font-bold mb-4">Quantized Preview</h2>
          <img src={`${API_BASE_URL}${sessionData.quantized_preview_url}`} alt="Quantized" className="max-w-full rounded" />
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
              {(() => {
                const w = getTotalWeightGrams(color.index)
                return w != null ? <div className="text-xs text-gray-500">{w.toFixed(2)} g</div> : null
              })()}
            </div>
          ))}
        </div>

        <div className="mb-4">
          <label className="block text-sm font-semibold mb-2">Paint library</label>
          {libraryGroupsLoaded && libraryGroups.length > 0 ? (
            <select
              value={selectedLibraryGroup}
              onChange={(e) => setSelectedLibraryGroup(e.target.value)}
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

        <div className="flex gap-2 mb-4">
          <button onClick={() => handleGenerateRecipes(false)} disabled={loadingRecipes} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded disabled:opacity-50">
            {loadingRecipes ? 'Generating…' : 'Generate Recipes'}
          </button>
          <button onClick={() => handleGenerateRecipes(true)} disabled={loadingRecipes} className="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded disabled:opacity-50">
            Force Regenerate
          </button>
        </div>

        {recipes.length > 0 && (
          <div className="space-y-3 mt-4">
            <h3 className="text-xl font-bold">Mixing Recipes</h3>
            {recipes.map((recipeData: any) => {
              const color = sessionData.palette.find((p) => p.index === recipeData.palette_index)
              if (!color) return null
              const recipe = recipeData.recipe
              const errorInfo = recipe && recipeData.type !== 'chatgpt' && recipe.error != null ? getErrorLevel(recipe.error) : null
              return (
                <div key={recipeData.palette_index} className="flex items-center gap-4 p-4 bg-gray-700 rounded">
                  <div className="w-16 h-16 rounded border border-gray-600 flex-shrink-0" style={{ backgroundColor: color.hex }} />
                  <div className="flex-1">
                    <div className="font-bold">Palette Color {recipeData.palette_index}</div>
                    <div className="text-sm text-gray-300">{formatRecipe(recipeData)}</div>
                    {recipe && (recipeData.type === 'chatgpt' || recipe.type === 'chatgpt') && recipe.ingredients && (
                      <div className="text-xs text-gray-400 mt-2 space-y-1">
                        {recipe.mixing_strategy && <div><strong>Strategy:</strong> {recipe.mixing_strategy}</div>}
                        {recipe.expected_result && <div><strong>Expected:</strong> {recipe.expected_result}</div>}
                      </div>
                    )}
                    {recipe && recipeData.type !== 'chatgpt' && recipe.type !== 'chatgpt' && errorInfo && (
                      <span className="inline-block mt-1 px-2 py-0.5 rounded text-xs" style={{ backgroundColor: errorInfo.color === 'green' ? '#16a34a' : errorInfo.color === 'yellow' ? '#ca8a04' : '#dc2626' }}>
                        {recipe.error?.toFixed(2)} ΔE – {errorInfo.level}
                      </span>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
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
                  src={`${API_BASE_URL}${layer.mask_pure_url ?? `/api/sessions/${sessionId}/layer_${layer.layer_index}_pure_mask.png`}`}
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
                  <div className="text-sm text-gray-300">{formatRecipe(recipeData)}</div>
                </div>
              )
            })()}
          </div>
        </div>
      )}
    </div>
  )
}

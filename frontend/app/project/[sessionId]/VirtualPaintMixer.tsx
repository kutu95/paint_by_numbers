'use client'

import { useState, useEffect, useMemo } from 'react'
import type { SessionData } from './types'
import { API_BASE_URL } from '@/lib/config'

interface Paint {
  id: string
  name: string
  type?: string
  hex_approx?: string
  hex?: string
}

/** Recipe item as stored in SessionResultsContent (palette_index, recipe with ingredients). */
interface RecipeItem {
  palette_index: number
  recipe?: {
    ingredients?: Array<{ paint_id?: string; paint_name?: string; percentage?: number }>
    white_ratio?: number
    pigment_id?: string
    pigment_ratio?: number
    pigment1_id?: string
    pigment2_id?: string
    pigment1_ratio?: number
    pigment2_ratio?: number
    pigment_ids?: string[]
    pigment_ratios?: number[]
    error?: number
    predicted_hex?: string
  }
}

interface VirtualPaintMixerProps {
  sessionData: SessionData
  selectedLibraryGroup: string
  /** Recipes from session (for applying palette recipe to sliders when a palette colour is selected). */
  recipes?: RecipeItem[]
}

function hexFromRgb(r: number, g: number, b: number): string {
  const R = Math.max(0, Math.min(255, Math.round(r)))
  const G = Math.max(0, Math.min(255, Math.round(g)))
  const B = Math.max(0, Math.min(255, Math.round(b)))
  return `#${R.toString(16).padStart(2, '0')}${G.toString(16).padStart(2, '0')}${B.toString(16).padStart(2, '0')}`
}

/** Linear blend of paints by relative amounts (0–10). Total is summed; each paint's percentage = value / total. */
function mixHexFromSliders(
  paints: Paint[],
  sliderValues: Record<string, number>
): string {
  let total = 0
  for (const p of paints) {
    total += sliderValues[p.id] ?? 0
  }
  if (total <= 0) return '#808080'
  let r = 0, g = 0, b = 0
  for (const p of paints) {
    const v = sliderValues[p.id] ?? 0
    if (v <= 0 || !p.hex_approx) continue
    const hex = p.hex_approx.replace(/^#/, '')
    const pr = parseInt(hex.slice(0, 2), 16)
    const pg = parseInt(hex.slice(2, 4), 16)
    const pb = parseInt(hex.slice(4, 6), 16)
    const pct = v / total
    r += pct * pr
    g += pct * pg
    b += pct * pb
  }
  return hexFromRgb(r, g, b)
}

/** Normalize to #RRGGBB or return null if invalid. */
function parseHex(input: string): string | null {
  const s = (input || '').trim().replace(/^#/, '')
  if (!/^[0-9a-fA-F]{6}$/.test(s)) return null
  return '#' + s.toUpperCase()
}

/** Build slider values (0–10) from recipe ingredients so proportions match. Paints not in recipe get 0. */
function sliderValuesFromRecipe(
  recipe: RecipeItem['recipe'],
  paintIds: string[]
): Record<string, number> | null {
  if (!recipe) return null
  const out: Record<string, number> = {}
  for (const id of paintIds) {
    out[id] = 0
  }
  const ingredients = recipe.ingredients
  if (Array.isArray(ingredients) && ingredients.length > 0) {
    for (const ing of ingredients) {
      const id = ing.paint_id ?? ing.paint_name
      const pct = ing.percentage
      if (id != null && typeof pct === 'number' && pct >= 0) {
        out[id] = (pct / 100) * 10
      }
    }
    return out
  }
  const whiteRatio = recipe.white_ratio
  const whiteId = 'white'
  if (typeof whiteRatio === 'number' && whiteRatio > 0 && paintIds.includes(whiteId)) {
    out[whiteId] = (whiteRatio * 10)
  }
  if (recipe.pigment_id != null && typeof recipe.pigment_ratio === 'number') {
    out[recipe.pigment_id] = (recipe.pigment_ratio * 10)
  }
  if (recipe.pigment1_id != null && typeof recipe.pigment1_ratio === 'number') {
    out[recipe.pigment1_id] = (recipe.pigment1_ratio * 10)
  }
  if (recipe.pigment2_id != null && typeof recipe.pigment2_ratio === 'number') {
    out[recipe.pigment2_id] = (recipe.pigment2_ratio * 10)
  }
  if (Array.isArray(recipe.pigment_ids) && Array.isArray(recipe.pigment_ratios)) {
    recipe.pigment_ids.forEach((id, i) => {
      const r = recipe.pigment_ratios?.[i]
      if (id && typeof r === 'number') out[id] = (r * 10)
    })
  }
  return out
}

export function VirtualPaintMixer({ sessionData, selectedLibraryGroup, recipes = [] }: VirtualPaintMixerProps) {
  const [paints, setPaints] = useState<Paint[]>([])
  const [loading, setLoading] = useState(true)
  const [sliderValues, setSliderValues] = useState<Record<string, number>>({})
  const [collapsed, setCollapsed] = useState(true)
  const [selectedPaletteIndex, setSelectedPaletteIndex] = useState<number | null>(null)
  const [customCompareHex, setCustomCompareHex] = useState('')

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    fetch(`${API_BASE_URL}/api/paint/library?group=${encodeURIComponent(selectedLibraryGroup)}`, { cache: 'no-store' })
      .then((res) => res.ok ? res.json() : Promise.reject(new Error('Failed to load library')))
      .then((data) => {
        if (cancelled) return
        const list = (data.paints || []).filter((p: Paint) => p && (p.hex_approx || p.hex))
        const withHex = list.map((p: Paint) => ({ ...p, hex_approx: p.hex_approx || p.hex }))
        setPaints(withHex)
        setSliderValues((prev) => {
          const next = { ...prev }
          for (const p of withHex) {
            if (next[p.id] === undefined) next[p.id] = 0
          }
          return next
        })
      })
      .catch(() => {
        if (!cancelled) setPaints([])
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [selectedLibraryGroup])

  useEffect(() => {
    if (selectedPaletteIndex == null || paints.length === 0) return
    const recipeItem = recipes.find((r) => r.palette_index === selectedPaletteIndex)
    const next = sliderValuesFromRecipe(recipeItem?.recipe, paints.map((p) => p.id))
    if (next) {
      setSliderValues(next)
    }
  }, [selectedPaletteIndex, recipes, paints])

  const mixHexLinear = useMemo(() => mixHexFromSliders(paints, sliderValues), [paints, sliderValues])
  const selectedPaletteColor = selectedPaletteIndex != null ? sessionData.palette.find((p) => p.index === selectedPaletteIndex) : null
  const customHexValid = parseHex(customCompareHex)
  const compareTargetHex = customHexValid ?? selectedPaletteColor?.hex ?? null
  const [predictResult, setPredictResult] = useState<{ predicted_hex: string; delta_e: number } | null>(null)
  const recipeForSelected = selectedPaletteIndex != null ? recipes.find((r) => r.palette_index === selectedPaletteIndex) : null
  const recipeError = recipeForSelected?.recipe && typeof recipeForSelected.recipe.error === 'number' ? recipeForSelected.recipe.error : null

  // Use backend calibration-based prediction so blend color and ΔE match recipe generation (same method).
  useEffect(() => {
    if (!compareTargetHex || paints.length === 0) {
      setPredictResult(null)
      return
    }
    const components = paints
      .filter((p) => (sliderValues[p.id] ?? 0) > 0)
      .map((p) => ({ paint_id: p.id, ratio: sliderValues[p.id] ?? 0 }))
    if (components.length === 0) {
      setPredictResult(null)
      return
    }
    let cancelled = false
    const norm = (h: string) => h.startsWith('#') ? h : '#' + h
    const url = `${API_BASE_URL}/api/paint/recipes/predict-mix`
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        library_group: selectedLibraryGroup,
        target_hex: norm(compareTargetHex),
        components,
      }),
      cache: 'no-store',
    })
      .then((res) => res.ok ? res.json() : Promise.reject(new Error('Failed to predict mix')))
      .then((data: { predicted_hex?: string; delta_e?: number }) => {
        if (!cancelled && typeof data?.predicted_hex === 'string' && typeof data?.delta_e === 'number') {
          setPredictResult({ predicted_hex: data.predicted_hex, delta_e: data.delta_e })
        } else {
          setPredictResult(null)
        }
      })
      .catch(() => {
        if (!cancelled) setPredictResult(null)
      })
    return () => { cancelled = true }
  }, [compareTargetHex, selectedLibraryGroup, paints, sliderValues])

  const blendDisplayHex = predictResult?.predicted_hex ?? mixHexLinear
  const mixDeltaE = predictResult?.delta_e ?? null

  const setSlider = (paintId: string, value: number) => {
    setSliderValues((prev) => ({ ...prev, [paintId]: Math.max(0, Math.min(10, value)) }))
  }

  return (
    <div className="border border-gray-600 rounded-lg overflow-hidden bg-gray-800/80">
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        className="w-full flex items-center justify-between px-4 py-2 text-left font-semibold bg-gray-700/80 hover:bg-gray-700"
      >
        <span>Virtual paint mixer</span>
        <span className="text-gray-400">{collapsed ? '▼' : '▲'}</span>
      </button>
      {!collapsed && (
        <div className="p-4 space-y-4">
          {/* Mix preview + palette selector */}
          <div className="flex flex-wrap items-end gap-4">
            <div className="flex flex-col items-center gap-1">
              <div
                className="w-24 h-24 rounded border-2 border-gray-600 flex-shrink-0"
                style={{ backgroundColor: blendDisplayHex }}
                title={blendDisplayHex}
              />
              <span className="text-xs font-mono text-gray-400">{blendDisplayHex.toUpperCase()}</span>
            </div>
            <div className="flex flex-col gap-2 min-w-[200px]">
              <label className="text-sm font-medium text-gray-300">Compare to palette colour</label>
              <select
                value={selectedPaletteIndex ?? ''}
                onChange={(e) => setSelectedPaletteIndex(e.target.value === '' ? null : Number(e.target.value))}
                className="px-3 py-2 bg-gray-700 rounded border border-gray-600 text-sm"
              >
                <option value="">— Select —</option>
                {sessionData.palette.map((c) => (
                  <option key={c.index} value={c.index}>
                    {c.index}: {c.hex.toUpperCase()} ({c.coverage.toFixed(1)}%)
                  </option>
                ))}
              </select>
              <label className="text-sm font-medium text-gray-300 mt-1">Or custom hex</label>
              <input
                type="text"
                value={customCompareHex}
                onChange={(e) => setCustomCompareHex(e.target.value)}
                placeholder="#68A616 or 68A616"
                className="px-3 py-2 bg-gray-700 rounded border border-gray-600 text-sm font-mono placeholder:text-gray-500"
              />
              {compareTargetHex && (
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-24 h-24 rounded border-2 border-gray-600 flex-shrink-0"
                      style={{ backgroundColor: compareTargetHex }}
                    />
                    <div className="flex flex-col gap-0.5">
                      <span className="text-sm text-gray-300">
                        Mix ΔE = <strong>{mixDeltaE != null ? mixDeltaE.toFixed(2) : '…'}</strong>
                        {customHexValid ? ' (vs custom)' : ' (vs palette)'}
                      </span>
                      {!customHexValid && recipeError != null && (
                        <span className="text-xs text-gray-400">
                          Recipe: {recipeError.toFixed(2)} ΔE
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Vertical sliders: 0–10, one per paint */}
          {loading ? (
            <div className="text-sm text-gray-400">Loading paints…</div>
          ) : paints.length === 0 ? (
            <div className="text-sm text-gray-400">No paints in this library.</div>
          ) : (
            <div className="flex flex-wrap gap-3 items-end">
              {paints.map((p) => (
                <div key={p.id} className="flex flex-col items-center gap-1">
                  <div
                    className="w-6 h-6 rounded border border-gray-600 flex-shrink-0"
                    style={{ backgroundColor: p.hex_approx || '#888' }}
                    title={p.hex_approx}
                  />
                  <div className="h-24 w-6 flex items-center justify-center" style={{ transform: 'rotate(-90deg)' }}>
                    <input
                      type="range"
                      min={0}
                      max={10}
                      step={0.5}
                      value={sliderValues[p.id] ?? 0}
                      onChange={(e) => setSlider(p.id, Number(e.target.value))}
                      className="w-24 h-4 accent-gray-500"
                    />
                  </div>
                  <span className="text-[10px] text-gray-400 max-w-[4rem] truncate" title={p.name}>
                    {p.name}
                  </span>
                  <span className="text-[10px] text-gray-500">{(sliderValues[p.id] ?? 0).toFixed(1)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

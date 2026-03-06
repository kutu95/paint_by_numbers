'use client'

import { useState, useEffect, useMemo } from 'react'
import type { SessionData } from './types'
import { API_BASE_URL } from '@/lib/config'
import { hexToLab, deltaE } from '@/lib/colorUtils'

interface Paint {
  id: string
  name: string
  type?: string
  hex_approx?: string
  hex?: string
}

interface VirtualPaintMixerProps {
  sessionData: SessionData
  selectedLibraryGroup: string
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

export function VirtualPaintMixer({ sessionData, selectedLibraryGroup }: VirtualPaintMixerProps) {
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

  const mixHex = useMemo(() => mixHexFromSliders(paints, sliderValues), [paints, sliderValues])
  const selectedPaletteColor = selectedPaletteIndex != null ? sessionData.palette.find((p) => p.index === selectedPaletteIndex) : null
  const customHexValid = parseHex(customCompareHex)
  const compareTargetHex = customHexValid ?? selectedPaletteColor?.hex ?? null
  const deltaEValue = useMemo(() => {
    if (!compareTargetHex) return null
    const lab1 = hexToLab(mixHex)
    const lab2 = hexToLab(compareTargetHex)
    if (!lab1 || !lab2) return null
    return deltaE(lab1, lab2)
  }, [mixHex, compareTargetHex])

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
                style={{ backgroundColor: mixHex }}
                title={mixHex}
              />
              <span className="text-xs font-mono text-gray-400">{mixHex.toUpperCase()}</span>
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
              {compareTargetHex && deltaEValue != null && (
                <div className="flex items-center gap-2">
                  <div
                    className="w-8 h-8 rounded border border-gray-600 flex-shrink-0"
                    style={{ backgroundColor: compareTargetHex }}
                  />
                  <span className="text-sm text-gray-300">
                    ΔE = <strong>{deltaEValue.toFixed(2)}</strong>
                    {customHexValid ? ' (vs custom)' : ' (vs palette)'}
                  </span>
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

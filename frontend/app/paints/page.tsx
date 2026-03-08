'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { API_BASE_URL } from '@/lib/config'

function formatRecipe(recipeData: any): string {
  if (!recipeData?.recipe) return recipeData?.error || 'No recipe available'
  const recipe = recipeData.recipe

  if (recipe.ingredients && Array.isArray(recipe.ingredients) && recipe.ingredients.length > 0) {
    const parts = recipe.ingredients
      .map((ing: any) => {
        if (ing?.paint_name == null) return null
        const pct = ing.percentage != null ? Number(ing.percentage) : 0
        const gramsText = ing.grams != null ? ` (${Number(ing.grams).toFixed(2)} g)` : ''
        return `${ing.paint_name} ${pct.toFixed(2)}%${gramsText}`
      })
      .filter(Boolean)
    if (parts.length > 0) return parts.join(' + ')
  }
  if (recipe.instructions) return recipe.instructions
  return 'No recipe available'
}

const CHART_WIDTH = 420
const CHART_HEIGHT = 180
const PAD = { left: 36, right: 12, top: 12, bottom: 28 }

function CalibrationResultsView({ data }: { data: CalibrationData }) {
  const samples = [...(data.samples || [])].sort((a, b) => a.ratio - b.ratio)
  if (samples.length === 0) return <p className="text-gray-400">No samples.</p>

  const ratios = samples.map((s) => s.ratio)
  const L = samples.map((s) => s.lab[0])
  const a = samples.map((s) => s.lab[1])
  const b = samples.map((s) => s.lab[2])
  const minRatio = Math.min(...ratios)
  const maxRatio = Math.max(...ratios)
  const minL = Math.min(0, ...L)
  const maxL = Math.max(100, ...L)
  const abExtent = Math.max(1, ...a.map(Math.abs), ...b.map(Math.abs), 50)
  const minAb = -abExtent
  const maxAb = abExtent

  // X: logarithmic scale, reversed (100% left, most diluted right); equal spacing per dilution step
  const logMin = Math.log2(Math.max(minRatio, 1e-6))
  const logMax = Math.log2(maxRatio)
  const logRange = logMax - logMin || 1
  const innerW = CHART_WIDTH - PAD.left - PAD.right
  const toX = (ratio: number) => {
    const t = (logMax - Math.log2(Math.max(ratio, 1e-6))) / logRange
    return PAD.left + t * innerW
  }
  const toYL = (v: number) =>
    PAD.top + (1 - (v - minL) / (maxL - minL || 1)) * (CHART_HEIGHT - PAD.top - PAD.bottom)
  const toYAb = (v: number) =>
    PAD.top + (1 - (v - minAb) / (maxAb - minAb || 1)) * (CHART_HEIGHT - PAD.top - PAD.bottom)

  const pathL = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYL(s.lab[0])}`).join(' ')
  const pathA = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYAb(s.lab[1])}`).join(' ')
  const pathB = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYAb(s.lab[2])}`).join(' ')

  // Red / green from a* (red = +a*, green = -a*); yellow / blue from b* (yellow = +b*, blue = -b*)
  const redChroma = a.map((v) => Math.max(0, v))
  const greenChroma = a.map((v) => Math.max(0, -v))
  const yellowChroma = b.map((v) => Math.max(0, v))
  const blueChroma = b.map((v) => Math.max(0, -v))
  const maxChroma = Math.max(1, ...redChroma, ...greenChroma, ...yellowChroma, ...blueChroma)
  const toYChroma = (v: number) =>
    PAD.top + (1 - v / maxChroma) * (CHART_HEIGHT - PAD.top - PAD.bottom)
  const pathRed = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYChroma(redChroma[i])}`).join(' ')
  const pathGreen = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYChroma(greenChroma[i])}`).join(' ')
  const pathYellow = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYChroma(yellowChroma[i])}`).join(' ')
  const pathBlue = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYChroma(blueChroma[i])}`).join(' ')

  // RGB channels (0–255) vs dilution
  const R = samples.map((s) => s.rgb[0])
  const G = samples.map((s) => s.rgb[1])
  const B = samples.map((s) => s.rgb[2])
  const minRgb = Math.min(0, ...R, ...G, ...B)
  const maxRgb = Math.max(255, ...R, ...G, ...B)
  const rangeRgb = maxRgb - minRgb || 1
  const toYRgb = (v: number) =>
    PAD.top + (1 - (v - minRgb) / rangeRgb) * (CHART_HEIGHT - PAD.top - PAD.bottom)
  const pathR = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYRgb(s.rgb[0])}`).join(' ')
  const pathG = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYRgb(s.rgb[1])}`).join(' ')
  const pathB_Rgb = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYRgb(s.rgb[2])}`).join(' ')

  // C* (chroma) and h° (hue) from Lab: C* = sqrt(a*² + b*²), h° = atan2(b*, a*) in degrees [0, 360)
  const C = samples.map((s) => Math.sqrt(s.lab[1] * s.lab[1] + s.lab[2] * s.lab[2]))
  const hDeg = samples.map((s) => {
    const deg = (Math.atan2(s.lab[2], s.lab[1]) * 180) / Math.PI
    return deg < 0 ? deg + 360 : deg
  })
  const maxC = Math.max(1, ...C)
  const toYC = (v: number) =>
    PAD.top + (1 - v / maxC) * (CHART_HEIGHT - PAD.top - PAD.bottom)
  const pathC = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYC(C[i])}`).join(' ')
  const minH = Math.min(0, ...hDeg)
  const maxH = Math.max(360, ...hDeg)
  const rangeH = maxH - minH || 1
  const toYH = (v: number) =>
    PAD.top + (1 - (v - minH) / rangeH) * (CHART_HEIGHT - PAD.top - PAD.bottom)
  const pathH = samples.map((s, i) => `${i === 0 ? 'M' : 'L'} ${toX(s.ratio)} ${toYH(hDeg[i])}`).join(' ')

  // a*b* plane: map a*, b* to SVG coords (same abExtent as chroma chart)
  const innerH = CHART_HEIGHT - PAD.top - PAD.bottom
  const toXAb = (aVal: number) => PAD.left + ((aVal - minAb) / (maxAb - minAb)) * innerW
  const toYAbPlane = (bVal: number) => CHART_HEIGHT - PAD.bottom - ((bVal - minAb) / (maxAb - minAb)) * innerH

  const rgbToHex = (r: number, g: number, blue: number) =>
    '#' + [r, g, blue].map((x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, '0')).join('')

  return (
    <div className="space-y-4">
      {data.created_at && (
        <p className="text-sm text-gray-400">
          Calibrated: {new Date(data.created_at).toLocaleString()}
        </p>
      )}
      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-2">Samples ({samples.length})</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-400 border-b border-gray-600">
                <th className="py-1 pr-3">Ratio</th>
                <th className="py-1 pr-2">Color</th>
                <th className="py-1 pr-2">Hex</th>
                <th className="py-1 pr-2">L</th>
                <th className="py-1 pr-2">a*</th>
                <th className="py-1">b*</th>
              </tr>
            </thead>
            <tbody>
              {samples.map((s, i) => (
                <tr key={i} className="border-b border-gray-700">
                  {(() => {
                    const hex = rgbToHex(s.rgb[0], s.rgb[1], s.rgb[2])
                    return (
                      <>
                  <td className="py-1.5 pr-3">{(s.ratio * 100).toFixed(1)}%</td>
                  <td className="py-1.5 pr-2">
                    <span
                      className="inline-block w-6 h-6 rounded border border-gray-600"
                      style={{ backgroundColor: hex }}
                      title={hex}
                    />
                  </td>
                  <td className="py-1.5 pr-2 font-mono text-xs">{hex.toUpperCase()}</td>
                  <td className="py-1.5 pr-2">{s.lab[0].toFixed(1)}</td>
                  <td className="py-1.5 pr-2">{s.lab[1].toFixed(1)}</td>
                  <td className="py-1.5">{s.lab[2].toFixed(1)}</td>
                      </>
                    )
                  })()}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-1">Luminance (L) vs dilution</h4>
        <p className="text-xs text-gray-500 mb-1">L: 0 (dark) → 100 (light)</p>
        <svg width={CHART_WIDTH} height={CHART_HEIGHT} className="overflow-visible">
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <line x1={PAD.left} y1={CHART_HEIGHT - PAD.bottom} x2={CHART_WIDTH - PAD.right} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <text x={PAD.left - 8} y={PAD.top + 4} className="fill-gray-500 text-[10px]">100</text>
          <text x={PAD.left - 8} y={CHART_HEIGHT - PAD.bottom + 4} className="fill-gray-500 text-[10px]">0</text>
          {samples.map((s, i) => (
            <g key={i}>
              <line x1={toX(s.ratio)} y1={CHART_HEIGHT - PAD.bottom} x2={toX(s.ratio)} y2={CHART_HEIGHT - PAD.bottom + 4} stroke="#4b5563" strokeWidth="1" />
              <text x={toX(s.ratio)} y={CHART_HEIGHT - 4} textAnchor="middle" className="fill-gray-500 text-[10px]">{(s.ratio * 100).toFixed(s.ratio >= 0.1 ? 0 : 1)}%</text>
            </g>
          ))}
          <path d={pathL} fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-1">Chroma (a*, b*) vs dilution</h4>
        <p className="text-xs text-gray-500 mb-1">a* (green–red), b* (blue–yellow)</p>
        <svg width={CHART_WIDTH} height={CHART_HEIGHT} className="overflow-visible">
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <line x1={PAD.left} y1={CHART_HEIGHT - PAD.bottom} x2={CHART_WIDTH - PAD.right} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <text x={PAD.left - 8} y={PAD.top + 4} className="fill-gray-500 text-[10px]">{maxAb}</text>
          <text x={PAD.left - 8} y={CHART_HEIGHT - PAD.bottom + 4} className="fill-gray-500 text-[10px]">{minAb}</text>
          {samples.map((s, i) => (
            <g key={i}>
              <line x1={toX(s.ratio)} y1={CHART_HEIGHT - PAD.bottom} x2={toX(s.ratio)} y2={CHART_HEIGHT - PAD.bottom + 4} stroke="#4b5563" strokeWidth="1" />
              <text x={toX(s.ratio)} y={CHART_HEIGHT - 4} textAnchor="middle" className="fill-gray-500 text-[10px]">{(s.ratio * 100).toFixed(s.ratio >= 0.1 ? 0 : 1)}%</text>
            </g>
          ))}
          <path d={pathA} fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathB} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <text x={CHART_WIDTH - 80} y={PAD.top + 10} className="fill-[#ef4444] text-[10px]">a*</text>
          <text x={CHART_WIDTH - 50} y={PAD.top + 10} className="fill-[#3b82f6] text-[10px]">b*</text>
        </svg>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-1">Red / Green / Blue / Yellow chroma vs dilution</h4>
        <p className="text-xs text-gray-500 mb-1">From Lab: red = +a*, green = −a*, yellow = +b*, blue = −b*</p>
        <svg width={CHART_WIDTH} height={CHART_HEIGHT} className="overflow-visible">
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <line x1={PAD.left} y1={CHART_HEIGHT - PAD.bottom} x2={CHART_WIDTH - PAD.right} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <text x={PAD.left - 8} y={PAD.top + 4} className="fill-gray-500 text-[10px]">{maxChroma.toFixed(0)}</text>
          <text x={PAD.left - 8} y={CHART_HEIGHT - PAD.bottom + 4} className="fill-gray-500 text-[10px]">0</text>
          {samples.map((s, i) => (
            <g key={i}>
              <line x1={toX(s.ratio)} y1={CHART_HEIGHT - PAD.bottom} x2={toX(s.ratio)} y2={CHART_HEIGHT - PAD.bottom + 4} stroke="#4b5563" strokeWidth="1" />
              <text x={toX(s.ratio)} y={CHART_HEIGHT - 4} textAnchor="middle" className="fill-gray-500 text-[10px]">{(s.ratio * 100).toFixed(s.ratio >= 0.1 ? 0 : 1)}%</text>
            </g>
          ))}
          <path d={pathRed} fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathGreen} fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathYellow} fill="none" stroke="#eab308" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathBlue} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <text x={CHART_WIDTH - 120} y={PAD.top + 8} className="fill-[#ef4444] text-[10px]">red</text>
          <text x={CHART_WIDTH - 95} y={PAD.top + 8} className="fill-[#22c55e] text-[10px]">green</text>
          <text x={CHART_WIDTH - 68} y={PAD.top + 8} className="fill-[#eab308] text-[10px]">yellow</text>
          <text x={CHART_WIDTH - 42} y={PAD.top + 8} className="fill-[#3b82f6] text-[10px]">blue</text>
        </svg>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-1">RGB channels vs dilution</h4>
        <p className="text-xs text-gray-500 mb-1">R, G, B (0–255) by dilution</p>
        <svg width={CHART_WIDTH} height={CHART_HEIGHT} className="overflow-visible">
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <line x1={PAD.left} y1={CHART_HEIGHT - PAD.bottom} x2={CHART_WIDTH - PAD.right} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <text x={PAD.left - 8} y={PAD.top + 4} className="fill-gray-500 text-[10px]">{maxRgb}</text>
          <text x={PAD.left - 8} y={CHART_HEIGHT - PAD.bottom + 4} className="fill-gray-500 text-[10px]">{minRgb}</text>
          {samples.map((s, i) => (
            <g key={i}>
              <line x1={toX(s.ratio)} y1={CHART_HEIGHT - PAD.bottom} x2={toX(s.ratio)} y2={CHART_HEIGHT - PAD.bottom + 4} stroke="#4b5563" strokeWidth="1" />
              <text x={toX(s.ratio)} y={CHART_HEIGHT - 4} textAnchor="middle" className="fill-gray-500 text-[10px]">{(s.ratio * 100).toFixed(s.ratio >= 0.1 ? 0 : 1)}%</text>
            </g>
          ))}
          <path d={pathR} fill="none" stroke="#ef4444" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathG} fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <path d={pathB_Rgb} fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <text x={CHART_WIDTH - 70} y={PAD.top + 8} className="fill-[#ef4444] text-[10px]">R</text>
          <text x={CHART_WIDTH - 52} y={PAD.top + 8} className="fill-[#22c55e] text-[10px]">G</text>
          <text x={CHART_WIDTH - 34} y={PAD.top + 8} className="fill-[#3b82f6] text-[10px]">B</text>
        </svg>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-1">Chroma (C*) vs dilution</h4>
        <p className="text-xs text-gray-500 mb-1">C* = √(a*² + b*²); saturation</p>
        <svg width={CHART_WIDTH} height={CHART_HEIGHT} className="overflow-visible">
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <line x1={PAD.left} y1={CHART_HEIGHT - PAD.bottom} x2={CHART_WIDTH - PAD.right} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <text x={PAD.left - 8} y={PAD.top + 4} className="fill-gray-500 text-[10px]">{maxC.toFixed(0)}</text>
          <text x={PAD.left - 8} y={CHART_HEIGHT - PAD.bottom + 4} className="fill-gray-500 text-[10px]">0</text>
          {samples.map((s, i) => (
            <g key={i}>
              <line x1={toX(s.ratio)} y1={CHART_HEIGHT - PAD.bottom} x2={toX(s.ratio)} y2={CHART_HEIGHT - PAD.bottom + 4} stroke="#4b5563" strokeWidth="1" />
              <text x={toX(s.ratio)} y={CHART_HEIGHT - 4} textAnchor="middle" className="fill-gray-500 text-[10px]">{(s.ratio * 100).toFixed(s.ratio >= 0.1 ? 0 : 1)}%</text>
            </g>
          ))}
          <path d={pathC} fill="none" stroke="#a855f7" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-1">Hue (h°) vs dilution</h4>
        <p className="text-xs text-gray-500 mb-1">h° = atan2(b*, a*) in degrees (0–360)</p>
        <svg width={CHART_WIDTH} height={CHART_HEIGHT} className="overflow-visible">
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <line x1={PAD.left} y1={CHART_HEIGHT - PAD.bottom} x2={CHART_WIDTH - PAD.right} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <text x={PAD.left - 12} y={PAD.top + 4} className="fill-gray-500 text-[10px]">360</text>
          <text x={PAD.left - 12} y={CHART_HEIGHT - PAD.bottom + 4} className="fill-gray-500 text-[10px]">0</text>
          {samples.map((s, i) => (
            <g key={i}>
              <line x1={toX(s.ratio)} y1={CHART_HEIGHT - PAD.bottom} x2={toX(s.ratio)} y2={CHART_HEIGHT - PAD.bottom + 4} stroke="#4b5563" strokeWidth="1" />
              <text x={toX(s.ratio)} y={CHART_HEIGHT - 4} textAnchor="middle" className="fill-gray-500 text-[10px]">{(s.ratio * 100).toFixed(s.ratio >= 0.1 ? 0 : 1)}%</text>
            </g>
          ))}
          <path d={pathH} fill="none" stroke="#f97316" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      <div>
        <h4 className="text-sm font-medium text-gray-300 mb-1">a*b* plane (colour path)</h4>
        <p className="text-xs text-gray-500 mb-1">Path of tint in Lab a*b*; points coloured by sample, labelled by ratio</p>
        <svg width={CHART_WIDTH} height={CHART_HEIGHT} className="overflow-visible">
          <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <line x1={PAD.left} y1={CHART_HEIGHT - PAD.bottom} x2={CHART_WIDTH - PAD.right} y2={CHART_HEIGHT - PAD.bottom} stroke="#4b5563" strokeWidth="1" />
          <text x={PAD.left + innerW / 2 - 10} y={CHART_HEIGHT - 6} textAnchor="middle" className="fill-gray-500 text-[10px]">a*</text>
          <text x={PAD.left - 10} y={PAD.top + innerH / 2 + 4} textAnchor="middle" className="fill-gray-500 text-[10px]">b*</text>
          <text x={PAD.left} y={CHART_HEIGHT - PAD.bottom + 14} textAnchor="middle" className="fill-gray-500 text-[9px]">{minAb}</text>
          <text x={PAD.left + innerW} y={CHART_HEIGHT - PAD.bottom + 14} textAnchor="middle" className="fill-gray-500 text-[9px]">{maxAb}</text>
          <text x={PAD.left - 6} y={CHART_HEIGHT - PAD.bottom} textAnchor="end" className="fill-gray-500 text-[9px]">{minAb}</text>
          <text x={PAD.left - 6} y={PAD.top + 4} textAnchor="end" className="fill-gray-500 text-[9px]">{maxAb}</text>
          {samples.map((s, i) => {
            const hex = rgbToHex(s.rgb[0], s.rgb[1], s.rgb[2])
            const cx = toXAb(s.lab[1])
            const cy = toYAbPlane(s.lab[2])
            const label = s.ratio >= 0.1 ? `${(s.ratio * 100).toFixed(0)}%` : `${(s.ratio * 100).toFixed(1)}%`
            return (
              <g key={i}>
                <line
                  x1={i === 0 ? cx : toXAb(samples[i - 1].lab[1])}
                  y1={i === 0 ? cy : toYAbPlane(samples[i - 1].lab[2])}
                  x2={cx}
                  y2={cy}
                  stroke="#6b7280"
                  strokeWidth="1"
                  strokeDasharray="2,2"
                />
                <circle cx={cx} cy={cy} r={6} fill={hex} stroke="#374151" strokeWidth="1" />
                <text x={cx} y={cy - 10} textAnchor="middle" className="fill-gray-400 text-[9px]">{label}</text>
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}

interface Paint {
  id: string
  name: string
  type: string
  hex_approx: string
  notes: string
}

interface LibraryGroup {
  group: string
  paint_count: number
  calibrated_count: number
  name: string
  coverage_mg_per_cm2?: number | null
}

interface CalibrationSample {
  ratio: number
  rgb: number[]
  lab: number[]
}

interface CalibrationData {
  paint_id: string
  ratios: number[]
  samples: CalibrationSample[]
  reference_strip?: Record<string, { rgb: number[]; lab: number[] }>
  created_at?: string
  notes?: string
}

interface LibraryRecipeRow {
  hex: string
  last_modified?: string | null
  type?: string | null
  delta_e?: number | null
  ingredients: Array<{
    paint_id?: string
    paint_name?: string
    percentage?: number
  }>
}

export default function PaintsPage() {
  const router = useRouter()
  const [paints, setPaints] = useState<Paint[]>([])
  const [loading, setLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)
  const [editingPaint, setEditingPaint] = useState<Paint | null>(null)
  const [formData, setFormData] = useState({ name: '', hex_approx: '#000000', notes: '' })
  const [libraryCoverage, setLibraryCoverage] = useState<string>('')
  const [libraryGroups, setLibraryGroups] = useState<LibraryGroup[]>([])
  const [selectedGroup, setSelectedGroup] = useState<string>(() => {
    // Load last selected group from localStorage
    if (typeof window !== 'undefined') {
      return localStorage.getItem('lastSelectedPaintLibrary') || 'default'
    }
    return 'default'
  })
  const [showCreateGroup, setShowCreateGroup] = useState(false)
  const [newGroupName, setNewGroupName] = useState('')
  const [renamingGroup, setRenamingGroup] = useState<string | null>(null)
  const [renameGroupName, setRenameGroupName] = useState('')
  const [downloadingCalibrationExport, setDownloadingCalibrationExport] = useState(false)
  const [gamutL, setGamutL] = useState<number>(50)
  const [gamutLoading, setGamutLoading] = useState(false)
  const [gamutData, setGamutData] = useState<any | null>(null)
  const [selectedGamutCell, setSelectedGamutCell] = useState<any | null>(null)
  const gamutCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const [recipeRows, setRecipeRows] = useState<LibraryRecipeRow[]>([])
  const [recipeRowsLoading, setRecipeRowsLoading] = useState(false)
  const [recipeRowsPage, setRecipeRowsPage] = useState(1)
  const [recipeRowsTotalPages, setRecipeRowsTotalPages] = useState(1)
  const [recipeRowsTotal, setRecipeRowsTotal] = useState(0)
  const [calibrationPanelOpen, setCalibrationPanelOpen] = useState(false)
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null)
  const [calibrationLoading, setCalibrationLoading] = useState(false)
  const [calibrationError, setCalibrationError] = useState<string | null>(null)

  useEffect(() => {
    loadLibraryGroups()
  }, [])

  useEffect(() => {
    if (selectedGroup) {
      // Save selected group to localStorage
      if (typeof window !== 'undefined') {
        localStorage.setItem('lastSelectedPaintLibrary', selectedGroup)
      }
      loadPaints()
      setRecipeRowsPage(1)
    }
  }, [selectedGroup])

  useEffect(() => {
    if (!selectedGroup) return
    loadLibraryRecipes(recipeRowsPage)
  }, [selectedGroup, recipeRowsPage])

  useEffect(() => {
    if (!editingPaint) {
      setCalibrationPanelOpen(false)
      setCalibrationData(null)
      setCalibrationError(null)
      return
    }
    if (!calibrationPanelOpen) return
    setCalibrationLoading(true)
    setCalibrationError(null)
    fetch(`${API_BASE_URL}/api/paint/calibration/${editingPaint.id}?group=${encodeURIComponent(selectedGroup)}`)
      .then((res) => {
        if (!res.ok) {
          if (res.status === 404) throw new Error('No calibration data')
          throw new Error(res.statusText)
        }
        return res.json()
      })
      .then((data: CalibrationData) => setCalibrationData(data))
      .catch((err: Error) => setCalibrationError(err.message))
      .finally(() => setCalibrationLoading(false))
  }, [editingPaint, calibrationPanelOpen])

  const loadLibraryGroups = async () => {
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
      setLibraryGroups(data.groups || [])
      
      // If we have a saved group, verify it still exists, otherwise use first available
      if (data.groups && data.groups.length > 0) {
        const savedGroup = typeof window !== 'undefined' 
          ? localStorage.getItem('lastSelectedPaintLibrary') 
          : null
        const groupExists = savedGroup && data.groups.some((g: LibraryGroup) => g.group === savedGroup)
        
        if (groupExists && savedGroup) {
          setSelectedGroup(savedGroup)
        } else if (!selectedGroup || !data.groups.some((g: LibraryGroup) => g.group === selectedGroup)) {
          // Use first group if current selection doesn't exist
          setSelectedGroup(data.groups[0].group)
        }
      }
    } catch (error) {
      console.error(`Failed to load library groups from ${url}:`, error)
    }
  }

  const loadPaints = async () => {
    if (!selectedGroup) return
    setLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/library?group=${encodeURIComponent(selectedGroup)}`)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const data = await response.json()
      setPaints(data.paints || [])
      const cov = data.coverage_mg_per_cm2
      setLibraryCoverage(cov != null && cov > 0 ? String(cov) : '')
    } catch (error) {
      console.error('Failed to load paints:', error)
    } finally {
      setLoading(false)
    }
  }

  const saveLibraryCoverage = async () => {
    if (!selectedGroup) return
    const trimmed = libraryCoverage.trim()
    const val = trimmed === '' ? null : parseFloat(trimmed)
    const isValid = val != null && !Number.isNaN(val) && val > 0
    try {
      const form = new FormData()
      if (isValid) form.append('coverage_mg_per_cm2', String(val))
      const response = await fetch(`${API_BASE_URL}/api/paint/library/groups/${encodeURIComponent(selectedGroup)}/settings`, {
        method: 'PUT',
        body: form,
      })
      if (!response.ok) throw new Error('Failed to save')
      const data = await response.json()
      setLibraryGroups((prev) => prev.map((g) => g.group === selectedGroup ? { ...g, coverage_mg_per_cm2: data.coverage_mg_per_cm2 } : g))
    } catch (e) {
      console.error(e)
      alert('Failed to save library coverage')
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      const formDataObj = new FormData()
      formDataObj.append('name', formData.name)
      formDataObj.append('hex_approx', formData.hex_approx)
      formDataObj.append('notes', formData.notes)
      formDataObj.append('group', selectedGroup)

      if (editingPaint) {
        const response = await fetch(`${API_BASE_URL}/api/paint/library/${editingPaint.id}`, {
          method: 'PUT',
          body: formDataObj,
        })
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          throw new Error(errorData.detail || 'Failed to update paint')
        }
        const updatedPaint: Paint = await response.json()
        setPaints((prev) => prev.map((p) => (p.id === updatedPaint.id ? { ...p, ...updatedPaint } : p)))
      } else {
        const response = await fetch(`${API_BASE_URL}/api/paint/library`, {
          method: 'POST',
          body: formDataObj,
        })
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))
          const errorMessage = errorData.detail || 'Failed to add paint'
          if (response.status === 400 && errorMessage.includes('already exists')) {
            alert(`A paint with the name "${formData.name}" already exists in this library. Please use a different name or edit the existing paint.`)
          } else {
            throw new Error(errorMessage)
          }
          return
        }
      }

      setShowAddForm(false)
      setEditingPaint(null)
      setFormData({ name: '', hex_approx: '#000000', notes: '' })
      loadPaints()
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = error instanceof Error ? error.message : 'Failed to save paint'
      alert(errorMessage)
    }
  }

  const handleEdit = (paint: Paint) => {
    setEditingPaint(paint)
    setFormData({ name: paint.name, hex_approx: paint.hex_approx, notes: paint.notes })
    setShowAddForm(true)
  }

  const handleDelete = async (paintId: string) => {
    if (!confirm('Delete this paint? This will also delete its calibration data.')) return

    try {
      const response = await fetch(`${API_BASE_URL}/api/paint/library/${paintId}?group=${selectedGroup}`, {
        method: 'DELETE',
      })
      if (!response.ok) throw new Error('Failed to delete paint')
      loadPaints()
      loadLibraryGroups() // Refresh group info
    } catch (error) {
      console.error('Error:', error)
      alert('Failed to delete paint')
    }
  }

  const handleCreateGroup = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newGroupName.trim()) return

    try {
      const formData = new FormData()
      formData.append('name', newGroupName.trim())

      const response = await fetch(`${API_BASE_URL}/api/paint/library/groups`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to create library group')
      }

      setNewGroupName('')
      setShowCreateGroup(false)
      loadLibraryGroups()
      // Switch to the new group
      const data = await response.json()
      setSelectedGroup(data.group)
    } catch (error) {
      console.error('Error:', error)
      alert(error instanceof Error ? error.message : 'Failed to create library group')
    }
  }

  const handleRenameGroup = async (group: string, currentName: string) => {
    setRenamingGroup(group)
    setRenameGroupName(currentName)
  }

  const handleRenameGroupSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!renamingGroup || !renameGroupName.trim()) return

    try {
      const formData = new FormData()
      formData.append('name', renameGroupName.trim())

      const response = await fetch(`${API_BASE_URL}/api/paint/library/groups/${renamingGroup}`, {
        method: 'PUT',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Failed to rename library group')
      }

      setRenamingGroup(null)
      setRenameGroupName('')
      loadLibraryGroups()
    } catch (error) {
      console.error('Error:', error)
      alert(error instanceof Error ? error.message : 'Failed to rename library group')
    }
  }

  const handleDownloadCalibrationExport = async () => {
    if (!selectedGroup) return
    setDownloadingCalibrationExport(true)
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/paint/calibration-export?group=${encodeURIComponent(selectedGroup)}`
      )
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `calibration_export_${selectedGroup}.json`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to download calibration export:', error)
      alert('Failed to download calibration export')
    } finally {
      setDownloadingCalibrationExport(false)
    }
  }

  const loadGamutSlice = async (refresh: boolean = false) => {
    if (!selectedGroup) return
    setGamutLoading(true)
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/paint/gamut/slice?group=${encodeURIComponent(selectedGroup)}&l=${gamutL}&refresh=${refresh ? 'true' : 'false'}`,
        { cache: 'no-store' }
      )
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const data = await response.json()
      setGamutData(data)
      setSelectedGamutCell(null)
    } catch (error) {
      console.error('Failed to load gamut slice:', error)
      alert('Failed to load gamut slice')
    } finally {
      setGamutLoading(false)
    }
  }

  const loadLibraryRecipes = async (page: number) => {
    if (!selectedGroup) return
    setRecipeRowsLoading(true)
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/paint/library/recipes?group=${encodeURIComponent(selectedGroup)}&page=${page}&page_size=25`,
        { cache: 'no-store' }
      )
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const data = await response.json()
      setRecipeRows(Array.isArray(data.recipes) ? data.recipes : [])
      setRecipeRowsTotalPages(Math.max(1, Number(data.total_pages || 1)))
      setRecipeRowsTotal(Number(data.total || 0))
    } catch (error) {
      console.error('Failed to load library recipes:', error)
      setRecipeRows([])
      setRecipeRowsTotalPages(1)
      setRecipeRowsTotal(0)
    } finally {
      setRecipeRowsLoading(false)
    }
  }

  useEffect(() => {
    if (!gamutData || !gamutCanvasRef.current) return
    const canvas = gamutCanvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const step = Number(gamutData.step || 5)
    const aMin = Number(gamutData.a_min ?? -100)
    const aMax = Number(gamutData.a_max ?? 100)
    const bMin = Number(gamutData.b_min ?? -100)
    const bMax = Number(gamutData.b_max ?? 100)
    const cols = Math.floor((aMax - aMin) / step) + 1
    const rows = Math.floor((bMax - bMin) / step) + 1
    const cell = 9
    canvas.width = cols * cell
    canvas.height = rows * cell

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    for (const c of gamutData.cells || []) {
      const a = Number(c.a)
      const b = Number(c.b)
      const x = Math.round((a - aMin) / step)
      const yGrid = Math.round((b - bMin) / step)
      const y = (rows - 1 - yGrid)
      ctx.fillStyle = c.target_hex || '#000000'
      ctx.fillRect(x * cell, y * cell, cell, cell)

      let overlay = ''
      if (c.band === 'excellent') overlay = 'rgba(34,197,94,0.35)'
      else if (c.band === 'good') overlay = 'rgba(234,179,8,0.35)'
      else if (c.band === 'poor') overlay = 'rgba(239,68,68,0.45)'
      else if (c.band === 'mid') overlay = 'rgba(249,115,22,0.35)'
      if (overlay) {
        ctx.fillStyle = overlay
        ctx.fillRect(x * cell, y * cell, cell, cell)
      }
    }
  }, [gamutData])

  const handleGamutCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!gamutData || !gamutCanvasRef.current) return
    const rect = gamutCanvasRef.current.getBoundingClientRect()
    const xPx = e.clientX - rect.left
    const yPx = e.clientY - rect.top

    const step = Number(gamutData.step || 5)
    const aMin = Number(gamutData.a_min ?? -100)
    const aMax = Number(gamutData.a_max ?? 100)
    const bMin = Number(gamutData.b_min ?? -100)
    const bMax = Number(gamutData.b_max ?? 100)
    const cols = Math.floor((aMax - aMin) / step) + 1
    const rows = Math.floor((bMax - bMin) / step) + 1
    const cell = 9

    const x = Math.floor(xPx / cell)
    const y = Math.floor(yPx / cell)
    if (x < 0 || y < 0 || x >= cols || y >= rows) return

    const a = aMin + x * step
    const b = bMin + (rows - 1 - y) * step
    const hit = (gamutData.cells || []).find((c: any) => Number(c.a) === a && Number(c.b) === b) || null
    setSelectedGamutCell(hit)
  }


  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 text-white p-8">
        <div className="max-w-6xl mx-auto">Loading...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-6xl mx-auto">
        <Link href="/" className="inline-flex items-center text-gray-400 hover:text-white mb-6">
          ← Back to menu
        </Link>
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-4xl font-bold">Paint Library</h1>
          <div className="flex gap-4">
            <button
              onClick={() => router.push('/')}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
            >
              ← Back to Home
            </button>
            <button
              onClick={() => {
                setShowAddForm(true)
                setEditingPaint(null)
                    setFormData({ name: '', hex_approx: '#000000', notes: '' })
                  }}
                  className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
            >
              + Add Paint
            </button>
          </div>
        </div>

        {/* Library Group Selection */}
        <div className="mb-6 p-4 bg-gray-800 rounded">
          <div className="flex items-center gap-4 flex-wrap">
            <label className="font-semibold">Library Group:</label>
            <select
              value={selectedGroup}
              onChange={(e) => {
                const newGroup = e.target.value
                setSelectedGroup(newGroup)
                // Save to localStorage immediately
                if (typeof window !== 'undefined') {
                  localStorage.setItem('lastSelectedPaintLibrary', newGroup)
                }
              }}
              className="px-3 py-2 bg-gray-700 rounded border border-gray-600"
            >
              {libraryGroups.map((group) => (
                <option key={group.group} value={group.group}>
                  {group.name} ({group.paint_count} paints, {group.calibrated_count} calibrated)
                </option>
              ))}
            </select>
            <button
              onClick={() => {
                const currentGroup = libraryGroups.find(g => g.group === selectedGroup)
                if (currentGroup) {
                  handleRenameGroup(selectedGroup, currentGroup.name)
                }
              }}
              className="px-3 py-2 bg-gray-600 hover:bg-gray-500 rounded text-sm"
            >
              Rename Group
            </button>
            <button
              onClick={() => setShowCreateGroup(true)}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm"
            >
              + New Group
            </button>
            <button
              onClick={handleDownloadCalibrationExport}
              disabled={downloadingCalibrationExport}
              className="px-3 py-2 bg-emerald-700 hover:bg-emerald-600 rounded text-sm disabled:opacity-50"
            >
              {downloadingCalibrationExport ? 'Downloading…' : 'Download Calibrations JSON'}
            </button>
          </div>
          <div className="mt-4 pt-4 border-t border-gray-600">
            <label className="block font-semibold mb-2">Library coverage (g/cm²)</label>
            <p className="text-sm text-gray-400 mb-2">
              One value for this whole library: how many grams of paint cover 1 cm². Used for recipe weight calculations.
            </p>
            <div className="flex items-center gap-2 flex-wrap">
              <input
                type="text"
                inputMode="decimal"
                placeholder="e.g. 5 or 0.008"
                value={libraryCoverage}
                onChange={(e) => {
                  const v = e.target.value
                  if (v === '' || /^-?\d*\.?\d*$/.test(v)) setLibraryCoverage(v)
                }}
                className="w-24 px-3 py-2 bg-gray-700 rounded border border-gray-600 text-white"
              />
              <button
                type="button"
                onClick={saveLibraryCoverage}
                className="px-3 py-2 bg-green-600 hover:bg-green-500 rounded text-sm"
              >
                Save
              </button>
            </div>
          </div>
        </div>

        {/* Rename Group Form */}
        {renamingGroup && (
          <div className="mb-6 p-4 bg-gray-800 rounded">
            <h3 className="font-bold mb-3">Rename Library Group</h3>
            <form onSubmit={handleRenameGroupSubmit} className="flex gap-3">
              <input
                type="text"
                value={renameGroupName}
                onChange={(e) => setRenameGroupName(e.target.value)}
                placeholder="New library name"
                className="flex-1 px-3 py-2 bg-gray-700 rounded border border-gray-600"
                required
                autoFocus
              />
              <button
                type="submit"
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
              >
                Rename
              </button>
              <button
                type="button"
                onClick={() => {
                  setRenamingGroup(null)
                  setRenameGroupName('')
                }}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded"
              >
                Cancel
              </button>
            </form>
          </div>
        )}

        {/* Create New Group Form */}
        {showCreateGroup && (
          <div className="mb-6 p-4 bg-gray-800 rounded">
            <h3 className="font-bold mb-3">Create New Library Group</h3>
            <form onSubmit={handleCreateGroup} className="flex gap-3">
              <input
                type="text"
                value={newGroupName}
                onChange={(e) => setNewGroupName(e.target.value)}
                placeholder="Library name (e.g., Matisse, Dulux)"
                className="flex-1 px-3 py-2 bg-gray-700 rounded border border-gray-600"
                required
              />
              <button
                type="submit"
                className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
              >
                Create
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowCreateGroup(false)
                  setNewGroupName('')
                }}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded"
              >
                Cancel
              </button>
            </form>
          </div>
        )}

        {showAddForm && (
          <div className="mb-6 p-6 bg-gray-800 rounded">
            <h2 className="text-2xl font-bold mb-4">
              {editingPaint ? 'Edit Paint' : 'Add New Paint'}
            </h2>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block mb-2">Paint Name</label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                  className="w-full px-3 py-2 bg-gray-700 rounded text-white"
                />
              </div>
              <div>
                <label className="block mb-2">Approximate Color (Hex)</label>
                <div className="flex gap-2">
                  <input
                    type="color"
                    value={formData.hex_approx}
                    onChange={(e) => setFormData({ ...formData, hex_approx: e.target.value })}
                    className="h-10 w-20"
                  />
                  <input
                    type="text"
                    value={formData.hex_approx}
                    onChange={(e) => setFormData({ ...formData, hex_approx: e.target.value })}
                    required
                    className="flex-1 px-3 py-2 bg-gray-700 rounded text-white"
                  />
                </div>
              </div>
              <div>
                <label className="block mb-2">Notes (optional)</label>
                <textarea
                  value={formData.notes}
                  onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                  className="w-full px-3 py-2 bg-gray-700 rounded text-white"
                  rows={3}
                />
              </div>

              {editingPaint && (
                <div className="border border-gray-600 rounded overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setCalibrationPanelOpen((o) => !o)}
                    className="w-full px-4 py-3 flex items-center justify-between bg-gray-750 hover:bg-gray-700 text-left"
                  >
                    <span className="font-medium">Calibration results</span>
                    <span className="text-gray-400">{calibrationPanelOpen ? '▼' : '▶'}</span>
                  </button>
                  {calibrationPanelOpen && (
                    <div className="p-4 bg-gray-800/80 border-t border-gray-600 space-y-4">
                      {calibrationLoading && (
                        <p className="text-gray-400">Loading calibration…</p>
                      )}
                      {calibrationError && (
                        <p className="text-amber-400">{calibrationError}</p>
                      )}
                      {calibrationData && !calibrationLoading && (
                        <CalibrationResultsView data={calibrationData} />
                      )}
                    </div>
                  )}
                </div>
              )}

              <div className="flex gap-2">
                <button
                  type="submit"
                  className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded"
                >
                  {editingPaint ? 'Update' : 'Add'} Paint
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setShowAddForm(false)
                    setEditingPaint(null)
                    setFormData({ name: '', hex_approx: '#000000', notes: '' })
                  }}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {paints.map((paint) => (
            <div key={paint.id} className="p-4 bg-gray-800 rounded">
              <div className="flex items-center gap-3 mb-3">
                <div
                  className="w-16 h-16 rounded border border-gray-600"
                  style={{ backgroundColor: paint.hex_approx }}
                />
                <div className="flex-1">
                  <h3 className="text-lg font-bold">{paint.name}</h3>
                  <div className="text-sm text-gray-400">{paint.hex_approx}</div>
                </div>
              </div>
              {paint.notes && (
                <p className="text-sm text-gray-300 mb-3">{paint.notes}</p>
              )}
              <div className="flex gap-2">
                <button
                  onClick={() => router.push(`/paints/calibrate/${paint.id}?group=${encodeURIComponent(selectedGroup)}`)}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                >
                  Calibrate
                </button>
                <button
                  onClick={() => handleEdit(paint)}
                  className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm"
                >
                  Edit
                </button>
                <button
                  onClick={() => handleDelete(paint.id)}
                  className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>

        <div className="mt-8 p-4 bg-gray-800 rounded">
          <h2 className="text-2xl font-bold mb-3">Palette Gamut Analysis</h2>
          <p className="text-sm text-gray-400 mb-3">
            Fixed-L slice over Lab a/b. Pixel color is target color with ΔE heat overlay
            (green {'<'}2, yellow {'<'}5, red {'>'}10). Click a pixel to inspect the suggested recipe.
          </p>
          <div className="flex items-center gap-3 flex-wrap mb-3">
            <label className="text-sm font-semibold">L slice:</label>
            <select
              value={gamutL}
              onChange={(e) => setGamutL(Number(e.target.value))}
              className="px-3 py-2 bg-gray-700 rounded border border-gray-600"
            >
              {Array.from({ length: 21 }, (_, i) => i * 5).map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
            <button
              onClick={() => loadGamutSlice(false)}
              disabled={gamutLoading}
              className="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm disabled:opacity-50"
            >
              {gamutLoading ? 'Generating…' : 'Generate Slice'}
            </button>
            <button
              onClick={() => loadGamutSlice(true)}
              disabled={gamutLoading}
              className="px-3 py-2 bg-orange-600 hover:bg-orange-700 rounded text-sm disabled:opacity-50"
            >
              Refresh
            </button>
          </div>

          {gamutData && (
            <div className="flex flex-col lg:flex-row gap-4">
              <div>
                <canvas
                  ref={gamutCanvasRef}
                  onClick={handleGamutCanvasClick}
                  className="border border-gray-600 rounded cursor-crosshair max-w-full h-auto"
                />
                <div className="text-xs text-gray-400 mt-2">
                  a*: -100 → 100 (left to right), b*: 100 → -100 (top to bottom)
                </div>
              </div>
              <div className="flex-1 min-w-[260px] p-3 bg-gray-900 rounded border border-gray-700">
                <h3 className="font-bold mb-2">Selected Pixel</h3>
                {!selectedGamutCell && (
                  <p className="text-sm text-gray-400">Click a pixel on the heatmap to inspect the recommended recipe.</p>
                )}
                {selectedGamutCell && (
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-6 h-6 rounded border border-gray-600"
                        style={{ backgroundColor: selectedGamutCell.target_hex || '#000000' }}
                      />
                      <span className="font-mono">{selectedGamutCell.target_hex}</span>
                    </div>
                    <div>a*: {selectedGamutCell.a}, b*: {selectedGamutCell.b}, L*: {gamutL}</div>
                    <div>
                      ΔE: {selectedGamutCell.error != null ? Number(selectedGamutCell.error).toFixed(2) : 'N/A'}
                    </div>
                    <div className="pt-2 border-t border-gray-700">
                      <div className="font-semibold mb-1">Recommended recipe</div>
                      <div className="text-gray-300">
                        {selectedGamutCell.recipe_data
                          ? formatRecipe(selectedGamutCell.recipe_data)
                          : 'No recipe available'}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="mt-6 pt-4 border-t border-gray-700">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-lg font-bold">Library Recipe Cache</h3>
              <div className="text-sm text-gray-400">Total recipes: {recipeRowsTotal}</div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-400 border-b border-gray-600">
                    <th className="py-2 pr-3">Hex</th>
                    <th className="py-2 pr-3">Recipe</th>
                    <th className="py-2 pr-3">ΔE</th>
                    <th className="py-2">Last Modified</th>
                  </tr>
                </thead>
                <tbody>
                  {!recipeRowsLoading && recipeRows.length === 0 && (
                    <tr>
                      <td colSpan={4} className="py-4 text-gray-500">
                        No recipes cached for this library yet.
                      </td>
                    </tr>
                  )}
                  {recipeRows.map((row) => (
                    <tr key={row.hex} className="border-b border-gray-700 align-top">
                      <td className="py-2 pr-3">
                        <div className="flex items-center gap-2">
                          <span
                            className="inline-block w-5 h-5 rounded border border-gray-600"
                            style={{ backgroundColor: row.hex }}
                          />
                          <span className="font-mono">{row.hex}</span>
                        </div>
                      </td>
                      <td className="py-2 pr-3 text-gray-200">
                        {Array.isArray(row.ingredients) && row.ingredients.length > 0
                          ? row.ingredients
                              .map((ing) => `${ing.paint_name || ing.paint_id} ${Number(ing.percentage || 0).toFixed(2)}%`)
                              .join(' + ')
                          : 'No ingredients'}
                      </td>
                      <td className="py-2 pr-3">{row.delta_e != null ? Number(row.delta_e).toFixed(2) : '—'}</td>
                      <td className="py-2 text-gray-300">
                        {row.last_modified ? new Date(row.last_modified).toLocaleString() : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-3 flex items-center gap-2">
              <button
                onClick={() => setRecipeRowsPage((p) => Math.max(1, p - 1))}
                disabled={recipeRowsLoading || recipeRowsPage <= 1}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded disabled:opacity-50"
              >
                Prev
              </button>
              <div className="text-sm text-gray-300">
                Page {recipeRowsPage} / {recipeRowsTotalPages}
              </div>
              <button
                onClick={() => setRecipeRowsPage((p) => Math.min(recipeRowsTotalPages, p + 1))}
                disabled={recipeRowsLoading || recipeRowsPage >= recipeRowsTotalPages}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded disabled:opacity-50"
              >
                Next
              </button>
              {recipeRowsLoading && <span className="text-sm text-gray-400">Loading…</span>}
            </div>
          </div>
        </div>

        {paints.length === 0 && !showAddForm && (
          <div className="text-center py-12 text-gray-400">
            No paints in library. Click "Add Paint" to get started.
          </div>
        )}
      </div>
    </div>
  )
}

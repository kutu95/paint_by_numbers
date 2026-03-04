'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'

const DEFAULT_PAINT_MARGIN_PERCENT = 33
const DEFAULT_CANVAS_WIDTH_CM = 50
const DEFAULT_CANVAS_HEIGHT_CM = 40

export default function SettingsPage() {
  const [mounted, setMounted] = useState(false)
  const [paintMarginPercent, setPaintMarginPercent] = useState(DEFAULT_PAINT_MARGIN_PERCENT)
  const [canvasWidthCm, setCanvasWidthCm] = useState(DEFAULT_CANVAS_WIDTH_CM)
  const [canvasHeightCm, setCanvasHeightCm] = useState(DEFAULT_CANVAS_HEIGHT_CM)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted || typeof window === 'undefined') return
    try {
      const marginRaw = localStorage.getItem('layerpainter_recipe_margin')
      if (marginRaw !== null) {
        const multiplier = parseFloat(marginRaw)
        if (!Number.isNaN(multiplier) && multiplier >= 1) {
          setPaintMarginPercent(Math.round((multiplier - 1) * 100))
        }
      }
      const settingsRaw = localStorage.getItem('layerpainter_settings')
      if (settingsRaw) {
        const parsed = JSON.parse(settingsRaw)
        if (typeof parsed.canvasWidthCm === 'number' && parsed.canvasWidthCm > 0) {
          setCanvasWidthCm(parsed.canvasWidthCm)
        }
        if (typeof parsed.canvasHeightCm === 'number' && parsed.canvasHeightCm > 0) {
          setCanvasHeightCm(parsed.canvasHeightCm)
        }
      }
    } catch (e) {
      console.error('Failed to load settings:', e)
    }
  }, [mounted])

  const saveSettings = () => {
    if (typeof window === 'undefined') return
    try {
      const multiplier = 1 + paintMarginPercent / 100
      localStorage.setItem('layerpainter_recipe_margin', String(multiplier))
      const existing = localStorage.getItem('layerpainter_settings')
      let settings: Record<string, unknown> = {}
      if (existing) {
        try {
          settings = JSON.parse(existing) as Record<string, unknown>
        } catch (_) {}
      }
      settings.canvasWidthCm = canvasWidthCm
      settings.canvasHeightCm = canvasHeightCm
      localStorage.setItem('layerpainter_settings', JSON.stringify(settings))
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      console.error('Failed to save settings:', e)
    }
  }

  if (!mounted) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <p className="text-gray-400">Loading…</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-lg mx-auto">
        <Link
          href="/"
          className="inline-flex items-center text-gray-400 hover:text-white mb-8"
        >
          ← Back to menu
        </Link>

        <h1 className="text-2xl font-bold mb-6">Settings</h1>

        <div className="space-y-6">
          <div>
            <label htmlFor="paint-margin" className="block text-sm font-medium text-gray-300 mb-2">
              Paint margin (%)
            </label>
            <p className="text-sm text-gray-500 mb-2">
              Extra paint to order as a percentage (e.g. 33% = 1.33× calculated amount). Default 33%.
            </p>
            <input
              id="paint-margin"
              type="number"
              min={0}
              max={200}
              value={paintMarginPercent}
              onChange={(e) => setPaintMarginPercent(Number(e.target.value) || 0)}
              className="w-full px-4 py-2 rounded-lg bg-gray-800 border border-gray-600 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <div>
            <label htmlFor="canvas-width" className="block text-sm font-medium text-gray-300 mb-2">
              Default canvas width (cm)
            </label>
            <input
              id="canvas-width"
              type="number"
              min={1}
              step={0.1}
              value={canvasWidthCm}
              onChange={(e) => setCanvasWidthCm(Number(e.target.value) || 0)}
              className="w-full px-4 py-2 rounded-lg bg-gray-800 border border-gray-600 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <div>
            <label htmlFor="canvas-height" className="block text-sm font-medium text-gray-300 mb-2">
              Default canvas height (cm)
            </label>
            <input
              id="canvas-height"
              type="number"
              min={1}
              step={0.1}
              value={canvasHeightCm}
              onChange={(e) => setCanvasHeightCm(Number(e.target.value) || 0)}
              className="w-full px-4 py-2 rounded-lg bg-gray-800 border border-gray-600 text-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <button
            type="button"
            onClick={saveSettings}
            className="w-full py-3 px-4 rounded-lg bg-blue-600 hover:bg-blue-500 font-medium transition-colors"
          >
            {saved ? 'Saved' : 'Save settings'}
          </button>
        </div>
      </div>
    </div>
  )
}

'use client'

import { useState, useEffect, useCallback } from 'react'
import type { SessionData } from './types'
import {
  loadViewerHudState,
  saveViewerHudState,
  getViewerHudKey,
  DEFAULT_HUD_STATE,
  type ProjectionViewerHudState,
  type OutlineMode,
} from './viewer/projectionViewerState'

const PROJECTION_LAYER_KEY = (id: string) => `projection_current_layer_${id}`
const PROJECTION_SCALE_KEY = (id: string) => `projection_scale_${id}`
const DONE_LAYERS_KEY = (id: string) => `done_${id}`

export interface ProjectionHUDControlsProps {
  sessionId: string
  sessionData: SessionData
}

export function ProjectionHUDControls({ sessionId, sessionData }: ProjectionHUDControlsProps) {
  const [hudState, setHudState] = useState<ProjectionViewerHudState>(DEFAULT_HUD_STATE)
  const [currentLayer, setCurrentLayer] = useState(0)
  const [scale, setScale] = useState(1.0)
  const [doneLayers, setDoneLayers] = useState<Set<number>>(new Set())

  const loadState = useCallback(() => {
    if (typeof window === 'undefined') return
    setHudState(loadViewerHudState(sessionId))
    const layerStored = localStorage.getItem(PROJECTION_LAYER_KEY(sessionId))
    if (layerStored !== null) {
      const n = parseInt(layerStored, 10)
      if (!Number.isNaN(n) && n >= 0) setCurrentLayer(n)
    }
    const scaleStored = localStorage.getItem(PROJECTION_SCALE_KEY(sessionId))
    if (scaleStored != null) {
      const n = parseFloat(scaleStored)
      if (!Number.isNaN(n) && n >= 0.25 && n <= 2) setScale(n)
    }
    const doneStored = localStorage.getItem(DONE_LAYERS_KEY(sessionId))
    if (doneStored) setDoneLayers(new Set(JSON.parse(doneStored)))
  }, [sessionId])

  useEffect(() => {
    loadState()
  }, [loadState])

  useEffect(() => {
    const key = PROJECTION_LAYER_KEY(sessionId)
    const onStorage = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        const n = parseInt(e.newValue, 10)
        if (!Number.isNaN(n)) setCurrentLayer(n)
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [sessionId])

  useEffect(() => {
    const key = getViewerHudKey(sessionId)
    const onStorage = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        try {
          const next = JSON.parse(e.newValue) as ProjectionViewerHudState
          setHudState(next)
        } catch (_) {}
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [sessionId])

  useEffect(() => {
    const key = PROJECTION_SCALE_KEY(sessionId)
    const onStorage = (e: StorageEvent) => {
      if (e.key === key && e.newValue != null) {
        const n = parseFloat(e.newValue)
        if (!Number.isNaN(n) && n >= 0.25 && n <= 2) setScale(n)
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [sessionId])

  const updateHud = useCallback(
    (patch: Partial<ProjectionViewerHudState>) => {
      setHudState((prev) => {
        const next = { ...prev, ...patch }
        saveViewerHudState(sessionId, next)
        return next
      })
    },
    [sessionId]
  )

  const setLayer = useCallback(
    (n: number) => {
      if (n >= 0 && n < sessionData.layers.length) {
        setCurrentLayer(n)
        localStorage.setItem(PROJECTION_LAYER_KEY(sessionId), String(n))
      }
    },
    [sessionId, sessionData.layers.length]
  )

  const setScaleValue = useCallback(
    (v: number) => {
      const clamped = Math.max(0.25, Math.min(2, Math.round(v * 100) / 100))
      setScale(clamped)
      localStorage.setItem(PROJECTION_SCALE_KEY(sessionId), String(clamped))
    },
    [sessionId]
  )

  const toggleDone = useCallback(() => {
    const layer = sessionData.layers[currentLayer]
    if (layer?.is_finished) return
    setDoneLayers((prev) => {
      const next = new Set(prev)
      if (next.has(currentLayer)) next.delete(currentLayer)
      else next.add(currentLayer)
      localStorage.setItem(DONE_LAYERS_KEY(sessionId), JSON.stringify(Array.from(next)))
      return next
    })
  }, [sessionId, currentLayer, sessionData.layers])

  const currentLayerData = sessionData.layers[currentLayer]
  const layerLabel = !currentLayerData
    ? '—'
    : currentLayerData.is_finished
      ? 'Finished'
      : `Layer ${currentLayer + 1} / ${sessionData.layers.length}`
  const outlineModes: OutlineMode[] = ['off', 'thin', 'thick', 'glow']

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-lg font-semibold mb-4">Projection controls</h2>
      <p className="text-gray-400 text-sm mb-4">
        These controls sync with the projection window. Change layer, mask, opacity, scale, and other options here or use the keyboard in the projection window.
      </p>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">Current layer</label>
          <div className="flex flex-wrap gap-2">
            {sessionData.layers.map((layer, idx) => {
              if (layer.is_finished) return null
              const isCurrent = idx === currentLayer
              return (
                <button
                  key={layer.layer_index}
                  type="button"
                  onClick={() => setLayer(idx)}
                  className={`px-3 py-1 rounded text-sm ${isCurrent ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
                >
                  {idx + 1}
                </button>
              )
            })}
          </div>
          <p className="text-xs text-gray-500 mt-1">{layerLabel}</p>
        </div>

        {sessionData.quantized_preview_url && (
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => updateHud({ showFinalPreview: !hudState.showFinalPreview, showOriginalImage: hudState.showFinalPreview ? hudState.showOriginalImage : false })}
              className="px-3 py-1.5 rounded text-sm bg-amber-600 hover:bg-amber-500 text-white"
            >
              {hudState.showFinalPreview ? 'Back to layer (F)' : 'Show final (F)'}
            </button>
            {sessionData.original_url && (
              <button
                type="button"
                onClick={() => updateHud({ showOriginalImage: !hudState.showOriginalImage, showFinalPreview: hudState.showOriginalImage ? hudState.showFinalPreview : false })}
                className="px-3 py-1.5 rounded text-sm bg-emerald-600 hover:bg-emerald-500 text-white"
              >
                {hudState.showOriginalImage ? 'Back (G)' : 'Original (G)'}
              </button>
            )}
          </div>
        )}

        {currentLayerData && !currentLayerData.is_finished && (
          <>
            <div className="flex flex-wrap items-center gap-4">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={hudState.usePureMask}
                  onChange={(e) => updateHud({ usePureMask: e.target.checked })}
                  className="rounded bg-gray-700 border-gray-600"
                />
                <span className="text-sm">Mask: {hudState.usePureMask ? 'Pure' : 'Expanded'}</span>
              </label>
              <div className="flex items-center gap-2">
                <span className="text-sm">Opacity: {hudState.maskOpacity}%</span>
                <button type="button" onClick={() => updateHud({ maskOpacity: Math.max(40, hudState.maskOpacity - 5) })} className="px-2 py-0.5 rounded bg-gray-700 text-sm">−</button>
                <button type="button" onClick={() => updateHud({ maskOpacity: Math.min(100, hudState.maskOpacity + 5) })} className="px-2 py-0.5 rounded bg-gray-700 text-sm">+</button>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm">Outline:</span>
                <select
                  value={hudState.outlineMode}
                  onChange={(e) => updateHud({ outlineMode: e.target.value as OutlineMode })}
                  className="bg-gray-700 border border-gray-600 rounded px-2 py-1 text-sm"
                >
                  {outlineModes.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-4">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={hudState.showColor}
                  onChange={(e) => updateHud({ showColor: e.target.checked })}
                  className="rounded bg-gray-700 border-gray-600"
                />
                <span className="text-sm">Color</span>
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={hudState.inverted}
                  onChange={(e) => updateHud({ inverted: e.target.checked })}
                  className="rounded bg-gray-700 border-gray-600"
                />
                <span className="text-sm">Invert</span>
              </label>
            </div>
          </>
        )}

        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm">Scale: {Math.round(scale * 100)}%</span>
            <button type="button" onClick={() => setScaleValue(scale - 0.05)} className="px-2 py-0.5 rounded bg-gray-700 text-sm">−</button>
            <button type="button" onClick={() => setScaleValue(scale + 0.05)} className="px-2 py-0.5 rounded bg-gray-700 text-sm">+</button>
          </div>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={hudState.crosshairs}
              onChange={(e) => updateHud({ crosshairs: e.target.checked })}
              className="rounded bg-gray-700 border-gray-600"
            />
            <span className="text-sm">Crosshairs</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={hudState.grid}
              onChange={(e) => updateHud({ grid: e.target.checked })}
              className="rounded bg-gray-700 border-gray-600"
            />
            <span className="text-sm">Grid</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={hudState.registrationMode}
              onChange={(e) => updateHud({ registrationMode: e.target.checked })}
              className="rounded bg-gray-700 border-gray-600"
            />
            <span className="text-sm">Registration</span>
          </label>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={hudState.showDoneLayers}
              onChange={(e) => updateHud({ showDoneLayers: e.target.checked })}
              className="rounded bg-gray-700 border-gray-600"
            />
            <span className="text-sm">Show done</span>
          </label>
        </div>

        {currentLayerData && !currentLayerData.is_finished && (
          <div>
            <button
              type="button"
              onClick={toggleDone}
              className="px-3 py-1.5 rounded text-sm bg-gray-700 hover:bg-gray-600"
            >
              {doneLayers.has(currentLayer) ? '✓ Mark not done' : 'Mark done (D)'}
            </button>
          </div>
        )}

        <div className="text-xs text-gray-500 pt-2 border-t border-gray-700">
          ← → Space: Navigate | D: Done | C: Crosshairs | X: Grid | I: Invert | K: Color | L: Pure/Expanded | O: Outline | [ ]: Opacity | − +: Scale | F: Final | G: Original | R: Registration | B/W: Black/White | S: Show Done | H: HUD | E: End lasso | Esc: Close
        </div>
      </div>
    </div>
  )
}

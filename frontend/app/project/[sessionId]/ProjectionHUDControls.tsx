'use client'

import { useState, useEffect, useCallback } from 'react'
import type { SessionData } from './types'
import {
  loadViewerHudState,
  saveViewerHudState,
  DEFAULT_HUD_STATE,
  type ProjectionViewerHudState,
  type OutlineMode,
  type MaskDisplayMode,
} from './viewer/projectionViewerState'
import {
  cycleMaskDisplayMode,
  maskDisplayModeLabel,
  resolveMaskDisplayMode,
} from './viewer/maskDisplay'
import { PROJECTION_SHORTCUTS_LINE } from './viewer/projectionKeyboardHelp'
import {
  adjustProjectionScalePercent,
  percentToScale,
  PROJECTION_SCALE_COARSE_STEP_PCT,
  scaleToPercent,
} from './viewer/projectionZoom'
import { fetchProjectState, saveProjectState } from '@/lib/projectSession'

async function persistUiState(
  sessionId: string,
  patch: {
    currentLayer?: number
    projectionScale?: number
    doneLayers?: number[]
    projectionHud?: ProjectionViewerHudState
  }
) {
  const prev = await fetchProjectState(sessionId)
  await saveProjectState(sessionId, { ...prev, ...patch })
}

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
    void fetchProjectState(sessionId).then((ui) => {
      const hud = (ui.projectionHud ?? {}) as Partial<ProjectionViewerHudState>
      setHudState({
        ...DEFAULT_HUD_STATE,
        ...hud,
        maskDisplayMode: resolveMaskDisplayMode(hud),
      })
      if (typeof ui.currentLayer === 'number' && ui.currentLayer >= 0) setCurrentLayer(ui.currentLayer)
      if (typeof ui.projectionScale === 'number') {
        const n = percentToScale(scaleToPercent(ui.projectionScale))
        setScale(n)
      }
      if (Array.isArray(ui.doneLayers)) setDoneLayers(new Set(ui.doneLayers))
    })
  }, [sessionId])

  useEffect(() => {
    loadState()
  }, [loadState])

  // Storage events only fire in other tabs/windows; reload when this tab regains focus (e.g. after H in projection window).
  useEffect(() => {
    const onFocus = () => loadState()
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [loadState])

  useEffect(() => {
    const id = window.setInterval(() => loadState(), 5000)
    return () => clearInterval(id)
  }, [loadState])

  const updateHud = useCallback(
    (patch: Partial<ProjectionViewerHudState>) => {
      setHudState((prev) => {
        const next = { ...prev, ...patch }
        saveViewerHudState(sessionId, next)
        void persistUiState(sessionId, { projectionHud: next })
        return next
      })
    },
    [sessionId]
  )

  const setLayer = useCallback(
    (n: number) => {
      if (n >= 0 && n < sessionData.layers.length) {
        setCurrentLayer(n)
        void persistUiState(sessionId, { currentLayer: n })
      }
    },
    [sessionId, sessionData.layers.length]
  )

  const nudgeScale = useCallback(
    (deltaPct: number) => {
      setScale((prev) => {
        const next = percentToScale(adjustProjectionScalePercent(scaleToPercent(prev), deltaPct))
        void persistUiState(sessionId, { projectionScale: next })
        return next
      })
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
      void persistUiState(sessionId, { doneLayers: Array.from(next) })
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
              <div className="flex items-center gap-2">
                <span className="text-sm">Mask view:</span>
                <button
                  type="button"
                  onClick={() =>
                    updateHud({
                      maskDisplayMode: cycleMaskDisplayMode(hudState.maskDisplayMode),
                      ...(hudState.maskDisplayMode === 'white'
                        ? { inverted: false }
                        : {}),
                    })
                  }
                  className="px-2 py-1 rounded bg-gray-700 hover:bg-gray-600 text-sm"
                  title="Same as K in projection window"
                >
                  {maskDisplayModeLabel(hudState.maskDisplayMode)} (K)
                </button>
              </div>
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
            <span className="text-sm">Scale: {scaleToPercent(scale)}%</span>
            <button type="button" onClick={() => nudgeScale(-PROJECTION_SCALE_COARSE_STEP_PCT)} className="px-2 py-0.5 rounded bg-gray-700 text-sm">−</button>
            <button type="button" onClick={() => nudgeScale(PROJECTION_SCALE_COARSE_STEP_PCT)} className="px-2 py-0.5 rounded bg-gray-700 text-sm">+</button>
          </div>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={hudState.showHudOverlay}
              onChange={(e) => updateHud({ showHudOverlay: e.target.checked })}
              className="rounded bg-gray-700 border-gray-600"
            />
            <span className="text-sm">On-screen HUD (projection window)</span>
          </label>
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

        <div className="text-xs text-gray-500 pt-2 border-t border-gray-700 leading-relaxed">
          {PROJECTION_SHORTCUTS_LINE}
        </div>
      </div>
    </div>
  )
}

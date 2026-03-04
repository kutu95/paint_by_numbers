'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import type { SessionData } from '../../types'

const PROJECTION_LAYER_KEY = (id: string) => `projection_current_layer_${id}`

export default function HudOnlyWindow() {
  const params = useParams()
  const sessionId = params.sessionId as string
  const [sessionData, setSessionData] = useState<SessionData | null>(null)
  const [currentLayer, setCurrentLayer] = useState(0)

  useEffect(() => {
    const stored = localStorage.getItem(`session_${sessionId}`)
    if (stored) {
      try {
        setSessionData(JSON.parse(stored) as SessionData)
        const layerStored = localStorage.getItem(PROJECTION_LAYER_KEY(sessionId))
        if (layerStored !== null) {
          const n = parseInt(layerStored, 10)
          if (!Number.isNaN(n) && n >= 0) setCurrentLayer(n)
        }
      } catch (_) {}
    }
  }, [sessionId])

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

  if (!sessionData) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center p-4">
        Loading…
      </div>
    )
  }

  const currentLayerData = sessionData.layers[currentLayer]
  const layerLabel = !currentLayerData
    ? '—'
    : currentLayerData.is_finished
      ? 'Finished'
      : `Layer ${currentLayer + 1} / ${sessionData.layers.length}`

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4">
      <div className="bg-black/80 rounded shadow-xl p-4 text-sm max-w-md">
        <div className="font-semibold text-base mb-3 border-b border-white/20 pb-2">Projection HUD</div>
        <div className="space-y-2">
          <div>{layerLabel}</div>
          <div className="text-xs text-gray-400 pt-2 border-t border-white/20">
            ← → Space: Navigate | D: Done | C: Crosshairs | X: Grid | I: Invert | K: Color | L: Pure/Expanded | O: Outline | [ ]: Opacity | - +: Scale | F: Final | G: Original | R: Registration | B/W: Black/White | S: Show Done | H: HUD | Esc: Close
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-4">This window can be moved to any display. Layer updates in sync with the projection window.</p>
      </div>
    </div>
  )
}

'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import type { SessionData } from '../../types'
import { fetchProjectSession, fetchProjectState } from '@/lib/projectSession'
import { PROJECTION_SHORTCUTS_LINES } from '../projectionKeyboardHelp'

export default function HudOnlyWindow() {
  const params = useParams()
  const sessionId = params.sessionId as string
  const [sessionData, setSessionData] = useState<SessionData | null>(null)
  const [currentLayer, setCurrentLayer] = useState(0)

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    void (async () => {
      const [session, ui] = await Promise.all([
        fetchProjectSession(sessionId),
        fetchProjectState(sessionId),
      ])
      if (cancelled) return
      if (session) setSessionData(session)
      if (typeof ui.currentLayer === 'number' && ui.currentLayer >= 0) setCurrentLayer(ui.currentLayer)
    })()
    const id = window.setInterval(() => {
      void fetchProjectState(sessionId).then((ui) => {
        if (typeof ui.currentLayer === 'number') setCurrentLayer(ui.currentLayer!)
      })
    }, 2000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
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
          <div className="text-xs text-gray-400 pt-2 border-t border-white/20 space-y-1 leading-relaxed">
            {PROJECTION_SHORTCUTS_LINES.map((line) => (
              <div key={line}>{line}</div>
            ))}
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-4">This window can be moved to any display. Layer updates in sync with the projection window.</p>
      </div>
    </div>
  )
}

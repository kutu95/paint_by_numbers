'use client'

import { useState, useEffect } from 'react'
import { getLassoMode, setLassoMode, type LassoMode } from './viewer/projectionLassoState'

export interface ProjectionToolsPanelProps {
  sessionId: string
}

export function ProjectionToolsPanel({ sessionId }: ProjectionToolsPanelProps) {
  const [lassoMode, setLassoModeState] = useState<LassoMode>(() =>
    typeof window !== 'undefined' ? getLassoMode(sessionId) : ''
  )

  useEffect(() => {
    if (typeof window === 'undefined') return
    setLassoModeState(getLassoMode(sessionId))
  }, [sessionId])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const key = `projection_lasso_mode_${sessionId}`
    const onStorage = (e: StorageEvent) => {
      if (e.key === key) {
        const v = e.newValue
        setLassoModeState(v === 'drawing' || v === 'active' ? v : '')
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [sessionId])

  const startLasso = () => {
    setLassoMode(sessionId, 'drawing')
    setLassoModeState('drawing')
  }

  const endLasso = () => {
    setLassoMode(sessionId, '')
    setLassoModeState('')
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-lg font-semibold mb-4">Tools</h2>
      <p className="text-gray-400 text-sm mb-4">
        Lasso: draw a region in the projection window; only that area and layers with content inside it will be shown. Use End lasso to return to normal view.
      </p>
      <div className="flex flex-wrap gap-2">
        {lassoMode === '' || lassoMode === 'active' ? (
          <button
            type="button"
            onClick={startLasso}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-sm font-medium"
          >
            Start lasso
          </button>
        ) : null}
        {lassoMode === 'drawing' ? (
          <span className="px-4 py-2 bg-amber-700/50 text-amber-200 rounded text-sm">
            Draw the lasso in the projection window, then double‑click or press Enter to close.
          </span>
        ) : null}
        {lassoMode !== '' ? (
          <button
            type="button"
            onClick={endLasso}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm font-medium"
          >
            End lasso
          </button>
        ) : null}
      </div>
    </div>
  )
}

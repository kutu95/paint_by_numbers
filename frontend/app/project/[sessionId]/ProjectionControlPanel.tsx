'use client'

import { useState, useEffect, useCallback } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import type { SessionData } from './types'
import { getProjectBySessionId, removeProject, saveProject, syncProjectsFromServer } from '@/lib/projects'
import { SessionResultsContent } from './SessionResultsContent'
import { ProjectionHUDControls } from './ProjectionHUDControls'
import { ProjectionToolsPanel } from './ProjectionToolsPanel'
import { API_BASE_URL } from '@/lib/config'

const PROJECTION_LAYER_KEY = (id: string) => `projection_current_layer_${id}`

export type PanelMode = 'layers' | 'projection' | 'all'

export interface ProjectionControlPanelProps {
  sessionId: string
  /** When true, used inside home page: no Back/Edit/Delete links */
  embedMode?: boolean
  /** When 'layers': only layers content (no Projection window). When 'projection': only Projection window + HUD. When 'all' or undefined: everything. */
  panelMode?: PanelMode
  /** When provided (e.g. in embed mode), called after project is deleted instead of navigating */
  onProjectDeleted?: () => void
}

export function ProjectionControlPanel({
  sessionId,
  embedMode = false,
  panelMode = 'all',
  onProjectDeleted,
}: ProjectionControlPanelProps) {
  const router = useRouter()
  const [sessionData, setSessionData] = useState<SessionData | null>(null)
  const [sessionLoaded, setSessionLoaded] = useState(false)
  const [currentLayer, setCurrentLayer] = useState(0)
  const [projectName, setProjectName] = useState<string | null>(null)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [regenerating, setRegenerating] = useState(false)

  useEffect(() => {
    if (typeof window === 'undefined' || !sessionId) return
    void (async () => {
      await syncProjectsFromServer()
      const p = getProjectBySessionId(sessionId)
      setProjectName(p?.name ?? null)
    })()
  }, [sessionId])

  useEffect(() => {
    if (typeof window === 'undefined' || !sessionId || !sessionData) return
    const proj = getProjectBySessionId(sessionId)
    if (!proj) return

    const nextWidth =
      typeof sessionData.canvas_width_cm === 'number' && sessionData.canvas_width_cm > 0
        ? sessionData.canvas_width_cm
        : proj.canvasWidthCm
    const nextHeight =
      typeof sessionData.canvas_height_cm === 'number' && sessionData.canvas_height_cm > 0
        ? sessionData.canvas_height_cm
        : proj.canvasHeightCm

    if (nextWidth === proj.canvasWidthCm && nextHeight === proj.canvasHeightCm) return
    saveProject({
      ...proj,
      canvasWidthCm: nextWidth,
      canvasHeightCm: nextHeight,
    })
  }, [sessionId, sessionData])

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
      } catch (e) {
        console.error('Failed to load session data')
      }
    }
    setSessionLoaded(true)
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

  const openProjectionWindow = useCallback(() => {
    if (!sessionId || typeof window === 'undefined') return
    const url = `${window.location.origin}/project/${sessionId}/viewer`
    window.open(url, 'projection', 'width=1920,height=1080,menubar=no,toolbar=no,location=no,status=no')
  }, [sessionId])

  const handleDeleteProject = useCallback(() => {
    if (!sessionId || typeof window === 'undefined') return
    removeProject(sessionId)
    localStorage.removeItem(`session_${sessionId}`)
    localStorage.removeItem(PROJECTION_LAYER_KEY(sessionId))
    if (localStorage.getItem('current_session_id') === sessionId) {
      localStorage.removeItem('current_session_id')
    }
    setShowDeleteConfirm(false)
    if (onProjectDeleted) {
      onProjectDeleted()
    } else {
      router.push('/')
    }
  }, [sessionId, router, onProjectDeleted])

  const handleGenerateOrRegenerate = useCallback(async () => {
    if (!sessionId || typeof window === 'undefined') return
    const proj = getProjectBySessionId(sessionId)
    if (!proj) return
    if (sessionData) {
      setRegenerating(true)
      try {
        const nColors = proj.nColors ?? 16
        const overpaintMm = proj.overpaintMm ?? 5
        const orderMode = proj.orderMode ?? 'largest'
        const maxSide = proj.maxSide ?? 1920
        const formData = new FormData()
        formData.append('n_colors', String(nColors))
        formData.append('overpaint_mm', String(overpaintMm))
        formData.append('order_mode', orderMode)
        formData.append('max_side', String(maxSide))
        formData.append('canvas_width_cm', String(proj.canvasWidthCm))
        formData.append('canvas_height_cm', String(proj.canvasHeightCm))
        formData.append('saturation_boost', String(proj.saturationBoost))
        formData.append('detail_level', String(proj.detailLevel))
        const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/reprocess`, { method: 'POST', body: formData })
        if (!response.ok) {
          const err = await response.json().catch(() => ({}))
          throw new Error((err as { detail?: string }).detail || `Server error: ${response.status}`)
        }
        const data = await response.json() as SessionData & { session_id: string }
        localStorage.setItem(`session_${sessionId}`, JSON.stringify(data))
        setSessionData(data)
        saveProject({
          ...proj,
          canvasWidthCm:
            typeof data.canvas_width_cm === 'number' && data.canvas_width_cm > 0
              ? data.canvas_width_cm
              : proj.canvasWidthCm,
          canvasHeightCm:
            typeof data.canvas_height_cm === 'number' && data.canvas_height_cm > 0
              ? data.canvas_height_cm
              : proj.canvasHeightCm,
        })
      } catch (e) {
        const msg = e instanceof Error ? e.message : 'Regenerate failed'
        alert(msg)
      } finally {
        setRegenerating(false)
      }
    } else {
      if (embedMode && typeof window !== 'undefined' && window.top !== window.self) {
        window.top!.location.href = `/?tab=image&session=${sessionId}`
      } else {
        router.push(`/upload?edit=${sessionId}&returnTo=home`)
      }
    }
  }, [sessionId, sessionData, embedMode, router])

  if (!sessionLoaded) {
    return (
      <div className="flex items-center justify-center py-12 text-gray-400">
        Loading session...
      </div>
    )
  }

  if (!sessionData) {
    return (
      <div className="space-y-6">
        {!embedMode && (
          <div className="flex items-center justify-between flex-wrap gap-4">
            <Link href="/" className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded">← Back to home</Link>
            <h1 className="text-xl font-bold">{projectName ? projectName : `Project: ${sessionId.slice(0, 8)}…`}</h1>
          </div>
        )}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-lg font-semibold mb-2">Generate Layers</h2>
          <p className="text-gray-400 text-sm mb-4">
            No layers generated yet. Upload an image and set options on the Image tab, then generate.
          </p>
          <button
            type="button"
            onClick={handleGenerateOrRegenerate}
            className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded font-medium"
          >
            Open Image tab to generate
          </button>
        </div>
      </div>
    )
  }

  const currentLayerData = sessionData.layers[currentLayer]
  const showLayersContent = panelMode === 'all' || panelMode === 'layers'
  const showProjectionContent = panelMode === 'all' || panelMode === 'projection'

  return (
    <div className="space-y-6">
      {!embedMode && (
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-2"
            >
              ← Back to home
            </Link>
            <Link
              href={`/upload?edit=${sessionId}`}
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-2"
            >
              Edit image & settings
            </Link>
            <h1 className="text-xl font-bold">
              {projectName ? projectName : `Project: ${sessionId.slice(0, 8)}…`}
            </h1>
            <button
              type="button"
              onClick={() => setShowDeleteConfirm(true)}
              className="ml-auto px-4 py-2 bg-red-900/80 hover:bg-red-800 text-red-200 rounded"
            >
              Delete project
            </button>
          </div>
        </div>
      )}

      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full border border-gray-600">
            <h2 className="text-lg font-semibold mb-2">Delete project?</h2>
            <p className="text-gray-400 text-sm mb-6">
              This will remove &quot;{projectName || sessionId.slice(0, 8)}&quot; from your project list and clear its saved data. This cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                type="button"
                onClick={() => setShowDeleteConfirm(false)}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleDeleteProject}
                className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {showLayersContent && (
        <>
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Generate / Regenerate Layers</h2>
            <p className="text-gray-400 text-sm mb-4">
              Regenerate layers with the current project settings (image is stored on the server). To change image or options, use Edit image &amp; settings.
            </p>
            <button
              type="button"
              onClick={handleGenerateOrRegenerate}
              disabled={regenerating}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded font-medium"
            >
              {regenerating ? 'Regenerating…' : 'Regenerate Layers'}
            </button>
          </div>

          <SessionResultsContent sessionId={sessionId} sessionData={sessionData} />
        </>
      )}

      {showProjectionContent && (
        <>
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Projection window</h2>
            <p className="text-gray-400 text-sm mb-4">
              Open the projection in a separate window so you can move it to your second monitor or projector. Use the controls below to change layer, mask, opacity, and other options; they sync with the projection window.
            </p>
            <button
              type="button"
              onClick={openProjectionWindow}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded font-medium"
            >
              Open projection window
            </button>
          </div>

          <ProjectionToolsPanel sessionId={sessionId} />

          <ProjectionHUDControls sessionId={sessionId} sessionData={sessionData} />
        </>
      )}
    </div>
  )
}

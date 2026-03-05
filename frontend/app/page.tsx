'use client'

import { useState, useEffect, useCallback } from 'react'
import { useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { getProjects, getProjectBySessionId, saveProject, removeProject, syncProjectsFromServer, type Project } from '@/lib/projects'
import { ProjectionControlPanel } from '@/app/project/[sessionId]/ProjectionControlPanel'

const TABS = ['file', 'settings', 'image', 'paint', 'layers', 'projection'] as const
type TabId = (typeof TABS)[number]

const CURRENT_SESSION_KEY = 'layerpainter_current_session_id'
const DEFAULT_PAINT_MARGIN_PERCENT = 33
const DEFAULT_CANVAS_WIDTH_CM = 50
const DEFAULT_CANVAS_HEIGHT_CM = 40

export default function Home() {
  const searchParams = useSearchParams()
  const tabParam = searchParams.get('tab') as TabId | null
  const sessionParam = searchParams.get('session')

  const [activeTab, setActiveTab] = useState<TabId>(TABS.includes(tabParam as TabId) ? (tabParam as TabId) : 'file')
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [projects, setProjects] = useState<Project[]>([])
  const [mounted, setMounted] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [paintMarginPercent, setPaintMarginPercent] = useState(DEFAULT_PAINT_MARGIN_PERCENT)
  const [canvasWidthCm, setCanvasWidthCm] = useState(DEFAULT_CANVAS_WIDTH_CM)
  const [canvasHeightCm, setCanvasHeightCm] = useState(DEFAULT_CANVAS_HEIGHT_CM)
  const [settingsSaved, setSettingsSaved] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted || typeof window === 'undefined') return
    void (async () => {
      await syncProjectsFromServer()
      setProjects(getProjects())
    })()
  }, [mounted, activeTab])

  useEffect(() => {
    if (!mounted || typeof window === 'undefined') return
    const marginRaw = localStorage.getItem('layerpainter_recipe_margin')
    if (marginRaw !== null) {
      const multiplier = parseFloat(marginRaw)
      if (!Number.isNaN(multiplier) && multiplier >= 1) setPaintMarginPercent(Math.round((multiplier - 1) * 100))
    }
    const settingsRaw = localStorage.getItem('layerpainter_settings')
    if (settingsRaw) {
      try {
        const parsed = JSON.parse(settingsRaw) as { canvasWidthCm?: number; canvasHeightCm?: number }
        if (typeof parsed.canvasWidthCm === 'number' && parsed.canvasWidthCm > 0) setCanvasWidthCm(parsed.canvasWidthCm)
        if (typeof parsed.canvasHeightCm === 'number' && parsed.canvasHeightCm > 0) setCanvasHeightCm(parsed.canvasHeightCm)
      } catch (_) {}
    }
  }, [mounted, activeTab])

  const saveLocalSettings = useCallback(() => {
    if (typeof window === 'undefined') return
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

    // Keep current project's canvas dimensions in sync with saved settings when a project is active.
    if (currentSessionId) {
      const currentProject = getProjectBySessionId(currentSessionId)
      if (currentProject) {
        saveProject({
          ...currentProject,
          canvasWidthCm,
          canvasHeightCm,
        })
      }
    }

    setSettingsSaved(true)
    setTimeout(() => setSettingsSaved(false), 2000)
  }, [paintMarginPercent, canvasWidthCm, canvasHeightCm, currentSessionId])

  useEffect(() => {
    if (tabParam && TABS.includes(tabParam as TabId)) setActiveTab(tabParam as TabId)
  }, [tabParam])

  useEffect(() => {
    if (sessionParam) {
      setCurrentSessionId(sessionParam)
      if (typeof window !== 'undefined') localStorage.setItem(CURRENT_SESSION_KEY, sessionParam)
      return
    }
    if (typeof window === 'undefined') return
    const stored = localStorage.getItem(CURRENT_SESSION_KEY)
    setCurrentSessionId(stored)
  }, [sessionParam])

  const setSessionAndTab = useCallback((sessionId: string | null, tab: TabId) => {
    setCurrentSessionId(sessionId)
    if (typeof window !== 'undefined') {
      if (sessionId) localStorage.setItem(CURRENT_SESSION_KEY, sessionId)
      else localStorage.removeItem(CURRENT_SESSION_KEY)
    }
    setActiveTab(tab)
    const url = new URL(window.location.href)
    url.searchParams.set('tab', tab)
    if (sessionId) url.searchParams.set('session', sessionId)
    else url.searchParams.delete('session')
    window.history.replaceState({}, '', url.toString())
  }, [])

  const currentProject = currentSessionId ? getProjectBySessionId(currentSessionId) : null

  const handleSave = useCallback(() => {
    if (!currentSessionId || typeof window === 'undefined') return
    const sessionJson = localStorage.getItem(`session_${currentSessionId}`)
    const proj = getProjectBySessionId(currentSessionId)
    if (!proj) return
    if (sessionJson) {
      try {
        const session = JSON.parse(sessionJson) as { canvas_width_cm?: number; canvas_height_cm?: number }
        saveProject({
          ...proj,
          canvasWidthCm: session.canvas_width_cm ?? proj.canvasWidthCm,
          canvasHeightCm: session.canvas_height_cm ?? proj.canvasHeightCm,
        })
      } catch {
        saveProject(proj)
      }
    } else {
      saveProject(proj)
    }
    void (async () => {
      await syncProjectsFromServer(true)
      setProjects(getProjects())
    })()
  }, [currentSessionId])

  const handleDownloadJson = useCallback(() => {
    if (!currentSessionId || typeof window === 'undefined') return
    const sessionJson = localStorage.getItem(`session_${currentSessionId}`)
    const proj = getProjectBySessionId(currentSessionId)
    const payload = {
      project: proj ?? null,
      session: sessionJson ? JSON.parse(sessionJson) : null,
      exportedAt: new Date().toISOString(),
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `layerpainter-${proj?.name?.replace(/\s+/g, '-') ?? currentSessionId.slice(0, 8)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [currentSessionId])

  const handleDelete = useCallback(() => {
    if (!currentSessionId || typeof window === 'undefined') return
    removeProject(currentSessionId)
    localStorage.removeItem(`session_${currentSessionId}`)
    localStorage.removeItem(`projection_current_layer_${currentSessionId}`)
    if (localStorage.getItem(CURRENT_SESSION_KEY) === currentSessionId) {
      localStorage.removeItem(CURRENT_SESSION_KEY)
      setCurrentSessionId(null)
    }
    setShowDeleteConfirm(false)
    void (async () => {
      await syncProjectsFromServer(true)
      setProjects(getProjects())
    })()
  }, [currentSessionId])

  const imageIframeSrc = currentSessionId
    ? `/upload?edit=${currentSessionId}&returnTo=home`
    : '/upload?new=1&returnTo=home'

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      <header className="border-b border-gray-700 px-4 py-3 flex items-center justify-between flex-wrap gap-2">
        <h1 className="text-xl font-bold">LayerPainter</h1>
        <nav className="flex gap-1" aria-label="Tabs">
          {TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              onClick={() => {
                setActiveTab(tab)
                const url = new URL(window.location.href)
                url.searchParams.set('tab', tab)
                if (currentSessionId) url.searchParams.set('session', currentSessionId)
                window.history.replaceState({}, '', url.toString())
              }}
              className={`px-4 py-2 rounded capitalize ${activeTab === tab ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white'}`}
            >
              {tab === 'paint' ? 'Paint Library' : tab === 'layers' ? 'Layers' : tab === 'projection' ? 'Projection' : tab === 'settings' ? 'Settings' : tab}
            </button>
          ))}
        </nav>
      </header>

      <main className="flex-1 overflow-auto p-6">
        <div className={activeTab !== 'file' ? 'hidden' : ''}>
          <div className="max-w-2xl mx-auto space-y-6">
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => setSessionAndTab(null, 'image')}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded font-medium"
              >
                New project
              </button>
              {currentSessionId && (
                <>
                  <button
                    type="button"
                    onClick={handleSave}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                  >
                    Project save
                  </button>
                  <button
                    type="button"
                    onClick={() => setSessionAndTab(currentSessionId, 'image')}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                    title="Change name in Image tab and Generate to create a new project"
                  >
                    Project save as
                  </button>
                  <button
                    type="button"
                    onClick={handleDownloadJson}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                  >
                    Download project JSON
                  </button>
                  <button
                    type="button"
                    onClick={() => setShowDeleteConfirm(true)}
                    className="px-4 py-2 bg-red-900/80 hover:bg-red-800 text-red-200 rounded"
                  >
                    Project delete
                  </button>
                </>
              )}
            </div>
            {currentSessionId && currentProject && (
              <p className="text-gray-400 text-sm">
                Current project: <span className="text-white font-medium">{currentProject.name || 'Untitled'}</span>
              </p>
            )}
            <div>
              <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">Recent projects</h2>
              {!mounted ? (
                <p className="text-gray-500">Loading…</p>
              ) : projects.length === 0 ? (
                <p className="text-gray-500">No projects yet. Create one with New project.</p>
              ) : (
                <ul className="space-y-2">
                  {projects.map((p) => (
                    <li key={p.sessionId}>
                      <button
                        type="button"
                        onClick={() => setSessionAndTab(p.sessionId, activeTab)}
                        className={`block w-full py-3 px-4 rounded-lg text-left transition-colors ${currentSessionId === p.sessionId ? 'bg-gray-700 border border-gray-600' : 'bg-gray-800 hover:bg-gray-700 border border-gray-700 hover:border-gray-600'}`}
                      >
                        <span className="font-medium text-white">{p.name || 'Untitled'}</span>
                        <span className="block text-xs text-gray-500 mt-0.5">
                          {p.imageFileName} · {p.canvasWidthCm}×{p.canvasHeightCm} cm
                          {p.libraryGroup && p.libraryGroup !== 'default' && ` · ${p.libraryGroup}`}
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </div>

        <div className={activeTab !== 'settings' ? 'hidden' : ''}>
          <div className="max-w-lg mx-auto space-y-6">
            <h2 className="text-xl font-bold">Settings</h2>

            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-semibold">Paint mix margin</h3>
              <p className="text-sm text-gray-400">
                Extra paint to order as a percentage (e.g. 33% = 1.33× calculated amount). Default 33%.
              </p>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={0}
                  max={200}
                  value={paintMarginPercent}
                  onChange={(e) => setPaintMarginPercent(Number(e.target.value) || 0)}
                  className="w-24 px-4 py-2 rounded-lg bg-gray-700 border border-gray-600 text-white focus:border-blue-500"
                />
                <span className="text-gray-400">%</span>
              </div>
            </div>

            <div className="bg-gray-800 rounded-lg p-6 space-y-4">
              <h3 className="text-lg font-semibold">Default canvas size (new projects)</h3>
              <p className="text-sm text-gray-400">
                Default width and height in cm used when creating a new project.
              </p>
              <div className="flex flex-wrap gap-4">
                <div className="flex items-center gap-2">
                  <label htmlFor="settings-canvas-width" className="text-sm text-gray-400">Width (cm)</label>
                  <input
                    id="settings-canvas-width"
                    type="number"
                    min={1}
                    step={0.1}
                    value={canvasWidthCm}
                    onChange={(e) => setCanvasWidthCm(Number(e.target.value) || 0)}
                    className="w-24 px-4 py-2 rounded-lg bg-gray-700 border border-gray-600 text-white focus:border-blue-500"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <label htmlFor="settings-canvas-height" className="text-sm text-gray-400">Height (cm)</label>
                  <input
                    id="settings-canvas-height"
                    type="number"
                    min={1}
                    step={0.1}
                    value={canvasHeightCm}
                    onChange={(e) => setCanvasHeightCm(Number(e.target.value) || 0)}
                    className="w-24 px-4 py-2 rounded-lg bg-gray-700 border border-gray-600 text-white focus:border-blue-500"
                  />
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={saveLocalSettings}
              className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded font-medium"
            >
              {settingsSaved ? 'Saved' : 'Save paint margin & default canvas'}
            </button>
          </div>
        </div>

        <div className={activeTab !== 'image' ? 'hidden' : ''}>
          <div className="h-[calc(100vh-8rem)] w-full">
            <iframe
              src={imageIframeSrc}
              title="Edit image and settings"
              className="w-full h-full rounded border border-gray-700 bg-gray-900"
            />
          </div>
        </div>

        <div className={activeTab !== 'paint' ? 'hidden' : ''}>
          <div className="h-[calc(100vh-8rem)] w-full">
            <iframe
              src="/paints"
              title="Paint Library"
              className="w-full h-full rounded border border-gray-700 bg-gray-900"
            />
          </div>
        </div>

        <div className={activeTab !== 'layers' ? 'hidden' : ''}>
          <div className="max-w-4xl mx-auto">
            {currentSessionId ? (
              <ProjectionControlPanel
                sessionId={currentSessionId}
                embedMode
                panelMode="layers"
                onProjectDeleted={() => {
                  setCurrentSessionId(null)
                  setProjects(getProjects())
                  const url = new URL(window.location.href)
                  url.searchParams.delete('session')
                  window.history.replaceState({}, '', url.toString())
                }}
              />
            ) : (
              <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-400">
                <p className="mb-4">Select a project from the File tab to view layers and palette.</p>
                <button
                  type="button"
                  onClick={() => setActiveTab('file')}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                >
                  Go to File tab
                </button>
              </div>
            )}
          </div>
        </div>

        <div className={activeTab !== 'projection' ? 'hidden' : ''}>
          <div className="max-w-4xl mx-auto">
            {currentSessionId ? (
              <ProjectionControlPanel
                sessionId={currentSessionId}
                embedMode
                panelMode="projection"
                onProjectDeleted={() => {
                  setCurrentSessionId(null)
                  setProjects(getProjects())
                  const url = new URL(window.location.href)
                  url.searchParams.delete('session')
                  window.history.replaceState({}, '', url.toString())
                }}
              />
            ) : (
              <div className="bg-gray-800 rounded-lg p-8 text-center text-gray-400">
                <p className="mb-4">Select a project from the File tab to open the projection launcher.</p>
                <button
                  type="button"
                  onClick={() => setActiveTab('file')}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                >
                  Go to File tab
                </button>
              </div>
            )}
          </div>
        </div>
      </main>

      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full border border-gray-600">
            <h2 className="text-lg font-semibold mb-2">Delete project?</h2>
            <p className="text-gray-400 text-sm mb-6">
              This will remove &quot;{currentProject?.name || currentSessionId?.slice(0, 8)}&quot; from your project list and clear its saved data. This cannot be undone.
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
                onClick={handleDelete}
                className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

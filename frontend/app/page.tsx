'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { getProjects, getProjectBySessionId, saveProject, removeProject, syncProjectsFromServer, type Project } from '@/lib/projects'
import { fetchProjectSession, fetchProjectState } from '@/lib/projectSession'
import { ProjectionControlPanel } from '@/app/project/[sessionId]/ProjectionControlPanel'
import { ProjectFileList } from '@/components/ProjectFileList'
import { AccountBar } from '@/components/AccountBar'
import { ProjectCollaborators } from '@/components/ProjectCollaborators'
import { TechniqueGuide } from '@/components/TechniqueGuide'

const PROJECT_TABS = ['projects', 'image', 'layers', 'projection', 'technique'] as const
type ProjectTab = (typeof PROJECT_TABS)[number]
type AdminSection = 'library' | 'users' | 'defaults'
type WorkspaceTab = ProjectTab | 'admin'

const AUTH_REQUIRED = process.env.NEXT_PUBLIC_AUTH_REQUIRED === '1'

function projectTabFromParam(tab: string | null): ProjectTab {
  if (tab === 'file' || tab === 'projects') return 'projects'
  if (tab === 'image' || tab === 'layers' || tab === 'projection' || tab === 'technique') return tab
  return 'projects'
}

function adminSectionFromParam(tab: string | null, section: string | null): AdminSection {
  if (tab === 'paint' || section === 'library') return 'library'
  if (tab === 'settings' || section === 'defaults') return 'defaults'
  if (section === 'users' && AUTH_REQUIRED) return 'users'
  return 'library'
}

function isAdminRequest(tab: string | null): boolean {
  return tab === 'admin' || tab === 'paint' || tab === 'settings'
}

const CURRENT_SESSION_KEY = 'layerpainter_current_session_id'
const DEFAULT_PAINT_MARGIN_PERCENT = 33
const DEFAULT_CANVAS_WIDTH_CM = 50
const DEFAULT_CANVAS_HEIGHT_CM = 40

export default function Home() {
  const searchParams = useSearchParams()
  const tabParam = searchParams.get('tab')
  const sectionParam = searchParams.get('section')
  const sessionParam = searchParams.get('session')

  const [activeTab, setActiveTab] = useState<WorkspaceTab>(() => (
    isAdminRequest(tabParam) && !AUTH_REQUIRED ? 'admin' : projectTabFromParam(tabParam)
  ))
  const [adminSection, setAdminSection] = useState<AdminSection>(() => adminSectionFromParam(tabParam, sectionParam))
  const [isAdmin, setIsAdmin] = useState(!AUTH_REQUIRED)
  const [roleReady, setRoleReady] = useState(!AUTH_REQUIRED)
  const [projectMenuOpen, setProjectMenuOpen] = useState(false)
  const projectMenuRef = useRef<HTMLDivElement>(null)
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [projects, setProjects] = useState<Project[]>([])
  const [mounted, setMounted] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [paintMarginPercent, setPaintMarginPercent] = useState(DEFAULT_PAINT_MARGIN_PERCENT)
  const [canvasWidthCm, setCanvasWidthCm] = useState(DEFAULT_CANVAS_WIDTH_CM)
  const [canvasHeightCm, setCanvasHeightCm] = useState(DEFAULT_CANVAS_HEIGHT_CM)
  const [settingsSaved, setSettingsSaved] = useState(false)
  const [sessionRevision, setSessionRevision] = useState(0)
  const [currentUserId, setCurrentUserId] = useState<string | null>(null)

  useEffect(() => {
    setMounted(true)
  }, [])

  useEffect(() => {
    if (process.env.NEXT_PUBLIC_AUTH_REQUIRED !== '1') return
    let active = true
    fetch('/api/auth/me', { credentials: 'include', cache: 'no-store' })
      .then(async (response) => (response.ok ? response.json() : null))
      .then((body) => {
        if (!active) return
        if (body?.id) setCurrentUserId(body.id)
        setIsAdmin(body?.role === 'admin')
        setRoleReady(true)
      })
      .catch(() => {
        if (active) setRoleReady(true)
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    if (!mounted || typeof window === 'undefined') return
    void (async () => {
      await syncProjectsFromServer()
      setProjects(getProjects())
    })()
  }, [mounted])

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
    if (!roleReady) return
    if (isAdminRequest(tabParam)) {
      if (isAdmin) {
        setActiveTab('admin')
        setAdminSection(adminSectionFromParam(tabParam, sectionParam))
      } else {
        setActiveTab('projects')
      }
      return
    }
    if (tabParam) setActiveTab(projectTabFromParam(tabParam))
  }, [roleReady, isAdmin, tabParam, sectionParam])

  useEffect(() => {
    if (!projectMenuOpen) return
    const close = (event: MouseEvent) => {
      if (!projectMenuRef.current?.contains(event.target as Node)) setProjectMenuOpen(false)
    }
    window.addEventListener('mousedown', close)
    return () => window.removeEventListener('mousedown', close)
  }, [projectMenuOpen])

  useEffect(() => {
    if (activeTab === 'layers' || activeTab === 'projection') {
      setSessionRevision((r) => r + 1)
    }
  }, [activeTab, currentSessionId])

  useEffect(() => {
    if (typeof window === 'undefined') return
    const onMessage = (event: MessageEvent) => {
      if (event.origin !== window.location.origin) return
      if (event.data?.type !== 'layerpainter-session-updated') return
      const id = event.data.sessionId as string | undefined
      if (!id) return
      setCurrentSessionId(id)
      localStorage.setItem(CURRENT_SESSION_KEY, id)
      const url = new URL(window.location.href)
      url.searchParams.set('session', id)
      window.history.replaceState({}, '', url.toString())
      setSessionRevision((r) => r + 1)
      void syncProjectsFromServer().then(() => setProjects(getProjects()))
    }
    window.addEventListener('message', onMessage)
    return () => window.removeEventListener('message', onMessage)
  }, [])

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

  const showWorkspace = useCallback((tab: WorkspaceTab, section?: AdminSection) => {
    if (tab === 'admin' && section) setAdminSection(section)
    setActiveTab(tab)
    setProjectMenuOpen(false)
    const url = new URL(window.location.href)
    url.searchParams.set('tab', tab)
    if (tab === 'admin') url.searchParams.set('section', section || adminSection)
    else url.searchParams.delete('section')
    if (currentSessionId) url.searchParams.set('session', currentSessionId)
    window.history.replaceState({}, '', url.toString())
  }, [adminSection, currentSessionId])

  const setSessionAndTab = useCallback((sessionId: string | null, tab: ProjectTab) => {
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
    const proj = getProjectBySessionId(currentSessionId)
    if (!proj) return
    void (async () => {
      const session = await fetchProjectSession(currentSessionId)
      if (session) {
        saveProject({
          ...proj,
          canvasWidthCm: session.canvas_width_cm ?? proj.canvasWidthCm,
          canvasHeightCm: session.canvas_height_cm ?? proj.canvasHeightCm,
        })
      } else {
        saveProject(proj)
      }
      await syncProjectsFromServer(true)
      setProjects(getProjects())
    })()
  }, [currentSessionId])

  const handleDownloadJson = useCallback(() => {
    if (!currentSessionId || typeof window === 'undefined') return
    const proj = getProjectBySessionId(currentSessionId)
    void (async () => {
      const [session, state] = await Promise.all([
        fetchProjectSession(currentSessionId),
        fetchProjectState(currentSessionId),
      ])
      const payload = {
        project: proj ?? null,
        session,
        state,
        exportedAt: new Date().toISOString(),
      }
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `layerpainter-${proj?.name?.replace(/\s+/g, '-') ?? currentSessionId.slice(0, 8)}.json`
      a.click()
      URL.revokeObjectURL(url)
    })()
  }, [currentSessionId])

  const handleDelete = useCallback(() => {
    if (!currentSessionId || typeof window === 'undefined') return
    removeProject(currentSessionId)
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
    <div className="h-screen min-h-0 bg-gray-900 text-white flex flex-col overflow-hidden">
      <header className="flex-shrink-0 border-b border-gray-700 bg-gray-900 px-4 py-3 flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-4 flex-wrap">
          <h1 className="text-xl font-bold">LayerPainter</h1>
          <nav className="flex gap-1" aria-label="Project">
            {PROJECT_TABS.map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => showWorkspace(tab)}
                className={`px-4 py-2 rounded capitalize ${activeTab === tab ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white'}`}
              >
                {tab}
              </button>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-3">
          {isAdmin ? (
            <button
              type="button"
              onClick={() => showWorkspace('admin', adminSection)}
              className={`px-4 py-2 rounded ${activeTab === 'admin' ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white'}`}
            >
              Admin
            </button>
          ) : null}
          <AccountBar />
        </div>
      </header>

      <main className="flex-1 min-h-0 overflow-auto p-6">
        <div className={activeTab !== 'projects' ? 'hidden' : ''}>
          <div className="max-w-4xl mx-auto space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => setSessionAndTab(null, 'image')}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded font-medium"
              >
                New project
              </button>
              {currentSessionId && (
                <div className="relative" ref={projectMenuRef}>
                  <button
                    type="button"
                    onClick={() => setProjectMenuOpen((open) => !open)}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                    aria-expanded={projectMenuOpen}
                    aria-haspopup="menu"
                  >
                    Project
                  </button>
                  {projectMenuOpen ? (
                    <div className="absolute left-0 z-20 mt-1 w-56 rounded-lg border border-gray-600 bg-gray-800 py-1" role="menu">
                      <button type="button" role="menuitem" onClick={() => { setProjectMenuOpen(false); handleSave() }} className="block w-full px-3 py-2 text-left text-sm hover:bg-gray-700">Save</button>
                      <button type="button" role="menuitem" title="Opens Image so you can rename it and generate a new project" onClick={() => { setProjectMenuOpen(false); setSessionAndTab(currentSessionId, 'image') }} className="block w-full px-3 py-2 text-left text-sm hover:bg-gray-700">Save as new project</button>
                      <button type="button" role="menuitem" onClick={() => { setProjectMenuOpen(false); handleDownloadJson() }} className="block w-full px-3 py-2 text-left text-sm hover:bg-gray-700">Download JSON</button>
                      {(!currentUserId || !currentProject?.ownerId || currentProject.ownerId === currentUserId) ? (
                        <button type="button" role="menuitem" onClick={() => { setProjectMenuOpen(false); setShowDeleteConfirm(true) }} className="block w-full px-3 py-2 text-left text-sm text-red-200 hover:bg-gray-700">Delete</button>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              )}
            </div>
            {currentSessionId && currentProject && (
              <p className="text-gray-400 text-sm">
                Current project: <span className="text-white font-medium">{currentProject.name || 'Untitled'}</span>
                {currentUserId && currentProject.ownerId && currentProject.ownerId !== currentUserId && (currentProject.collaboratorIds || []).includes(currentUserId) ? (
                  <span className="ml-2 text-blue-300">Shared with you</span>
                ) : null}
                {isAdmin && currentProject.owner ? (
                  <span className="ml-2 text-gray-500">
                    Owner {currentProject.owner.displayName || currentProject.owner.email || 'Unknown'}
                    {(currentProject.collaborators || []).length > 0
                      ? ` · With ${(currentProject.collaborators || []).map((person) => person.displayName || person.email || 'Unknown').join(', ')}`
                      : ' · No collaborators'}
                  </span>
                ) : null}
              </p>
            )}
            {currentSessionId && currentUserId && currentProject?.ownerId ? (
              <ProjectCollaborators
                projectId={currentSessionId}
                isOwner={currentProject.ownerId === currentUserId}
                isAdmin={isAdmin}
              />
            ) : null}
            {!mounted ? (
              <p className="text-gray-500 text-sm">Loading…</p>
            ) : projects.length === 0 ? (
              <p className="text-gray-500 text-sm">No projects yet. Create one with New project.</p>
            ) : (
              <ProjectFileList
                projects={projects}
                currentSessionId={currentSessionId}
                currentUserId={currentUserId}
                showPeople={isAdmin}
                onSelect={(sessionId) => setSessionAndTab(sessionId, 'projects')}
              />
            )}
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

        {isAdmin ? (
        <div className={activeTab !== 'admin' ? 'hidden' : ''}>
          <div className="max-w-6xl mx-auto space-y-4">
            <div className="flex gap-1" role="tablist" aria-label="Admin">
              {([
                ['library', 'Paint library'],
                ...(AUTH_REQUIRED ? [['users', 'Users'] as const] : []),
                ['defaults', 'Defaults'],
              ] as const).map(([section, label]) => (
                <button
                  key={section}
                  type="button"
                  role="tab"
                  aria-selected={adminSection === section}
                  onClick={() => showWorkspace('admin', section)}
                  className={`px-3 py-1.5 rounded text-sm ${adminSection === section ? 'bg-gray-700 text-white' : 'text-gray-400 hover:bg-gray-800 hover:text-white'}`}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className={adminSection !== 'library' ? 'hidden' : 'h-[calc(100vh-11rem)] w-full'}>
              <iframe src="/paints?embed=1" title="Paint library" className="w-full h-full rounded border border-gray-700 bg-gray-900" />
            </div>
            {AUTH_REQUIRED ? (
              <div className={adminSection !== 'users' ? 'hidden' : 'h-[calc(100vh-11rem)] w-full'}>
                <iframe src="/manage-users?embed=1" title="Users" className="w-full h-full rounded border border-gray-700 bg-gray-900" />
              </div>
            ) : null}
            {adminSection === 'defaults' ? (
              <div className="max-w-lg space-y-6">
                <h2 className="text-xl font-bold">Defaults</h2>
                <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                  <h3 className="text-lg font-semibold">Paint mix margin</h3>
                  <p className="text-sm text-gray-400">Extra paint to order as a percentage (e.g. 33% = 1.33× calculated amount). Default 33%.</p>
                  <div className="flex items-center gap-2">
                    <input type="number" min={0} max={200} value={paintMarginPercent} onChange={(e) => setPaintMarginPercent(Number(e.target.value) || 0)} className="w-24 px-4 py-2 rounded-lg bg-gray-700 border border-gray-600 text-white focus:border-blue-500" />
                    <span className="text-gray-400">%</span>
                  </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-6 space-y-4">
                  <h3 className="text-lg font-semibold">Default canvas size (new projects)</h3>
                  <p className="text-sm text-gray-400">Default width and height in cm used when creating a new project.</p>
                  <div className="flex flex-wrap gap-4">
                    <div className="flex items-center gap-2">
                      <label htmlFor="settings-canvas-width" className="text-sm text-gray-400">Width (cm)</label>
                      <input id="settings-canvas-width" type="number" min={1} step={0.1} value={canvasWidthCm} onChange={(e) => setCanvasWidthCm(Number(e.target.value) || 0)} className="w-24 px-4 py-2 rounded-lg bg-gray-700 border border-gray-600 text-white focus:border-blue-500" />
                    </div>
                    <div className="flex items-center gap-2">
                      <label htmlFor="settings-canvas-height" className="text-sm text-gray-400">Height (cm)</label>
                      <input id="settings-canvas-height" type="number" min={1} step={0.1} value={canvasHeightCm} onChange={(e) => setCanvasHeightCm(Number(e.target.value) || 0)} className="w-24 px-4 py-2 rounded-lg bg-gray-700 border border-gray-600 text-white focus:border-blue-500" />
                    </div>
                  </div>
                </div>
                <button type="button" onClick={saveLocalSettings} className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded font-medium">
                  {settingsSaved ? 'Saved' : 'Save paint margin & default canvas'}
                </button>
              </div>
            ) : null}
          </div>
        </div>
        ) : null}

        <div className={activeTab !== 'layers' ? 'hidden' : ''}>
          <div className="max-w-4xl mx-auto">
            {currentSessionId ? (
              <ProjectionControlPanel
                sessionId={currentSessionId}
                sessionRevision={sessionRevision}
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
                <p className="mb-4">Select a project from Projects to view layers and palette.</p>
                <button
                  type="button"
                  onClick={() => showWorkspace('projects')}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                >
                  Go to Projects
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
                sessionRevision={sessionRevision}
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
                <p className="mb-4">Select a project from Projects to open the projection launcher.</p>
                <button
                  type="button"
                  onClick={() => showWorkspace('projects')}
                  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded"
                >
                  Go to Projects
                </button>
              </div>
            )}
          </div>
        </div>

        <div className={activeTab !== 'technique' ? 'hidden' : ''}>
          <TechniqueGuide />
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

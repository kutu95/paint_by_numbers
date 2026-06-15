/**
 * Project session + UI state from server bundle (single source of truth).
 */
import { API_BASE_URL } from '@/lib/config'
import type { SessionData } from '@/app/project/[sessionId]/types'

export interface ProjectInfo {
  session_id: string
  project_id?: string
  original_url?: string | null
  priority_region_url?: string | null
  has_stored_image?: boolean
  has_artifacts?: boolean
  has_priority_region?: boolean
  must_include_colors?: string[]
  favor_skin_tones?: boolean | null
  skin_tone_strength?: number | null
}

export interface ProjectUiState {
  currentLayer?: number
  doneLayers?: number[]
  projectionScale?: number
  projectionHud?: Record<string, unknown>
}

/** Accepts structured HUD state objects; serialized as plain JSON for the API. */
export type ProjectUiStateInput = {
  currentLayer?: number
  doneLayers?: number[]
  projectionScale?: number
  projectionHud?: Record<string, unknown> | object
}

function normalizeProjectUiState(state: ProjectUiStateInput): ProjectUiState {
  return {
    currentLayer: state.currentLayer,
    doneLayers: state.doneLayers,
    projectionScale: state.projectionScale,
    projectionHud: state.projectionHud as Record<string, unknown> | undefined,
  }
}

export async function fetchProjectInfo(projectId: string): Promise<ProjectInfo | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/projects/${projectId}/info`, { cache: 'no-store' })
    if (!res.ok) return null
    return (await res.json()) as ProjectInfo
  } catch {
    return null
  }
}

export async function fetchProjectSession(projectId: string): Promise<SessionData | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/projects/${projectId}/session`, { cache: 'no-store' })
    if (!res.ok) return null
    return (await res.json()) as SessionData
  } catch {
    return null
  }
}

export async function fetchProjectState(projectId: string): Promise<ProjectUiState> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/projects/${projectId}/state`, { cache: 'no-store' })
    if (!res.ok) return {}
    return (await res.json()) as ProjectUiState
  } catch {
    return {}
  }
}

export async function saveProjectState(projectId: string, state: ProjectUiStateInput): Promise<void> {
  try {
    await fetch(`${API_BASE_URL}/api/projects/${projectId}/state`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(normalizeProjectUiState(state)),
    })
  } catch {
    /* ignore */
  }
}

/** @deprecated Use fetchProjectInfo */
export const fetchSessionInfo = fetchProjectInfo

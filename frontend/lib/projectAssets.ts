import { API_BASE_URL } from '@/lib/config'

/** Append a cache-bust query when project artifacts are regenerated at the same URL. */
export function projectAssetUrl(
  path: string | undefined | null,
  version?: number | null
): string {
  if (!path) return ''
  const base = path.startsWith('http') ? path : `${API_BASE_URL}${path}`
  if (version == null || version <= 0) return base
  const sep = base.includes('?') ? '&' : '?'
  return `${base}${sep}v=${version}`
}

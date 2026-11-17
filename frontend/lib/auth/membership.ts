import { Pool } from 'pg'

const USER_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

let pool: Pool | null = null

export type Membership = {
  userId: string
  role: 'admin' | 'user'
  displayName: string | null
  createdAt: string | null
}

function database(): Pool {
  const connectionString = process.env.DATABASE_URL
  if (!connectionString) {
    throw new Error('DATABASE_URL is not set')
  }
  if (!pool) {
    pool = new Pool({ connectionString, max: 4 })
  }
  return pool
}

function asRole(value: unknown): 'admin' | 'user' | null {
  return value === 'admin' || value === 'user' ? value : null
}

export function isUserId(value: string): boolean {
  return USER_ID.test(value)
}

export async function membershipFor(userId: string): Promise<Membership | null> {
  if (!isUserId(userId)) return null
  const result = await database().query(
    'SELECT user_id, role, display_name, created_at FROM layerpainter.user_roles WHERE user_id = $1',
    [userId],
  )
  const row = result.rows[0]
  if (!row) return null
  const role = asRole(row.role)
  if (!role) return null
  return {
    userId: String(row.user_id),
    role,
    displayName: row.display_name ? String(row.display_name) : null,
    createdAt: row.created_at ? new Date(row.created_at).toISOString() : null,
  }
}

export async function listMemberships(): Promise<Membership[]> {
  const result = await database().query(
    'SELECT user_id, role, display_name, created_at FROM layerpainter.user_roles ORDER BY created_at',
  )
  return result.rows.flatMap((row) => {
    const role = asRole(row.role)
    if (!role) return []
    return [{
      userId: String(row.user_id),
      role,
      displayName: row.display_name ? String(row.display_name) : null,
      createdAt: row.created_at ? new Date(row.created_at).toISOString() : null,
    }]
  })
}

export async function insertMembership(userId: string, role: 'admin' | 'user', displayName: string | null): Promise<boolean> {
  const result = await database().query(
    `INSERT INTO layerpainter.user_roles (user_id, role, display_name)
     VALUES ($1, $2, $3)
     ON CONFLICT (user_id) DO NOTHING
     RETURNING user_id`,
    [userId, role, displayName],
  )
  return result.rowCount === 1
}

export async function updateMembershipRole(userId: string, role: 'admin' | 'user'): Promise<void> {
  await database().query(
    'UPDATE layerpainter.user_roles SET role = $2 WHERE user_id = $1',
    [userId, role],
  )
}

export async function deleteMembership(userId: string): Promise<void> {
  await database().query('DELETE FROM layerpainter.user_roles WHERE user_id = $1', [userId])
}

export async function otherAdminCount(userId: string): Promise<number> {
  const result = await database().query(
    `SELECT count(*)::int AS n FROM layerpainter.user_roles WHERE role = 'admin' AND user_id <> $1`,
    [userId],
  )
  return Number(result.rows[0]?.n || 0)
}

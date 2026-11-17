export const COOKIE_NAME: string
export const SESSION_TTL_SECONDS: number

export function signSession(
  payload: { uid: string; email: string; exp: number },
  secret: string,
): Promise<string>

export function readSession(
  token: string,
  secret: string,
): Promise<{ uid: string; email: string; exp: number } | null>

export function sessionCookieOptions(maxAge?: number): {
  httpOnly: boolean
  secure: boolean
  sameSite: 'lax'
  path: string
  maxAge: number
}

export {
  COOKIE_NAME,
  SESSION_TTL_SECONDS,
  readSession,
  signSession,
} from './session-cookie.mjs'

export function sessionCookieOptions(maxAge = 7 * 24 * 60 * 60) {
  return {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax' as const,
    path: '/',
    maxAge,
  }
}

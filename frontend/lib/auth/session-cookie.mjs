const COOKIE_NAME = 'lp_session'
const SESSION_TTL_SECONDS = 7 * 24 * 60 * 60

function bytesToBase64Url(bytes) {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)
  let binary = ''
  for (let index = 0; index < view.length; index += 1) {
    binary += String.fromCharCode(view[index])
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '')
}

function base64UrlToBytes(value) {
  const padded = value.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (value.length % 4)) % 4)
  const binary = atob(padded)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
}

function payloadJson(email, exp, uid) {
  return JSON.stringify({ email, exp, uid })
}

async function hmacSha256(secret, message) {
  const key = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  )
  return new Uint8Array(await crypto.subtle.sign('HMAC', key, new TextEncoder().encode(message)))
}

export async function signSession(payload, secret) {
  const body = bytesToBase64Url(new TextEncoder().encode(payloadJson(payload.email, payload.exp, payload.uid)))
  const signature = await hmacSha256(secret, body)
  return `${body}.${bytesToBase64Url(signature)}`
}

export async function readSession(token, secret) {
  if (!token || !secret || !token.includes('.')) return null
  const [body, signature] = token.split('.')
  if (!body || !signature) return null
  const expected = bytesToBase64Url(await hmacSha256(secret, body))
  if (expected.length !== signature.length) return null
  let mismatch = 0
  for (let index = 0; index < expected.length; index += 1) {
    mismatch |= expected.charCodeAt(index) ^ signature.charCodeAt(index)
  }
  if (mismatch !== 0) return null
  try {
    const payload = JSON.parse(new TextDecoder().decode(base64UrlToBytes(body)))
    if (!payload || typeof payload.uid !== 'string' || typeof payload.email !== 'string' || typeof payload.exp !== 'number') {
      return null
    }
    if (payload.exp < Math.floor(Date.now() / 1000)) return null
    return payload
  } catch {
    return null
  }
}

export function sessionCookieOptions(maxAge = SESSION_TTL_SECONDS) {
  return {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    path: '/',
    maxAge,
  }
}

export { COOKIE_NAME, SESSION_TTL_SECONDS }

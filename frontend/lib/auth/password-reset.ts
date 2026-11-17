import { Resend } from 'resend'
import type { User } from '@supabase/supabase-js'
import { membershipFor } from '@/lib/auth/membership'
import { serviceSupabase } from '@/lib/auth/supabase-server'

const PRODUCTION_ORIGIN = 'https://layerpainter.margies.app'

export function appOrigin(request: { headers: Headers; nextUrl: URL }) {
  const forwardedHost = request.headers.get('x-forwarded-host')?.split(',')[0]?.trim()
  const host = forwardedHost || request.headers.get('host')?.split(',')[0]?.trim() || ''
  const hostname = host.replace(/^\[|\]$/g, '').split(':')[0]
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    const proto = request.headers.get('x-forwarded-proto')?.split(',')[0]?.trim() || 'http'
    return `${proto}://${host}`
  }
  return PRODUCTION_ORIGIN
}

export function isPlausibleEmail(value: string) {
  return value.length <= 320 && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)
}

function escapeHtml(value: string) {
  return value.replace(/[&<>"']/g, (character) => {
    switch (character) {
      case '&':
        return '&amp;'
      case '<':
        return '&lt;'
      case '>':
        return '&gt;'
      case '"':
        return '&quot;'
      default:
        return '&#39;'
    }
  })
}

export type PasswordEmailKind = 'reset' | 'welcome'

export type PasswordEmailResult =
  | { status: 'sent' }
  | { status: 'missing' }
  | { status: 'failed'; httpStatus: number; error: string }

export async function findAuthUserByEmail(email: string): Promise<User | null> {
  const supabase = serviceSupabase()
  for (let page = 1; page <= 20; page += 1) {
    const { data, error } = await supabase.auth.admin.listUsers({ page, perPage: 200 })
    if (error) throw error
    const found = data.users.find((user) => user.email?.toLowerCase() === email)
    if (found) return found
    if (data.users.length < 200) return null
  }
  return null
}

export async function sendPasswordEmail(
  request: { headers: Headers; nextUrl: URL },
  email: string,
  kind: PasswordEmailKind,
): Promise<PasswordEmailResult> {
  if (!process.env.RESEND_API_KEY || !process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
    console.error('Password email was not sent because email or auth is not configured')
    return { status: 'failed', httpStatus: 503, error: 'Could not send the email. Try again later.' }
  }

  let user: User | null
  try {
    user = await findAuthUserByEmail(email)
  } catch (error) {
    console.error('Could not look up the account:', error instanceof Error ? error.message : 'unknown')
    return { status: 'failed', httpStatus: 503, error: 'Could not send the email. Try again later.' }
  }
  if (!user) return { status: 'missing' }

  try {
    const membership = await membershipFor(user.id)
    if (!membership) return { status: 'missing' }
  } catch (error) {
    console.error('Could not check LayerPainter membership:', error instanceof Error ? error.message : 'unknown')
    return { status: 'failed', httpStatus: 503, error: 'Could not send the email. Try again later.' }
  }

  const supabase = serviceSupabase()
  const { data, error } = await supabase.auth.admin.generateLink({
    type: 'recovery',
    email,
  })
  if (error || !data.properties?.hashed_token) {
    console.error('Could not create a password link:', error?.message || 'missing token')
    return { status: 'failed', httpStatus: 503, error: 'Could not send the email. Try again later.' }
  }

  const resetUrl = new URL('/reset-password', appOrigin(request))
  resetUrl.searchParams.set('token_hash', data.properties.hashed_token)
  resetUrl.searchParams.set('type', 'recovery')
  const safeUrl = escapeHtml(resetUrl.toString())
  const subject = kind === 'welcome' ? 'Your LayerPainter account' : 'Reset your LayerPainter password'
  const intro = kind === 'welcome'
    ? 'An account has been created for you on LayerPainter. Choose a password to sign in.'
    : 'We received a request to reset the password for your LayerPainter account.'
  const closing = kind === 'welcome'
    ? 'This link opens the page where you set your password. It is valid for a short time. Ask an admin to send a new link if this one expires.'
    : 'This link is valid for a short time. If you did not request this, you can ignore this email. Your current password stays the same until you choose a new one. This is the same password used by the other apps on this login.'

  const { error: emailError } = await new Resend(process.env.RESEND_API_KEY).emails.send({
    from: 'LayerPainter <john@margies.app>',
    to: [email],
    subject,
    html: `
      <p>Hello,</p>
      <p>${intro}</p>
      <p><a href="${safeUrl}">Choose a password</a></p>
      <p>${closing}</p>
      <p>If the button does not work, copy this link into your browser:</p>
      <p>${safeUrl}</p>
      <p>Best regards,<br>LayerPainter</p>
    `,
  })
  if (emailError) {
    console.error('Resend password email error:', emailError)
    return { status: 'failed', httpStatus: 503, error: 'Could not send the email. Try again later.' }
  }
  return { status: 'sent' }
}

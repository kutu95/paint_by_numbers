import { NextResponse, type NextRequest } from 'next/server'
import { COOKIE_NAME, readSession } from '@/lib/auth/session-cookie'

const PUBLIC_PATHS = ['/login', '/forgot-password', '/reset-password']

function isPublic(pathname: string) {
  return PUBLIC_PATHS.some((path) => pathname === path || pathname.startsWith(`${path}/`))
}

export async function middleware(request: NextRequest) {
  if (process.env.NEXT_PUBLIC_AUTH_REQUIRED !== '1') {
    return NextResponse.next()
  }
  const { pathname, search } = request.nextUrl
  if (isPublic(pathname) || pathname.startsWith('/api/auth/')) {
    return NextResponse.next()
  }
  const secret = process.env.AUTH_SECRET || ''
  const token = request.cookies.get(COOKIE_NAME)?.value || ''
  const session = secret ? await readSession(token, secret) : null
  if (session) return NextResponse.next()
  if (pathname.startsWith('/api/')) {
    return NextResponse.json({ error: 'Sign in required' }, { status: 401 })
  }
  const url = request.nextUrl.clone()
  url.pathname = '/login'
  url.search = ''
  url.searchParams.set('redirectTo', `${pathname}${search}`)
  return NextResponse.redirect(url)
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico|icon.svg|icon-192.png|icon-512.png|apple-touch-icon.png|site.webmanifest).*)'],
}

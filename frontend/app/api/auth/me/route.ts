import { NextResponse } from 'next/server'
import { requireMember } from '@/lib/auth/member'

export const dynamic = 'force-dynamic'

export async function GET() {
  const auth = await requireMember()
  if (auth.error || !auth.member) return auth.error
  return NextResponse.json({
    id: auth.member.id,
    email: auth.member.email,
    role: auth.member.role,
    displayName: auth.member.displayName,
  })
}

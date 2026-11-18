import { redirect } from 'next/navigation'

export default function SettingsPage() {
  redirect('/?tab=admin&section=defaults')
}

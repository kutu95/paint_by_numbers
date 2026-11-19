/** @type {import('next').NextConfig} */
// Use IPv4 loopback to avoid intermittent localhost (::1 vs 127.0.0.1) proxy failures.
const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // fallback runs after static files and dynamic App Router routes.
    // afterFiles runs before dynamic routes, so it was stealing
    // /api/projects/[projectId]/collaborators and FastAPI then rejected the email.
    return {
      fallback: [
        {
          source: '/api/:path*',
          destination: `${backendUrl}/api/:path*`,
        },
      ],
    }
  },
}

module.exports = nextConfig

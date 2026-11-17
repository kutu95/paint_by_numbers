/** @type {import('next').NextConfig} */
// Use IPv4 loopback to avoid intermittent localhost (::1 vs 127.0.0.1) proxy failures.
const backendUrl = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // afterFiles: Next.js auth routes under /api/auth and /api/users stay in this app.
    // Every other /api request is proxied to FastAPI on the same origin, so the session cookie is sent.
    return {
      afterFiles: [
        {
          source: '/api/:path*',
          destination: `${backendUrl}/api/:path*`,
        },
      ],
    }
  },
}

module.exports = nextConfig

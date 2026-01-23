/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  env: process.env.NEXT_PUBLIC_API_URL ? {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  } : {},
  
  async rewrites() {
    if (process.env.NEXT_PUBLIC_USE_PROXY === 'true') {
      const backendUrl = process.env.BACKEND_URL || 'http://localhost:8000'
      return [
        {
          source: '/api/:path*',
          destination: `${backendUrl}/api/:path*`,
        },
      ]
    }
    return []
  },
}

module.exports = nextConfig

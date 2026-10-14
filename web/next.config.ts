import type { NextConfig } from "next";

const API = process.env.API_ORIGIN || "http://localhost:8000";

// The app is often reached through a tunnel rather than at localhost, and the
// dev server treats those as cross-origin when it serves /_next/* assets.
// Today that is a warning; a future major version refuses it, which would
// leave a shared link loading a page with no stylesheet and no JavaScript.
const TUNNEL_HOSTS = [
  process.env.VS_NGROK_DOMAIN,
  ".ngrok-free.dev",
  ".ngrok-free.app",
  ".trycloudflare.com",
].filter(Boolean) as string[];

const nextConfig: NextConfig = {
  // In dev, proxy /api/* to the FastAPI backend so the browser talks to one
  // origin. In production, set API_ORIGIN or put a reverse proxy in front.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
  allowedDevOrigins: TUNNEL_HOSTS,
};

export default nextConfig;

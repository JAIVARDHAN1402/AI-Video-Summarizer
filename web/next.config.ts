import type { NextConfig } from "next";

const API = process.env.API_ORIGIN || "http://localhost:8000";

const nextConfig: NextConfig = {
  // In dev, proxy /api/* to the FastAPI backend so the browser talks to one
  // origin. In production, set API_ORIGIN or put a reverse proxy in front.
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;

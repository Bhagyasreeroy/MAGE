import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactCompiler: true,
  // Required for the multi-stage Docker build (copies only what's needed)
  output: "standalone",
};

export default nextConfig;

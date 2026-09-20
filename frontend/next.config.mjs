/** @type {import('next').NextConfig} */
import { dirname } from "node:path"
import { fileURLToPath } from "node:url"

const frontendRoot = dirname(fileURLToPath(import.meta.url))

const nextConfig = {
  output: "standalone",
  // Avoid Next inferring a parent workspace from unrelated lockfiles.
  turbopack: {
    root: frontendRoot,
  },
  outputFileTracingRoot: frontendRoot,
  images: {
    unoptimized: true,
  },
}

export default nextConfig

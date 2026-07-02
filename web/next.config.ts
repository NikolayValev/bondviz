import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The command-panel routes import the package's client entry (for the
  // component registry), which pulls @ai-sdk/react → swr; swr's `react-server`
  // build has no default export, so bundling it into a route handler breaks.
  // Externalizing defers to Node's own resolution (normal `import` conditions)
  // at runtime, where the default export exists.
  serverExternalPackages: ["@nikolayvalev/command-panel"],
};

export default nextConfig;

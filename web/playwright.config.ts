import { defineConfig, devices } from "@playwright/test";

// Browser tests against a production build: run `npm run build` first, then `npm run e2e`.
// Every API route the pages call is mocked per test from e2e/fixtures (see e2e/mocks.ts), so a
// run never depends on the Treasury feed, Polygon or the LLM being reachable.
const PORT = 3200;

export default defineConfig({
  testDir: "e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  workers: process.env.CI ? 2 : undefined,
  reporter: "list",
  use: {
    baseURL: `http://localhost:${PORT}`,
    colorScheme: "dark",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1280, height: 900 } } },
    { name: "mobile", use: { ...devices["Pixel 5"] } },
  ],
  webServer: {
    command: `npx next start -p ${PORT}`,
    url: `http://localhost:${PORT}/about`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});

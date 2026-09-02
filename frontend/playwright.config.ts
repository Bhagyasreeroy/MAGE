import { defineConfig } from '@playwright/test';

/**
 * End-to-end tests run against an already-running stack (docker compose up),
 * not a server Playwright starts itself: the app needs the backend, Postgres
 * and a real OCR API key, which compose already wires together.
 *
 * `channel: 'chrome'` uses the system Chrome instead of downloading a
 * browser, so `npm ci` on a fresh machine stays fast and no post-install
 * browser fetch is required.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? 'http://localhost:3000',
    channel: 'chrome',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'setup', testMatch: /auth\.setup\.ts/ },
    {
      name: 'e2e',
      testIgnore: /auth\.setup\.ts/,
      dependencies: ['setup'],
      use: { storageState: 'e2e/.auth/user.json' },
    },
  ],
});

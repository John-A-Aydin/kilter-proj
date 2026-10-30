import { defineConfig } from '@playwright/test';
import { existsSync } from 'node:fs';

const port = process.env.PLAYWRIGHT_PORT || '4174';

export default defineConfig({
  testDir: './tests', testMatch: '**/*.spec.mjs', timeout: 90000, workers: 1,
  use: { baseURL: `http://127.0.0.1:${port}`, browserName: 'chromium',
    launchOptions: { executablePath: process.env.CHROME_PATH || (existsSync('/opt/google/chrome/chrome') ? '/opt/google/chrome/chrome' : undefined),
      args: ['--no-sandbox'] },
    viewport: { width: 1440, height: 1100 }, screenshot: 'only-on-failure',
  },
  webServer: { command: 'node server.mjs', url: `http://127.0.0.1:${port}`, env: { PORT: port }, reuseExistingServer: false },
});

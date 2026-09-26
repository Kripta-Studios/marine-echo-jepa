import { defineConfig } from '@playwright/test';
import { resolve } from 'node:path';
const copiedRelease = process.env.MARINE_RELEASE_PATH;
const artifactRoot = process.env.MARINE_ARTIFACT_ROOT || resolve(import.meta.dirname, '../release/demo-20260926-r2/artifacts');
const command = copiedRelease
  ? `powershell.exe -NoProfile -ExecutionPolicy Bypass -File "${resolve(copiedRelease, 'Run-Demo.ps1')}" -Port 8767`
  : `"${resolve(import.meta.dirname, '../.venv/Scripts/python.exe')}" -m marine_echo.serving.cli serve --port 8767 --artifact-root "${artifactRoot}"`;
export default defineConfig({
  testDir: './tests', timeout: 30000, workers: 1,
  use: { baseURL: 'http://127.0.0.1:8767', browserName: 'chromium' },
  webServer: { command, port: 8767, reuseExistingServer: false, timeout: 30000 },
  reporter: [['list'], ['json', {outputFile:'../evidence/browser/playwright-results.json'}]],
});

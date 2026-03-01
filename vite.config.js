import { defineConfig, searchForWorkspaceRoot } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import { execSync } from 'child_process';

const useLocalSource = process.env.TRANSFORMERS_LOCAL === 'true';

function readJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function getShortCommitHash(repoRoot) {
  try {
    return execSync('git rev-parse --short HEAD', {
      encoding: 'utf8',
      cwd: repoRoot,
    }).trim();
  } catch {
    return null;
  }
}

const appRoot = __dirname;
const transformersRepoRoot = path.resolve(__dirname, '../transformers.js');
const transformersWebEntry = path.resolve(
  transformersRepoRoot,
  'packages/transformers/dist/transformers.web.js',
);
const localPkg = readJson(path.resolve(transformersRepoRoot, 'packages/transformers/package.json'));
const npmPkg = readJson(path.resolve(appRoot, 'node_modules/@huggingface/transformers/package.json'));
const localVersion = localPkg?.version;
const npmVersion = npmPkg?.version;

const shortHash = getShortCommitHash(transformersRepoRoot);
let transformersVersion = useLocalSource ? localVersion : npmVersion;
let transformersSource = useLocalSource ? (shortHash ? `dev-${shortHash}` : 'dev') : 'npm';
if (!transformersVersion) {
  transformersVersion = localVersion || npmVersion || 'unknown';
  transformersSource = localVersion ? (shortHash ? `dev-${shortHash}` : 'dev') : 'unknown';
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    fs: {
      allow: [appRoot, transformersRepoRoot, searchForWorkspaceRoot(process.cwd())],
    },
  },
  resolve: {
    alias: useLocalSource
      ? {
        // Local browser-safe entrypoint from your transformers.js checkout.
        '@huggingface/transformers': transformersWebEntry,
      }
      : {},
  },
  define: {
    __TRANSFORMERS_VERSION__: JSON.stringify(transformersVersion),
    __TRANSFORMERS_SOURCE__: JSON.stringify(transformersSource),
  },
});

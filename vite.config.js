import { defineConfig, searchForWorkspaceRoot } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import { execSync } from 'child_process';
import { createRequire } from 'module';

const useLocalSource = process.env.TRANSFORMERS_LOCAL === 'true';
const basePath = process.env.VITE_BASE_PATH || '/';
const require = createRequire(import.meta.url);

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

function resolvePackageRoot(packageName, fromDir) {
  try {
    const resolved = require.resolve(packageName, { paths: [fromDir] });
    let dir = path.dirname(resolved);
    for (let i = 0; i < 8; i++) {
      if (fs.existsSync(path.join(dir, 'package.json'))) {
        return dir;
      }
      const parent = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  } catch {
    // Fall back to direct node_modules path if resolve fails.
  }
  return path.resolve(fromDir, 'node_modules', packageName);
}

function ensureOrtRuntimeAssets(ortRoot, appRoot) {
  const distDir = path.join(ortRoot, 'dist');
  const publicOrtDir = path.join(appRoot, 'public', 'ort');
  const files = [
    'ort-wasm-simd-threaded.asyncify.mjs',
    'ort-wasm-simd-threaded.asyncify.wasm',
  ];
  if (!fs.existsSync(distDir)) return;
  fs.mkdirSync(publicOrtDir, { recursive: true });
  for (const name of files) {
    const src = path.join(distDir, name);
    const dst = path.join(publicOrtDir, name);
    if (!fs.existsSync(src)) continue;
    // Update if missing or source changed.
    const needsCopy =
      !fs.existsSync(dst) ||
      fs.statSync(src).size !== fs.statSync(dst).size ||
      fs.statSync(src).mtimeMs > fs.statSync(dst).mtimeMs;
    if (needsCopy) {
      fs.copyFileSync(src, dst);
    }
  }
}

const appRoot = __dirname;
const transformersRepoRoot = path.resolve(__dirname, '../transformers.js');
const transformersWebEntry = path.resolve(
  transformersRepoRoot,
  'packages/transformers/dist/transformers.web.js',
);
const transformersPackageRoot = path.resolve(transformersRepoRoot, 'packages/transformers');
const transformersOrtWeb = resolvePackageRoot('onnxruntime-web', transformersPackageRoot);
const transformersOrtCommon = resolvePackageRoot('onnxruntime-common', transformersPackageRoot);
const transformersOrtWebgpuEntry = path.resolve(
  transformersOrtWeb,
  'dist/ort.webgpu.bundle.min.mjs',
);
ensureOrtRuntimeAssets(transformersOrtWeb, appRoot);
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
  base: basePath,
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
        // transformers.web.js imports this subpath directly.
        'onnxruntime-web/webgpu': transformersOrtWebgpuEntry,
        // Keep common aligned with local transformers workspace.
        'onnxruntime-common': transformersOrtCommon,
      }
      : {},
  },
  define: {
    __TRANSFORMERS_VERSION__: JSON.stringify(transformersVersion),
    __TRANSFORMERS_SOURCE__: JSON.stringify(transformersSource),
  },
});

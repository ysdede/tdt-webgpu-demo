# Transformers.js v4 Parakeet TDT Demo

[![Deploy Demo to GitHub Pages](https://github.com/ysdede/tdt-webgpu-demo/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/ysdede/tdt-webgpu-demo/actions/workflows/deploy-pages.yml)
[![Live Demo](https://img.shields.io/badge/Live-GitHub%20Pages-2ea44f?logo=github)](https://ysdede.github.io/tdt-webgpu-demo/)

Live demo: https://ysdede.github.io/tdt-webgpu-demo/

This project is a React + Vite web app for automatic speech recognition with Nemo Conformer TDT models using [Transformers.js](https://huggingface.co/docs/transformers.js) v4.

## Features

- Run Parakeet-style TDT ASR models in the browser, for example `ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4`.
- Choose encoder backend and dtype settings (WebGPU or WASM).
- Transcribe sample or uploaded audio.
- Inspect transcript, timestamps, confidence data, and raw JSON output.
- Run a Node.js CLI test script for quick non-UI checks.

## Requirements

- Node.js 18 or newer.
- A modern browser. WebGPU (Chrome or Edge) is recommended for best encoder performance.
- Optional local development setup for `transformers.js` as a sibling folder at `../transformers.js`.

## Install

```bash
git clone <this-repo-url>
cd transformers-v4-parakeet-demo
npm install
```

## Run

### NPM mode (default)

Uses `@huggingface/transformers@next` from npm:

```bash
npm run dev
```

Then open the URL shown by Vite (typically `http://localhost:5173`).

### Local source mode

Use this when you want to test local `transformers.js` changes without publishing a package.

1. Keep both repositories as siblings:
   - `.../transformers.js/`
   - `.../transformers-v4-parakeet-demo/`
2. Build transformers from the `transformers.js` root:
```bash
cd path/to/transformers.js
pnpm --filter @huggingface/transformers run build
```
3. Start the demo in local mode:
```bash
cd path/to/transformers-v4-parakeet-demo
npm run dev:local
```

`dev:local` sets `TRANSFORMERS_LOCAL=true` and aliases `@huggingface/transformers` to `../transformers.js/packages/transformers/dist/transformers.web.js`.

## Production Build

```bash
npm run build
npm run preview
```

## GitHub Pages Deployment

Deployment is handled by `.github/workflows/deploy-pages.yml`.

The workflow:

1. Checks out this demo repository.
2. Checks out `transformers.js` into a sibling directory.
3. Builds `@huggingface/transformers` from source with `pnpm`.
4. Builds this demo with `TRANSFORMERS_LOCAL=true`.
5. Publishes `dist/` to GitHub Pages.

Repository settings:

- Enable Pages and select `GitHub Actions` as the source.
- Optional repository variable `TRANSFORMERS_REPO` (default `ysdede/transformers.js`).
- Optional repository variable `TRANSFORMERS_REPO_REF` (default `v4-nemo-conformer-tdt-main`).
- Optional secret `TRANSFORMERS_REPO_TOKEN` if `transformers.js` is private.

Notes:

- GitHub Actions can only build commits that are pushed to GitHub.
- `workflow_dispatch` supports a `transformers_ref` input for one-off branch/tag/SHA overrides.
- The Vite base path is set automatically for both project pages and user pages.

## Node CLI Test

Run a quick transcription from the terminal:

```bash
npm run test:node -- --model ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4 --audio <path-to-wav-file> --encoder-device webgpu
```

By default, this script loads the local transformers build from `../transformers.js/packages/transformers/dist/transformers.node.mjs`.
Use `--npm` to use the installed npm package instead.
Node CLI input must be WAV (`.wav`).

| Option | Description |
|--------|-------------|
| `--model <id-or-path>` | Model ID or local model path |
| `--audio <wav-path>` | WAV file path |
| `--encoder-device <webgpu\|cpu>` | Encoder device (`cpu` is safer for Node) |
| `--encoder-dtype`, `--decoder-dtype` | Examples: `fp16`, `int8`, `fp32` |
| `--timestamps` | Request word-level timestamps |
| `--loop <n>` | Repeat transcription `n` times |
| `--npm` | Use `@huggingface/transformers` from `node_modules` |
| `--local-module <path>` | Path to a local transformers node build |

## Included Sample

Sample audio file used by the UI: `public/assets/Harvard-L2-1.ogg`.

## Additional Notes

- [Conformer TDT return granularity details](./docs/return-granularity.md)

## UI Notes

- Model configuration supports load mode, model ID, device, dtype, and WASM thread tuning.
- Transcription options include direct Nemo API mode and detailed output flags.
- UI includes transcript view, raw JSON output, metrics, and local history.
- Settings and theme preferences are persisted in `localStorage`.

## License

See the repository license.

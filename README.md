# Transformers.js v4 — Nemo Conformer TDT Demo

A React + Vite demo for **automatic speech recognition** using the Nemo Conformer TDT (Token-and-Duration Transducer) model with [Hugging Face Transformers.js](https://huggingface.co/docs/transformers.js) v4.

## What it does

- Load and run Parakeet-style TDT ASR models (e.g. `ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4`) in the browser.
- Configure encoder (WebGPU or WASM) and decoder (WASM) device/dtype.
- Transcribe sample or uploaded audio; view transcript and raw JSON output with timestamps and confidence.
- Optional Node script for quick CLI transcription without the browser.

## Requirements

- **Node.js** 18+
- **Browser**: WebGPU support recommended for best encoder performance (Chrome/Edge). WASM-only works without WebGPU.
- For **local transformers.js development**: a sibling checkout of [transformers.js](https://github.com/huggingface/transformers.js) at `../transformers.js` (see [Local source mode](#local-source-mode) below).

## Install

```bash
git clone <this-repo-url>
cd transformers-v4-parakeet-demo
npm install
```

## Run

### NPM package mode (default)

Uses `@huggingface/transformers@next` from npm. No extra setup.

```bash
npm run dev
```

Open the URL shown (e.g. http://localhost:5173). Load the model, then run the sample or upload an audio file.

### Local source mode (optional)

If you develop on [transformers.js](https://github.com/huggingface/transformers.js) and want to try changes without publishing:

1. Clone or place the transformers.js repo so this demo repo is a **sibling** of it, e.g.:
   - `…/transformers.js/`
   - `…/transformers-v4-parakeet-demo/`
2. Build the transformers.js package from the **transformers.js repo root**:
   ```bash
   cd path/to/transformers.js
   pnpm --filter @huggingface/transformers run build
   ```
3. Run the demo with the local build:
   ```bash
   cd path/to/transformers-v4-parakeet-demo
   npm run dev:local
   ```

`dev:local` sets `TRANSFORMERS_LOCAL=true` and aliases `@huggingface/transformers` to `../transformers.js/packages/transformers/dist/transformers.web.js`. Restart the dev server (or hard-refresh) after rebuilding transformers.js. The app header shows the loaded version and source (e.g. `4.0.0-next.x (dev-abc1234)` or `npm`).

## Build for production

```bash
npm run build
npm run preview   # optional: preview the built app
```

## Node quick test

Run transcription from the CLI (no browser):

```bash
npm run test:node -- --model ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4 --audio public/assets/life_Jim.wav --encoder-device webgpu
```

By default the script uses the local transformers build from `../transformers.js/packages/transformers/dist/transformers.node.mjs`. Use `--npm` to use the package from `node_modules` instead.

**Options:**

| Option | Description |
|--------|-------------|
| `--model <id-or-path>` | Model ID (e.g. on Hugging Face) or local path |
| `--audio <wav-path>` | Path to a WAV file |
| `--encoder-device <webgpu\|cpu>` | Encoder device (Node has no WebGPU; use `cpu` if needed) |
| `--encoder-dtype`, `--decoder-dtype` | e.g. `fp16`, `int8`, `fp32` |
| `--timestamps` | Request word-level timestamps |
| `--loop <n>` | Run transcription n times (for rough benchmarks) |
| `--npm` | Use `@huggingface/transformers` from node_modules |
| `--local-module <path>` | Path to a custom transformers build |

In Node, the decoder runs on CPU (WASM is browser-only). Use this for quick regression checks; browser/WebGPU issues still need to be tested in the browser.

## Sample audio

Included sample: `public/assets/life_Jim.wav`. You can replace it or point the UI/CLI to your own WAV files.

## UI overview

- **Model configuration**: Load mode (pipeline vs explicit), model ID, encoder/decoder device and dtype, WASM thread count.
- **Transcription options**: Direct Nemo API, timestamps, and detail flags (words, tokens, metrics, etc.).
- **Test & transcribe**: Sample button and drag-and-drop upload; performance metrics row; transcript and output JSON panels.
- **History**: Recent transcriptions (cleared on reload).

Settings and dark/light theme are persisted in `localStorage`.

## License

See the repository for license information.

# Transformers.js v4 Nemo Conformer TDT Demo

React + Vite demo app for testing your Parakeet TDT port in `transformers.js` v4.

## Location

`N:\github\ysdede\transformers-v4-parakeet-demo`

## Install

```bash
cd N:\github\ysdede\transformers-v4-parakeet-demo
npm install
```

## Run

### NPM package mode

Uses `@huggingface/transformers@next` from npm:

```bash
npm run dev
```

### Local source mode

Uses your local checkout at `N:\github\ysdede\transformers.js` via Vite alias:

```bash
npm run dev:local
```

This sets `TRANSFORMERS_LOCAL=true` and aliases:

- `@huggingface/transformers` -> `../transformers.js/packages/transformers/dist/transformers.web.js`

If you are actively editing `transformers.js`, run its local build/watch first so `dist/transformers.web.js` is up to date.

## Notes

- The UI exposes Nemo Conformer TDT timestamp controls (`timestamp_granularity`).
- For non-`nemo-conformer-tdt` models, only `return_timestamps` is sent.
- Encoder and decoder sessions are configurable independently in the UI:
  - `encoder_model`: device + dtype
  - `decoder_model_merged`: device + dtype
- Sample audio file is at `public/assets/life_Jim.wav`.

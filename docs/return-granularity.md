# Return granularity in Conformer TDT (transformers.js)

The Nemo Conformer TDT (Token-and-Duration Transducer) implementation in transformers.js does **not** use a single “granularity” option. Output detail is controlled by decode options passed to the model’s `transcribe()` (or indirectly via the ASR pipeline).

---

## 1. Where it’s set

- **Pipeline** (`pipeline('automatic-speech-recognition', ...)` then `pipe(audio, options)`): only **`return_timestamps`** is exposed. When `true`, the pipeline calls the model with utterance-level timestamps plus words and metrics (see below).
- **Direct model call** (`model.transcribe(inputs, decode_options)`): you pass **decode_options** to control exactly what is returned (utterance, words, tokens, metrics, debug).

---

## 2. Decode options (granularity levels)

These are the options that effectively define “return granularity” for Conformer TDT.

| Option | Type | Default | Effect |
|--------|------|--------|--------|
| **`return_timestamps`** | `boolean` | `true` | Utterance-level: adds `utterance_confidence`, `utterance_timestamp`, `confidence_scores` (token/word/frame averages). Base for words/tokens. |
| **`return_words`** | `boolean` | `false` | Word-level: adds `words` array. **Requires `return_timestamps`**. |
| **`return_tokens`** | `boolean` | `false` | Token-level: adds `tokens` array. **Requires `return_timestamps`**. |
| **`return_metrics`** | `boolean` | `false` | Timing: adds `metrics` (preprocess_ms, encode_ms, decode_ms, etc.). Independent of timestamps. |

**Output by level:**

- **`return_timestamps: false`** → `{ text, is_final }` only (plus `metrics` if `return_metrics`).
- **`return_timestamps: true`** → same + `utterance_confidence`, `utterance_timestamp`, `confidence_scores`.
- **`return_timestamps: true` + `return_words: true`** → also `words` (each with `text`, `start_time`, `end_time`, optional `confidence`).
- **`return_timestamps: true` + `return_tokens: true`** → also `tokens` (id, token, raw_token, is_word_start, start_time, end_time, optional confidence).

Optional debug flags (independent): `returnFrameConfidences`, `returnFrameIndices`, `returnLogProbs`, `returnTdtSteps`.  
Optional: **`timeOffset`** (seconds) added to all timestamps.

---

## 3. Pipeline vs direct call

**Pipeline (limited granularity):**

```js
const pipe = await pipeline('automatic-speech-recognition', modelId);
// Only return_timestamps is a pipeline argument:
const out = await pipe(audio, { return_timestamps: true });
// Pipeline internally uses: return_timestamps, return_words: true, return_metrics: true
```

You cannot set `return_tokens` or turn off words/metrics from the pipeline kwargs.

**Direct model call (full control):**

```js
const pipe = await pipeline('automatic-speech-recognition', modelId);
const inputs = await pipe.processor(audio);
const out = await pipe.model.transcribe(inputs, {
  tokenizer: pipe.tokenizer,
  return_timestamps: true,   // utterance-level
  return_words: true,        // word-level
  return_tokens: true,       // token-level
  return_metrics: true,
  timeOffset: 0,
});
```

Use direct `model.transcribe(..., decode_options)` when you need token-level output or want to disable words/metrics.

---

## 4. Example project (this repo)

The **transformers-v4-parakeet-demo** app shows both usages and exposes the decode options in the UI:

- **Transcription options** panel: “Direct Nemo call” uses `model.transcribe()` with:
  - **Return timestamps** → `return_timestamps`
  - **Words** → `return_words`
  - **Tokens** → `return_tokens`
  - **Metrics** → `return_metrics`
  - **Frame conf. / Frame idx / Log probs / TDT steps** → debug flags
  - **Time offset** → `timeOffset`

- With “Direct Nemo call” **off**, the app uses the pipeline and only **Return timestamps** is applied (pipeline then adds words + metrics internally).

Relevant code:

- **App (direct call):** `src/App.jsx` — `transcribeInput()` builds `decode_options` from UI state and calls `t.model.transcribe(inputs, { tokenizer, return_timestamps: rt, return_words: returnWords, return_tokens: returnTokens, ... })`.
- **Node script:** `scripts/node-asr-test.mjs` — uses pipeline only; `--timestamps` maps to `return_timestamps: true` in the pipeline call.

**Summary:** To set “return granularity” in transformers.js Conformer TDT, use **`return_timestamps`**, **`return_words`**, and **`return_tokens`** in the **decode_options** of `model.transcribe()`. The pipeline only exposes `return_timestamps`; for word/token level and full control, call `model.transcribe()` directly as in the demo.

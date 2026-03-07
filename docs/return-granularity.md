# Return granularity in Conformer TDT (transformers.js)

The Nemo Conformer TDT implementation now follows the shared `automatic-speech-recognition` pipeline contract more closely:

- **pipeline mode** returns `{ text }` by default and `{ text, chunks }` when timestamps are requested
- **direct model calls** keep the richer Nemo-native outputs such as `words`, `tokens`, confidences, metrics, and debug fields

---

## 1. Pipeline mode

Pipeline mode is for task-level compatibility:

```js
const pipe = await pipeline('automatic-speech-recognition', modelId);
```

Supported timestamp behaviors:

| Option | Output |
|--------|--------|
| `return_timestamps: false` | `{ text }` |
| `return_timestamps: true` | `{ text, chunks }` with segment-level timestamped chunks |
| `return_timestamps: 'word'` | `{ text, chunks }` with word-level timestamped chunks |

Examples:

```js
await pipe(audio);
// { text }

await pipe(audio, { return_timestamps: true });
// { text, chunks: [{ text, timestamp: [start, end] }, ...] }

await pipe(audio, { return_timestamps: 'word' });
// { text, chunks: [{ text, timestamp: [start, end] }, ...] }
```

Notes:

- Pipeline mode does **not** expose `return_words`, `return_tokens`, or the debug flags.
- For long audio, Nemo may internally use windowing/merge, but the public pipeline shape remains the same.

---

## 2. Direct model call

Direct calls expose the full Nemo decode options:

```js
const pipe = await pipeline('automatic-speech-recognition', modelId);
const inputs = await pipe.processor(audio);
const out = await pipe.model.transcribe(inputs, {
  tokenizer: pipe.tokenizer,
  return_timestamps: true,
  return_words: true,
  return_tokens: true,
  return_metrics: true,
  timeOffset: 0,
});
```

Main options:

| Option | Effect |
|--------|--------|
| `return_timestamps` | Enables utterance timestamps/confidence output |
| `return_words` | Adds `words` |
| `return_tokens` | Adds `tokens` |
| `return_metrics` | Adds timing metrics |
| `returnFrameConfidences` | Debug frame confidences |
| `returnFrameIndices` | Debug frame indices |
| `returnLogProbs` | Debug log probs |
| `returnTdtSteps` | Debug TDT step trace |
| `timeOffset` | Adds a global timestamp offset in seconds |

Typical direct outputs:

- `{ text }`
- `{ text, utterance_timestamp, utterance_confidence, confidence_scores }`
- same plus `words`
- same plus `tokens`
- same plus `metrics` and debug fields

---

## 3. Demo app behavior

The demo app in this repo now mirrors that split:

- **Direct Nemo call: on**
  - `Return timestamps` is a boolean
  - `Words`, `Tokens`, `Metrics`, and debug flags are available
- **Direct Nemo call: off**
  - the app uses pipeline mode
  - `Pipeline timestamps` has three options:
    - `off`
    - `segments`
    - `words`

Relevant code:

- [App.jsx](N:\github\ysdede\transformers-v4-parakeet-demo\src\App.jsx)

---

## 4. Summary

Use **pipeline mode** when you want compatibility with the standard ASR task API.  
Use **direct `model.transcribe()`** when you want the full Nemo-native outputs.

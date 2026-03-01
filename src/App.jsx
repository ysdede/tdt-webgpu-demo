import { useEffect, useMemo, useRef, useState } from 'react';
import * as Transformers from '@huggingface/transformers';
import './App.css';

const MODEL_DEFAULT = 'ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4';
const SAMPLE = '/assets/life_Jim.wav';
const DECODER_DEVICE = 'wasm';
const DTYPES = ['fp16', 'int8', 'fp32'];

const SETTINGS_STORAGE_KEY = 'nemo-tdt-demo.settings.v1';

const VERSION = typeof __TRANSFORMERS_VERSION__ !== 'undefined' ? __TRANSFORMERS_VERSION__ : 'unknown';
const SOURCE = typeof __TRANSFORMERS_SOURCE__ !== 'undefined' ? __TRANSFORMERS_SOURCE__ : 'unknown';

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch {
    return {};
  }
}

function saveSettings(settings) {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Ignore storage failures (private mode/quota).
  }
}

const detectMaxCores = () => {
  const c = Number(globalThis?.navigator?.hardwareConcurrency ?? 1);
  return Number.isFinite(c) && c > 0 ? Math.floor(c) : 1;
};

const clampThreadCount = (value, maxCores) => {
  const n = Number.parseInt(String(value), 10);
  if (!Number.isFinite(n)) return 1;
  return Math.max(1, Math.min(maxCores, n));
};

const MAX_WASM_CORES = detectMaxCores();
const DEFAULT_WASM_THREADS = Math.max(1, Math.floor(MAX_WASM_CORES / 2));
let ortWasmBlobInitPromise = null;
let ortWasmBlobUrls = null;

if (typeof window !== 'undefined') {
  const { env } = Transformers;
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
  if (env?.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.proxy = false;
    env.backends.onnx.wasm.wasmPaths = '/ort/';
  }
}

async function ensureLocalOrtWasmBlobs() {
  if (ortWasmBlobInitPromise) return ortWasmBlobInitPromise;
  ortWasmBlobInitPromise = (async () => {
    const { env } = Transformers;
    if (!env?.backends?.onnx?.wasm) return;
    if (ortWasmBlobUrls) {
      env.backends.onnx.wasm.wasmPaths = ortWasmBlobUrls;
      return;
    }

    const mjsSrc = '/ort/ort-wasm-simd-threaded.asyncify.mjs';
    const wasmSrc = '/ort/ort-wasm-simd-threaded.asyncify.wasm';
    const [mjsResp, wasmResp] = await Promise.all([fetch(mjsSrc), fetch(wasmSrc)]);
    if (!mjsResp.ok) throw new Error(`Failed to fetch local ORT wasm factory: ${mjsResp.status}`);
    if (!wasmResp.ok) throw new Error(`Failed to fetch local ORT wasm binary: ${wasmResp.status}`);

    const [mjsText, wasmBlobRaw] = await Promise.all([mjsResp.text(), wasmResp.blob()]);
    const mjsBlob = new Blob([mjsText], { type: 'text/javascript' });
    const wasmBlob = new Blob([wasmBlobRaw], { type: 'application/wasm' });
    ortWasmBlobUrls = {
      mjs: URL.createObjectURL(mjsBlob),
      wasm: URL.createObjectURL(wasmBlob),
    };
    env.backends.onnx.wasm.wasmPaths = ortWasmBlobUrls;
  })();
  return ortWasmBlobInitPromise;
}

const toJSON = (x) => { try { return JSON.stringify(x, null, 2); } catch { return String(x); } };
const textOf = (o) => (typeof o === 'string' ? o : o?.text ?? '');

async function decodeAudio(input, sampleRate) {
  if (input instanceof Float32Array) return input;
  let ab;
  if (typeof input === 'string') {
    const r = await fetch(input);
    if (!r.ok) throw new Error(`Audio fetch failed: ${r.status}`);
    ab = await r.arrayBuffer();
  } else {
    ab = await input.arrayBuffer();
  }
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx({ sampleRate });
  try {
    const b = await ctx.decodeAudioData(ab.slice(0));
    const out = new Float32Array(b.length);
    for (let c = 0; c < b.numberOfChannels; c++) {
      const d = b.getChannelData(c);
      for (let i = 0; i < b.length; i++) out[i] += d[i] / b.numberOfChannels;
    }
    return out;
  } finally { await ctx.close(); }
}

function Toggle({ id, checked, onChange, disabled }) {
  return (
    <div className="relative inline-block w-10 mr-2 align-middle select-none transition duration-200 ease-in">
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        disabled={disabled}
        className="toggle-checkbox absolute block w-5 h-5 rounded-full bg-white appearance-none cursor-pointer border border-border-light dark:border-border-dark shadow-none checked:right-0 disabled:opacity-40"
        id={id}
      />
      <label
        htmlFor={id}
        className="toggle-label block overflow-hidden h-5 rounded-full bg-gray-300 dark:bg-gray-600 cursor-pointer"
      />
    </div>
  );
}

function SelectField({ label, value, onChange, disabled, children }) {
  return (
    <div>
      <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
        {label}
      </label>
      <div className="relative">
        <select
          value={value}
          onChange={onChange}
          disabled={disabled}
          className="w-full bg-gray-50 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-muted focus:border-primary-muted dark:focus:ring-accent-muted dark:focus:border-accent-muted dark:text-white appearance-none"
        >
          {children}
        </select>
        <span className="material-icons-outlined absolute right-2 top-2 text-gray-400 pointer-events-none text-lg">
          expand_more
        </span>
      </div>
    </div>
  );
}

function MetricRow({ items }) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1 text-sm">
      {items.map(({ label, value }) => (
        <span key={label} className="flex items-baseline gap-2 min-w-0">
          <span className="text-primary-muted dark:text-accent-muted font-medium shrink-0">{label}</span>
          <span className="font-mono text-gray-900 dark:text-white tabular-nums truncate">
            {value != null ? value : '-'}
          </span>
        </span>
      ))}
    </div>
  );
}

function PerformanceMetrics({ stats }) {
  const fmt = (v, unit) => (v != null && Number.isFinite(v) ? `${v.toFixed(1)}${unit}` : '-');
  const items = [
    { label: 'Preprocess', value: fmt(stats.preprocessMs, 'ms') },
    { label: 'Encode', value: fmt(stats.encodeMs, 'ms') },
    { label: 'Decode', value: fmt(stats.decodeMs, 'ms') },
    { label: 'Tokenize', value: fmt(stats.tokenizeMs, 'ms') },
    { label: 'Total', value: fmt(stats.totalMs, 'ms') },
    { label: 'RTFx', value: stats.rtfx != null && Number.isFinite(stats.rtfx) ? `${stats.rtfx.toFixed(1)}x` : null },
  ];
  return (
    <div className="bg-card-light dark:bg-card-dark rounded-xl border border-border-light dark:border-border-dark px-4 py-3">
      <MetricRow items={items} />
    </div>
  );
}

export default function App() {
  const initialSettings = loadSettings();
  const tRef = useRef(null);
  const fileInputRef = useRef(null);
  const { pipeline } = Transformers;
  const [modelId, setModelId] = useState(initialSettings.modelId || MODEL_DEFAULT);
  const [mode, setMode] = useState(initialSettings.mode || 'pipeline');
  const [encDev, setEncDev] = useState(initialSettings.encDev || 'webgpu');
  const [encDtype, setEncDtype] = useState(initialSettings.encDtype || 'fp16');
  const [decDtype, setDecDtype] = useState(initialSettings.decDtype || 'int8');
  const [mType, setMType] = useState('not-loaded');
  const [maxWasmCores] = useState(MAX_WASM_CORES);
  const [wasmThreads, setWasmThreads] = useState(
    Number.isInteger(initialSettings.wasmThreads)
      ? clampThreadCount(initialSettings.wasmThreads, MAX_WASM_CORES)
      : DEFAULT_WASM_THREADS
  );

  const [direct, setDirect] = useState(initialSettings.direct !== undefined ? Boolean(initialSettings.direct) : true);
  const [rt, setRt] = useState(initialSettings.rt !== undefined ? Boolean(initialSettings.rt) : true);
  const [metrics, setMetrics] = useState(initialSettings.metrics !== undefined ? Boolean(initialSettings.metrics) : true);

  const [returnWords, setReturnWords] = useState(initialSettings.returnWords !== undefined ? Boolean(initialSettings.returnWords) : true);
  const [returnTokens, setReturnTokens] = useState(initialSettings.returnTokens !== undefined ? Boolean(initialSettings.returnTokens) : true);
  const [returnFrameConf, setReturnFrameConf] = useState(Boolean(initialSettings.returnFrameConf));
  const [frameIdx, setFrameIdx] = useState(Boolean(initialSettings.frameIdx));
  const [logProbs, setLogProbs] = useState(Boolean(initialSettings.logProbs));
  const [tdtSteps, setTdtSteps] = useState(Boolean(initialSettings.tdtSteps));
  const [offset, setOffset] = useState(initialSettings.offset !== undefined ? String(initialSettings.offset) : '0');

  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [status, setStatus] = useState('Idle');
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedFileName, setSelectedFileName] = useState('');
  const [darkMode, setDarkMode] = useState(Boolean(initialSettings.darkMode));
  const [downloadProgress, setDownloadProgress] = useState(null); // {pct, loaded, total, file}
  const [isCached, setIsCached] = useState(null); // null=unknown, true/false

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  useEffect(() => {
    const { env } = Transformers;
    if (!env?.backends?.onnx?.wasm) return;
    env.backends.onnx.wasm.numThreads = clampThreadCount(wasmThreads, maxWasmCores);
  }, [wasmThreads, maxWasmCores]);

  useEffect(() => {
    saveSettings({
      modelId, mode, encDev, encDtype, decDtype, wasmThreads,
      direct, rt, metrics,
      returnWords, returnTokens, returnFrameConf, frameIdx, logProbs, tdtSteps,
      offset, darkMode,
    });
  }, [
    modelId, mode, encDev, encDtype, decDtype, wasmThreads,
    direct, rt, metrics,
    returnWords, returnTokens, returnFrameConf, frameIdx, logProbs, tdtSteps,
    offset, darkMode,
  ]);

  const stats = useMemo(() => {
    const words = Array.isArray(result?.words) ? result.words.length : null;
    const tokens = Array.isArray(result?.tokens) ? result.tokens.length : null;
    const metricsOut = result?.metrics ?? null;
    const rtf = metricsOut?.rtf ?? null;
    const rtfx = metricsOut?.rtf_x ?? (rtf && Number.isFinite(rtf) && rtf > 0 ? 1 / rtf : null);
    return {
      textLen: textOf(result).length,
      words,
      tokens,
      tAvg: result?.confidence_scores?.token_avg ?? null,
      wAvg: result?.confidence_scores?.word_avg ?? null,
      preprocessMs: metricsOut?.preprocess_ms ?? null,
      encodeMs: metricsOut?.encode_ms ?? null,
      decodeMs: metricsOut?.decode_ms ?? null,
      tokenizeMs: metricsOut?.tokenize_ms ?? null,
      totalMs: metricsOut?.total_ms ?? null,
      rtf,
      rtfx,
    };
  }, [result]);

  async function checkCached(id) {
    try {
      const { ModelRegistry } = Transformers;
      if (!ModelRegistry?.is_pipeline_cached) { setIsCached(null); return; }
      const opts = { dtype: { encoder_model: encDtype, decoder_model_merged: decDtype } };
      const result = await ModelRegistry.is_pipeline_cached('automatic-speech-recognition', id, opts);
      setIsCached(result?.allCached ?? null);
    } catch {
      setIsCached(null);
    }
  }

  async function load() {
    setLoading(true); setError(''); setStatus('Loading...'); setResult(null); setDownloadProgress(null);
    await checkCached(modelId);
    try {
      await ensureLocalOrtWasmBlobs();

      const onProgress = (info) => {
        if (info.status === 'progress_total') {
          // Aggregate progress across all files (from upstream ModelRegistry commit)
          setDownloadProgress({
            pct: Math.round(info.progress),
            loaded: info.loaded,
            total: info.total,
            file: info.name ?? '',
          });
        } else if (info.status === 'progress') {
          // Fallback: per-file progress when aggregate total isn't available
          const pct = info.total > 0 ? Math.round((info.loaded / info.total) * 100) : 0;
          setDownloadProgress((prev) => ({
            pct,
            loaded: info.loaded,
            total: info.total,
            file: info.file ?? prev?.file ?? '',
          }));
        } else if (info.status === 'ready') {
          setDownloadProgress(null);
        }
      };

      const opts = {
        progress_callback: onProgress,
        device: { encoder_model: encDev, decoder_model_merged: DECODER_DEVICE },
        dtype: { encoder_model: encDtype, decoder_model_merged: decDtype },
        session_options: { logId: `demo-${Date.now()}`, logSeverityLevel: 2 },
      };
      const AutoProcessorCtor = Transformers['AutoProcessor'];
      const AutoTokenizerCtor = Transformers['AutoTokenizer'];
      const NemoCtor = Transformers['NemoConformerForTDT'];
      const AsrPipelineCtor = Transformers['AutomaticSpeechRecognitionPipeline'];
      const canExplicit =
        typeof AutoProcessorCtor?.from_pretrained === 'function' &&
        typeof AutoTokenizerCtor?.from_pretrained === 'function' &&
        typeof NemoCtor?.from_pretrained === 'function' &&
        typeof AsrPipelineCtor === 'function';

      let t;
      if (mode === 'explicit' && canExplicit) {
        const [processor, tokenizer, model] = await Promise.all([
          AutoProcessorCtor.from_pretrained(modelId, { progress_callback: onProgress }),
          AutoTokenizerCtor.from_pretrained(modelId, { progress_callback: onProgress }),
          NemoCtor.from_pretrained(modelId, opts),
        ]);
        t = new AsrPipelineCtor({
          task: 'automatic-speech-recognition',
          model,
          processor,
          tokenizer,
        });
      } else {
        t = await pipeline('automatic-speech-recognition', modelId, opts);
        if (mode === 'explicit' && !canExplicit) {
          console.warn('[App] Explicit mode unavailable, fell back to pipeline()');
        }
      }
      tRef.current = t;
      setMType(t?.model?.config?.model_type || 'unknown');

      setStatus('Verifying...');
      const expectedText = 'it is not life as we know or understand it';
      try {
        const pcm = await decodeAudio(SAMPLE, 16000);
        const useDirect = t?.model?.config?.model_type === 'nemo-conformer-tdt';
        let warmupText = '';
        if (useDirect) {
          const inputs = await t.processor(pcm);
          const res = await t.model.transcribe(inputs, { tokenizer: t.tokenizer });
          warmupText = res.text ?? '';
        } else {
          const res = await t(pcm);
          warmupText = textOf(res);
        }
        const normalize = (s) => s.toLowerCase().replace(/[.,/#!$%^&*;:{}=\-_`~()]/g, '').trim();
        if (normalize(warmupText).includes(normalize(expectedText))) {
          console.log('[App] Warm-up verification passed');
          setModelLoaded(true);
          setStatus('Model ready');
        } else {
          console.warn(`[App] Warm-up mismatch. Expected "${expectedText}", got "${warmupText}"`);
          setModelLoaded(true);
          setStatus('Model ready (warm-up text mismatch)');
        }
      } catch (warmupErr) {
        console.error('[App] Warm-up failed:', warmupErr);
        setModelLoaded(true);
        setStatus('Model ready (warm-up failed)');
      }
    } catch (e) {
      setError(e?.message || String(e));
      setStatus('Load failed');
    } finally { setLoading(false); setDownloadProgress(null); }
  }

  async function transcribeInput(input, name) {
    if (!tRef.current) return;
    setRunning(true); setError(''); setStatus(`Transcribing ${name}...`);
    try {
      const t = tRef.current;
      const useDirect = direct && mType === 'nemo-conformer-tdt';
      let out;
      if (useDirect) {
        const sr = t.processor.feature_extractor?.config?.sampling_rate;
        const mono = await decodeAudio(input, sr);
        const inputs = await t.processor(mono);
        out = await t.model.transcribe(inputs, {
          tokenizer: t.tokenizer,
          return_timestamps: rt,
          return_words: returnWords,
          return_tokens: returnTokens,
          return_metrics: metrics,
          returnFrameConfidences: returnFrameConf,
          returnFrameIndices: frameIdx,
          returnLogProbs: logProbs,
          returnTdtSteps: tdtSteps,
          timeOffset: Number(offset) || 0,
        });
      } else {
        const p = rt ? { return_timestamps: true } : {};
        out = await t(input, p);
      }
      setResult(out);
      setHistory((h) => [{ id: `${Date.now()}`, name, mode: useDirect ? 'direct' : 'pipeline', text: textOf(out) }, ...h].slice(0, 25));
      setStatus('Done');
    } catch (e) {
      setError(e?.message || String(e));
      setStatus('Failed');
    } finally { setRunning(false); }
  }

  function handleFileUpload(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setSelectedFileName(f.name);
    const u = URL.createObjectURL(f);
    transcribeInput(u, f.name).finally(() => URL.revokeObjectURL(u));
  }

  function copyToClipboard(text) {
    navigator.clipboard.writeText(text);
  }

  const canRun = !!tRef.current && modelLoaded && !loading && !running;
  const isModelReady = modelLoaded;
  const configDisabled = loading || running || modelLoaded;

  return (
    <div className="bg-background-light dark:bg-background-dark text-gray-800 dark:text-gray-200 font-sans min-h-screen p-6 md:p-10 transition-colors duration-300">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
              Nemo TDT Demo
            </h1>
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              transformers.js {VERSION} ({SOURCE})
            </div>
          </div>
          <button
            className="flex items-center justify-center p-2 rounded-full bg-gray-200 dark:bg-card-dark border border-border-light dark:border-border-dark hover:bg-primary-muted/20 dark:hover:bg-accent-muted/30 transition-colors"
            onClick={() => setDarkMode(!darkMode)}
          >
            <span className="material-icons-outlined text-primary-muted dark:text-accent-muted">
              brightness_4
            </span>
          </button>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          {/* Left Column - Model + Options */}
          <div className="lg:col-span-1 flex flex-col gap-4">
            {/* Model Configuration Card */}
            <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
              <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-5">
                Model Configuration
              </h2>
              <div className="space-y-4">
                <SelectField label="Load Mode" value={mode} onChange={(e) => setMode(e.target.value)} disabled={configDisabled}>
                  <option value="pipeline">pipeline (auto)</option>
                  <option value="explicit">explicit (local export)</option>
                </SelectField>

                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                    Model ID
                  </label>
                  <input
                    value={modelId}
                    onChange={(e) => setModelId(e.target.value)}
                    disabled={configDisabled}
                    className="w-full bg-gray-50 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-muted focus:border-primary-muted dark:focus:ring-accent-muted dark:focus:border-accent-muted dark:text-white"
                  />
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <SelectField label="Enc. Device" value={encDev} onChange={(e) => setEncDev(e.target.value)} disabled={configDisabled}>
                    <option value="webgpu">webgpu</option>
                    <option value="wasm">wasm</option>
                  </SelectField>
                  <SelectField label="Enc. Dtype" value={encDtype} onChange={(e) => setEncDtype(e.target.value)} disabled={configDisabled}>
                    {DTYPES.map((d) => <option key={d} value={d}>{d}</option>)}
                  </SelectField>
                  <SelectField label="Dec. Dtype" value={decDtype} onChange={(e) => setDecDtype(e.target.value)} disabled={configDisabled}>
                    {DTYPES.map((d) => <option key={d} value={d}>{d}</option>)}
                  </SelectField>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                    WASM Threads (1-{maxWasmCores})
                  </label>
                  <input
                    type="number"
                    min={1}
                    max={maxWasmCores}
                    step={1}
                    value={wasmThreads}
                    onChange={(e) => setWasmThreads(clampThreadCount(e.target.value, maxWasmCores))}
                    disabled={configDisabled}
                    className="w-full bg-gray-50 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-muted focus:border-primary-muted dark:focus:ring-accent-muted dark:focus:border-accent-muted dark:text-white"
                  />
                </div>

                <div className="flex gap-3 pt-1">
                  <button
                    onClick={load}
                    disabled={configDisabled}
                    className="flex-1 bg-primary hover:bg-primary-hover dark:bg-primary dark:hover:bg-primary-muted text-white font-medium py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <span className="material-icons-outlined text-sm">bolt</span>
                    {modelLoaded ? 'Model Loaded' : loading ? 'Loading...' : 'Load Model'}
                  </button>
                  <button
                    onClick={() => { tRef.current = null; setResult(null); setMType('not-loaded'); setModelLoaded(false); setStatus('Idle'); setIsCached(null); }}
                    disabled={loading || running}
                    className="bg-primary-muted hover:bg-primary dark:bg-border-dark dark:hover:bg-accent-muted text-white font-medium py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <span className="material-icons-outlined text-sm">power_settings_new</span>
                    Dispose
                  </button>
                </div>

                {/* Cache status badge */}
                {isCached !== null && !modelLoaded && (
                  <div className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg border ${isCached
                    ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800 text-green-700 dark:text-green-400'
                    : 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800 text-yellow-700 dark:text-yellow-400'
                    }`}>
                    <span className="material-icons-outlined text-sm">{isCached ? 'offline_bolt' : 'cloud_download'}</span>
                    {isCached ? 'Model cached — instant load' : 'Model not cached — will download'}
                  </div>
                )}

                {/* Download progress bar */}
                {downloadProgress && (
                  <div>
                    <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
                      <span className="truncate max-w-[70%]" title={downloadProgress.file}>
                        {downloadProgress.file ? `↓ ${downloadProgress.file.split('/').pop()}` : 'Downloading...'}
                      </span>
                      <span className="font-mono tabular-nums">{downloadProgress.pct}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                      <div
                        className="bg-primary dark:bg-accent-muted h-1.5 rounded-full transition-all duration-150"
                        style={{ width: `${downloadProgress.pct}%` }}
                      />
                    </div>
                    {downloadProgress.total > 0 && (
                      <div className="text-right text-[0.65rem] text-gray-400 mt-0.5 font-mono">
                        {(downloadProgress.loaded / 1e6).toFixed(1)} / {(downloadProgress.total / 1e6).toFixed(1)} MB
                      </div>
                    )}
                  </div>
                )}

              </div>
            </div>

            {/* Options Card */}
            <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
              <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-5">
                Transcription Options
              </h2>
              <div className="flex flex-col gap-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700 dark:text-gray-300">Direct Nemo call</span>
                  <Toggle id="direct" checked={direct} onChange={(e) => setDirect(e.target.checked)} />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-700 dark:text-gray-300">Return timestamps</span>
                  <Toggle id="rt" checked={rt} onChange={(e) => setRt(e.target.checked)} />
                </div>

                <div className="border-t border-border-light dark:border-border-dark pt-3 mt-1">
                  <p className="text-[0.65rem] font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-2">Detail Flags</p>
                  <div className="grid grid-cols-2 gap-y-2 gap-x-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-700 dark:text-gray-300">Words</span>
                      <Toggle id="words" checked={returnWords} onChange={(e) => setReturnWords(e.target.checked)} disabled={!direct || !rt} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-700 dark:text-gray-300">Tokens</span>
                      <Toggle id="tokens" checked={returnTokens} onChange={(e) => setReturnTokens(e.target.checked)} disabled={!direct || !rt} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-700 dark:text-gray-300">Metrics</span>
                      <Toggle id="metrics" checked={metrics} onChange={(e) => setMetrics(e.target.checked)} disabled={!direct} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-700 dark:text-gray-300">Frame conf.</span>
                      <Toggle id="frameConf" checked={returnFrameConf} onChange={(e) => setReturnFrameConf(e.target.checked)} disabled={!direct} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-700 dark:text-gray-300">Frame idx</span>
                      <Toggle id="frameIdx" checked={frameIdx} onChange={(e) => setFrameIdx(e.target.checked)} disabled={!direct} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-700 dark:text-gray-300">Log probs</span>
                      <Toggle id="logProbs" checked={logProbs} onChange={(e) => setLogProbs(e.target.checked)} disabled={!direct} />
                    </div>
                    <div className="flex items-center justify-between col-span-2">
                      <span className="text-sm text-gray-700 dark:text-gray-300">TDT steps</span>
                      <Toggle id="tdtSteps" checked={tdtSteps} onChange={(e) => setTdtSteps(e.target.checked)} disabled={!direct} />
                    </div>
                  </div>
                </div>

                <div className="border-t border-border-light dark:border-border-dark pt-3 mt-1">
                  <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                    Time Offset
                  </label>
                  <input
                    value={offset}
                    onChange={(e) => setOffset(e.target.value)}
                    disabled={!direct}
                    className="w-full bg-gray-50 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-muted focus:border-primary-muted dark:focus:ring-accent-muted dark:focus:border-accent-muted dark:text-white disabled:opacity-50"
                  />
                </div>
              </div>
            </div>

            {/* Status */}
            <div className="flex items-center gap-2 px-1">
              <span className="font-medium text-gray-900 dark:text-white">Status:</span>
              <span className={`font-medium ${isModelReady ? 'text-gray-700 dark:text-gray-300' : 'text-gray-600 dark:text-gray-400'}`}>
                {status}
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2 px-1 text-xs text-gray-500 dark:text-gray-400">
              <span className="bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full border border-border-light dark:border-border-dark text-primary-muted dark:text-accent-muted">
                model_type: {mType}
              </span>
              <span className="bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full border border-border-light dark:border-border-dark text-primary-muted dark:text-accent-muted">
                wasm_threads: {clampThreadCount(wasmThreads, maxWasmCores)}
              </span>
            </div>
            {!!error && (
              <div className="mx-1 p-3 bg-gray-100 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg text-sm text-gray-800 dark:text-gray-200 break-words">
                {error}
              </div>
            )}
          </div>

          {/* Right Column - Test & Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* Transcribe Card */}
            <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
              <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-5">
                Test & Transcribe
              </h2>

              <div className="flex flex-col md:flex-row gap-4 mb-6 items-end">
                <div className="flex gap-3">
                  <button
                    onClick={() => transcribeInput(SAMPLE, 'life_Jim.wav')}
                    disabled={!canRun}
                    className="bg-primary hover:bg-primary-hover dark:bg-primary dark:hover:bg-primary-muted text-white font-medium py-2 px-4 rounded-lg whitespace-nowrap transition-colors text-sm h-[38px] disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    <span className="material-icons-outlined text-sm">play_arrow</span>
                    Sample
                  </button>
                </div>
              </div>

              {/* File Upload Area */}
              <div className="upload-zone relative border-2 border-dashed border-border-light dark:border-border-dark rounded-xl bg-gray-50 dark:bg-gray-800/50 p-6 flex flex-col items-center justify-center text-center cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors group">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  disabled={!canRun}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
                />
                <span className="material-icons-outlined text-4xl text-primary-muted dark:text-accent-muted group-hover:text-primary dark:group-hover:text-primary-muted mb-2 transition-colors">
                  cloud_upload
                </span>
                <p className="text-gray-500 dark:text-gray-400 font-medium">
                  Drag & drop audio file here, or click to select
                </p>
                <p className="text-xs text-gray-400 mt-1">Supports .wav, .mp3, .ogg</p>
              </div>

              {!!selectedFileName && (
                <p className="mt-3 text-sm text-gray-500 dark:text-gray-400">
                  Selected: <span className="font-medium text-gray-700 dark:text-gray-200">{selectedFileName}</span>
                </p>
              )}
            </div>

            {/* Performance row at top */}
            <PerformanceMetrics stats={stats} />

            {/* Transcript then Output JSON */}
            <div className="space-y-4">
              <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted">
                    Transcript
                  </h2>
                  {textOf(result) && (
                    <button
                      onClick={() => copyToClipboard(textOf(result))}
                      className="flex items-center gap-1 text-xs font-medium text-primary-muted hover:text-primary dark:text-accent-muted dark:hover:text-primary-muted bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded border border-border-light dark:border-border-dark transition-colors"
                    >
                      <span className="material-icons-outlined text-xs">content_copy</span>
                      Copy
                    </button>
                  )}
                </div>
                <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 border border-border-light dark:border-border-dark min-h-[120px]">
                  <p className="text-lg leading-relaxed text-gray-800 dark:text-gray-200 font-medium">
                    {textOf(result) || 'No text yet.'}
                  </p>
                </div>
              </div>

              <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted">
                    Output JSON
                  </h2>
                  {result && (
                    <button
                      onClick={() => copyToClipboard(toJSON(result))}
                      className="flex items-center gap-1 text-xs font-medium text-primary-muted hover:text-primary dark:text-accent-muted dark:hover:text-primary-muted bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded border border-border-light dark:border-border-dark transition-colors"
                    >
                      <span className="material-icons-outlined text-xs">content_copy</span>
                      Copy
                    </button>
                  )}
                </div>
                <div className="json-output bg-gray-100 dark:bg-gray-800 rounded-lg p-4 border border-border-light dark:border-border-dark min-h-[200px]">
                  <pre className="text-xs leading-relaxed text-gray-800 dark:text-gray-200 font-mono whitespace-pre-wrap break-words">
                    {result ? toJSON(result) : 'No transcription yet.'}
                  </pre>
                </div>
              </div>
            </div>

            {/* History */}
            {history.length > 0 && (
              <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted">
                    History
                  </h2>
                  <button
                    onClick={() => setHistory([])}
                    className="text-xs font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
                  >
                    Clear All
                  </button>
                </div>
                <div className="space-y-3 max-h-[400px] overflow-y-auto">
                  {history.map((h) => (
                    <div key={h.id} className="p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-sm font-medium text-gray-900 dark:text-white">
                          {h.name}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full border border-gray-200 dark:border-gray-700">
                          {h.mode}
                        </span>
                      </div>
                      <div className="text-sm text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-900/30 p-3 rounded border border-gray-200 dark:border-gray-700">
                        {h.text || '(empty)'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

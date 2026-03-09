import { useEffect, useMemo, useRef, useState } from 'react';
import * as Transformers from '@huggingface/transformers';
import './App.css';
import {
  CUSTOM_RESAMPLER_QUALITY_OPTIONS,
  DEFAULT_CUSTOM_RESAMPLER_QUALITY,
  prepareBrowserAudioInput,
} from './audio-prep';
import {
  benchmarkSummaryToCsv,
  buildRandomSamplePlan,
  buildTargetSamplePlan,
  benchmarkRunsToCsv,
  findEffectiveRtfxSweetSpot,
  formatSeconds,
  getDurationBucket,
  mergeTargetDurations,
  parseDurationTargets,
  refineTargetsFromRuns,
  suggestDurationTargets,
  summarizeBenchmarkOverall,
  summarizeBenchmarkRuns,
} from './benchmark-utils';

const MODEL_DEFAULT = 'ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4';
const DECODER_DEVICE = 'wasm';
const DTYPES = ['fp16', 'int8', 'fp32'];
const AUDIO_PREP_BACKEND_DEFAULT = 'custom-js';
const AUDIO_PREP_CUSTOM_RESAMPLER_DEFAULT = DEFAULT_CUSTOM_RESAMPLER_QUALITY;
const NEMO_PIPELINE_WINDOW_DEFAULT_S = 90;
const NEMO_PIPELINE_WINDOW_MIN_S = 20;
const NEMO_PIPELINE_WINDOW_MAX_S = 180;

const SETTINGS_STORAGE_KEY = 'nemo-tdt-demo.settings.v1';
const BENCHMARK_HANDLE_DB = 'nemo-tdt-demo.benchmark-folder';
const BENCHMARK_HANDLE_STORE = 'handles';
const BENCHMARK_HANDLE_KEY = 'last-folder';
const BENCHMARK_CATALOG_STORE = 'catalogs';
const BASE_URL = import.meta.env?.BASE_URL || '/';
const withBaseUrl = (relativePath) => {
  const base = BASE_URL.endsWith('/') ? BASE_URL : `${BASE_URL}/`;
  return `${base}${String(relativePath).replace(/^\/+/, '')}`;
};
const SAMPLE = withBaseUrl('assets/Harvard-L2-1.ogg');

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

function parseThemeValue(value) {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().toLowerCase();
  if (normalized === 'dark') return true;
  if (normalized === 'light') return false;
  return null;
}

function getHfThemeFromQuery() {
  if (typeof window === 'undefined') return null;
  const params = new URLSearchParams(window.location.search);
  return parseThemeValue(params.get('__theme') || params.get('theme') || params.get('color'));
}

function getThemeFromMessage(data) {
  if (!data) return null;
  if (typeof data === 'string') return parseThemeValue(data);
  if (typeof data !== 'object') return null;
  return (
    parseThemeValue(data.theme) ??
    parseThemeValue(data.colorMode) ??
    parseThemeValue(data.mode) ??
    parseThemeValue(data?.payload?.theme)
  );
}

function getInitialDarkMode(settings) {
  const hfTheme = getHfThemeFromQuery();
  if (hfTheme !== null) return hfTheme;
  if (typeof settings.darkMode === 'boolean') return settings.darkMode;
  if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  return false;
}

const detectMaxCores = () => {
  const c = Number(globalThis?.navigator?.hardwareConcurrency ?? 1);
  return Number.isFinite(c) && c > 0 ? Math.floor(c) : 1;
};

const clampPipelineWindowSec = (value) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return NEMO_PIPELINE_WINDOW_DEFAULT_S;
  return Math.max(NEMO_PIPELINE_WINDOW_MIN_S, Math.min(NEMO_PIPELINE_WINDOW_MAX_S, numeric));
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
const isInIframe = (() => {
  if (typeof window === 'undefined') return false;
  try {
    return window.self !== window.top;
  } catch {
    return true;
  }
})();

if (typeof window !== 'undefined') {
  const { env } = Transformers;
  env.allowRemoteModels = true;
  env.allowLocalModels = false;
  if (env?.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.proxy = false;
    env.backends.onnx.wasm.wasmPaths = withBaseUrl('ort/');
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

    const mjsSrc = withBaseUrl('ort/ort-wasm-simd-threaded.asyncify.mjs');
    const wasmSrc = withBaseUrl('ort/ort-wasm-simd-threaded.asyncify.wasm');
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
const indentCode = (text, spaces = 2) => String(text)
  .split('\n')
  .map((line) => `${' '.repeat(spaces)}${line}`)
  .join('\n');
const averageOf = (values) => {
  const valid = values.filter(Number.isFinite);
  if (valid.length === 0) return null;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
};
const stddevOf = (values) => {
  const valid = values.filter(Number.isFinite);
  if (valid.length < 2) return null;
  const mean = averageOf(valid);
  const variance = valid.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / valid.length;
  return Math.sqrt(variance);
};
const formatCompactDuration = (value) => {
  if (!Number.isFinite(value)) return '-';
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = value / 60;
  if (minutes < 10) return `${minutes.toFixed(1)}m`;
  return `${Math.round(minutes)}m`;
};
const formatClockDuration = (value) => {
  if (!Number.isFinite(value) || value < 0) return '0:00';
  const totalSeconds = Math.round(value);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
};
const parseDurationBound = (value) => {
  if (value === '' || value == null) return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) && numeric >= 0 ? numeric : null;
};
const bytesPerType = (type) => ({
  float32: 4,
  float16: 2,
  int8: 1,
  uint8: 1,
  bool: 1,
  int16: 2,
  uint16: 2,
  int32: 4,
  uint32: 4,
  int64: 8,
  uint64: 8,
}[String(type || '').toLowerCase()] || 4);
const bytesPerEncoderDtype = (dtype) => ({
  fp32: 4,
  fp16: 2,
  int8: 1,
}[String(dtype || '').toLowerCase()] || 4);
const toMiB = (bytes) => (Number.isFinite(bytes) ? bytes / (1024 * 1024) : null);
const shuffleArray = (items) => {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
};
const buildDurationHistogram = (samples, bucketSizeSec = 30) => {
  const grouped = new Map();

  for (const duration of samples
    .map((sample) => sample?.durationSec)
    .filter((value) => Number.isFinite(value) && value > 0)) {
    const bucket = getDurationBucket(duration, bucketSizeSec);
    const key = `${bucket.start}-${bucket.end}`;
    if (!grouped.has(key)) {
      grouped.set(key, {
        start: bucket.start,
        end: bucket.end,
        count: 0,
        label: `${formatCompactDuration(bucket.start)}-${formatCompactDuration(bucket.end)}`,
      });
    }
    grouped.get(key).count += 1;
  }

  return Array.from(grouped.values()).sort((a, b) => a.start - b.start);
};

function estimateInputPayloadMetrics({ mono, sampleRate, inputs, encoderDtype }) {
  const inputFeatures = inputs?.input_features;
  const attentionMask = inputs?.attention_mask;
  const featureDims = Array.isArray(inputFeatures?.dims) ? inputFeatures.dims : null;
  const featureElements = featureDims ? featureDims.reduce((product, value) => product * value, 1) : null;
  const featureFrames = featureDims?.at(-2) ?? null;
  const featureBins = featureDims?.at(-1) ?? null;
  const featureBytes = featureElements != null
    ? featureElements * bytesPerType(inputFeatures?.type || 'float32')
    : null;
  const attentionMaskBytes = Array.isArray(attentionMask?.dims)
    ? attentionMask.dims.reduce((product, value) => product * value, 1) * bytesPerType(attentionMask?.type || 'int64')
    : null;
  const encoderPayloadBytes = featureElements != null
    ? featureElements * bytesPerEncoderDtype(encoderDtype)
    : null;
  const audioPcmBytes = mono instanceof Float32Array ? mono.byteLength : null;

  return {
    sampleRate,
    audioSamples: mono?.length ?? null,
    audioPcmBytes,
    audioPcmMiB: toMiB(audioPcmBytes),
    featureShape: featureDims ? featureDims.join('x') : null,
    featureFrames,
    featureBins,
    featureElements,
    featureBytes,
    featureMiB: toMiB(featureBytes),
    attentionMaskBytes,
    attentionMaskMiB: toMiB(attentionMaskBytes),
    encoderPayloadBytes,
    encoderPayloadMiB: toMiB(encoderPayloadBytes),
  };
}

async function prepareAudioInput(
  input,
  sampleRate,
  backend = AUDIO_PREP_BACKEND_DEFAULT,
  customResamplerQuality = AUDIO_PREP_CUSTOM_RESAMPLER_DEFAULT,
) {
  return await prepareBrowserAudioInput(input, {
    targetSampleRate: sampleRate,
    backend,
    readAudio: Transformers.read_audio,
    customResamplerQuality,
  });
}

function readUInt32LE(view, offset) {
  return view.getUint32(offset, true);
}

function readAscii(view, offset, length) {
  let out = '';
  for (let i = 0; i < length; i += 1) {
    out += String.fromCharCode(view.getUint8(offset + i));
  }
  return out;
}

async function getWavDuration(file) {
  const header = await file.slice(0, Math.min(file.size, 1024 * 1024)).arrayBuffer();
  const view = new DataView(header);

  if (view.byteLength < 44 || readAscii(view, 0, 4) !== 'RIFF' || readAscii(view, 8, 4) !== 'WAVE') {
    return null;
  }

  let offset = 12;
  let byteRate = null;
  let dataSize = null;
  while (offset + 8 <= view.byteLength) {
    const chunkId = readAscii(view, offset, 4);
    const chunkSize = readUInt32LE(view, offset + 4);
    const chunkDataOffset = offset + 8;

    if (chunkId === 'fmt ' && chunkDataOffset + 12 <= view.byteLength) {
      // WAV fmt layout: audioFormat(2), numChannels(2), sampleRate(4), byteRate(4), ...
      byteRate = readUInt32LE(view, chunkDataOffset + 8);
    } else if (chunkId === 'data') {
      dataSize = chunkSize;
      break;
    }

    offset = chunkDataOffset + chunkSize + (chunkSize % 2);
  }

  if (!Number.isFinite(byteRate) || !Number.isFinite(dataSize) || byteRate <= 0) {
    return null;
  }

  return dataSize / byteRate;
}

async function getAudioDurationSeconds(file) {
  if (/\.wav$/i.test(file.name)) {
    const wavDuration = await getWavDuration(file);
    if (Number.isFinite(wavDuration)) {
      return wavDuration;
    }
  }

  const audio = new Audio();
  const objectUrl = URL.createObjectURL(file);

  try {
    const duration = await new Promise((resolve, reject) => {
      const cleanup = () => {
        audio.removeEventListener('loadedmetadata', onLoaded);
        audio.removeEventListener('error', onError);
      };
      const onLoaded = () => {
        cleanup();
        resolve(audio.duration);
      };
      const onError = () => {
        cleanup();
        reject(new Error(`Unable to read audio metadata for ${file.name}`));
      };

      audio.addEventListener('loadedmetadata', onLoaded);
      audio.addEventListener('error', onError);
      audio.src = objectUrl;
    });

    return duration;
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function downloadTextFile(filename, content, mimeType = 'application/json') {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function isAudioFileName(name) {
  return /\.(wav|mp3|ogg|flac|m4a|webm)$/i.test(name);
}

async function collectAudioEntriesFromDirectoryHandle(directoryHandle, prefix = '') {
  const entries = [];
  for await (const handle of directoryHandle.values()) {
    if (handle.kind === 'file') {
      if (!isAudioFileName(handle.name)) continue;
      const file = await handle.getFile();
      entries.push({
        file,
        name: file.name,
        path: prefix ? `${prefix}/${file.name}` : file.name,
        sizeBytes: file.size,
        lastModified: file.lastModified,
      });
      continue;
    }
    if (handle.kind === 'directory') {
      const nested = await collectAudioEntriesFromDirectoryHandle(
        handle,
        prefix ? `${prefix}/${handle.name}` : handle.name
      );
      entries.push(...nested);
    }
  }
  return entries;
}

function openHandleDb() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(BENCHMARK_HANDLE_DB, 2);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(BENCHMARK_HANDLE_STORE)) {
        db.createObjectStore(BENCHMARK_HANDLE_STORE);
      }
      if (!db.objectStoreNames.contains(BENCHMARK_CATALOG_STORE)) {
        db.createObjectStore(BENCHMARK_CATALOG_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error('Failed to open benchmark handle database.'));
  });
}

async function readStoredDirectoryHandle() {
  const db = await openHandleDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(BENCHMARK_HANDLE_STORE, 'readonly');
    const store = tx.objectStore(BENCHMARK_HANDLE_STORE);
    const request = store.get(BENCHMARK_HANDLE_KEY);
    request.onsuccess = () => {
      resolve(request.result ?? null);
      db.close();
    };
    request.onerror = () => {
      reject(request.error || new Error('Failed to read stored benchmark folder handle.'));
      db.close();
    };
  });
}

async function writeStoredDirectoryHandle(handle) {
  const db = await openHandleDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(BENCHMARK_HANDLE_STORE, 'readwrite');
    tx.objectStore(BENCHMARK_HANDLE_STORE).put(handle, BENCHMARK_HANDLE_KEY);
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error || new Error('Failed to store benchmark folder handle.'));
    };
  });
}

async function clearStoredDirectoryHandle() {
  const db = await openHandleDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(BENCHMARK_HANDLE_STORE, 'readwrite');
    tx.objectStore(BENCHMARK_HANDLE_STORE).delete(BENCHMARK_HANDLE_KEY);
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error || new Error('Failed to clear stored benchmark folder handle.'));
    };
  });
}

function buildCatalogEntrySignature(entry) {
  return `${entry.path}::${entry.sizeBytes ?? entry.file?.size ?? 0}::${entry.lastModified ?? entry.file?.lastModified ?? 0}`;
}

function buildCatalogCacheKey(entries) {
  let hash = 2166136261;
  const signatures = entries
    .map((entry) => buildCatalogEntrySignature(entry))
    .sort();

  for (const signature of signatures) {
    for (let i = 0; i < signature.length; i += 1) {
      hash ^= signature.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
  }

  return `catalog:${signatures.length}:${(hash >>> 0).toString(16)}`;
}

async function readStoredCatalog(cacheKey) {
  const db = await openHandleDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(BENCHMARK_CATALOG_STORE, 'readonly');
    const store = tx.objectStore(BENCHMARK_CATALOG_STORE);
    const request = store.get(cacheKey);
    request.onsuccess = () => {
      resolve(request.result ?? null);
      db.close();
    };
    request.onerror = () => {
      reject(request.error || new Error('Failed to read cached benchmark catalog.'));
      db.close();
    };
  });
}

async function writeStoredCatalog(cacheKey, payload) {
  const db = await openHandleDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(BENCHMARK_CATALOG_STORE, 'readwrite');
    tx.objectStore(BENCHMARK_CATALOG_STORE).put(payload, cacheKey);
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error || new Error('Failed to store benchmark catalog cache.'));
    };
  });
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

function MetricGrid({ items }) {
  const visibleItems = items.filter(({ value }) => value != null);
  if (visibleItems.length === 0) return null;

  return (
    <div className="grid grid-cols-1 gap-x-4 gap-y-1.5 text-[12px] leading-5 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
      {visibleItems.map(({ label, value }) => (
        <div key={label} className="flex min-w-0 items-baseline justify-between gap-3">
          <span className="shrink-0 text-[11px] font-medium uppercase tracking-wide text-primary-muted dark:text-accent-muted">
            {label}
          </span>
          <span className="shrink-0 whitespace-nowrap text-right font-mono tabular-nums text-gray-900 dark:text-white">
            {value}
          </span>
        </div>
      ))}
    </div>
  );
}

function PerformanceMetrics({ stats }) {
  const fmt = (v, unit) => (v != null && Number.isFinite(v) ? `${v.toFixed(1)}${unit}` : '-');
  const isDirectMode = stats.mode === 'direct';
  const hasModelInternals = [
    stats.preprocessMs,
    stats.encodeMs,
    stats.decodeMs,
    stats.tokenizeMs,
    stats.totalMs,
    stats.rtfx,
  ].some((value) => value != null && Number.isFinite(value));

  const overviewItems = [
    { label: 'Audio', value: stats.audioDurationSec != null ? formatCompactDuration(stats.audioDurationSec) : '-' },
    { label: 'Processor', value: fmt(stats.processorMs, 'ms') },
    { label: 'Infer wall', value: fmt(stats.inferenceWallMs, 'ms') },
    { label: 'End-to-end', value: fmt(stats.endToEndMs, 'ms') },
    { label: 'Infer RTFx', value: stats.inferenceRtfx != null && Number.isFinite(stats.inferenceRtfx) ? `${stats.inferenceRtfx.toFixed(1)}x` : null },
    { label: 'End-to-end RTFx', value: stats.endToEndRtfx != null && Number.isFinite(stats.endToEndRtfx) ? `${stats.endToEndRtfx.toFixed(1)}x` : null },
  ];
  const audioItems = [
    { label: 'Total', value: fmt(stats.audioPrepMs, 'ms') },
    { label: 'Input', value: stats.audioInputSampleRate != null ? `${stats.audioInputSampleRate} Hz` : null },
    { label: 'Output', value: stats.audioOutputSampleRate != null ? `${stats.audioOutputSampleRate} Hz` : null },
    { label: 'Decode', value: fmt(stats.audioDecodeMs, 'ms') },
    { label: 'Downmix', value: fmt(stats.audioDownmixMs, 'ms') },
    { label: 'Resample', value: fmt(stats.audioResampleMs, 'ms') },
    { label: 'Backend', value: stats.audioPrepBackend || null },
    { label: 'Resampler', value: stats.audioResampler || null },
    { label: 'Quality', value: stats.audioResamplerQuality || null },
  ];
  const modelItems = [
    { label: 'Preprocess', value: fmt(stats.preprocessMs, 'ms') },
    { label: 'Encode', value: fmt(stats.encodeMs, 'ms') },
    { label: 'Decode', value: fmt(stats.decodeMs, 'ms') },
    { label: 'Tokenize', value: fmt(stats.tokenizeMs, 'ms') },
    { label: 'Total', value: fmt(stats.totalMs, 'ms') },
    { label: 'RTFx', value: stats.rtfx != null && Number.isFinite(stats.rtfx) ? `${stats.rtfx.toFixed(1)}x` : null },
    { label: 'Window', value: stats.pipelineWindowSec != null ? `${stats.pipelineWindowSec.toFixed(0)}s` : null },
  ];
  return (
    <div className="rounded-xl border border-border-light bg-card-light px-4 py-3 dark:border-border-dark dark:bg-card-dark">
      <div className="space-y-2.5">
        {!isDirectMode && (
          <div className="space-y-1.5">
            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-gray-500 dark:text-gray-400">
              Run Timing
            </div>
            <MetricGrid items={overviewItems} />
          </div>
        )}
        <div className="border-t border-border-light pt-2 dark:border-border-dark">
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-gray-500 dark:text-gray-400">
            Audio Prep
          </div>
          <MetricGrid items={audioItems} />
        </div>
        {isDirectMode && hasModelInternals ? (
          <div className="border-t border-border-light pt-2 dark:border-border-dark">
            <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-[0.16em] text-gray-500 dark:text-gray-400">
              Direct Model Internals
            </div>
            <MetricGrid items={modelItems} />
          </div>
        ) : !isDirectMode ? null : (
          <div className="border-t border-border-light pt-2 text-[12px] text-gray-500 dark:border-border-dark dark:text-gray-400">
            Direct-only model internals appear here when available.
          </div>
        )}
      </div>
    </div>
  );
}

function InfoHint({ text, className = '' }) {
  return (
    <span
      title={text}
      className={`inline-flex h-4 w-4 items-center justify-center rounded-full border border-border-light dark:border-border-dark text-[10px] text-gray-500 dark:text-gray-400 cursor-help ${className}`}
      aria-label={text}
    >
      <span className="material-icons-outlined text-[12px] leading-none">info</span>
    </span>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  disabled = false,
  formatValue = (next) => String(next),
}) {
  const safeMax = Math.max(min, max);
  const clampedValue = Math.min(safeMax, Math.max(min, value));
  return (
    <div className="rounded-xl border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800/70 px-4 py-3">
      <div className="flex items-center justify-between gap-3 mb-2">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</label>
        <span className="text-sm font-mono text-gray-900 dark:text-white">{formatValue(clampedValue)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={safeMax}
        step={step}
        value={clampedValue}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className="w-full accent-primary dark:accent-accent-muted disabled:opacity-50"
      />
      <div className="mt-2 flex items-center justify-between text-[11px] text-gray-500 dark:text-gray-400">
        <span>{formatValue(min)}</span>
        <span>{formatValue(safeMax)}</span>
      </div>
    </div>
  );
}

function DurationRangeField({
  minValue,
  maxValue,
  limit,
  step = 5,
  onMinChange,
  onMaxChange,
  disabled = false,
}) {
  const safeLimit = Math.max(step, limit);
  const safeMin = Math.max(0, Math.min(minValue, maxValue));
  const safeMax = Math.min(safeLimit, Math.max(minValue, maxValue));
  const effectiveMin = safeMin <= 0 ? null : safeMin;
  const effectiveMax = safeMax >= safeLimit ? null : safeMax;

  const pctMin = (safeMin / safeLimit) * 100;
  const pctMax = (safeMax / safeLimit) * 100;

  return (
    <div className="rounded-xl border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800/70 px-4 py-3">
      <div className="flex items-center justify-between gap-3 mb-2">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Duration Range</label>
        <span className="text-sm font-mono text-gray-900 dark:text-white">
          {effectiveMin == null ? '0:00' : formatClockDuration(effectiveMin)} – {effectiveMax == null ? formatClockDuration(safeLimit) : formatClockDuration(effectiveMax)}
        </span>
      </div>
      <div className="dual-range-wrap">
        <div className="dual-range-track">
          <div
            className="dual-range-fill"
            style={{ left: `${pctMin}%`, width: `${pctMax - pctMin}%` }}
          />
        </div>
        <input
          type="range"
          min={0}
          max={safeLimit}
          step={step}
          value={safeMin}
          onChange={(e) => onMinChange(Math.min(Number(e.target.value), safeMax))}
          disabled={disabled}
          className="dual-range-input"
        />
        <input
          type="range"
          min={0}
          max={safeLimit}
          step={step}
          value={safeMax}
          onChange={(e) => onMaxChange(Math.max(Number(e.target.value), safeMin))}
          disabled={disabled}
          className="dual-range-input"
        />
      </div>
      <div className="mt-1 flex items-center justify-between text-[11px] text-gray-500 dark:text-gray-400">
        <span>0:00</span>
        <span>{formatClockDuration(safeLimit)}</span>
      </div>
    </div>
  );
}

function BenchmarkMetrics({ overall }) {
  const fmt = (v, unit = '') => (v != null && Number.isFinite(v) ? `${v.toFixed(unit ? 1 : 0)}${unit}` : '-');
  const items = [
    { label: 'Files', value: overall.count != null ? overall.count : '-' },
    { label: 'Failed', value: overall.failed != null ? overall.failed : '-' },
    { label: 'Total Audio', value: overall.totalAudioSec != null ? formatSeconds(overall.totalAudioSec) : '-' },
    { label: 'Wall Avg', value: overall.avgWallRtfx != null ? `${overall.avgWallRtfx.toFixed(1)}x` : '-' },
    { label: 'Wall Med', value: overall.medianWallRtfx != null ? `${overall.medianWallRtfx.toFixed(1)}x` : '-' },
    { label: 'Model Avg', value: overall.avgModelRtfx != null ? `${overall.avgModelRtfx.toFixed(1)}x` : '-' },
    { label: 'Encode', value: fmt(overall.avgEncodeMs, 'ms') },
    { label: 'Decode', value: fmt(overall.avgDecodeMs, 'ms') },
  ];

  return (
    <div className="bg-card-light dark:bg-card-dark rounded-xl border border-border-light dark:border-border-dark px-4 py-3">
      <MetricRow items={items} />
    </div>
  );
}

function svgPathFromPoints(points, width, height, domainX, domainY, { logY = false } = {}) {
  if (!Array.isArray(points) || points.length === 0) return '';
  const [minX, maxX] = domainX;
  const [minY, maxY] = domainY;
  const spanX = Math.max(maxX - minX, 1e-9);
  const spanY = Math.max(maxY - minY, 1e-9);
  const minLog = logY ? Math.log(minY) : null;
  const maxLog = logY ? Math.log(maxY) : null;
  const spanLog = logY ? Math.max(maxLog - minLog, 1e-9) : null;

  return points
    .map((point, index) => {
      const x = ((point.x - minX) / spanX) * width;
      const y = logY
        ? height - ((Math.log(point.y) - minLog) / spanLog) * height
        : height - ((point.y - minY) / spanY) * height;
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
}

function BenchmarkChart({
  title,
  subtitle,
  series,
  yLabel,
  formatY = (value) => value.toFixed(1),
  height = 220,
}) {
  const [logY, setLogY] = useState(false);
  const [visibleSeries, setVisibleSeries] = useState(() =>
    Object.fromEntries((series || []).map((entry) => [entry.label, true]))
  );
  useEffect(() => {
    setVisibleSeries((current) => {
      const next = {};
      for (const entry of series || []) {
        next[entry.label] = current[entry.label] ?? true;
      }
      return next;
    });
  }, [series]);
  const width = 640;
  const preparedSeries = series
    .map((entry) => ({
      ...entry,
      points: (entry.points || []).filter(
        (point) => Number.isFinite(point?.x) && Number.isFinite(point?.y) && (!logY || point.y > 0)
      ),
    }));
  const validSeries = preparedSeries.filter((entry) => visibleSeries[entry.label] !== false && entry.points.length > 0);
  const hasVisibleSeries = validSeries.length > 0;

  const allPoints = validSeries.flatMap((entry) => entry.points);
  const xs = allPoints.map((point) => point.x);
  const ys = allPoints.map((point) => point.y);
  const minX = hasVisibleSeries ? Math.min(...xs) : 0;
  const maxX = hasVisibleSeries ? Math.max(...xs) : 1;
  const positiveYs = ys.filter((value) => value > 0);
  const minY = hasVisibleSeries
    ? (logY ? Math.max(Math.min(...positiveYs) * 0.9, 1e-6) : 0)
    : 0;
  const maxY = hasVisibleSeries
    ? (logY ? Math.max(...positiveYs) * 1.1 || 1 : Math.max(...ys) * 1.1 || 1)
    : 1;
  const ticks = 4;
  const mapY = (value) => {
    if (logY) {
      const minLog = Math.log(minY);
      const maxLog = Math.log(maxY);
      const spanLog = Math.max(maxLog - minLog, 1e-9);
      return height - ((Math.log(value) - minLog) / spanLog) * height;
    }
    return height - ((value - minY) / Math.max(maxY - minY, 1e-9)) * height;
  };
  const tickValues = hasVisibleSeries
    ? Array.from({ length: ticks + 1 }, (_, tick) => {
      if (logY) {
        const ratio = 1 - (tick / ticks);
        return Math.exp(Math.log(minY) + (Math.log(maxY) - Math.log(minY)) * ratio);
      }
      return maxY - ((maxY - minY) / ticks) * tick;
    })
    : [];

  return (
    <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-5">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{title}</h3>
          {subtitle && <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{subtitle}</p>}
        </div>
        <div className="flex flex-wrap items-center justify-end gap-3 text-xs">
          <button
            type="button"
            onClick={() => setLogY((value) => !value)}
            className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 transition-colors ${logY
              ? 'border-teal-600 bg-teal-50 text-teal-700 dark:border-teal-500 dark:bg-teal-950/40 dark:text-teal-300'
              : 'border-border-light bg-gray-100 text-gray-600 dark:border-border-dark dark:bg-gray-800 dark:text-gray-300'
              }`}
          >
            <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: logY ? '#0f766e' : '#94a3b8' }} />
            {logY ? 'Log: On' : 'Log: Off'}
          </button>
          {(series || []).map((entry) => {
            const enabled = visibleSeries[entry.label] !== false;
            return (
              <button
                key={entry.label}
                type="button"
                onClick={() => setVisibleSeries((current) => ({ ...current, [entry.label]: !enabled }))}
                className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 transition-colors ${enabled
                  ? 'border-border-light bg-white text-gray-700 dark:border-border-dark dark:bg-gray-900/40 dark:text-gray-200'
                  : 'border-border-light bg-gray-100 text-gray-400 dark:border-border-dark dark:bg-gray-800 dark:text-gray-500'
                  }`}
              >
                <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: enabled ? entry.color : '#94a3b8' }} />
                {enabled ? 'Hide' : 'Show'} {entry.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${width + 80} ${height + 40}`} className="min-w-[680px] w-full h-auto">
          <g transform="translate(56 10)">
            {tickValues.map((value, tick) => {
              const y = (height / ticks) * tick;
              return (
                <g key={tick}>
                  <line x1="0" y1={y} x2={width} y2={y} stroke="currentColor" opacity="0.12" />
                  <text x="-10" y={y + 4} textAnchor="end" fontSize="11" fill="currentColor" opacity="0.7">
                    {formatY(value)}
                  </text>
                </g>
              );
            })}

            {hasVisibleSeries && [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
              const x = width * ratio;
              const value = minX + (maxX - minX) * ratio;
              return (
                <g key={ratio}>
                  <line x1={x} y1="0" x2={x} y2={height} stroke="currentColor" opacity="0.08" />
                  <text x={x} y={height + 18} textAnchor="middle" fontSize="11" fill="currentColor" opacity="0.7">
                    {formatSeconds(value)}
                  </text>
                </g>
              );
            })}

            {validSeries.map((entry) => (
              <g key={entry.label}>
                <path
                  d={svgPathFromPoints(entry.points, width, height, [minX, maxX], [minY, maxY], { logY })}
                  fill="none"
                  stroke={entry.color}
                  strokeWidth="2"
                  strokeLinejoin="round"
                  strokeLinecap="round"
                />
                {entry.points.map((point, index) => {
                  const x = ((point.x - minX) / Math.max(maxX - minX, 1e-9)) * width;
                  const y = mapY(point.y);
                  return (
                    <circle key={`${entry.label}-${index}`} cx={x} cy={y} r="3" fill={entry.color}>
                      <title>{`${entry.label}: ${formatY(point.y)} at ${formatSeconds(point.x)}`}</title>
                    </circle>
                  );
                })}
              </g>
            ))}

            {!hasVisibleSeries && (
              <text x={width / 2} y={height / 2} textAnchor="middle" fontSize="14" fill="currentColor" opacity="0.55">
                All series hidden. Re-enable a series to draw the chart.
              </text>
            )}
            <text x={width / 2} y={height + 34} textAnchor="middle" fontSize="12" fill="currentColor" opacity="0.8">
              Duration
            </text>
            <text
              x={-40}
              y={height / 2}
              textAnchor="middle"
              fontSize="12"
              fill="currentColor"
              opacity="0.8"
              transform={`rotate(-90 -40 ${height / 2})`}
            >
              {logY ? `${yLabel} (log)` : yLabel}
            </text>
          </g>
        </svg>
      </div>
    </div>
  );
}

function BucketBoxChart({
  title,
  subtitle,
  buckets,
  series,
  annotation = null,
  yLabel,
  formatY = (value) => value.toFixed(1),
  height = 240,
}) {
  const defaultSeriesVisibility = (entries) =>
    Object.fromEntries((entries || []).map((entry) => [entry.label, !/wall/i.test(entry.label)]));

  const [visibleSeries, setVisibleSeries] = useState(() =>
    defaultSeriesVisibility(series)
  );

  useEffect(() => {
    setVisibleSeries((current) => {
      const next = {};
      for (const entry of series || []) {
        next[entry.label] = current[entry.label] ?? defaultSeriesVisibility([entry])[entry.label];
      }
      return next;
    });
  }, [series]);

  const validSeries = (series || []).filter((entry) => visibleSeries[entry.label] !== false);
  if (!Array.isArray(buckets) || buckets.length === 0) return null;

  const width = Math.max(720, buckets.length * 80);
  const tickCount = 4;
  const values = validSeries.flatMap((entry) =>
    buckets.flatMap((bucket) => {
      const distribution = entry.distribution(bucket);
      return distribution
        ? [distribution.min, distribution.q1, distribution.median, distribution.q3, distribution.max].filter(Number.isFinite)
        : [];
    })
  );
  const hasVisibleSeries = values.length > 0;
  const maxY = hasVisibleSeries ? (Math.max(...values, 0) * 1.1 || 1) : 1;
  const groupWidth = width / buckets.length;
  const bucketGap = Math.min(4, Math.max(1, groupWidth * 0.04));
  const baseBoxWidth = Math.max(12, groupWidth - bucketGap);
  const overlayInsetStep = validSeries.length > 1
    ? Math.min(4, Math.max(1.5, groupWidth * 0.035))
    : 0;
  const tickValues = hasVisibleSeries
    ? Array.from({ length: tickCount + 1 }, (_, tick) => maxY - ((maxY / tickCount) * tick))
    : [];
  const labelStep = Math.max(1, Math.ceil(buckets.length / 10));
  const mapY = (value) => height - ((value / Math.max(maxY, 1e-9)) * height);

  return (
    <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-5">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{title}</h3>
          {subtitle && <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{subtitle}</p>}
        </div>
        <div className="flex flex-wrap items-center justify-end gap-3 text-xs">
          {(series || []).map((entry) => {
            const enabled = visibleSeries[entry.label] !== false;
            return (
              <button
                key={entry.label}
                type="button"
                onClick={() => setVisibleSeries((current) => ({ ...current, [entry.label]: !enabled }))}
                className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 transition-colors ${enabled
                  ? 'border-border-light bg-white text-gray-700 dark:border-border-dark dark:bg-gray-900/40 dark:text-gray-200'
                  : 'border-border-light bg-gray-100 text-gray-400 dark:border-border-dark dark:bg-gray-800 dark:text-gray-500'
                  }`}
              >
                <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ backgroundColor: enabled ? entry.color : '#94a3b8' }} />
                {enabled ? 'Hide' : 'Show'} {entry.label}
              </button>
            );
          })}
        </div>
      </div>

      <div className="overflow-x-auto">
        <svg viewBox={`0 -28 ${width + 80} ${height + 116}`} className="min-w-[760px] w-full h-auto">
          <g transform="translate(56 12)">
            {tickValues.map((value, index) => {
              const y = (height / tickCount) * index;
              return (
                <g key={index}>
                  <line x1="0" y1={y} x2={width} y2={y} stroke="currentColor" opacity="0.12" />
                  <text x="-10" y={y + 4} textAnchor="end" fontSize="11" fill="currentColor" opacity="0.7">
                    {formatY(value)}
                  </text>
                </g>
              );
            })}

            {buckets.map((bucket, bucketIndex) => {
              const rangeLabel = `${formatClockDuration(bucket.start)}-${formatClockDuration(bucket.end)}`;
              const tickLabel = formatClockDuration(bucket.end);
              const bucketX = bucketIndex * groupWidth + (bucketGap / 2);
              return (
                <g key={bucket.label}>
                  <line x1={bucketIndex * groupWidth} y1="0" x2={bucketIndex * groupWidth} y2={height} stroke="currentColor" opacity="0.05" />
                  {validSeries.map((entry, seriesIndex) => {
                    const distribution = entry.distribution(bucket);
                    if (!distribution) return null;
                    const overlayInset = (validSeries.length - seriesIndex - 1) * overlayInsetStep;
                    const boxWidth = Math.max(8, baseBoxWidth - (overlayInset * 2));
                    const x = bucketX + overlayInset;
                    const centerX = x + (boxWidth / 2);
                    const minY = mapY(distribution.min);
                    const q1Y = mapY(distribution.q1);
                    const medianY = mapY(distribution.median);
                    const q3Y = mapY(distribution.q3);
                    const maxDistY = mapY(distribution.max);
                    return (
                      <g key={`${bucket.label}-${entry.label}`}>
                        <line x1={centerX} y1={maxDistY} x2={centerX} y2={minY} stroke={entry.color} strokeWidth="1.5" opacity="0.8" />
                        <line x1={centerX - 5} y1={maxDistY} x2={centerX + 5} y2={maxDistY} stroke={entry.color} strokeWidth="1.5" opacity="0.9" />
                        <line x1={centerX - 5} y1={minY} x2={centerX + 5} y2={minY} stroke={entry.color} strokeWidth="1.5" opacity="0.9" />
                        <rect
                          x={x}
                          y={q3Y}
                          width={boxWidth}
                          height={Math.max(2, q1Y - q3Y)}
                          rx="3"
                          fill={entry.color}
                          opacity="0.22"
                          stroke={entry.color}
                          strokeWidth="1.25"
                        >
                          <title>{`${entry.label}
${rangeLabel}
count: ${bucket.count}
min: ${formatY(distribution.min)}
q1: ${formatY(distribution.q1)}
median: ${formatY(distribution.median)}
q3: ${formatY(distribution.q3)}
max: ${formatY(distribution.max)}`}</title>
                        </rect>
                        <line x1={x} y1={medianY} x2={x + boxWidth} y2={medianY} stroke={entry.color} strokeWidth="2" />
                      </g>
                    );
                  })}
                  {bucketIndex % labelStep === 0 && (
                    <text
                      x={bucketIndex * groupWidth + (groupWidth / 2)}
                      y={height + 16}
                      textAnchor="middle"
                      fontSize="11"
                      fill="currentColor"
                      opacity="0.7"
                    >
                      {tickLabel}
                    </text>
                  )}
                </g>
              );
            })}

            {annotation && Number.isFinite(annotation.bucketIndex) && annotation.bucketIndex >= 0 && annotation.bucketIndex < buckets.length && (
              <g>
                <line
                  x1={(annotation.bucketIndex * groupWidth) + (groupWidth / 2)}
                  y1="-2"
                  x2={(annotation.bucketIndex * groupWidth) + (groupWidth / 2)}
                  y2="14"
                  stroke={annotation.color || '#b45309'}
                  strokeWidth="1.5"
                  opacity="0.95"
                />
                <polygon
                  points={`${(annotation.bucketIndex * groupWidth) + (groupWidth / 2) - 5},14 ${(annotation.bucketIndex * groupWidth) + (groupWidth / 2) + 5},14 ${(annotation.bucketIndex * groupWidth) + (groupWidth / 2)},22`}
                  fill={annotation.color || '#b45309'}
                  opacity="0.95"
                />
                <text
                  x={(annotation.bucketIndex * groupWidth) + (groupWidth / 2)}
                  y="-8"
                  textAnchor="middle"
                  fontSize="11"
                  fontWeight="600"
                  fill={annotation.color || '#b45309'}
                >
                  {annotation.label}
                </text>
                {annotation.detail && (
                  <text
                    x={(annotation.bucketIndex * groupWidth) + (groupWidth / 2)}
                    y="-20"
                    textAnchor="middle"
                    fontSize="10"
                    fill={annotation.color || '#b45309'}
                    opacity="0.8"
                  >
                    {annotation.detail}
                  </text>
                )}
              </g>
            )}

            {!hasVisibleSeries && (
              <text x={width / 2} y={height / 2} textAnchor="middle" fontSize="14" fill="currentColor" opacity="0.55">
                All series hidden. Re-enable a series to draw the chart.
              </text>
            )}
            <text x={width / 2} y={height + 44} textAnchor="middle" fontSize="12" fill="currentColor" opacity="0.8">
              Bucket End
            </text>
            <text
              x={-40}
              y={height / 2}
              textAnchor="middle"
              fontSize="12"
              fill="currentColor"
              opacity="0.8"
              transform={`rotate(-90 -40 ${height / 2})`}
            >
              {yLabel}
            </text>
          </g>
        </svg>
      </div>
    </div>
  );
}

function InlineDistributionBars({ bins, expanded = false, onClick }) {
  const width = 248;
  const height = 42;
  const validBins = bins.filter((bin) => Number.isFinite(bin?.count));
  if (validBins.length === 0) return null;

  const maxBars = 14;
  const groupedBins = [];
  const groupSize = Math.max(1, Math.ceil(validBins.length / maxBars));
  for (let i = 0; i < validBins.length; i += groupSize) {
    const slice = validBins.slice(i, i + groupSize);
    groupedBins.push({
      start: slice[0].start,
      end: slice[slice.length - 1].end,
      count: slice.reduce((sum, item) => sum + item.count, 0),
      label: `${formatCompactDuration(slice[0].start)}-${formatCompactDuration(slice[slice.length - 1].end)}`,
    });
  }

  const maxCount = Math.max(...groupedBins.map((bin) => bin.count), 1);
  const peakBin = groupedBins.reduce((best, bin) => (bin.count > best.count ? bin : best), groupedBins[0]);
  const gap = 4;
  const barWidth = Math.max(10, (width - gap * (groupedBins.length - 1)) / Math.max(groupedBins.length, 1));
  const rangeLabel = `${formatCompactDuration(groupedBins[0].start)}-${formatCompactDuration(groupedBins[groupedBins.length - 1].end)}`;

  const Wrapper = onClick ? 'button' : 'div';

  return (
    <Wrapper
      type={onClick ? 'button' : undefined}
      onClick={onClick}
      className={`ml-auto shrink-0 rounded-lg border border-border-light dark:border-border-dark bg-white/70 dark:bg-gray-900/40 px-2 py-1.5 text-left transition-colors ${onClick ? 'cursor-pointer hover:bg-white dark:hover:bg-gray-900/60' : ''
        }`}
      title={onClick ? (expanded ? 'Hide detailed distribution' : 'Show detailed distribution') : undefined}
    >
      <div className="mb-1 flex items-center justify-between gap-3 text-[10px] text-gray-500 dark:text-gray-400">
        <span className="uppercase tracking-[0.12em]">Distribution</span>
        <span>{expanded ? 'hide detail' : 'show detail'}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="block h-10 w-[248px]">
        <rect x="0" y="0" width={width} height={height} rx="6" fill="rgba(15,118,110,0.08)" />
        {groupedBins.map((bin, index) => {
          const x = index * (barWidth + gap);
          const barHeight = Math.max(4, (bin.count / maxCount) * (height - 2));
          const y = height - barHeight;
          return (
            <rect key={`${bin.start}-${bin.end}`} x={x} y={y} width={barWidth} height={barHeight} rx="2" fill="#0f766e" opacity="0.85">
              <title>{`${bin.label}: ${bin.count} sample${bin.count === 1 ? '' : 's'}`}</title>
            </rect>
          );
        })}
      </svg>
      <div className="mt-1 flex items-center justify-between text-[10px] text-gray-500 dark:text-gray-400">
        <span>{rangeLabel}</span>
        <span>{validBins.length} buckets</span>
      </div>
    </Wrapper>
  );
}

function DetailedDistributionHistogram({ title, subtitle, bins }) {
  const validBins = bins.filter((bin) => Number.isFinite(bin?.count));
  if (validBins.length === 0) return null;

  const width = Math.max(760, validBins.length * 54);
  const height = 240;
  const maxCount = Math.max(...validBins.map((bin) => bin.count), 1);
  const gap = 10;
  const barWidth = Math.max(20, (width - gap * (validBins.length - 1)) / Math.max(validBins.length, 1));

  return (
    <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-5">
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-white">{title}</h3>
        {subtitle && <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{subtitle}</p>}
      </div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${width + 80} ${height + 110}`} className="min-w-[840px] w-full h-auto">
          <g transform="translate(56 12)">
            {[0, 1, 2, 3, 4].map((tick) => {
              const y = (height / 4) * tick;
              const value = Math.round(maxCount - (maxCount * tick) / 4);
              return (
                <g key={tick}>
                  <line x1="0" y1={y} x2={width} y2={y} stroke="currentColor" opacity="0.12" />
                  <text x="-10" y={y + 4} textAnchor="end" fontSize="11" fill="currentColor" opacity="0.72">
                    {value}
                  </text>
                </g>
              );
            })}
            {validBins.map((bin, index) => {
              const x = index * (barWidth + gap);
              const barHeight = Math.max(3, (bin.count / maxCount) * height);
              const y = height - barHeight;
              return (
                <g key={`${bin.start}-${bin.end}`}>
                  <rect x={x} y={y} width={barWidth} height={barHeight} rx="4" fill="#0f766e" opacity="0.88">
                    <title>{`${bin.label}: ${bin.count} sample${bin.count === 1 ? '' : 's'}`}</title>
                  </rect>
                  <text x={x + barWidth / 2} y={y - 6} textAnchor="middle" fontSize="10" fill="currentColor" opacity="0.82">
                    {bin.count}
                  </text>
                  <text
                    x={x + barWidth / 2}
                    y={height + 14}
                    textAnchor="end"
                    fontSize="10"
                    fill="currentColor"
                    opacity="0.72"
                    transform={`rotate(-35 ${x + barWidth / 2} ${height + 14})`}
                  >
                    {bin.label}
                  </text>
                </g>
              );
            })}
            <text x={width / 2} y={height + 64} textAnchor="middle" fontSize="12" fill="currentColor" opacity="0.82">
              Duration Buckets
            </text>
            <text
              x={-36}
              y={height / 2}
              textAnchor="middle"
              fontSize="12"
              fill="currentColor"
              opacity="0.82"
              transform={`rotate(-90 -36 ${height / 2})`}
            >
              Samples
            </text>
          </g>
        </svg>
      </div>
    </div>
  );
}

export default function App() {
  const initialSettings = loadSettings();
  const tRef = useRef(null);
  const fileInputRef = useRef(null);
  const benchmarkFileInputRef = useRef(null);
  const benchmarkFolderInputRef = useRef(null);
  const benchmarkCancelRef = useRef(false);
  const { pipeline } = Transformers;
  const [activeView, setActiveView] = useState(initialSettings.activeView || 'transcribe');
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
  const [threadingStatus, setThreadingStatus] = useState({ sab: false, threads: 1 });

  const [direct, setDirect] = useState(initialSettings.direct !== undefined ? Boolean(initialSettings.direct) : true);
  const [rt, setRt] = useState(initialSettings.rt !== undefined ? Boolean(initialSettings.rt) : true);
  const [audioPrepBackend, setAudioPrepBackend] = useState(
    initialSettings.audioPrepBackend || AUDIO_PREP_BACKEND_DEFAULT
  );
  const [audioPrepQuality, setAudioPrepQuality] = useState(
    initialSettings.audioPrepQuality || AUDIO_PREP_CUSTOM_RESAMPLER_DEFAULT
  );
  const [pipelineTimestampMode, setPipelineTimestampMode] = useState(
    initialSettings.pipelineTimestampMode || (initialSettings.rt ? 'segments' : 'none')
  );
  const [pipelineWindowOverrideEnabled, setPipelineWindowOverrideEnabled] = useState(
    initialSettings.pipelineWindowOverrideEnabled !== undefined
      ? Boolean(initialSettings.pipelineWindowOverrideEnabled)
      : false
  );
  const [pipelineWindowOverrideSec, setPipelineWindowOverrideSec] = useState(
    clampPipelineWindowSec(initialSettings.pipelineWindowOverrideSec)
  );
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
  const [lastRunMetrics, setLastRunMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedFileName, setSelectedFileName] = useState('');
  const [benchmarkFiles, setBenchmarkFiles] = useState([]);
  const [benchmarkCatalog, setBenchmarkCatalog] = useState([]);
  const [benchmarkFolderHandleSupported] = useState(
    typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function'
  );
  const [benchmarkStoredFolderAvailable, setBenchmarkStoredFolderAvailable] = useState(false);
  const [benchmarkStoredFolderName, setBenchmarkStoredFolderName] = useState('');
  const [benchmarkTargetMode, setBenchmarkTargetMode] = useState(initialSettings.benchmarkTargetMode || 'auto');
  const [benchmarkRandomCount, setBenchmarkRandomCount] = useState(
    Number.isFinite(initialSettings.benchmarkRandomCount)
      ? Math.max(1, Math.floor(Number(initialSettings.benchmarkRandomCount)))
      : 12
  );
  const [benchmarkBucketSize, setBenchmarkBucketSize] = useState(
    Number.isFinite(initialSettings.benchmarkBucketSize)
      ? Math.max(5, Number(initialSettings.benchmarkBucketSize))
      : 30
  );
  const [benchmarkOverlapSec, setBenchmarkOverlapSec] = useState(
    Number.isFinite(initialSettings.benchmarkOverlapSec)
      ? Math.max(0, Number(initialSettings.benchmarkOverlapSec))
      : 6
  );
  const [benchmarkMinDurationSec, setBenchmarkMinDurationSec] = useState(
    Number.isFinite(initialSettings.benchmarkMinDurationSec)
      ? Math.max(0, Number(initialSettings.benchmarkMinDurationSec))
      : ''
  );
  const [benchmarkMaxDurationSec, setBenchmarkMaxDurationSec] = useState(
    Number.isFinite(initialSettings.benchmarkMaxDurationSec)
      ? Math.max(0, Number(initialSettings.benchmarkMaxDurationSec))
      : ''
  );
  const [benchmarkTargetsText, setBenchmarkTargetsText] = useState(initialSettings.benchmarkTargetsText || '');
  const [benchmarkSamplesPerTarget, setBenchmarkSamplesPerTarget] = useState(
    Number.isFinite(initialSettings.benchmarkSamplesPerTarget)
      ? Math.max(1, Math.floor(Number(initialSettings.benchmarkSamplesPerTarget)))
      : 1
  );
  const [benchmarkMeasurementRepeats, setBenchmarkMeasurementRepeats] = useState(
    Number.isFinite(initialSettings.benchmarkMeasurementRepeats)
      ? Math.max(1, Math.floor(Number(initialSettings.benchmarkMeasurementRepeats)))
      : 1
  );
  const [benchmarkLongRepeatBoostEnabled, setBenchmarkLongRepeatBoostEnabled] = useState(
    initialSettings.benchmarkLongRepeatBoostEnabled !== undefined
      ? Boolean(initialSettings.benchmarkLongRepeatBoostEnabled)
      : false
  );
  const [benchmarkLongRepeatThresholdSec, setBenchmarkLongRepeatThresholdSec] = useState(
    Number.isFinite(initialSettings.benchmarkLongRepeatThresholdSec)
      ? Math.max(5, Number(initialSettings.benchmarkLongRepeatThresholdSec))
      : 180
  );
  const [benchmarkLongRepeatMultiplier, setBenchmarkLongRepeatMultiplier] = useState(
    Number.isFinite(initialSettings.benchmarkLongRepeatMultiplier)
      ? Math.max(1, Math.floor(Number(initialSettings.benchmarkLongRepeatMultiplier)))
      : 2
  );
  const [benchmarkWarmup, setBenchmarkWarmup] = useState(
    initialSettings.benchmarkWarmup !== undefined ? Boolean(initialSettings.benchmarkWarmup) : true
  );
  const [benchmarkRandomizeExecution, setBenchmarkRandomizeExecution] = useState(
    initialSettings.benchmarkRandomizeExecution !== undefined
      ? Boolean(initialSettings.benchmarkRandomizeExecution)
      : true
  );
  const [benchmarkIndexing, setBenchmarkIndexing] = useState(false);
  const [benchmarkRuns, setBenchmarkRuns] = useState([]);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const [benchmarkStatus, setBenchmarkStatus] = useState('Load a model, then select audio files to benchmark.');
  const [benchmarkCurrent, setBenchmarkCurrent] = useState(null);
  const [benchmarkStopOnError, setBenchmarkStopOnError] = useState(Boolean(initialSettings.benchmarkStopOnError));
  const [benchmarkRandomSeed, setBenchmarkRandomSeed] = useState(
    Number.isFinite(initialSettings.benchmarkRandomSeed)
      ? Math.floor(Number(initialSettings.benchmarkRandomSeed))
      : Date.now()
  );
  const [benchmarkRandomPlan, setBenchmarkRandomPlan] = useState([]);
  const [benchmarkRandomDistributionExpanded, setBenchmarkRandomDistributionExpanded] = useState(false);
  const [benchmarkRestoringFolder, setBenchmarkRestoringFolder] = useState(false);
  const [darkMode, setDarkMode] = useState(getInitialDarkMode(initialSettings));
  const [downloadProgress, setDownloadProgress] = useState(null); // {pct, loaded, total, file}
  const [isCached, setIsCached] = useState(null); // null=unknown, true/false
  const usingNpmTransformers = SOURCE === 'npm';

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Follow HF/parent theme when provided; otherwise fallback to system changes
  // if user has no saved explicit theme preference.
  useEffect(() => {
    const hfTheme = getHfThemeFromQuery();
    if (hfTheme !== null) {
      setDarkMode(hfTheme);
    }

    const onMessage = (event) => {
      const messageTheme = getThemeFromMessage(event?.data);
      if (messageTheme !== null) {
        setDarkMode(messageTheme);
      }
    };

    window.addEventListener('message', onMessage);

    if (typeof window.matchMedia !== 'function') {
      return () => window.removeEventListener('message', onMessage);
    }

    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const onMediaChange = (event) => {
      if (getHfThemeFromQuery() !== null) return;
      if (typeof initialSettings.darkMode === 'boolean') return;
      setDarkMode(event.matches);
    };

    if (typeof mediaQuery.addEventListener === 'function') {
      mediaQuery.addEventListener('change', onMediaChange);
    } else {
      mediaQuery.addListener(onMediaChange);
    }

    return () => {
      window.removeEventListener('message', onMessage);
      if (typeof mediaQuery.removeEventListener === 'function') {
        mediaQuery.removeEventListener('change', onMediaChange);
      } else {
        mediaQuery.removeListener(onMediaChange);
      }
    };
  }, [initialSettings.darkMode]);

  useEffect(() => {
    const sabAvailable = typeof SharedArrayBuffer !== 'undefined';
    const coiAvailable = typeof window !== 'undefined' && window.crossOriginIsolated === true;
    const multiThreadReady = sabAvailable && coiAvailable;
    setThreadingStatus({
      sab: multiThreadReady,
      threads: multiThreadReady ? maxWasmCores : 1,
    });
  }, [maxWasmCores]);

  useEffect(() => {
    const { env } = Transformers;
    if (!env?.backends?.onnx?.wasm) return;
    const allowedThreads = threadingStatus.sab ? maxWasmCores : 1;
    env.backends.onnx.wasm.numThreads = clampThreadCount(wasmThreads, allowedThreads);
  }, [wasmThreads, maxWasmCores, threadingStatus.sab]);

  useEffect(() => {
    saveSettings({
      activeView,
      modelId, mode, encDev, encDtype, decDtype, wasmThreads,
      direct, rt, audioPrepBackend, audioPrepQuality, pipelineTimestampMode, pipelineWindowOverrideEnabled, pipelineWindowOverrideSec, metrics,
      returnWords, returnTokens, returnFrameConf, frameIdx, logProbs, tdtSteps,
      benchmarkTargetMode, benchmarkRandomCount, benchmarkRandomSeed, benchmarkBucketSize, benchmarkOverlapSec,
      benchmarkMinDurationSec: effectiveBenchmarkMinDurationSec,
      benchmarkMaxDurationSec: effectiveBenchmarkMaxDurationSec,
      benchmarkTargetsText, benchmarkSamplesPerTarget, benchmarkMeasurementRepeats,
      benchmarkLongRepeatBoostEnabled, benchmarkLongRepeatThresholdSec, benchmarkLongRepeatMultiplier,
      benchmarkWarmup, benchmarkRandomizeExecution, benchmarkStopOnError,
      offset, darkMode,
    });
  }, [
    activeView,
    modelId, mode, encDev, encDtype, decDtype, wasmThreads,
    direct, rt, audioPrepBackend, audioPrepQuality, pipelineTimestampMode, pipelineWindowOverrideEnabled, pipelineWindowOverrideSec, metrics,
    returnWords, returnTokens, returnFrameConf, frameIdx, logProbs, tdtSteps,
    benchmarkTargetMode, benchmarkRandomCount, benchmarkRandomSeed, benchmarkBucketSize, benchmarkOverlapSec,
    benchmarkMinDurationSec, benchmarkMaxDurationSec,
    benchmarkTargetsText, benchmarkSamplesPerTarget, benchmarkMeasurementRepeats,
    benchmarkLongRepeatBoostEnabled, benchmarkLongRepeatThresholdSec, benchmarkLongRepeatMultiplier,
    benchmarkWarmup, benchmarkRandomizeExecution, benchmarkStopOnError,
    offset, darkMode,
  ]);

  const stats = useMemo(() => {
    const words = Array.isArray(result?.words) ? result.words.length : null;
    const tokens = Array.isArray(result?.tokens) ? result.tokens.length : null;
    const metricsOut = result?.metrics ?? null;
    const rtf = metricsOut?.rtf ?? null;
    const rtfx = metricsOut?.rtfX ?? (rtf && Number.isFinite(rtf) && rtf > 0 ? 1 / rtf : null);
    const audioDurationSec = lastRunMetrics?.durationSec ?? null;
    const inferenceRtf = lastRunMetrics?.inferenceRtf ?? rtf ?? null;
    const inferenceRtfx =
      lastRunMetrics?.inferenceRtfx ??
      (inferenceRtf && Number.isFinite(inferenceRtf) && inferenceRtf > 0 ? 1 / inferenceRtf : null);
    const endToEndRtf = lastRunMetrics?.endToEndRtf ?? null;
    const endToEndRtfx =
      lastRunMetrics?.endToEndRtfx ??
      (endToEndRtf && Number.isFinite(endToEndRtf) && endToEndRtf > 0 ? 1 / endToEndRtf : null);
    return {
      mode: lastRunMetrics?.mode ?? (direct ? 'direct' : 'pipeline'),
      textLen: textOf(result).length,
      words,
      tokens,
      audioDurationSec,
      audioPrepMs: lastRunMetrics?.audioPrepMs ?? null,
      audioPrepBackend: lastRunMetrics?.audioPrepBackend ?? null,
      audioPrepStrategy: lastRunMetrics?.audioPrepStrategy ?? null,
      audioDecodeMs: lastRunMetrics?.audioDecodeMs ?? null,
      audioDownmixMs: lastRunMetrics?.audioDownmixMs ?? null,
      audioResampleMs: lastRunMetrics?.audioResampleMs ?? null,
      audioResampler: lastRunMetrics?.audioResampler ?? null,
      audioResamplerQuality: lastRunMetrics?.audioResamplerQuality ?? null,
      audioInputSampleRate: lastRunMetrics?.audioInputSampleRate ?? null,
      audioOutputSampleRate: lastRunMetrics?.audioOutputSampleRate ?? null,
      processorMs: lastRunMetrics?.processorMs ?? null,
      callMs: lastRunMetrics?.callMs ?? null,
      inferenceWallMs: lastRunMetrics?.inferenceWallMs ?? null,
      endToEndMs: lastRunMetrics?.endToEndMs ?? null,
      inferenceRtfx,
      endToEndRtfx,
      utteranceConfidence: result?.confidence?.utterance ?? null,
      wordConfidenceAverage: result?.confidence?.wordAverage ?? null,
      preprocessMs: metricsOut?.preprocessMs ?? null,
      encodeMs: metricsOut?.encodeMs ?? null,
      decodeMs: metricsOut?.decodeMs ?? null,
      tokenizeMs: metricsOut?.tokenizeMs ?? null,
      totalMs: metricsOut?.totalMs ?? null,
      rtf,
      rtfx,
      pipelineWindowSec:
        !direct && pipelineWindowOverrideEnabled
          ? pipelineWindowOverrideSec
          : null,
    };
  }, [result, lastRunMetrics, direct, pipelineWindowOverrideEnabled, pipelineWindowOverrideSec]);

  const pipelineCallOptions = useMemo(() => {
    const options = (
      pipelineTimestampMode === 'segments'
        ? { return_timestamps: true }
        : pipelineTimestampMode === 'words'
          ? { return_timestamps: 'word' }
          : {}
    );
    if (pipelineWindowOverrideEnabled) {
      options.chunk_length_s = pipelineWindowOverrideSec;
    }
    return options;
  }, [pipelineTimestampMode, pipelineWindowOverrideEnabled, pipelineWindowOverrideSec]);

  const pipelineContractSummary = useMemo(() => {
    if (pipelineTimestampMode === 'segments') {
      return {
        title: 'Pipeline sentences',
        shape: '{ text, chunks }',
        detail: 'Returns finalized sentence-like chunks with [start, end] timestamps.',
      };
    }
    if (pipelineTimestampMode === 'words') {
      return {
        title: 'Pipeline words',
        shape: '{ text, chunks }',
        detail: 'Returns a flat word list with [start, end] timestamps.',
      };
    }
    return {
      title: 'Pipeline text only',
      shape: '{ text }',
      detail: 'Returns only the merged transcript with no public chunk list.',
    };
  }, [pipelineTimestampMode]);

  const directOutputFields = useMemo(() => {
    const fields = ['text'];
    if (rt) {
      fields.push('utteranceTimestamp', 'confidence');
      if (returnWords) fields.push('words');
      if (returnTokens) fields.push('tokens');
    }
    if (metrics) fields.push('metrics');
    if (returnFrameConf || frameIdx || logProbs || tdtSteps) fields.push('debug');
    return fields;
  }, [rt, returnWords, returnTokens, metrics, returnFrameConf, frameIdx, logProbs, tdtSteps]);

  const directContractSummary = useMemo(() => ({
    title: 'Direct Nemo output',
    shape: `{ ${directOutputFields.join(', ')} }`,
    detail: 'Calls model.transcribe() and keeps the richer NeMo-style JS output surface.',
  }), [directOutputFields]);

  const currentContractSummary = direct ? directContractSummary : pipelineContractSummary;

  const loadOptionsSnippet = useMemo(() => toJSON({
    device: { encoder_model: encDev, decoder_model_merged: DECODER_DEVICE },
    dtype: { encoder_model: encDtype, decoder_model_merged: decDtype },
  }), [encDev, encDtype, decDtype]);

  const pipelineOptionsSnippet = useMemo(
    () => toJSON(pipelineCallOptions),
    [pipelineCallOptions]
  );

  const directOptionsSnippet = useMemo(() => [
    '{',
    '  tokenizer: transcriber.tokenizer,',
    `  returnTimestamps: ${rt},`,
    `  returnWords: ${rt ? returnWords : false},`,
    `  returnTokens: ${rt ? returnTokens : false},`,
    `  returnMetrics: ${metrics},`,
    `  returnFrameConfidences: ${returnFrameConf},`,
    `  returnFrameIndices: ${frameIdx},`,
    `  returnLogProbs: ${logProbs},`,
    `  returnTdtSteps: ${tdtSteps},`,
    `  timeOffset: ${Number(offset) || 0},`,
    '}',
  ].join('\n'), [rt, returnWords, returnTokens, metrics, returnFrameConf, frameIdx, logProbs, tdtSteps, offset]);

  const currentOptionsSnippet = direct ? directOptionsSnippet : pipelineOptionsSnippet;

  const currentExampleSnippet = useMemo(() => {
    const modelIdLiteral = JSON.stringify(modelId);
    const loadOptionsBlock = `const loadOptions = ${loadOptionsSnippet};`;
    const loadSnippet = mode === 'explicit'
      ? [
        "import { AutoProcessor, AutoTokenizer, NemoConformerForTDT, AutomaticSpeechRecognitionPipeline } from '@huggingface/transformers';",
        '',
        `const modelId = ${modelIdLiteral};`,
        'const audio = /* Float32Array, URL, Blob, or File */;',
        '',
        loadOptionsBlock,
        '',
        'const [processor, tokenizer, model] = await Promise.all([',
        '  AutoProcessor.from_pretrained(modelId),',
        '  AutoTokenizer.from_pretrained(modelId),',
        '  NemoConformerForTDT.from_pretrained(modelId, loadOptions),',
        ']);',
        '',
        'const transcriber = new AutomaticSpeechRecognitionPipeline({',
        "  task: 'automatic-speech-recognition',",
        '  model,',
        '  processor,',
        '  tokenizer,',
        '});',
      ].join('\n')
      : [
        "import { pipeline } from '@huggingface/transformers';",
        '',
        `const modelId = ${modelIdLiteral};`,
        'const audio = /* Float32Array, URL, Blob, or File */;',
        '',
        loadOptionsBlock,
        '',
        "const transcriber = await pipeline('automatic-speech-recognition', modelId, loadOptions);",
      ].join('\n');

    const inferenceSnippet = direct
      ? [
        'const inputs = await transcriber.processor(audio);',
        'const output = await transcriber.model.transcribe(',
        '  inputs,',
        indentCode(directOptionsSnippet, 2),
        ');',
      ].join('\n')
      : [
        'const output = await transcriber(',
        '  audio,',
        indentCode(pipelineOptionsSnippet, 2),
        ');',
      ].join('\n');

    return `${loadSnippet}\n\n${inferenceSnippet}`;
  }, [modelId, loadOptionsSnippet, mode, direct, directOptionsSnippet, pipelineOptionsSnippet]);

  const benchmarkSummary = useMemo(
    () => summarizeBenchmarkRuns(benchmarkRuns, benchmarkBucketSize, benchmarkOverlapSec),
    [benchmarkRuns, benchmarkBucketSize, benchmarkOverlapSec]
  );

  const benchmarkEffectiveSweetSpot = useMemo(
    () => findEffectiveRtfxSweetSpot(benchmarkSummary, benchmarkOverlapSec),
    [benchmarkSummary, benchmarkOverlapSec]
  );

  const benchmarkOverall = useMemo(
    () => summarizeBenchmarkOverall(benchmarkRuns),
    [benchmarkRuns]
  );

  const benchmarkChartData = useMemo(() => {
    const validRuns = benchmarkRuns
      .filter((run) => !run?.error && Number.isFinite(run?.durationSec))
      .sort((a, b) => a.durationSec - b.durationSec);

    return {
      rtfxSeries: [
        {
          label: 'Wall RTFx',
          color: '#1d4ed8',
          points: validRuns
            .filter((run) => Number.isFinite(run?.wallRtfx))
            .map((run) => ({ x: run.durationSec, y: run.wallRtfx })),
        },
        {
          label: 'Model RTFx',
          color: '#0f766e',
          points: validRuns
            .filter((run) => Number.isFinite(run?.modelRtfx))
            .map((run) => ({ x: run.durationSec, y: run.modelRtfx })),
        },
      ],
      phaseSeries: [
        {
          label: 'Encode ms',
          color: '#b45309',
          points: validRuns
            .filter((run) => Number.isFinite(run?.encodeMs))
            .map((run) => ({ x: run.durationSec, y: run.encodeMs })),
        },
        {
          label: 'Decode ms',
          color: '#7c3aed',
          points: validRuns
            .filter((run) => Number.isFinite(run?.decodeMs))
            .map((run) => ({ x: run.durationSec, y: run.decodeMs })),
        },
      ],
    };
  }, [benchmarkRuns]);

  const benchmarkCatalogDurationLimit = useMemo(() => {
    const durations = benchmarkCatalog
      .map((entry) => entry?.durationSec)
      .filter((value) => Number.isFinite(value) && value > 0);
    if (durations.length === 0) return 600;
    return Math.max(30, Math.ceil(Math.max(...durations) / 5) * 5);
  }, [benchmarkCatalog]);

  const effectiveBenchmarkMinDurationSec = useMemo(() => {
    const value = parseDurationBound(benchmarkMinDurationSec);
    return value != null && value > 0 ? Math.min(value, benchmarkCatalogDurationLimit) : null;
  }, [benchmarkMinDurationSec, benchmarkCatalogDurationLimit]);

  const effectiveBenchmarkMaxDurationSec = useMemo(() => {
    const value = parseDurationBound(benchmarkMaxDurationSec);
    if (value == null) return null;
    const clamped = Math.min(value, benchmarkCatalogDurationLimit);
    return clamped >= benchmarkCatalogDurationLimit ? null : clamped;
  }, [benchmarkMaxDurationSec, benchmarkCatalogDurationLimit]);

  const benchmarkTargets = useMemo(() => {
    const sourceCatalog = benchmarkCatalog.filter((entry) => {
      if (!Number.isFinite(entry?.durationSec)) return false;
      if (effectiveBenchmarkMinDurationSec != null && entry.durationSec < effectiveBenchmarkMinDurationSec) return false;
      if (effectiveBenchmarkMaxDurationSec != null && entry.durationSec > effectiveBenchmarkMaxDurationSec) return false;
      return true;
    });
    if (benchmarkTargetMode === 'auto') {
      return suggestDurationTargets(sourceCatalog);
    }
    if (benchmarkTargetMode === 'random') {
      return [];
    }
    return parseDurationTargets(benchmarkTargetsText);
  }, [benchmarkTargetMode, benchmarkCatalog, benchmarkTargetsText, effectiveBenchmarkMinDurationSec, effectiveBenchmarkMaxDurationSec]);

  const benchmarkFilteredCatalog = useMemo(() => {
    return benchmarkCatalog.filter((entry) => {
      if (!Number.isFinite(entry?.durationSec)) return false;
      if (effectiveBenchmarkMinDurationSec != null && entry.durationSec < effectiveBenchmarkMinDurationSec) return false;
      if (effectiveBenchmarkMaxDurationSec != null && entry.durationSec > effectiveBenchmarkMaxDurationSec) return false;
      return true;
    });
  }, [benchmarkCatalog, effectiveBenchmarkMinDurationSec, effectiveBenchmarkMaxDurationSec]);

  const benchmarkPlan = useMemo(
    () => buildTargetSamplePlan(benchmarkFilteredCatalog, benchmarkTargets, benchmarkSamplesPerTarget),
    [benchmarkFilteredCatalog, benchmarkTargets, benchmarkSamplesPerTarget]
  );

  const benchmarkResolvedPlan = benchmarkTargetMode === 'random' ? benchmarkRandomPlan : benchmarkPlan;

  const benchmarkRandomDistribution = useMemo(
    () => buildDurationHistogram(benchmarkResolvedPlan, benchmarkBucketSize),
    [benchmarkResolvedPlan, benchmarkBucketSize]
  );

  useEffect(() => {
    setBenchmarkRandomPlan([]);
    setBenchmarkRandomDistributionExpanded(false);
  }, [benchmarkCatalog, benchmarkMinDurationSec, benchmarkMaxDurationSec, benchmarkBucketSize, benchmarkRandomCount]);

  useEffect(() => {
    if (!benchmarkFolderHandleSupported) return;

    let cancelled = false;
    (async () => {
      try {
        const handle = await readStoredDirectoryHandle();
        if (!handle || cancelled) return;

        setBenchmarkStoredFolderAvailable(true);
        setBenchmarkStoredFolderName(handle.name || 'saved folder');
        setBenchmarkRestoringFolder(true);
        const restored = await restoreBenchmarkFolderFromHandle(handle, {
          requestAccess: false,
          sourceLabel: handle.name || 'saved folder',
        });
        if (!restored && !cancelled) {
          setBenchmarkStatus(`Saved folder "${handle.name || 'saved folder'}" is available. Click reconnect to reuse it.`);
        }
      } catch {
        // Ignore stored handle restore failures on startup.
      } finally {
        if (!cancelled) {
          setBenchmarkRestoringFolder(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [benchmarkFolderHandleSupported]);

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
    setLoading(true); setError(''); setStatus('Loading...'); setResult(null); setLastRunMetrics(null); setDownloadProgress(null);
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
      const expectedPrefix = 'The boy was there when the sun rose.';
      try {
        const { audio: pcm } = await prepareAudioInput(SAMPLE, 16000, audioPrepBackend, audioPrepQuality);
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
        console.log('[App] Warm-up raw transcription:', warmupText);
        const normalize = (s) => s.toLowerCase().replace(/[^\w\s]/g, '').trim();
        if (normalize(warmupText).startsWith(normalize(expectedPrefix))) {
          console.log('[App] Warm-up verification passed');
          setModelLoaded(true);
          setStatus('Model ready');
        } else {
          console.warn(`[App] Warm-up mismatch. Expected prefix "${expectedPrefix}", got "${warmupText}"`);
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

  async function inferInput(input, { forceMetrics = false, expectedDurationSec = null } = {}) {
    if (!tRef.current) {
      throw new Error('Load a model first.');
    }

    const t = tRef.current;
    const sampleRate = t.processor?.feature_extractor?.config?.sampling_rate ?? 16000;
    const endToEndStart = performance.now();
    const { audio: mono, profile: audioPrepProfile } = await prepareAudioInput(
      input,
      sampleRate,
      audioPrepBackend,
      audioPrepQuality,
    );
    const audioPrepMs = audioPrepProfile?.totalMs ?? 0;
    const durationSec = Number.isFinite(expectedDurationSec) && expectedDurationSec > 0
      ? expectedDurationSec
      : mono.length / sampleRate;
    const useDirect = direct && mType === 'nemo-conformer-tdt';
    let inputEstimate = null;
    let processorMs = null;
    let callMs = null;

    let out;
    if (useDirect) {
      const processorStart = performance.now();
      const inputs = await t.processor(mono);
      processorMs = performance.now() - processorStart;
      inputEstimate = estimateInputPayloadMetrics({
        mono,
        sampleRate,
        inputs,
        encoderDtype: encDtype,
      });
      const callStart = performance.now();
      out = await t.model.transcribe(inputs, {
        tokenizer: t.tokenizer,
        returnTimestamps: rt,
        returnWords: returnWords,
        returnTokens: returnTokens,
        returnMetrics: forceMetrics ? true : metrics,
        returnFrameConfidences: returnFrameConf,
        returnFrameIndices: frameIdx,
        returnLogProbs: logProbs,
        returnTdtSteps: tdtSteps,
        timeOffset: Number(offset) || 0,
      });
      callMs = performance.now() - callStart;
    } else {
      const p = { ...pipelineCallOptions };
      const callStart = performance.now();
      out = await t(mono, p);
      callMs = performance.now() - callStart;
    }
    const inferenceWallMs = (processorMs ?? 0) + (callMs ?? 0);
    const endToEndMs = performance.now() - endToEndStart;
    const inferenceRtf = durationSec > 0 ? inferenceWallMs / 1000 / durationSec : null;
    const endToEndRtf = durationSec > 0 ? endToEndMs / 1000 / durationSec : null;

    return {
      out,
      useDirect,
      durationSec,
      audioPrepMs,
      audioPrepBackend: audioPrepProfile?.backend ?? audioPrepBackend,
      audioPrepStrategy: audioPrepProfile?.strategy ?? null,
      audioDecodeMs: audioPrepProfile?.decodeMs ?? null,
      audioDownmixMs: audioPrepProfile?.downmixMs ?? null,
      audioResampleMs: audioPrepProfile?.resampleMs ?? null,
      audioResampler: audioPrepProfile?.resampler ?? null,
      audioResamplerQuality: audioPrepProfile?.resamplerQuality ?? null,
      audioInputSampleRate: audioPrepProfile?.inputSampleRate ?? null,
      audioOutputSampleRate: audioPrepProfile?.outputSampleRate ?? sampleRate,
      processorMs,
      callMs,
      inferenceWallMs,
      endToEndMs,
      inferenceRtf,
      inferenceRtfx: inferenceRtf && Number.isFinite(inferenceRtf) && inferenceRtf > 0 ? 1 / inferenceRtf : null,
      endToEndRtf,
      endToEndRtfx: endToEndRtf && Number.isFinite(endToEndRtf) && endToEndRtf > 0 ? 1 / endToEndRtf : null,
      inputEstimate,
      wallMs: inferenceWallMs,
    };
  }

  async function transcribeInput(input, name) {
    if (!tRef.current) return;
    setRunning(true); setError(''); setStatus(`Transcribing ${name}...`); setLastRunMetrics(null);
    try {
      const measured = await inferInput(input, { forceMetrics: metrics });
      const { out, useDirect } = measured;
      setResult(out);
      setLastRunMetrics({
        mode: useDirect ? 'direct' : 'pipeline',
        durationSec: measured.durationSec,
        audioPrepMs: measured.audioPrepMs,
        audioPrepBackend: measured.audioPrepBackend,
        audioPrepStrategy: measured.audioPrepStrategy,
        audioDecodeMs: measured.audioDecodeMs,
        audioDownmixMs: measured.audioDownmixMs,
        audioResampleMs: measured.audioResampleMs,
        audioResampler: measured.audioResampler,
        audioResamplerQuality: measured.audioResamplerQuality,
        audioInputSampleRate: measured.audioInputSampleRate,
        audioOutputSampleRate: measured.audioOutputSampleRate,
        processorMs: measured.processorMs,
        callMs: measured.callMs,
        inferenceWallMs: measured.inferenceWallMs,
        endToEndMs: measured.endToEndMs,
        inferenceRtf: measured.inferenceRtf,
        inferenceRtfx: measured.inferenceRtfx,
        endToEndRtf: measured.endToEndRtf,
        endToEndRtfx: measured.endToEndRtfx,
      });
      setHistory((h) => [{ id: `${Date.now()}`, name, mode: useDirect ? 'direct' : 'pipeline', text: textOf(out) }, ...h].slice(0, 25));
      setStatus('Done');
    } catch (e) {
      setLastRunMetrics(null);
      setError(e?.message || String(e));
      setStatus('Failed');
    } finally { setRunning(false); }
  }

  function handleFileUpload(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setSelectedFileName(f.name);
    transcribeInput(f, f.name);
  }

  async function indexBenchmarkEntries(entries, sourceLabel = 'selection') {
    if (!Array.isArray(entries) || entries.length === 0) {
      setBenchmarkFiles([]);
      setBenchmarkCatalog([]);
      setBenchmarkStatus('Load a model, then select audio files to benchmark.');
      return;
    }

    setBenchmarkFiles(entries);
    setBenchmarkCatalog([]);
    setBenchmarkIndexing(true);
    setBenchmarkStatus(`Loading benchmark catalog for ${entries.length} audio files from ${sourceLabel}...`);

    const indexed = [];
    const updateEvery = Math.max(10, Math.floor(entries.length / 40));
    const cacheKey = buildCatalogCacheKey(entries);

    try {
      let cachedCatalog = null;
      try {
        cachedCatalog = await readStoredCatalog(cacheKey);
      } catch {
        cachedCatalog = null;
      }

      const cachedDurationBySignature = new Map(
        Array.isArray(cachedCatalog?.entries)
          ? cachedCatalog.entries.map((entry) => [entry.signature, entry.durationSec])
          : []
      );
      const cacheHitCount = entries.reduce(
        (count, entry) => count + (Number.isFinite(cachedDurationBySignature.get(buildCatalogEntrySignature(entry))) ? 1 : 0),
        0
      );

      if (cacheHitCount > 0) {
        setBenchmarkStatus(
          cacheHitCount === entries.length
            ? `Loaded cached benchmark index for ${entries.length} audio files from ${sourceLabel}.`
            : `Loaded ${cacheHitCount} cached durations. Refreshing the remaining ${entries.length - cacheHitCount} files...`
        );
      }

      for (let i = 0; i < entries.length; i += 1) {
        const entry = entries[i];
        const signature = buildCatalogEntrySignature(entry);
        const cachedDuration = cachedDurationBySignature.get(signature);
        const durationSec = Number.isFinite(cachedDuration)
          ? cachedDuration
          : await getAudioDurationSeconds(entry.file);
        indexed.push({
          file: entry.file,
          name: entry.name,
          path: entry.path,
          durationSec,
          sizeBytes: entry.sizeBytes,
          lastModified: entry.lastModified ?? entry.file?.lastModified ?? 0,
        });

        if ((!Number.isFinite(cachedDuration)) && ((i + 1) % updateEvery === 0 || i === entries.length - 1)) {
          setBenchmarkStatus(`Indexed ${i + 1} / ${entries.length} audio files from ${sourceLabel}...`);
        }
      }

      await writeStoredCatalog(cacheKey, {
        savedAt: Date.now(),
        entryCount: indexed.length,
        entries: indexed.map((entry) => ({
          signature: buildCatalogEntrySignature(entry),
          path: entry.path,
          durationSec: entry.durationSec,
        })),
      });

      setBenchmarkCatalog(indexed);
      setBenchmarkStatus(
        `${cacheHitCount === entries.length ? 'Restored' : 'Indexed'} ${indexed.length} audio files from ${sourceLabel}. ${benchmarkTargetMode === 'random'
          ? 'Pick random samples to build a plan.'
          : benchmarkTargets.length > 0
            ? 'Sample plan is ready.'
            : 'Add target durations to build a sample plan.'
        }`
      );
    } catch (indexError) {
      setBenchmarkStatus(`Indexing failed: ${indexError?.message || String(indexError)}`);
    } finally {
      setBenchmarkIndexing(false);
    }
  }

  function handleBenchmarkFileUpload(e) {
    const files = Array.from(e.target.files || []).filter((file) => file.type.startsWith('audio/') || isAudioFileName(file.name));
    e.target.value = '';
    const entries = files.map((file) => ({
      file,
      name: file.name,
      path: file.webkitRelativePath || file.name,
      sizeBytes: file.size,
      lastModified: file.lastModified,
    }));
    indexBenchmarkEntries(entries, 'selected files');
  }

  async function restoreBenchmarkFolderFromHandle(directoryHandle, { requestAccess = false, sourceLabel = null } = {}) {
    if (!directoryHandle) return false;

    try {
      const permission = requestAccess
        ? await directoryHandle.requestPermission({ mode: 'read' })
        : await directoryHandle.queryPermission({ mode: 'read' });

      if (permission !== 'granted') {
        setBenchmarkStoredFolderAvailable(true);
        setBenchmarkStoredFolderName(directoryHandle.name || 'saved folder');
        if (requestAccess) {
          setBenchmarkStatus('Folder access was not granted.');
        }
        return false;
      }

      setBenchmarkStoredFolderAvailable(true);
      setBenchmarkStoredFolderName(directoryHandle.name || 'saved folder');
      const entries = await collectAudioEntriesFromDirectoryHandle(directoryHandle);
      await indexBenchmarkEntries(entries, sourceLabel || directoryHandle.name || 'saved folder');
      return true;
    } catch (error) {
      setBenchmarkStatus(`Folder restore failed: ${error?.message || String(error)}`);
      return false;
    }
  }

  async function pickBenchmarkFolderWithHandle() {
    if (!benchmarkFolderHandleSupported) {
      benchmarkFolderInputRef.current?.click();
      return;
    }

    try {
      const directoryHandle = await window.showDirectoryPicker({ mode: 'read' });
      await writeStoredDirectoryHandle(directoryHandle);
      setBenchmarkStoredFolderAvailable(true);
      setBenchmarkStoredFolderName(directoryHandle.name || 'saved folder');
      await restoreBenchmarkFolderFromHandle(directoryHandle, {
        requestAccess: true,
        sourceLabel: directoryHandle.name || 'picked folder',
      });
    } catch (error) {
      if (error?.name === 'AbortError') return;
      setBenchmarkStatus(`Folder pick failed: ${error?.message || String(error)}`);
    }
  }

  async function reconnectStoredBenchmarkFolder() {
    setBenchmarkRestoringFolder(true);
    try {
      const handle = await readStoredDirectoryHandle();
      if (!handle) {
        setBenchmarkStoredFolderAvailable(false);
        setBenchmarkStoredFolderName('');
        setBenchmarkStatus('No saved folder handle was found.');
        return;
      }
      const restored = await restoreBenchmarkFolderFromHandle(handle, {
        requestAccess: true,
        sourceLabel: handle.name || 'saved folder',
      });
      if (!restored) {
        setBenchmarkStatus('Saved folder found, but access still requires permission.');
      }
    } catch (error) {
      setBenchmarkStatus(`Reconnect failed: ${error?.message || String(error)}`);
    } finally {
      setBenchmarkRestoringFolder(false);
    }
  }

  async function forgetStoredBenchmarkFolder() {
    try {
      await clearStoredDirectoryHandle();
      setBenchmarkStoredFolderAvailable(false);
      setBenchmarkStoredFolderName('');
      setBenchmarkStatus('Saved benchmark folder was cleared.');
    } catch (error) {
      setBenchmarkStatus(`Failed to clear saved folder: ${error?.message || String(error)}`);
    }
  }

  async function runBenchmark({ warmupOverride } = {}) {
    if (!tRef.current || !modelLoaded) {
      setBenchmarkStatus('Load a model before running benchmarks.');
      return;
    }
    if (benchmarkResolvedPlan.length === 0) {
      setBenchmarkStatus(
        benchmarkTargetMode === 'random'
          ? 'Pick random samples first.'
          : 'Select a folder or files and define target durations first.'
      );
      return;
    }

    const useWarmup = warmupOverride ?? benchmarkWarmup;
    const baseRepeats = Math.max(1, benchmarkMeasurementRepeats);
    const planLabel = benchmarkTargetMode === 'random' ? 'random samples' : 'sampled targets';
    const planToRun = benchmarkRandomizeExecution ? shuffleArray(benchmarkResolvedPlan) : benchmarkResolvedPlan;

    benchmarkCancelRef.current = false;
    setBenchmarkRunning(true);
    setError('');
    setBenchmarkRuns([]);
    setBenchmarkCurrent(null);
    setBenchmarkStatus(
      `Benchmarking 0 / ${planToRun.length} ${planLabel}${useWarmup ? ' with warmup' : ' without warmup'}${benchmarkRandomizeExecution ? ' in random order' : ' in planned order'}...`
    );

    const collectedRuns = [];
    try {
      for (let i = 0; i < planToRun.length; i += 1) {
        if (benchmarkCancelRef.current) break;

        const sample = planToRun[i];
        const file = sample.file;
        const boostedRepeats = benchmarkLongRepeatBoostEnabled && sample.durationSec >= benchmarkLongRepeatThresholdSec
          ? Math.max(baseRepeats, baseRepeats * Math.max(1, benchmarkLongRepeatMultiplier))
          : baseRepeats;
        setBenchmarkCurrent({ index: i + 1, total: planToRun.length, name: file.name });
        setBenchmarkStatus(`Benchmarking ${i + 1} / ${planToRun.length}: ${file.name}`);

        try {
          if (useWarmup) {
            await inferInput(file, { forceMetrics: true, expectedDurationSec: sample.durationSec });
            if (benchmarkCancelRef.current) break;
          }

          const measurements = [];
          let lastOut = null;
          let lastUseDirect = false;
          let lastDurationSec = sample.durationSec;

          for (let repeatIndex = 0; repeatIndex < boostedRepeats; repeatIndex += 1) {
            if (benchmarkCancelRef.current) break;
            if (boostedRepeats > 1) {
              setBenchmarkStatus(
                `Benchmarking ${i + 1} / ${planToRun.length}: ${file.name} (repeat ${repeatIndex + 1} / ${boostedRepeats})`
              );
            }

            const {
              out,
              useDirect,
              durationSec,
              audioPrepMs,
              audioPrepBackend,
              audioPrepStrategy,
              audioDecodeMs,
              audioDownmixMs,
              audioResampleMs,
              audioResampler,
              audioResamplerQuality,
              audioInputSampleRate,
              audioOutputSampleRate,
              processorMs,
              callMs,
              inferenceWallMs,
              endToEndMs,
              inferenceRtfx,
              endToEndRtfx,
              inputEstimate,
              wallMs,
            } = await inferInput(file, {
              forceMetrics: true,
              expectedDurationSec: sample.durationSec,
            });
            const metricsOut = out?.metrics ?? null;
            const modelRtf = metricsOut?.rtf ?? (
              Number.isFinite(metricsOut?.totalMs) && durationSec > 0
                ? metricsOut.totalMs / 1000 / durationSec
                : null
            );
            const wallRtf = durationSec > 0 ? wallMs / 1000 / durationSec : null;

            measurements.push({
              durationSec,
              audioDecodeMs,
              audioPrepMs,
              audioPrepBackend,
              audioPrepStrategy,
              audioDownmixMs,
              audioResampleMs,
              audioResampler,
              audioResamplerQuality,
              audioInputSampleRate,
              audioOutputSampleRate,
              processorMs,
              callMs,
              wallMs,
              inferenceWallMs,
              endToEndMs,
              wallRtfx: wallRtf && Number.isFinite(wallRtf) && wallRtf > 0 ? 1 / wallRtf : null,
              inferenceRtfx,
              endToEndRtfx,
              modelTotalMs: metricsOut?.totalMs ?? null,
              modelRtfx: metricsOut?.rtfX ?? (modelRtf && Number.isFinite(modelRtf) && modelRtf > 0 ? 1 / modelRtf : null),
              encodeMs: metricsOut?.encodeMs ?? null,
              decodeMs: metricsOut?.decodeMs ?? null,
              tokenizeMs: metricsOut?.tokenizeMs ?? null,
              inputEstimate,
            });
            lastOut = out;
            lastUseDirect = useDirect;
            lastDurationSec = durationSec;
          }

          if (measurements.length === 0) {
            break;
          }

          const run = {
            id: `${Date.now()}-${i}`,
            name: file.name,
            sourcePath: sample.path,
            sizeBytes: file.size,
            selectionMode: benchmarkTargetMode,
            randomIndex: sample.randomIndex ?? null,
            targetSec: sample.targetSec,
            sampledDurationSec: sample.durationSec,
            sampledDeltaSec: sample.deltaSec,
            repeatCount: measurements.length,
            baseRepeatCount: baseRepeats,
            boostedRepeatCount: boostedRepeats,
            repeatBoostApplied: boostedRepeats > baseRepeats,
            durationSec: lastDurationSec,
            mode: lastUseDirect ? 'direct' : 'pipeline',
            textLen: textOf(lastOut).length,
            wordCount: Array.isArray(lastOut?.words) ? lastOut.words.length : null,
            tokenCount: Array.isArray(lastOut?.tokens) ? lastOut.tokens.length : null,
            audioDecodeMs: averageOf(measurements.map((entry) => entry.audioDecodeMs)),
            audioPrepMs: averageOf(measurements.map((entry) => entry.audioPrepMs)),
            audioPrepBackend: measurements[0]?.audioPrepBackend ?? null,
            audioPrepStrategy: measurements[0]?.audioPrepStrategy ?? null,
            audioDownmixMs: averageOf(measurements.map((entry) => entry.audioDownmixMs)),
            audioResampleMs: averageOf(measurements.map((entry) => entry.audioResampleMs)),
            audioResampler: measurements[0]?.audioResampler ?? null,
            audioResamplerQuality: measurements[0]?.audioResamplerQuality ?? null,
            audioInputSampleRate: measurements[0]?.audioInputSampleRate ?? null,
            audioOutputSampleRate: measurements[0]?.audioOutputSampleRate ?? null,
            processorMs: averageOf(measurements.map((entry) => entry.processorMs)),
            callMs: averageOf(measurements.map((entry) => entry.callMs)),
            wallMs: averageOf(measurements.map((entry) => entry.wallMs)),
            inferenceWallMs: averageOf(measurements.map((entry) => entry.inferenceWallMs)),
            endToEndMs: averageOf(measurements.map((entry) => entry.endToEndMs)),
            wallRtfx: averageOf(measurements.map((entry) => entry.wallRtfx)),
            wallRtfxStddev: stddevOf(measurements.map((entry) => entry.wallRtfx)),
            inferenceRtfx: averageOf(measurements.map((entry) => entry.inferenceRtfx)),
            endToEndRtfx: averageOf(measurements.map((entry) => entry.endToEndRtfx)),
            modelTotalMs: averageOf(measurements.map((entry) => entry.modelTotalMs)),
            modelRtfx: averageOf(measurements.map((entry) => entry.modelRtfx)),
            modelRtfxStddev: stddevOf(measurements.map((entry) => entry.modelRtfx)),
            encodeMs: averageOf(measurements.map((entry) => entry.encodeMs)),
            decodeMs: averageOf(measurements.map((entry) => entry.decodeMs)),
            tokenizeMs: averageOf(measurements.map((entry) => entry.tokenizeMs)),
            featureShape: measurements[0]?.inputEstimate?.featureShape ?? null,
            featureFrames: measurements[0]?.inputEstimate?.featureFrames ?? null,
            featureBins: measurements[0]?.inputEstimate?.featureBins ?? null,
            featureMiB: measurements[0]?.inputEstimate?.featureMiB ?? null,
            attentionMaskMiB: measurements[0]?.inputEstimate?.attentionMaskMiB ?? null,
            encoderPayloadMiB: measurements[0]?.inputEstimate?.encoderPayloadMiB ?? null,
            audioPcmMiB: measurements[0]?.inputEstimate?.audioPcmMiB ?? null,
            measurements,
            error: null,
          };
          collectedRuns.push(run);
          setBenchmarkRuns([...collectedRuns]);
        } catch (runError) {
          const failedRun = {
            id: `${Date.now()}-${i}`,
            name: file.name,
            sourcePath: sample.path,
            sizeBytes: file.size,
            selectionMode: benchmarkTargetMode,
            randomIndex: sample.randomIndex ?? null,
            targetSec: sample.targetSec,
            sampledDurationSec: sample.durationSec,
            sampledDeltaSec: sample.deltaSec,
            repeatCount: repeats,
            durationSec: null,
            mode: direct && mType === 'nemo-conformer-tdt' ? 'direct' : 'pipeline',
            textLen: null,
            wordCount: null,
            tokenCount: null,
            audioDecodeMs: null,
            audioPrepMs: null,
            processorMs: null,
            callMs: null,
            wallMs: null,
            inferenceWallMs: null,
            endToEndMs: null,
            wallRtfx: null,
            inferenceRtfx: null,
            endToEndRtfx: null,
            modelTotalMs: null,
            modelRtfx: null,
            encodeMs: null,
            decodeMs: null,
            tokenizeMs: null,
            featureShape: null,
            featureFrames: null,
            featureBins: null,
            featureMiB: null,
            attentionMaskMiB: null,
            encoderPayloadMiB: null,
            audioPcmMiB: null,
            measurements: [],
            error: runError?.message || String(runError),
          };
          collectedRuns.push(failedRun);
          setBenchmarkRuns([...collectedRuns]);
          if (benchmarkStopOnError) {
            throw runError;
          }
        }
      }

      if (benchmarkCancelRef.current) {
        setBenchmarkStatus(`Benchmark canceled after ${collectedRuns.length} files.`);
      } else {
        const failures = collectedRuns.filter((run) => run.error).length;
        setBenchmarkStatus(
          `${benchmarkTargetMode === 'random' ? 'Random benchmark' : 'Target benchmark'} complete. ${collectedRuns.length - failures} succeeded, ${failures} failed.`
        );
      }
    } catch (e) {
      setBenchmarkStatus(`Benchmark stopped: ${e?.message || String(e)}`);
    } finally {
      benchmarkCancelRef.current = false;
      setBenchmarkCurrent(null);
      setBenchmarkRunning(false);
    }
  }

  function cancelBenchmark() {
    if (!benchmarkRunning) return;
    benchmarkCancelRef.current = true;
    setBenchmarkStatus('Cancel requested. Finishing current file...');
  }

  function applySuggestedTargets() {
    const suggested = suggestDurationTargets(benchmarkFilteredCatalog);
    if (suggested.length === 0) {
      setBenchmarkStatus('Index a folder and choose a range with matching files first.');
      return;
    }
    setBenchmarkTargetMode('manual');
    setBenchmarkTargetsText(suggested.join(','));
    setBenchmarkStatus(`Generated ${suggested.length} auto targets from dataset coverage.`);
  }

  function pickRandomBenchmarkSamples(reseed = false) {
    if (benchmarkFilteredCatalog.length === 0) {
      setBenchmarkStatus('Index a folder and choose a range with matching files first.');
      return;
    }
    const nextSeed = reseed ? Date.now() : benchmarkRandomSeed;
    const plan = buildRandomSamplePlan(benchmarkFilteredCatalog, benchmarkRandomCount, nextSeed, benchmarkBucketSize);
    if (plan.length === 0) {
      setBenchmarkStatus('No valid audio durations were indexed for random sampling.');
      return;
    }
    setBenchmarkTargetMode('random');
    setBenchmarkRandomSeed(nextSeed);
    setBenchmarkRandomPlan(plan);
    setBenchmarkStatus(
      `Picked ${plan.length} random samples${reseed ? ' with a fresh shuffle' : ''}.`
    );
  }

  function refineTargetsFromBenchmarkRuns() {
    const refined = refineTargetsFromRuns(benchmarkRuns);
    if (refined.length === 0) {
      setBenchmarkStatus('Run a benchmark first, then refine from the measured gaps.');
      return;
    }
    const merged = mergeTargetDurations(benchmarkTargets, refined);
    setBenchmarkTargetMode('manual');
    setBenchmarkTargetsText(merged.join(','));
    setBenchmarkStatus(`Added ${refined.length} refinement targets based on the previous run.`);
  }

  function copyToClipboard(text) {
    navigator.clipboard?.writeText(text).catch(() => {
      setError('Copy failed. Your browser may block clipboard access on this page.');
    });
  }

  function exportSamplePlanJson() {
    if (benchmarkResolvedPlan.length === 0) {
      setBenchmarkStatus('No sampled plan is available to export.');
      return;
    }
    const payload = {
      exportedAt: new Date().toISOString(),
      selectionMode: benchmarkTargetMode,
      bucketSizeSec: benchmarkBucketSize,
      minDurationSec: effectiveBenchmarkMinDurationSec,
      maxDurationSec: effectiveBenchmarkMaxDurationSec,
      randomSeed: benchmarkTargetMode === 'random' ? benchmarkRandomSeed : null,
      count: benchmarkResolvedPlan.length,
      samples: benchmarkResolvedPlan.map((sample) => ({
        name: sample.name,
        path: sample.path,
        durationSec: sample.durationSec,
        sizeBytes: sample.sizeBytes,
        targetSec: sample.targetSec ?? null,
        deltaSec: sample.deltaSec ?? null,
        randomIndex: sample.randomIndex ?? null,
        duplicateOfPath: sample.duplicateOfPath ?? null,
        duplicateOrdinal: sample.duplicateOrdinal ?? null,
      })),
    };
    downloadTextFile(`sample-plan-${Date.now()}.json`, JSON.stringify(payload, null, 2));
    setBenchmarkStatus(`Exported ${benchmarkResolvedPlan.length} sampled reports as JSON.`);
  }

  function exportBenchmarkSummaryJson() {
    if (benchmarkSummary.length === 0) {
      setBenchmarkStatus('No duration bucket summary is available to export.');
      return;
    }
    const payload = {
      exportedAt: new Date().toISOString(),
      bucketSizeSec: benchmarkBucketSize,
      minDurationSec: effectiveBenchmarkMinDurationSec,
      maxDurationSec: effectiveBenchmarkMaxDurationSec,
      count: benchmarkSummary.length,
      buckets: benchmarkSummary,
    };
    downloadTextFile(`benchmark-summary-${Date.now()}.json`, JSON.stringify(payload, null, 2));
    setBenchmarkStatus(`Exported ${benchmarkSummary.length} duration buckets as JSON.`);
  }

  function exportBenchmarkSummaryCsv() {
    if (benchmarkSummary.length === 0) {
      setBenchmarkStatus('No duration bucket summary is available to export.');
      return;
    }
    downloadTextFile(`benchmark-summary-${Date.now()}.csv`, benchmarkSummaryToCsv(benchmarkSummary), 'text/csv;charset=utf-8');
    setBenchmarkStatus(`Exported ${benchmarkSummary.length} duration buckets as CSV.`);
  }

  const busy = loading || running || benchmarkRunning || benchmarkIndexing;
  const canRun = !!tRef.current && modelLoaded && !busy;
  const isModelReady = modelLoaded;
  const configDisabled = busy || modelLoaded;

  return (
    <div className="bg-background-light dark:bg-background-dark text-gray-800 dark:text-gray-200 font-sans min-h-screen p-4 md:p-6 transition-colors duration-300" style={{ fontSize: '90%' }}>
      <div className={`${activeView === 'benchmark' ? 'max-w-[1800px]' : 'max-w-[1860px]'} mx-auto`}>
        {/* Tabs at the very top */}
        <div className="mb-3 flex items-center justify-between">
          <div className="inline-flex rounded-xl border border-border-light dark:border-border-dark bg-card-light dark:bg-card-dark p-1">
            <button
              onClick={() => setActiveView('transcribe')}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${activeView === 'transcribe'
                ? 'bg-primary text-white dark:bg-accent-muted'
                : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
            >
              Transcribe
            </button>
            <button
              onClick={() => setActiveView('benchmark')}
              className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${activeView === 'benchmark'
                ? 'bg-primary text-white dark:bg-accent-muted'
                : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
            >
              Benchmark Lab
            </button>
          </div>
          <button
            className="flex items-center justify-center p-2 rounded-full bg-gray-200 dark:bg-card-dark border border-border-light dark:border-border-dark hover:bg-primary-muted/20 dark:hover:bg-accent-muted/30 transition-colors"
            onClick={() => setDarkMode(!darkMode)}
          >
            <span className="material-icons-outlined text-primary-muted dark:text-accent-muted">
              brightness_4
            </span>
          </button>
        </div>

        {/* Header */}
        <header className="mb-4">
          <h1 className="text-2xl font-bold tracking-tight text-gray-900 dark:text-white">
            Nemo TDT Demo
          </h1>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
            transformers.js {VERSION} ({SOURCE})
          </div>
        </header>

        {usingNpmTransformers && (
          <div className="mb-4 rounded-xl border border-amber-300/70 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/30 px-4 py-3 text-sm text-amber-900 dark:text-amber-100">
            <div className="font-semibold mb-1">Running npm package build</div>
            <p className="leading-relaxed">
              This app is currently using <code className="font-mono">@huggingface/transformers</code> from
              <code className="font-mono"> node_modules</code>, not your local
              <code className="font-mono"> ../transformers.js</code> checkout. If you are validating local NeMo pipeline
              fixes, run <code className="font-mono">npm run dev:local</code> or
              <code className="font-mono"> npm run build:local</code> after rebuilding
              <code className="font-mono"> ../transformers.js/packages/transformers</code>.
            </p>
          </div>
        )}

        <div className={`grid grid-cols-1 gap-4 items-start ${activeView === 'benchmark' ? 'xl:grid-cols-12' : 'xl:grid-cols-[minmax(280px,0.72fr)_minmax(500px,1.18fr)_minmax(420px,1fr)]'}`}>
          {/* Left Column - Model + Options */}
          <div className={`${activeView === 'benchmark' ? 'xl:col-span-12 grid gap-3 xl:grid-cols-2 2xl:grid-cols-4' : 'flex flex-col'} gap-3`}>
            {/* Model Configuration Card */}
            <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-4">
              <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-3">
                Model Configuration
              </h2>
              <div className="space-y-4">
                <SelectField
                  label={(
                    <span className="inline-flex items-center gap-2">
                      <span>Load Mode</span>
                      <InfoHint text="Changes how the transcriber is constructed. It does not decide whether the demo uses pipeline inference or direct model.transcribe()." />
                    </span>
                  )}
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  disabled={configDisabled}
                >
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

                <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
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
                  <div>
                    <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                      Threads
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={threadingStatus.sab ? maxWasmCores : 1}
                      step={1}
                      value={wasmThreads}
                      onChange={(e) => setWasmThreads(clampThreadCount(e.target.value, threadingStatus.sab ? maxWasmCores : 1))}
                      disabled={configDisabled || !threadingStatus.sab}
                      className="w-full bg-gray-50 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-muted focus:border-primary-muted dark:focus:ring-accent-muted dark:focus:border-accent-muted dark:text-white"
                    />
                    <div className="mt-1 text-[11px] text-gray-500 dark:text-gray-400">
                      1-{threadingStatus.sab ? maxWasmCores : 1}
                    </div>
                  </div>
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
                    onClick={() => { tRef.current = null; setResult(null); setLastRunMetrics(null); setMType('not-loaded'); setModelLoaded(false); setStatus('Idle'); setIsCached(null); }}
                    disabled={busy}
                    className="bg-primary-muted hover:bg-primary dark:bg-border-dark dark:hover:bg-accent-muted text-white font-medium py-2.5 px-4 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <span className="material-icons-outlined text-sm">power_settings_new</span>
                    Dispose
                  </button>
                </div>

              </div>
            </div>

            {/* Options Card */}
            <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-4">
              <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-3">
                Transcription Options
              </h2>
              <div className="flex flex-col gap-2">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-700 dark:text-gray-300">Inference mode</span>
                    <InfoHint
                      text={direct
                        ? 'Direct mode calls model.transcribe() and exposes rich NeMo fields like words, tokens, confidence, metrics, and debug traces.'
                        : 'Pipeline mode uses the HF-compatible ASR pipeline and returns text with optional timestamp chunks.'}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-2 rounded-xl border border-border-light dark:border-border-dark bg-gray-50 p-1 dark:bg-gray-900/40">
                    <button
                      type="button"
                      onClick={() => setDirect(false)}
                      aria-pressed={!direct}
                      className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                        !direct
                          ? 'bg-primary text-white shadow-sm dark:bg-accent dark:text-gray-950'
                          : 'text-gray-600 hover:bg-white hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800 dark:hover:text-white'
                      }`}
                    >
                      Pipeline
                    </button>
                    <button
                      type="button"
                      onClick={() => setDirect(true)}
                      aria-pressed={direct}
                      className={`rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                        direct
                          ? 'bg-primary text-white shadow-sm dark:bg-accent dark:text-gray-950'
                          : 'text-gray-600 hover:bg-white hover:text-gray-900 dark:text-gray-300 dark:hover:bg-gray-800 dark:hover:text-white'
                      }`}
                    >
                      Direct model.transcribe()
                    </button>
                  </div>
                </div>

                {direct ? (
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-700 dark:text-gray-300">Return timestamps</span>
                    <Toggle id="rt" checked={rt} onChange={(e) => setRt(e.target.checked)} />
                  </div>
                ) : (
                  <SelectField
                    label="Pipeline timestamps"
                    value={pipelineTimestampMode}
                    onChange={(e) => setPipelineTimestampMode(e.target.value)}
                  >
                    <option value="none">off ({'{ text }'})</option>
                    <option value="segments">sentences ({'{ text, chunks }'})</option>
                    <option value="words">words ({'{ text, chunks }'})</option>
                  </SelectField>
                )}
                {!direct && (
                  <>
                    <div className="rounded-xl border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800/70 px-4 py-3">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <div className="inline-flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300">
                            <span>Window size override</span>
                            <InfoHint text="Override NeMo long-audio chunk_length_s to compare window size versus measured RTFx. The sentence-based path ignores stride_length_s." />
                          </div>
                        </div>
                        <Toggle
                          id="pipelineWindowOverrideEnabled"
                          checked={pipelineWindowOverrideEnabled}
                          onChange={(e) => setPipelineWindowOverrideEnabled(e.target.checked)}
                        />
                      </div>
                      {pipelineWindowOverrideEnabled && (
                        <div className="mt-3">
                          <SliderField
                            label="chunk_length_s"
                            value={pipelineWindowOverrideSec}
                            min={NEMO_PIPELINE_WINDOW_MIN_S}
                            max={NEMO_PIPELINE_WINDOW_MAX_S}
                            step={5}
                            onChange={(next) => setPipelineWindowOverrideSec(clampPipelineWindowSec(next))}
                            formatValue={(next) => `${next.toFixed(0)}s`}
                          />
                        </div>
                      )}
                    </div>
                  </>
                )}

                {direct && (
                  <div className="border-t border-border-light dark:border-border-dark pt-2 mt-1">
                    <p className="text-[0.65rem] font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-2">Detail Flags</p>
                    <div className="grid grid-cols-2 gap-y-1.5 gap-x-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-700 dark:text-gray-300">Words</span>
                        <Toggle id="words" checked={returnWords} onChange={(e) => setReturnWords(e.target.checked)} disabled={!rt} />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-700 dark:text-gray-300">Tokens</span>
                        <Toggle id="tokens" checked={returnTokens} onChange={(e) => setReturnTokens(e.target.checked)} disabled={!rt} />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-700 dark:text-gray-300">Metrics</span>
                        <Toggle id="metrics" checked={metrics} onChange={(e) => setMetrics(e.target.checked)} />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-700 dark:text-gray-300">Frame conf.</span>
                        <Toggle id="frameConf" checked={returnFrameConf} onChange={(e) => setReturnFrameConf(e.target.checked)} />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-700 dark:text-gray-300">Frame idx</span>
                        <Toggle id="frameIdx" checked={frameIdx} onChange={(e) => setFrameIdx(e.target.checked)} />
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-700 dark:text-gray-300">Log probs</span>
                        <Toggle id="logProbs" checked={logProbs} onChange={(e) => setLogProbs(e.target.checked)} />
                      </div>
                      <div className="flex items-center justify-between col-span-2">
                        <span className="text-sm text-gray-700 dark:text-gray-300">TDT steps</span>
                        <Toggle id="tdtSteps" checked={tdtSteps} onChange={(e) => setTdtSteps(e.target.checked)} />
                      </div>
                    </div>
                  </div>
                )}

                <div className="border-t border-border-light dark:border-border-dark pt-2 mt-1">
                  <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                    Time Offset
                  </label>
                  <input
                    value={offset}
                    onChange={(e) => setOffset(e.target.value)}
                    disabled={!direct}
                    className="w-full bg-gray-50 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-primary-muted focus:border-primary-muted dark:focus:ring-accent-muted dark:focus:border-accent-muted dark:text-white disabled:opacity-50"
                  />
                </div>

                <div className="border-t border-border-light dark:border-border-dark pt-2 mt-1">
                  <SelectField
                    label={(
                      <span className="inline-flex items-center gap-2">
                        <span>Audio prep</span>
                        <InfoHint
                          text={
                            audioPrepBackend === 'custom-js'
                              ? 'Default path. WAV files bypass browser decode, other formats use browser codecs plus one explicit final resample to 16 kHz, and decode/downmix/resample time is profiled separately.'
                              : audioPrepBackend === 'transformers'
                                ? 'Uses transformers.js read_audio(), which stays on the browser decode/resample path.'
                                : 'Legacy demo path. Uses AudioContext at target sample rate plus simple browser downmix.'
                          }
                        />
                      </span>
                    )}
                    value={audioPrepBackend}
                    onChange={(e) => setAudioPrepBackend(e.target.value)}
                  >
                    <option value="custom-js">custom JS audio prep</option>
                    <option value="transformers">transformers.js</option>
                    <option value="demo">demo AudioContext</option>
                  </SelectField>
                  {audioPrepBackend === 'custom-js' && (
                    <div className="mt-2">
                      <SelectField
                        label={(
                          <span className="inline-flex items-center gap-2">
                            <span>Custom resampler</span>
                            <InfoHint text="Linear parity matches the current Python and Node parity scripts. The sinc modes use libsamplerate-js. Linear avoids that extra WASM SRC overhead and is now the default custom mode." />
                          </span>
                        )}
                        value={audioPrepQuality}
                        onChange={(e) => setAudioPrepQuality(e.target.value)}
                      >
                        {CUSTOM_RESAMPLER_QUALITY_OPTIONS.map((option) => (
                          <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                      </SelectField>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Merged Status Bar */}
            <div className="col-span-full flex flex-wrap items-center gap-2 px-1 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-800/50 border border-border-light dark:border-border-dark text-xs">
              <span className={`inline-flex items-center gap-1 font-medium px-2 py-0.5 rounded-full ${threadingStatus.sab
                ? 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
                : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-300'
                }`}>
                <span className="material-icons-outlined text-xs">
                  {threadingStatus.sab ? 'check_circle' : 'warning'}
                </span>
                {threadingStatus.sab ? `${threadingStatus.threads} cores` : '1 thread'}
              </span>
              <span className="text-gray-400 dark:text-gray-500">·</span>
              <span className={`font-medium ${isModelReady ? 'text-gray-700 dark:text-gray-300' : 'text-gray-500 dark:text-gray-400'}`}>
                {status}
              </span>
              <span className="text-gray-400 dark:text-gray-500">·</span>
              <span className="text-primary-muted dark:text-accent-muted">
                {mType}
              </span>
              <span className="text-gray-400 dark:text-gray-500">·</span>
              <span className="text-primary-muted dark:text-accent-muted">
                wasm:{clampThreadCount(wasmThreads, threadingStatus.sab ? maxWasmCores : 1)}
              </span>
              {/* Download progress inline */}
              {downloadProgress && (
                <>
                  <span className="text-gray-400 dark:text-gray-500">·</span>
                  <span className="truncate max-w-[120px]" title={downloadProgress.file}>
                    {downloadProgress.file ? `↓ ${downloadProgress.file.split('/').pop()}` : 'Downloading...'}
                  </span>
                  <span className="font-mono tabular-nums">{downloadProgress.pct}%</span>
                  <div className="flex-1 min-w-[60px] max-w-[150px] bg-gray-200 dark:bg-gray-700 rounded-full h-1">
                    <div
                      className="bg-primary dark:bg-accent-muted h-1 rounded-full transition-all duration-150"
                      style={{ width: `${downloadProgress.pct}%` }}
                    />
                  </div>
                </>
              )}
              {isCached !== null && !modelLoaded && (
                <>
                  <span className="text-gray-400 dark:text-gray-500">·</span>
                  <span className={`inline-flex items-center gap-1 ${isCached ? 'text-green-700 dark:text-green-400' : 'text-yellow-700 dark:text-yellow-400'}`}>
                    <span className="material-icons-outlined text-xs">{isCached ? 'offline_bolt' : 'cloud_download'}</span>
                    {isCached ? 'cached' : 'not cached'}
                  </span>
                </>
              )}
            </div>
            {!threadingStatus.sab && isInIframe && (
              <div className="col-span-full px-1 text-xs text-gray-600 dark:text-gray-400">
                Open this Space directly (outside the Hugging Face wrapper) to enable multi-threading.
              </div>
            )}
            {!!error && (
              <div className="col-span-full mx-1 p-2 bg-gray-100 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg text-sm text-gray-800 dark:text-gray-200 break-words">
                {error}
              </div>
            )}
          </div>

          {/* Workspace Columns */}
          <div className={`${activeView === 'benchmark' ? 'xl:col-span-12' : 'space-y-6'}`}>
            {activeView === 'transcribe' ? (
              <>
                <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
                  <div className="mb-5 flex items-center justify-between gap-4">
                    <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted">
                      Test & Transcribe
                    </h2>
                    <div className="flex gap-3">
                      <button
                        onClick={() => transcribeInput(SAMPLE, 'Harvard-L2-1.ogg')}
                        disabled={!canRun}
                        className="bg-primary hover:bg-primary-hover dark:bg-primary dark:hover:bg-primary-muted text-white font-medium py-2 px-4 rounded-lg whitespace-nowrap transition-colors text-sm h-[38px] disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      >
                        <span className="material-icons-outlined text-sm">play_arrow</span>
                        Sample
                      </button>
                    </div>
                  </div>

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

                  <div className="mt-6 pt-5 border-t border-border-light dark:border-border-dark">
                    <PerformanceMetrics stats={stats} />
                  </div>
                </div>

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
              </>
            ) : (
              <>
                <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
                  <div className="flex items-start justify-between gap-4 mb-5">
                    <div>
                      <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-2">
                        Browser Benchmark Lab
                      </h2>
                      <p className="text-sm text-gray-600 dark:text-gray-400 max-w-3xl">
                        Runs the currently loaded browser model sequentially over many local audio files using ORT Web in this tab, then groups RTFx by duration bucket.
                      </p>
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-3 py-2 rounded-lg border border-border-light dark:border-border-dark">
                      {direct && mType === 'nemo-conformer-tdt'
                        ? 'Direct Nemo metrics enabled'
                        : 'Fallback: wall-clock benchmarking'}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6 gap-4 mb-5">
                    <div>
                      <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                        Sampling Mode
                      </label>
                      <select
                        value={benchmarkTargetMode}
                        onChange={(e) => setBenchmarkTargetMode(e.target.value)}
                        disabled={benchmarkRunning || benchmarkIndexing}
                        className="w-full bg-gray-50 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-muted focus:border-primary-muted dark:focus:ring-accent-muted dark:focus:border-accent-muted dark:text-white appearance-none"
                      >
                        <option value="auto">auto (recommended)</option>
                        <option value="manual">manual</option>
                        <option value="random">random</option>
                      </select>
                    </div>
                    <SliderField
                      label="Random Samples"
                      value={benchmarkRandomCount}
                      min={10}
                      max={Math.max(25, Math.min(1000, benchmarkFilteredCatalog.length || benchmarkCatalog.length || 400))}
                      step={10}
                      onChange={(value) => setBenchmarkRandomCount(Math.max(1, value))}
                      disabled={benchmarkRunning || benchmarkIndexing}
                    />
                    <SliderField
                      label="Bucket Size"
                      value={benchmarkBucketSize}
                      min={5}
                      max={120}
                      step={5}
                      onChange={(value) => setBenchmarkBucketSize(Math.max(5, value))}
                      disabled={benchmarkRunning || benchmarkIndexing}
                      formatValue={(value) => formatClockDuration(value)}
                    />
                    <SliderField
                      label="Avg Overlap"
                      value={benchmarkOverlapSec}
                      min={0}
                      max={20}
                      step={1}
                      onChange={(value) => setBenchmarkOverlapSec(Math.max(0, value))}
                      disabled={benchmarkRunning || benchmarkIndexing}
                      formatValue={(value) => formatClockDuration(value)}
                    />
                    <DurationRangeField
                      minValue={effectiveBenchmarkMinDurationSec ?? 0}
                      maxValue={effectiveBenchmarkMaxDurationSec ?? benchmarkCatalogDurationLimit}
                      limit={benchmarkCatalogDurationLimit}
                      step={5}
                      onMinChange={(value) => setBenchmarkMinDurationSec(value)}
                      onMaxChange={(value) => setBenchmarkMaxDurationSec(value)}
                      disabled={benchmarkRunning || benchmarkIndexing}
                    />
                    <SliderField
                      label="Samples / Target"
                      value={benchmarkSamplesPerTarget}
                      min={1}
                      max={12}
                      step={1}
                      onChange={(value) => setBenchmarkSamplesPerTarget(Math.max(1, value))}
                      disabled={benchmarkRunning || benchmarkIndexing}
                    />
                    <div>
                      <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                        Measured Repeats
                      </label>
                      <input
                        type="number"
                        min={1}
                        step={1}
                        value={benchmarkMeasurementRepeats}
                        onChange={(e) => setBenchmarkMeasurementRepeats(Math.max(1, Number(e.target.value) || 1))}
                        disabled={benchmarkRunning || benchmarkIndexing}
                        className="w-full bg-gray-50 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-muted focus:border-primary-muted dark:focus:ring-accent-muted dark:focus:border-accent-muted dark:text-white"
                      />
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800 px-3 py-2 md:mt-6">
                      <span className="text-sm text-gray-700 dark:text-gray-300">Boost Long-Bin Repeats</span>
                      <Toggle
                        id="benchmarkLongRepeatBoostEnabled"
                        checked={benchmarkLongRepeatBoostEnabled}
                        onChange={(e) => setBenchmarkLongRepeatBoostEnabled(e.target.checked)}
                        disabled={benchmarkRunning || benchmarkIndexing}
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                        Long Repeat Threshold (sec)
                      </label>
                      <input
                        type="number"
                        min={5}
                        step={5}
                        value={benchmarkLongRepeatThresholdSec}
                        onChange={(e) => setBenchmarkLongRepeatThresholdSec(Math.max(5, Number(e.target.value) || 5))}
                        disabled={benchmarkRunning || benchmarkIndexing || !benchmarkLongRepeatBoostEnabled}
                        className="w-full bg-gray-50 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-muted focus:border-primary-muted dark:focus:ring-accent-muted dark:focus:border-accent-muted dark:text-white disabled:opacity-60"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium mb-1 text-gray-700 dark:text-gray-300">
                        Long Repeat Multiplier
                      </label>
                      <input
                        type="number"
                        min={1}
                        step={1}
                        value={benchmarkLongRepeatMultiplier}
                        onChange={(e) => setBenchmarkLongRepeatMultiplier(Math.max(1, Math.floor(Number(e.target.value) || 1)))}
                        disabled={benchmarkRunning || benchmarkIndexing || !benchmarkLongRepeatBoostEnabled}
                        className="w-full bg-gray-50 dark:bg-gray-800 border border-border-light dark:border-border-dark rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-primary-muted focus:border-primary-muted dark:focus:ring-accent-muted dark:focus:border-accent-muted dark:text-white disabled:opacity-60"
                      />
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800 px-3 py-2 md:mt-6">
                      <span className="text-sm text-gray-700 dark:text-gray-300">Warmup Each Sample</span>
                      <Toggle
                        id="benchmarkWarmup"
                        checked={benchmarkWarmup}
                        onChange={(e) => setBenchmarkWarmup(e.target.checked)}
                        disabled={benchmarkRunning || benchmarkIndexing}
                      />
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800 px-3 py-2 md:mt-6">
                      <span className="text-sm text-gray-700 dark:text-gray-300">Randomize Run Order</span>
                      <Toggle
                        id="benchmarkRandomizeExecution"
                        checked={benchmarkRandomizeExecution}
                        onChange={(e) => setBenchmarkRandomizeExecution(e.target.checked)}
                        disabled={benchmarkRunning || benchmarkIndexing}
                      />
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800 px-3 py-2 md:mt-6">
                      <span className="text-sm text-gray-700 dark:text-gray-300">Stop On Error</span>
                      <Toggle
                        id="benchmarkStopOnError"
                        checked={benchmarkStopOnError}
                        onChange={(e) => setBenchmarkStopOnError(e.target.checked)}
                        disabled={benchmarkRunning}
                      />
                    </div>
                    <div className="rounded-lg border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800 px-3 py-2 md:mt-6">
                      <div className="text-xs uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-1">Queue</div>
                      <div className="text-sm text-gray-700 dark:text-gray-300">
                        {benchmarkCatalog.length > 0
                          ? `${benchmarkFilteredCatalog.length} in range / ${benchmarkCatalog.length} indexed`
                          : `${benchmarkFiles.length} selected`}
                      </div>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-3 mb-5">
                    <button
                      onClick={() => benchmarkFileInputRef.current?.click()}
                      disabled={benchmarkRunning}
                      className="bg-primary hover:bg-primary-hover dark:bg-primary dark:hover:bg-primary-muted text-white font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <span className="material-icons-outlined text-sm">library_music</span>
                      Choose Audio Files
                    </button>
                    <button
                      onClick={pickBenchmarkFolderWithHandle}
                      disabled={benchmarkRunning || benchmarkIndexing}
                      className="bg-primary hover:bg-primary-hover dark:bg-primary dark:hover:bg-primary-muted text-white font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <span className="material-icons-outlined text-sm">folder_open</span>
                      Pick Folder
                    </button>
                    {benchmarkStoredFolderAvailable && (
                      <button
                        onClick={reconnectStoredBenchmarkFolder}
                        disabled={benchmarkRunning || benchmarkIndexing || benchmarkRestoringFolder}
                        className="bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      >
                        <span className="material-icons-outlined text-sm">restore</span>
                        {benchmarkRestoringFolder ? 'Reconnecting...' : `Reconnect ${benchmarkStoredFolderName || 'Saved Folder'}`}
                      </button>
                    )}
                    {benchmarkStoredFolderAvailable && (
                      <button
                        onClick={forgetStoredBenchmarkFolder}
                        disabled={benchmarkRunning || benchmarkIndexing}
                        className="bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      >
                        <span className="material-icons-outlined text-sm">delete</span>
                        Forget Saved Folder
                      </button>
                    )}
                    <button
                      onClick={applySuggestedTargets}
                      disabled={benchmarkRunning || benchmarkIndexing || benchmarkCatalog.length === 0 || benchmarkTargetMode === 'random'}
                      className="bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <span className="material-icons-outlined text-sm">auto_awesome</span>
                      Auto Targets
                    </button>
                    <button
                      onClick={() => pickRandomBenchmarkSamples(false)}
                      disabled={benchmarkRunning || benchmarkIndexing || benchmarkCatalog.length === 0 || benchmarkTargetMode !== 'random'}
                      className="bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <span className="material-icons-outlined text-sm">casino</span>
                      Pick Random Samples
                    </button>
                    <button
                      onClick={() => pickRandomBenchmarkSamples(true)}
                      disabled={benchmarkRunning || benchmarkIndexing || benchmarkCatalog.length === 0 || benchmarkTargetMode !== 'random'}
                      className="bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <span className="material-icons-outlined text-sm">shuffle</span>
                      Reshuffle
                    </button>
                    <button
                      onClick={refineTargetsFromBenchmarkRuns}
                      disabled={benchmarkRunning || benchmarkRuns.length < 2 || benchmarkTargetMode === 'random'}
                      className="bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <span className="material-icons-outlined text-sm">tune</span>
                      Refine Targets
                    </button>
                    <button
                      onClick={() => runBenchmark()}
                      disabled={!canRun || benchmarkResolvedPlan.length === 0}
                      className="bg-primary-muted hover:bg-primary dark:bg-border-dark dark:hover:bg-accent-muted text-white font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <span className="material-icons-outlined text-sm">speed</span>
                      {benchmarkRunning ? 'Benchmarking...' : benchmarkTargetMode === 'random' ? 'Run Random Plan' : 'Run Sample Plan'}
                    </button>
                    <button
                      onClick={exportSamplePlanJson}
                      disabled={benchmarkResolvedPlan.length === 0 || benchmarkRunning || benchmarkIndexing}
                      className="bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <span className="material-icons-outlined text-sm">download</span>
                      Export Sample JSON
                    </button>
                    <button
                      onClick={cancelBenchmark}
                      disabled={!benchmarkRunning}
                      className="bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                      <span className="material-icons-outlined text-sm">stop_circle</span>
                      Cancel
                    </button>
                    <button
                      onClick={() => { setBenchmarkRuns([]); setBenchmarkStatus('Benchmark results cleared.'); }}
                      disabled={benchmarkRunning || benchmarkRuns.length === 0}
                      className="bg-gray-200 hover:bg-gray-300 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-800 dark:text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Clear Results
                    </button>
                    <input
                      ref={benchmarkFileInputRef}
                      type="file"
                      accept="audio/*"
                      multiple
                      onChange={handleBenchmarkFileUpload}
                      className="hidden"
                    />
                    <input
                      ref={benchmarkFolderInputRef}
                      type="file"
                      accept="audio/*"
                      multiple
                      webkitdirectory=""
                      directory=""
                      onChange={handleBenchmarkFileUpload}
                      className="hidden"
                    />
                  </div>

                  <div className="upload-zone border-2 border-dashed border-border-light dark:border-border-dark rounded-xl bg-gray-50 dark:bg-gray-800/50 p-6">
                    <div className="flex items-center justify-between gap-3 mb-3">
                      <div className="min-w-0 flex-1 text-sm text-gray-600 dark:text-gray-400">
                        {benchmarkStatus}
                      </div>
                      {benchmarkTargetMode === 'random' && benchmarkResolvedPlan.length > 0 && (
                        <InlineDistributionBars
                          bins={benchmarkRandomDistribution}
                          expanded={benchmarkRandomDistributionExpanded}
                          onClick={() => setBenchmarkRandomDistributionExpanded((value) => !value)}
                        />
                      )}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                      {benchmarkTargetMode === 'random'
                        ? `Random seed: ${benchmarkRandomSeed}`
                        : `Targets: ${benchmarkTargets.length > 0 ? benchmarkTargets.join(', ') : 'none'}`}
                    </div>
                    {(effectiveBenchmarkMinDurationSec != null || effectiveBenchmarkMaxDurationSec != null) && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                        Sampling range: {effectiveBenchmarkMinDurationSec != null ? formatClockDuration(effectiveBenchmarkMinDurationSec) : '0:00'} to {effectiveBenchmarkMaxDurationSec != null ? formatClockDuration(effectiveBenchmarkMaxDurationSec) : 'no max'}
                      </div>
                    )}
                    {benchmarkTargetMode === 'random' && benchmarkResolvedPlan.length > 0 && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                        Random mode anchors shortest 2 and longest 2 files, then fills shared duration buckets with smoothed quotas so the histogram stays flatter without faking long-form coverage.
                      </div>
                    )}
                    {benchmarkLongRepeatBoostEnabled && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                        Buckets at or above {formatCompactDuration(benchmarkLongRepeatThresholdSec)} use up to {benchmarkLongRepeatMultiplier}x measured repeats for extra long-form stability.
                      </div>
                    )}
                    {benchmarkStoredFolderAvailable && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                        Saved folder: {benchmarkStoredFolderName || 'saved folder'}
                        {benchmarkFolderHandleSupported ? '' : ' (folder persistence not supported in this browser)'}
                      </div>
                    )}
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                      Sample plan: {benchmarkResolvedPlan.length} file{benchmarkResolvedPlan.length === 1 ? '' : 's'}
                    </div>
                    {benchmarkCurrent && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 mb-3 font-mono">
                        {benchmarkCurrent.index} / {benchmarkCurrent.total}: {benchmarkCurrent.name}
                      </div>
                    )}
                    {benchmarkResolvedPlan.length > 0 ? (
                      <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                        {benchmarkResolvedPlan.slice(0, 8).map((sample) => (
                          <div key={`${sample.path}-${sample.targetSec}`}>
                            {benchmarkTargetMode === 'random'
                              ? `random #${sample.randomIndex}: ${sample.path} (${sample.durationSec.toFixed(1)}s)`
                              : `target ${sample.targetSec}s -> ${sample.path} (${sample.durationSec.toFixed(1)}s, delta ${sample.deltaSec.toFixed(1)}s)`}
                          </div>
                        ))}
                        {benchmarkResolvedPlan.length > 8 && (
                          <div>... and {benchmarkResolvedPlan.length - 8} more planned samples</div>
                        )}
                      </div>
                    ) : benchmarkFiles.length > 0 && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
                        {benchmarkFiles.slice(0, 8).map((file) => (
                          <div key={`${file.name}-${file.lastModified}`}>{file.path || file.name}</div>
                        ))}
                        {benchmarkFiles.length > 8 && (
                          <div>... and {benchmarkFiles.length - 8} more</div>
                        )}
                      </div>
                    )}
                  </div>
                </div>

                {benchmarkTargetMode === 'random' && benchmarkResolvedPlan.length > 0 && benchmarkRandomDistributionExpanded && (
                  <DetailedDistributionHistogram
                    title="Picked Random Sample Distribution"
                    subtitle={`Full-size histogram for the selected random plan using the shared ${benchmarkBucketSize}s bucket size.`}
                    bins={benchmarkRandomDistribution}
                  />
                )}

                <BenchmarkMetrics overall={benchmarkOverall} />

                {benchmarkRuns.length > 0 && (
                  <div className="flex flex-col gap-6">
                    <BenchmarkChart
                      title="RTFx By Duration"
                      subtitle="Measured runs plotted against actual file duration so short- and long-form throughput trends are visible."
                      series={benchmarkChartData.rtfxSeries}
                      yLabel="RTFx"
                      formatY={(value) => `${value.toFixed(1)}x`}
                    />
                    <BenchmarkChart
                      title="Model Phase Time By Duration"
                      subtitle="Encoder and decoder timings for each successful run."
                      series={benchmarkChartData.phaseSeries}
                      yLabel="Milliseconds"
                      formatY={(value) => `${value.toFixed(0)}ms`}
                    />
                  </div>
                )}

                {benchmarkSummary.length > 0 && (
                  <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted">
                        Duration Buckets
                      </h2>
                      <div className="flex flex-wrap gap-2">
                        <button
                          onClick={() => copyToClipboard(benchmarkSummaryToCsv(benchmarkSummary))}
                          className="flex items-center gap-1 text-xs font-medium text-primary-muted hover:text-primary dark:text-accent-muted dark:hover:text-primary-muted bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded border border-border-light dark:border-border-dark transition-colors"
                        >
                          <span className="material-icons-outlined text-xs">table_view</span>
                          Copy CSV
                        </button>
                        <button
                          onClick={() => copyToClipboard(toJSON(benchmarkSummary))}
                          className="flex items-center gap-1 text-xs font-medium text-primary-muted hover:text-primary dark:text-accent-muted dark:hover:text-primary-muted bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded border border-border-light dark:border-border-dark transition-colors"
                        >
                          <span className="material-icons-outlined text-xs">content_copy</span>
                          Copy JSON
                        </button>
                        <button
                          onClick={exportBenchmarkSummaryCsv}
                          className="flex items-center gap-1 text-xs font-medium text-primary-muted hover:text-primary dark:text-accent-muted dark:hover:text-primary-muted bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded border border-border-light dark:border-border-dark transition-colors"
                        >
                          <span className="material-icons-outlined text-xs">download</span>
                          Export CSV
                        </button>
                        <button
                          onClick={exportBenchmarkSummaryJson}
                          className="flex items-center gap-1 text-xs font-medium text-primary-muted hover:text-primary dark:text-accent-muted dark:hover:text-primary-muted bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded border border-border-light dark:border-border-dark transition-colors"
                        >
                          <span className="material-icons-outlined text-xs">download</span>
                          Export JSON
                        </button>
                      </div>
                    </div>
                    <div className="flex flex-col gap-6 mb-6">
                      <BucketBoxChart
                        title="Bucketed RTFx"
                        subtitle={`Per-bucket throughput stability using min, quartiles, median, and max across the bucket range. Effective sweet spot uses a dry 90-minute overlap simulation with ${formatClockDuration(benchmarkOverlapSec)} average overlap and bucket averages for duration and model RTFx.`}
                        buckets={benchmarkSummary}
                        annotation={benchmarkEffectiveSweetSpot
                          ? {
                            bucketIndex: benchmarkSummary.findIndex((bucket) => bucket.label === benchmarkEffectiveSweetSpot.label),
                            color: '#b45309',
                            label: `${Math.round(benchmarkEffectiveSweetSpot.recommendedWindowSec || 0)}s avg sweet spot`,
                            detail: `${benchmarkEffectiveSweetSpot.avgModelRtfx?.toFixed(1) ?? '-'}x raw -> ${benchmarkEffectiveSweetSpot.simulatedEffectiveModelRtfx?.toFixed(1) ?? '-'}x sim (${benchmarkEffectiveSweetSpot.simulatedOverheadRatio?.toFixed(1) ?? '-'}% overhead)`,
                          }
                          : null}
                        series={[
                          {
                            label: 'Bucket Wall',
                            color: '#2563eb',
                            distribution: (bucket) => bucket.wallRtfxDistribution,
                          },
                          {
                            label: 'Bucket Model',
                            color: '#0f766e',
                            distribution: (bucket) => bucket.modelRtfxDistribution,
                          },
                        ]}
                        yLabel="RTFx"
                        formatY={(value) => `${value.toFixed(1)}x`}
                      />
                    </div>
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-sm">
                        <thead className="text-left text-gray-500 dark:text-gray-400">
                          <tr className="border-b border-border-light dark:border-border-dark">
                            <th className="py-2 pr-4">Bucket</th>
                            <th className="py-2 pr-4">Files</th>
                            <th className="py-2 pr-4">Total Audio</th>
                            <th className="py-2 pr-4">Avg / File</th>
                            <th className="py-2 pr-4">Wall Avg</th>
                            <th className="py-2 pr-4">Wall Med</th>
                            <th className="py-2 pr-4">Model Avg</th>
                            <th className="py-2 pr-4">Encode</th>
                            <th className="py-2">Decode</th>
                          </tr>
                        </thead>
                        <tbody>
                          {benchmarkSummary.map((bucket) => (
                            <tr key={bucket.label} className="border-b border-border-light/70 dark:border-border-dark/70">
                              <td className="py-2 pr-4 font-medium text-gray-900 dark:text-white">{bucket.label}</td>
                              <td className="py-2 pr-4">{bucket.count}</td>
                              <td className="py-2 pr-4">{formatSeconds(bucket.totalAudioSec)}</td>
                              <td className="py-2 pr-4">{bucket.avgAudioSec != null ? formatSeconds(bucket.avgAudioSec) : '-'}</td>
                              <td className="py-2 pr-4">{bucket.avgWallRtfx != null ? `${bucket.avgWallRtfx.toFixed(1)}x` : '-'}</td>
                              <td className="py-2 pr-4">{bucket.medianWallRtfx != null ? `${bucket.medianWallRtfx.toFixed(1)}x` : '-'}</td>
                              <td className="py-2 pr-4">{bucket.avgModelRtfx != null ? `${bucket.avgModelRtfx.toFixed(1)}x` : '-'}</td>
                              <td className="py-2 pr-4">{bucket.avgEncodeMs != null ? `${bucket.avgEncodeMs.toFixed(1)}ms` : '-'}</td>
                              <td className="py-2">{bucket.avgDecodeMs != null ? `${bucket.avgDecodeMs.toFixed(1)}ms` : '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {benchmarkRuns.length > 0 && (
                  <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted">
                        Benchmark Runs
                      </h2>
                      <div className="flex gap-2">
                        <button
                          onClick={() => copyToClipboard(benchmarkRunsToCsv(benchmarkRuns))}
                          className="flex items-center gap-1 text-xs font-medium text-primary-muted hover:text-primary dark:text-accent-muted dark:hover:text-primary-muted bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded border border-border-light dark:border-border-dark transition-colors"
                        >
                          <span className="material-icons-outlined text-xs">table_view</span>
                          Copy CSV
                        </button>
                        <button
                          onClick={() => copyToClipboard(toJSON(benchmarkRuns))}
                          className="flex items-center gap-1 text-xs font-medium text-primary-muted hover:text-primary dark:text-accent-muted dark:hover:text-primary-muted bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded border border-border-light dark:border-border-dark transition-colors"
                        >
                          <span className="material-icons-outlined text-xs">content_copy</span>
                          Copy JSON
                        </button>
                      </div>
                    </div>
                    <div className="overflow-x-auto max-h-[520px]">
                      <table className="min-w-full text-sm">
                        <thead className="sticky top-0 bg-card-light dark:bg-card-dark text-left text-gray-500 dark:text-gray-400">
                          <tr className="border-b border-border-light dark:border-border-dark">
                            <th className="py-2 pr-4">File</th>
                            <th className="py-2 pr-4">Target</th>
                            <th className="py-2 pr-4">Repeats</th>
                            <th className="py-2 pr-4">Delta</th>
                            <th className="py-2 pr-4">Duration</th>
                            <th className="py-2 pr-4">Wall</th>
                            <th className="py-2 pr-4">Wall SD</th>
                            <th className="py-2 pr-4">Model</th>
                            <th className="py-2 pr-4">Model SD</th>
                            <th className="py-2 pr-4">Encode</th>
                            <th className="py-2 pr-4">Decode</th>
                            <th className="py-2 pr-4">Feat</th>
                            <th className="py-2 pr-4">Payload</th>
                            <th className="py-2 pr-4">Text</th>
                            <th className="py-2">Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {benchmarkRuns.map((run) => (
                            <tr key={run.id} className="border-b border-border-light/70 dark:border-border-dark/70 align-top">
                              <td className="py-2 pr-4 font-medium text-gray-900 dark:text-white">{run.name}</td>
                              <td className="py-2 pr-4">
                                {run.selectionMode === 'random'
                                  ? `rnd #${run.randomIndex ?? '-'}`
                                  : run.targetSec != null
                                    ? formatSeconds(run.targetSec)
                                    : '-'}
                              </td>
                              <td className="py-2 pr-4">{run.repeatCount ?? 1}</td>
                              <td className="py-2 pr-4">{run.sampledDeltaSec != null ? `${run.sampledDeltaSec.toFixed(1)}s` : '-'}</td>
                              <td className="py-2 pr-4">{run.durationSec != null ? formatSeconds(run.durationSec) : '-'}</td>
                              <td className="py-2 pr-4">{run.wallRtfx != null ? `${run.wallRtfx.toFixed(1)}x` : '-'}</td>
                              <td className="py-2 pr-4">{run.wallRtfxStddev != null ? `${run.wallRtfxStddev.toFixed(1)}x` : '-'}</td>
                              <td className="py-2 pr-4">{run.modelRtfx != null ? `${run.modelRtfx.toFixed(1)}x` : '-'}</td>
                              <td className="py-2 pr-4">{run.modelRtfxStddev != null ? `${run.modelRtfxStddev.toFixed(1)}x` : '-'}</td>
                              <td className="py-2 pr-4">{run.encodeMs != null ? `${run.encodeMs.toFixed(1)}ms` : '-'}</td>
                              <td className="py-2 pr-4">{run.decodeMs != null ? `${run.decodeMs.toFixed(1)}ms` : '-'}</td>
                              <td className="py-2 pr-4" title={run.featureShape || undefined}>
                                {run.featureMiB != null ? `${run.featureMiB.toFixed(2)} MiB` : '-'}
                              </td>
                              <td className="py-2 pr-4">{run.encoderPayloadMiB != null ? `${run.encoderPayloadMiB.toFixed(2)} MiB` : '-'}</td>
                              <td className="py-2 pr-4">{run.textLen != null ? run.textLen : '-'}</td>
                              <td className="py-2">
                                {run.error ? (
                                  <span className="text-red-600 dark:text-red-400">{run.error}</span>
                                ) : (
                                  <span className="text-green-700 dark:text-green-400">{run.mode}</span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {activeView === 'transcribe' && (
            <div className="space-y-6">
              <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-sm border border-border-light dark:border-border-dark p-6">
                <div className="mb-4">
                  <div>
                    <h2 className="text-xs font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-2">
                      API Contract
                    </h2>
                    <p className="text-sm text-gray-600 dark:text-gray-400">
                      The demo now mirrors the updated NeMo TDT split: pipeline mode stays task-compatible,
                      while direct mode exposes the full model output.
                    </p>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="rounded-xl border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800/50 p-4">
                    <div className="text-[0.65rem] font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-2">
                      Current Output
                    </div>
                    <div className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                      {currentContractSummary.title}
                    </div>
                    <div className="font-mono text-xs text-gray-700 dark:text-gray-300 mb-2">
                      {currentContractSummary.shape}
                    </div>
                    <p className="text-sm leading-relaxed text-gray-600 dark:text-gray-400">
                      {currentContractSummary.detail}
                    </p>
                  </div>

                  {!direct && (
                    <div className="rounded-xl border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800/50 p-4">
                      <div className="text-[0.65rem] font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-2">
                        Current Options
                      </div>
                      <pre className="text-xs leading-relaxed text-gray-800 dark:text-gray-200 font-mono whitespace-pre-wrap break-words max-h-[260px] overflow-auto">
                        {currentOptionsSnippet}
                      </pre>
                      <button
                        onClick={() => copyToClipboard(currentOptionsSnippet)}
                        className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-primary-muted hover:text-primary dark:text-accent-muted dark:hover:text-primary-muted bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded border border-border-light dark:border-border-dark transition-colors"
                      >
                        <span className="material-icons-outlined text-xs">content_copy</span>
                        Copy options
                      </button>
                    </div>
                  )}

                  <div className="rounded-xl border border-border-light dark:border-border-dark bg-gray-50 dark:bg-gray-800/50 p-4">
                    <div className="flex items-center justify-between gap-3 mb-3">
                      <div>
                        <div className="text-[0.65rem] font-bold uppercase tracking-wider text-primary-muted dark:text-accent-muted mb-1">
                          Copyable Example
                        </div>
                        <p className="text-sm text-gray-600 dark:text-gray-400">
                          Uses the exact load mode and inference toggles selected in the UI.
                        </p>
                      </div>
                      <button
                        onClick={() => copyToClipboard(currentExampleSnippet)}
                        className="inline-flex items-center gap-1 text-xs font-medium text-primary-muted hover:text-primary dark:text-accent-muted dark:hover:text-primary-muted bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded border border-border-light dark:border-border-dark transition-colors"
                      >
                        <span className="material-icons-outlined text-xs">content_copy</span>
                        Copy JS
                      </button>
                    </div>
                    <pre className="text-xs leading-relaxed text-gray-800 dark:text-gray-200 font-mono whitespace-pre-wrap break-words max-h-[420px] overflow-auto">
                      {currentExampleSnippet}
                    </pre>
                  </div>
                </div>
              </div>

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
                  <div className="space-y-3 max-h-[520px] overflow-y-auto pr-1">
                    {history.map((h) => (
                      <div key={h.id} className="p-4 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="flex justify-between items-start mb-2 gap-3">
                          <span className="text-sm font-medium text-gray-900 dark:text-white break-words">
                            {h.name}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded-full border border-gray-200 dark:border-gray-700 shrink-0">
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
          )}
        </div>
      </div>
    </div>
  );
}

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  env,
  pipeline,
  AutoProcessor,
  AutoTokenizer,
  NemoConformerForTDT,
  AutomaticSpeechRecognitionPipeline,
} from '@huggingface/transformers';
import './App.css';

const DEFAULT_MODEL_ID = 'ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4';
const SAMPLE_AUDIO_URL = '/assets/life_Jim.wav';
const DEFAULT_DTYPES = ['fp16', 'int8', 'fp32'];
const DTYPE_PRIORITY = ['fp16', 'int8', 'fp32', 'q8', 'q4', 'q4f16'];
const MODEL_PRESETS = [
  { label: 'v2 tfjs4', value: 'ysdede/parakeet-tdt-0.6b-v2-onnx-tfjs4' },
  { label: 'v3 tfjs4', value: 'ysdede/parakeet-tdt-0.6b-v3-onnx-tfjs4' },
];

let originFetch = null;
if (typeof window !== 'undefined' && !originFetch) {
  originFetch = window.fetch;
}

const TRANSFORMERS_VERSION =
  typeof __TRANSFORMERS_VERSION__ !== 'undefined' ? __TRANSFORMERS_VERSION__ : 'unknown';
const TRANSFORMERS_SOURCE = typeof __TRANSFORMERS_SOURCE__ !== 'undefined' ? __TRANSFORMERS_SOURCE__ : 'unknown';
const DEFAULT_VERBOSE_LOGGING = false;
const MAX_DIAGNOSTIC_LOGS = 160;
const PROGRESS_LOG_STEP = 5;

if (typeof window !== 'undefined') {
  env.allowRemoteModels = true;
  env.allowLocalModels = false;

  if (env?.backends?.onnx?.wasm) {
    env.backends.onnx.wasm.numThreads = 1;
  }
}

function summarizeRuntimeFlags() {
  const onnx = env?.backends?.onnx;

  if (!onnx) return { available: false };

  const wasmPaths = onnx?.wasm?.wasmPaths;
  return {
    available: true,
    logLevel: onnx.logLevel ?? null,
    debug: onnx.debug ?? null,
    trace: onnx.trace ?? null,
    wasm: onnx.wasm
      ? {
        numThreads: onnx.wasm.numThreads ?? null,
        proxy: onnx.wasm.proxy ?? null,
        trace: onnx.wasm.trace ?? null,
        wasmPaths:
          typeof wasmPaths === 'string'
            ? wasmPaths
            : wasmPaths
              ? { wasm: wasmPaths.wasm ?? null, mjs: wasmPaths.mjs ?? null }
              : null,
      }
      : null,
    webgpu: onnx.webgpu
      ? {
        powerPreference: onnx.webgpu.powerPreference ?? null,
        forceFallbackAdapter: onnx.webgpu.forceFallbackAdapter ?? null,
        validateInputContent: onnx.webgpu.validateInputContent ?? null,
        profilingMode: onnx.webgpu.profiling?.mode ?? onnx.webgpu.profilingMode ?? null,
      }
      : null,
  };
}

function summarizeSession(session) {
  if (!session) return null;
  return {
    inputCount: Array.isArray(session.inputNames) ? session.inputNames.length : null,
    outputCount: Array.isArray(session.outputNames) ? session.outputNames.length : null,
    inputNames: Array.isArray(session.inputNames) ? session.inputNames : null,
    outputNames: Array.isArray(session.outputNames) ? session.outputNames : null,
  };
}

function summarizePipeline(transcriber) {
  const model = transcriber?.model;
  const sessions =
    model?.sessions && typeof model.sessions === 'object' && !Array.isArray(model.sessions)
      ? model.sessions
      : {};
  const sessionEntries = Object.entries(sessions);

  const summarizedSessions = Object.fromEntries(
    sessionEntries.map(([name, session]) => [name, summarizeSession(session)]),
  );
  const maybeSingleSession = summarizeSession(model?.session);

  return {
    modelType: model?.config?.model_type ?? null,
    modelClass: model?.constructor?.name ?? null,
    sessionKeys: sessionEntries.map(([name]) => name),
    sessions: summarizedSessions,
    singleSession: maybeSingleSession,
  };
}

function summarizeAudioInput(audioInput) {
  if (typeof audioInput === 'string') {
    const sourceType = audioInput.startsWith('blob:')
      ? 'blob-url'
      : audioInput.startsWith('http')
        ? 'http-url'
        : 'path-or-url';
    return {
      type: 'string',
      sourceType,
      value: audioInput,
    };
  }
  if (audioInput instanceof Blob) {
    return {
      type: 'blob',
      mimeType: audioInput.type || null,
      size: audioInput.size ?? null,
    };
  }
  return { type: typeof audioInput };
}

function summarizeError(error) {
  if (!error) return { message: 'Unknown error' };

  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack ?? null,
      cause: error.cause ? summarizeError(error.cause) : null,
    };
  }

  if (typeof error === 'object') {
    const fallbackMessage = 'message' in error ? String(error.message) : String(error);
    return { message: fallbackMessage };
  }

  return { message: String(error) };
}

function summarizeOutput(output) {
  if (typeof output === 'string') {
    return { type: 'string', textLength: output.length };
  }
  if (Array.isArray(output)) {
    return {
      type: 'array',
      itemCount: output.length,
      firstItemKeys: output[0] && typeof output[0] === 'object' ? Object.keys(output[0]) : null,
    };
  }
  if (output && typeof output === 'object') {
    return {
      type: 'object',
      keys: Object.keys(output),
      textLength: typeof output.text === 'string' ? output.text.length : null,
      chunkCount: Array.isArray(output.chunks) ? output.chunks.length : null,
    };
  }
  return { type: typeof output };
}

function formatProgress(progressInfo) {
  if (!progressInfo) return 'Loading...';
  const file = progressInfo.file || progressInfo.name || 'artifact';
  const status = progressInfo.status || 'downloading';
  const pct =
    Number.isFinite(progressInfo.progress) && progressInfo.progress >= 0
      ? ` (${Math.round(progressInfo.progress)}%)`
      : '';
  return `${status}: ${file}${pct}`;
}

function pretty(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function extractText(output) {
  if (!output) return '';
  if (typeof output === 'string') return output;
  if (Array.isArray(output)) {
    return output.map((x) => x?.text ?? '').filter(Boolean).join('\n');
  }
  return output.text ?? '';
}

export default function App() {
  const transcriberRef = useRef(null);
  const logSequenceRef = useRef(0);
  const lastProgressLogRef = useRef({ status: '', file: '', pct: -1 });

  const [modelId, setModelId] = useState(DEFAULT_MODEL_ID);
  const [encoderDevice, setEncoderDevice] = useState('webgpu');
  const [decoderDevice, setDecoderDevice] = useState('wasm');
  const [encoderDtype, setEncoderDtype] = useState('fp16');
  const [decoderDtype, setDecoderDtype] = useState('int8');
  const [availableEncoderDtypes, setAvailableEncoderDtypes] = useState(DEFAULT_DTYPES);
  const [availableDecoderDtypes, setAvailableDecoderDtypes] = useState(DEFAULT_DTYPES);
  const [returnTimestamps, setReturnTimestamps] = useState(false);
  const [timestampGranularity, setTimestampGranularity] = useState('word');
  // Local folder loader state
  const [isLocalMode, setIsLocalMode] = useState(false);
  const [localFilesCount, setLocalFilesCount] = useState(0);
  const localFilesMapRef = useRef(new Map());
  const folderInputRef = useRef(null);

  // Temporary fetch interceptor for local mode
  useEffect(() => {
    if (!isLocalMode || localFilesMapRef.current.size === 0) {
      if (originFetch) window.fetch = originFetch;
      return;
    }
    window.fetch = async (url, options) => {
      const urlStr = typeof url === 'string' ? url : url.url;
      if (urlStr.includes('resolve/main/') || urlStr.includes('resolve/')) {
        let requestedFile = urlStr.includes('resolve/main/')
          ? urlStr.substring(urlStr.lastIndexOf('resolve/main/') + 13)
          : urlStr.substring(urlStr.lastIndexOf('resolve/') + 8).split('/').slice(1).join('/');

        // Match with local blobs
        for (const [localPath, fileBlob] of localFilesMapRef.current.entries()) {
          const lp = localPath.replace(/\\/g, '/');
          if (lp.endsWith(requestedFile) || requestedFile.endsWith(lp)) {
            console.log(`[Local Intercept] Serving local file mapping for ${requestedFile} -> ${localPath}`);
            return new Response(fileBlob, {
              status: 200, statusText: 'OK', headers: new Headers({
                'content-length': fileBlob.size.toString(),
              })
            });
          }
        }
      }
      return originFetch(url, options);
    };
    return () => {
      if (originFetch) window.fetch = originFetch;
    };
  }, [isLocalMode, localFilesCount]);

  function handleFolderSelect(e) {
    const files = Array.from(e.target.files);
    localFilesMapRef.current.clear();
    for (const f of files) {
      const p = f.webkitRelativePath || f.name;
      localFilesMapRef.current.set(p, f);
    }
    setLocalFilesCount(files.length);
    setIsLocalMode(true);
    setStatus(`Local folder mapped (${files.length} files).`);
  }

  const [loadMode, setLoadMode] = useState('pipeline'); // 'pipeline' | 'explicit'
  const [modelType, setModelType] = useState('not-loaded');
  const [status, setStatus] = useState('Idle');
  const [progressText, setProgressText] = useState('');
  const [progressPct, setProgressPct] = useState(null);
  const [error, setError] = useState('');
  const [selectedFileName, setSelectedFileName] = useState('');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [verboseLogging, setVerboseLogging] = useState(DEFAULT_VERBOSE_LOGGING);
  const [diagnostics, setDiagnostics] = useState([]);

  const appendDiagnostic = useCallback(
    (level, event, payload = undefined) => {
      if (level === 'debug' && !verboseLogging) return;

      const timestamp = new Date().toISOString();
      const entry = {
        id: `${Date.now()}-${logSequenceRef.current++}`,
        timestamp,
        level,
        event,
        payload: payload ?? null,
      };
      setDiagnostics((prev) => [entry, ...prev].slice(0, MAX_DIAGNOSTIC_LOGS));

      const prefix = `[parakeet-demo][${timestamp}][${level}] ${event}`;
      if (level === 'error') {
        payload !== undefined ? console.error(prefix, payload) : console.error(prefix);
      } else if (level === 'warn') {
        payload !== undefined ? console.warn(prefix, payload) : console.warn(prefix);
      } else if (level === 'debug') {
        payload !== undefined ? console.debug(prefix, payload) : console.debug(prefix);
      } else {
        payload !== undefined ? console.info(prefix, payload) : console.info(prefix);
      }
    },
    [verboseLogging],
  );

  const configureRuntimeLogging = useCallback(() => {
    const onnx = env?.backends?.onnx;
    if (!onnx) {
      appendDiagnostic('warn', 'ONNX runtime environment is not ready yet.');
      return;
    }

    onnx.logLevel = verboseLogging ? 'verbose' : 'warning';
    onnx.debug = verboseLogging;
    onnx.trace = verboseLogging;

    if (onnx.wasm) {
      onnx.wasm.trace = verboseLogging;
    }
    if (onnx.webgpu) {
      onnx.webgpu.validateInputContent = verboseLogging;
    }

    appendDiagnostic('info', 'Applied ONNX runtime logging flags', summarizeRuntimeFlags());
  }, [appendDiagnostic, verboseLogging]);

  const canTranscribe = !!transcriberRef.current && !isLoadingModel && !isTranscribing;
  const headline = useMemo(() => {
    return `transformers.js ${TRANSFORMERS_VERSION} (${TRANSFORMERS_SOURCE})`;
  }, []);

  useEffect(() => {
    configureRuntimeLogging();
  }, [configureRuntimeLogging]);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;

    const onWindowError = (event) => {
      appendDiagnostic('error', 'window.error', {
        message: event.message ?? null,
        filename: event.filename ?? null,
        line: event.lineno ?? null,
        column: event.colno ?? null,
        error: summarizeError(event.error),
      });
    };

    const onUnhandledRejection = (event) => {
      appendDiagnostic('error', 'window.unhandledrejection', {
        reason: summarizeError(event.reason),
      });
    };

    window.addEventListener('error', onWindowError);
    window.addEventListener('unhandledrejection', onUnhandledRejection);

    appendDiagnostic('info', 'Attached global error listeners.');

    return () => {
      window.removeEventListener('error', onWindowError);
      window.removeEventListener('unhandledrejection', onUnhandledRejection);
    };
  }, [appendDiagnostic]);

  useEffect(() => {
    let isCancelled = false;

    async function refreshAvailableDtypes() {
      const trimmed = modelId.trim();
      if (!trimmed || !trimmed.includes('/')) {
        setAvailableEncoderDtypes(DEFAULT_DTYPES);
        setAvailableDecoderDtypes(DEFAULT_DTYPES);
        return;
      }

      try {
        const [owner, ...repoParts] = trimmed.split('/').filter(Boolean);
        if (!owner || repoParts.length === 0) {
          setAvailableEncoderDtypes(DEFAULT_DTYPES);
          setAvailableDecoderDtypes(DEFAULT_DTYPES);
          return;
        }
        const repo = repoParts.join('/');
        const response = await fetch(
          `https://huggingface.co/api/models/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`,
        );
        if (!response.ok) throw new Error(`HF API request failed: ${response.status}`);

        const data = await response.json();
        const siblings = Array.isArray(data?.siblings) ? data.siblings : [];
        const files = siblings.map((x) => x?.rfilename).filter(Boolean);

        const encoder = new Set();
        const decoder = new Set();
        const pattern = /^onnx\/(encoder_model|decoder_model_merged)(?:_(fp16|int8|q8|q4|q4f16))?\.onnx$/;
        for (const file of files) {
          const match = file.match(pattern);
          if (!match) continue;
          const role = match[1];
          const suffix = match[2];
          const mapped = suffix || 'fp32';
          if (role === 'encoder_model') encoder.add(mapped);
          if (role === 'decoder_model_merged') decoder.add(mapped);
        }

        const nextEncoderDtypes = DTYPE_PRIORITY.filter((x) => encoder.has(x));
        const nextDecoderDtypes = DTYPE_PRIORITY.filter((x) => decoder.has(x));
        const encoderChoices = nextEncoderDtypes.length > 0 ? nextEncoderDtypes : DEFAULT_DTYPES;
        const decoderChoices = nextDecoderDtypes.length > 0 ? nextDecoderDtypes : DEFAULT_DTYPES;

        if (!isCancelled) {
          setAvailableEncoderDtypes(encoderChoices);
          setAvailableDecoderDtypes(decoderChoices);
          appendDiagnostic('debug', 'Resolved available dtypes for model', {
            modelId: trimmed,
            encoderChoices,
            decoderChoices,
          });

          if (!encoderChoices.includes(encoderDtype)) {
            setEncoderDtype(encoderChoices[0]);
          }
          if (!decoderChoices.includes(decoderDtype)) {
            setDecoderDtype(decoderChoices[0]);
          }
        }
      } catch (dtypeError) {
        if (!isCancelled) {
          setAvailableEncoderDtypes(DEFAULT_DTYPES);
          setAvailableDecoderDtypes(DEFAULT_DTYPES);
        }
        appendDiagnostic('warn', 'Failed to resolve dtypes from HF model metadata, using defaults.', {
          modelId: trimmed,
          error: summarizeError(dtypeError),
        });
      }
    }

    refreshAvailableDtypes();
    return () => {
      isCancelled = true;
    };
  }, [modelId, encoderDtype, decoderDtype, appendDiagnostic]);

  async function disposeExistingPipeline(reason = 'unspecified') {
    const hasPipeline = !!transcriberRef.current;
    appendDiagnostic('info', 'Disposing existing pipeline', { reason, hasPipeline });

    if (transcriberRef.current?.dispose) {
      try {
        await transcriberRef.current.dispose();
        appendDiagnostic('info', 'Pipeline dispose() completed', { reason });
      } catch (disposeError) {
        appendDiagnostic('error', 'Pipeline dispose() failed', {
          reason,
          error: summarizeError(disposeError),
        });
        throw disposeError;
      }
    }
    transcriberRef.current = null;
  }

  async function loadModel() {
    setError('');
    setStatus('Loading model...');
    setProgressText('');
    setResult(null);
    setIsLoadingModel(true);
    configureRuntimeLogging();
    lastProgressLogRef.current = { status: '', file: '', pct: -1 };

    const startedAt = performance.now();
    const sessionOptions = {
      logId: `parakeet-demo-${Date.now()}`,
      logSeverityLevel: verboseLogging ? 0 : 2,
      logVerbosityLevel: verboseLogging ? 1 : 0,
      enableMemPattern: false,
      enableCpuMemArena: false,
      freeDimensionOverrides: {
        batch_size: 1
      },
    };

    const pipelineOptions = {
      device: {
        encoder_model: encoderDevice,
        decoder_model_merged: decoderDevice,
      },
      dtype: {
        encoder_model: encoderDtype,
        decoder_model_merged: decoderDtype,
      },
      session_options: sessionOptions,
      progress_callback: (info) => {
        setProgressText(formatProgress(info));
        if (Number.isFinite(info?.progress) && info.progress >= 0) {
          setProgressPct(Math.max(0, Math.min(100, Math.round(info.progress))));
        } else {
          setProgressPct(null);
        }

        const currentStatus = info?.status || 'downloading';
        const currentFile = info?.file || info?.name || 'artifact';
        const pct = Number.isFinite(info?.progress) ? Math.round(info.progress) : null;
        const last = lastProgressLogRef.current;
        const statusChanged = currentStatus !== last.status || currentFile !== last.file;
        const pctChanged =
          pct !== null &&
          (last.pct < 0 || Math.abs(pct - last.pct) >= PROGRESS_LOG_STEP || pct >= 100);

        if (statusChanged || pctChanged) {
          appendDiagnostic('info', 'Model load progress', {
            status: currentStatus,
            file: currentFile,
            progress: pct,
            loaded: info?.loaded ?? null,
            total: info?.total ?? null,
          });
          lastProgressLogRef.current = {
            status: currentStatus,
            file: currentFile,
            pct: pct ?? last.pct,
          };
        }
      },
    };
    const pipelineOptionsForLog = {
      ...pipelineOptions,
      progress_callback: '[function progress_callback]',
    };

    appendDiagnostic('info', 'Starting pipeline load', {
      modelId: modelId.trim(),
      pipelineOptions: pipelineOptionsForLog,
      runtime: summarizeRuntimeFlags(),
    });

    try {
      await disposeExistingPipeline('load-model');
      let transcriber;

      if (loadMode === 'explicit') {
        appendDiagnostic('info', 'Loading model explicitly (NemoConformerForTDT + processor + tokenizer)');
        const [processor, tokenizer, model] = await Promise.all([
          AutoProcessor.from_pretrained(modelId.trim(), { progress_callback: pipelineOptions.progress_callback }),
          AutoTokenizer.from_pretrained(modelId.trim(), { progress_callback: pipelineOptions.progress_callback }),
          NemoConformerForTDT.from_pretrained(modelId.trim(), pipelineOptions),
        ]);
        transcriber = new AutomaticSpeechRecognitionPipeline({
          model,
          processor,
          tokenizer,
          task: 'automatic-speech-recognition',
        });
        appendDiagnostic('info', 'Explicit pipeline constructed');
      } else {
        transcriber = await pipeline('automatic-speech-recognition', modelId.trim(), pipelineOptions);
      }

      transcriberRef.current = transcriber;
      const loadedType = transcriber?.model?.config?.model_type || 'unknown';
      setModelType(loadedType);
      setStatus(`Loaded (${loadedType})${loadMode === 'explicit' ? ' [explicit]' : ''}`);
      setProgressPct(null);
      appendDiagnostic('info', 'Pipeline loaded successfully', {
        modelType: loadedType,
        loadMode,
        runtime: summarizeRuntimeFlags(),
        pipeline: summarizePipeline(transcriber),
      });
    } catch (loadError) {
      setStatus('Load failed');
      setError(loadError?.message || String(loadError));
      setProgressPct(null);
      appendDiagnostic('error', 'Pipeline load failed', {
        modelId: modelId.trim(),
        pipelineOptions: pipelineOptionsForLog,
        runtime: summarizeRuntimeFlags(),
        error: summarizeError(loadError),
      });
    } finally {
      setIsLoadingModel(false);
      appendDiagnostic('info', 'Model load finished', {
        elapsedMs: Math.round(performance.now() - startedAt),
      });
    }
  }

  function onAudioPicked(event) {
    const file = event.target.files?.[0] || null;
    setSelectedFileName(file ? file.name : '');
    setResult(null);
    setError('');
    appendDiagnostic('info', 'Audio file selected', {
      fileName: file?.name ?? null,
      fileSize: file?.size ?? null,
      fileType: file?.type ?? null,
    });
  }

  async function runTranscription(audioInput, displayName) {
    if (!transcriberRef.current) {
      setError('Load a model first.');
      return;
    }

    setIsTranscribing(true);
    setError('');
    setStatus(`Transcribing ${displayName}...`);
    setProgressText('');

    try {
      const startedAt = performance.now();
      const options = {};
      if (returnTimestamps) {
        options.return_timestamps = true;
        if (modelType === 'nemo-conformer-tdt') {
          options.timestamp_granularity = timestampGranularity;
        }
      }
      appendDiagnostic('info', 'Transcription started', {
        displayName,
        options,
        audioInput: summarizeAudioInput(audioInput),
        pipeline: summarizePipeline(transcriberRef.current),
      });

      const output = await transcriberRef.current(audioInput, options);
      setResult(output);
      const elapsedMs = performance.now() - startedAt;
      const text = extractText(output);
      setHistory((prev) => [
        {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          when: new Date().toLocaleTimeString(),
          file: displayName,
          dtype: `enc:${encoderDtype} dec:${decoderDtype}`,
          elapsedMs,
          text,
        },
        ...prev,
      ]);
      setStatus('Transcription complete');
      appendDiagnostic('info', 'Transcription completed', {
        displayName,
        elapsedMs: Math.round(elapsedMs),
        output: summarizeOutput(output),
      });
    } catch (transcribeError) {
      setStatus('Transcription failed');
      setError(transcribeError?.message || String(transcribeError));
      appendDiagnostic('error', 'Transcription failed', {
        displayName,
        audioInput: summarizeAudioInput(audioInput),
        runtime: summarizeRuntimeFlags(),
        error: summarizeError(transcribeError),
      });
    } finally {
      setIsTranscribing(false);
    }
  }

  async function transcribeUploadedFile(event) {
    const file = event.target.files?.[0] || null;
    if (!file) return;

    const objectUrl = URL.createObjectURL(file);
    try {
      appendDiagnostic('debug', 'Created object URL for uploaded audio', {
        fileName: file.name,
        objectUrl,
      });
      await runTranscription(objectUrl, file.name);
    } finally {
      URL.revokeObjectURL(objectUrl);
      appendDiagnostic('debug', 'Revoked object URL for uploaded audio', {
        fileName: file.name,
      });
    }
  }

  async function transcribeSample() {
    appendDiagnostic('info', 'Starting sample transcription', { source: SAMPLE_AUDIO_URL });
    await runTranscription(SAMPLE_AUDIO_URL, 'life_Jim.wav');
  }

  async function resetModel() {
    setError('');
    setResult(null);
    setStatus('Idle');
    setProgressText('');
    setProgressPct(null);
    setModelType('not-loaded');
    await disposeExistingPipeline('manual-reset');
    appendDiagnostic('info', 'Model reset completed');
  }

  return (
    <div className="app">
      <div className="card">
        <h1>Transformers.js v4 Nemo Conformer TDT ASR Demo</h1>
        <p className="sub">{headline}</p>
        <p className="hint">
          Use <code>npm run dev:local</code> to load your local{' '}
          <code>N:\github\ysdede\transformers.js</code> web build.
        </p>
        <p className="hint">
          If you see &quot;Buffer used in submit while destroyed&quot; on WebGPU, try <strong>Encoder device: wasm</strong> to confirm it is WebGPU-specific; use <strong>Load mode: Model + processor (explicit)</strong> to test without auto pipeline.
        </p>

        <div className="row">
          <label>Load mode</label>
          <select
            value={loadMode}
            onChange={(e) => setLoadMode(e.target.value)}
            disabled={isLoadingModel || isTranscribing}
          >
            <option value="pipeline">Pipeline (auto)</option>
            <option value="explicit">Model + processor (explicit)</option>
          </select>
          <small>{loadMode === 'explicit' ? 'Loads NemoConformerForTDT, AutoProcessor, AutoTokenizer explicitly.' : 'Uses pipeline() to auto-select model from repo config.'}</small>
        </div>

        <div className="row">
          <label>Load Local Folder</label>
          <input
            type="file"
            webkitdirectory="true"
            directory="true"
            ref={folderInputRef}
            onChange={handleFolderSelect}
            disabled={isLoadingModel || isTranscribing}
          />
          {localFilesCount > 0 && <small>{localFilesCount} local files mapped.</small>}
        </div>

        <div className="row">
          <label>Model Preset</label>
          <select
            value={MODEL_PRESETS.some((x) => x.value === modelId) ? modelId : ''}
            onChange={(e) => {
              if (e.target.value) setModelId(e.target.value);
            }}
            disabled={isLoadingModel || isTranscribing}
          >
            <option value="">custom</option>
            {MODEL_PRESETS.map((preset) => (
              <option key={preset.value} value={preset.value}>
                {preset.label}
              </option>
            ))}
          </select>
        </div>

        <div className="row">
          <label>Model ID</label>
          <input
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            placeholder="hf-repo/model-id"
            disabled={isLoadingModel || isTranscribing}
          />
        </div>

        <div className="grid3">
          <div className="row">
            <label>Encoder device</label>
            <select
              value={encoderDevice}
              onChange={(e) => setEncoderDevice(e.target.value)}
              disabled={isLoadingModel}
            >
              <option value="webgpu">webgpu</option>
              <option value="wasm">wasm</option>
            </select>
          </div>

          <div className="row">
            <label>Encoder dtype</label>
            <select
              value={encoderDtype}
              onChange={(e) => setEncoderDtype(e.target.value)}
              disabled={isLoadingModel}
            >
              {availableEncoderDtypes.map((x) => (
                <option key={x} value={x}>
                  {x}
                </option>
              ))}
            </select>
          </div>

          <div className="row">
            <label>Model type (config.json)</label>
            <input value={modelType} disabled />
          </div>
        </div>

        <div className="grid2">
          <div className="row">
            <label>Decoder device</label>
            <select
              value={decoderDevice}
              onChange={(e) => setDecoderDevice(e.target.value)}
              disabled={isLoadingModel}
            >
              <option value="wasm">wasm</option>
              <option value="webgpu">webgpu</option>
            </select>
          </div>
          <div className="row">
            <label>Decoder dtype</label>
            <select
              value={decoderDtype}
              onChange={(e) => setDecoderDtype(e.target.value)}
              disabled={isLoadingModel}
            >
              {availableDecoderDtypes.map((x) => (
                <option key={x} value={x}>
                  {x}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="grid2">
          <label className="check">
            <input
              type="checkbox"
              checked={returnTimestamps}
              onChange={(e) => setReturnTimestamps(e.target.checked)}
              disabled={isLoadingModel || isTranscribing}
            />
            return_timestamps
          </label>
          <div className="row">
            <label>timestamp_granularity (Nemo Conformer TDT)</label>
            <select
              value={timestampGranularity}
              onChange={(e) => setTimestampGranularity(e.target.value)}
              disabled={isLoadingModel || isTranscribing || !returnTimestamps}
            >
              <option value="utterance">utterance</option>
              <option value="word">word</option>
              <option value="token">token</option>
              <option value="all">all</option>
            </select>
          </div>
        </div>

        <div className="grid2">
          <label className="check">
            <input
              type="checkbox"
              checked={verboseLogging}
              onChange={(e) => setVerboseLogging(e.target.checked)}
              disabled={isLoadingModel}
            />
            verbose_runtime_logs
          </label>
          <div className="buttons">
            <button
              onClick={() => setDiagnostics([])}
              disabled={diagnostics.length === 0 || isLoadingModel || isTranscribing}
            >
              Clear Diagnostics
            </button>
          </div>
        </div>

        <div className="buttons">
          <button onClick={loadModel} disabled={isLoadingModel || isTranscribing}>
            {isLoadingModel ? 'Loading...' : 'Load Pipeline'}
          </button>
          <button onClick={resetModel} disabled={isLoadingModel || isTranscribing}>
            Dispose
          </button>
        </div>

        <div className="buttons">
          <label className="fileBtn">
            Upload Audio
            <input
              type="file"
              accept="audio/*"
              onChange={(e) => {
                onAudioPicked(e);
                transcribeUploadedFile(e);
              }}
              disabled={!canTranscribe}
            />
          </label>
          <button onClick={transcribeSample} disabled={!canTranscribe}>
            Transcribe Sample
          </button>
        </div>

        <p className="status">Status: {status}</p>
        {selectedFileName ? <p className="status">Selected: {selectedFileName}</p> : null}
        {progressText ? <p className="status">{progressText}</p> : null}
        {progressPct !== null ? (
          <div className="progressWrap">
            <div className="progressBar">
              <div className="progressFill" style={{ width: `${progressPct}%` }} />
            </div>
            <span className="progressLabel">{progressPct}%</span>
          </div>
        ) : null}
        {error ? <pre className="error">{error}</pre> : null}

        <div className="result">
          <h2>Output</h2>
          <pre>{result ? pretty(result) : 'No transcription yet.'}</pre>
        </div>

        <div className="result">
          <h2>Transcription History</h2>
          {history.length === 0 ? (
            <pre>No transcriptions yet.</pre>
          ) : (
            <div className="historyList">
              {history.map((item) => (
                <div key={item.id} className="historyItem">
                  <div className="historyMeta">
                    <span>{item.when}</span>
                    <span>{item.file}</span>
                    <span>{item.dtype}</span>
                    <span>{item.elapsedMs.toFixed(0)} ms</span>
                  </div>
                  <pre>{item.text || '(empty text)'}</pre>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="result">
          <h2>Diagnostics</h2>
          {diagnostics.length === 0 ? (
            <pre>No diagnostics yet.</pre>
          ) : (
            <div className="historyList">
              {diagnostics.map((entry) => (
                <div key={entry.id} className="historyItem">
                  <div className="historyMeta">
                    <span>{entry.timestamp}</span>
                    <span>{entry.level}</span>
                    <span>{entry.event}</span>
                  </div>
                  {entry.payload ? <pre>{pretty(entry.payload)}</pre> : null}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

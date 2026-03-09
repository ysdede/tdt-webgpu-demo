const DEFAULT_BACKEND = 'transformers';
export const DEFAULT_CUSTOM_RESAMPLER_QUALITY = 'linear';
export const CUSTOM_RESAMPLER_QUALITY_OPTIONS = [
  { value: 'linear', label: 'linear parity (Node/Python)' },
  { value: 'balanced', label: 'sinc medium' },
  { value: 'fast', label: 'sinc fastest' },
  { value: 'best', label: 'sinc best' },
  { value: 'hold', label: 'zero-order hold' },
];
const WAV_HEADER_SIZE = 12;
const LANCZOS_A = 3;
const resamplerCache = new Map();
let libsamplerateModulePromise = null;

function nowMs() {
  return globalThis?.performance?.now?.() ?? Date.now();
}

function isUrlLike(input) {
  return typeof input === 'string' || input instanceof URL;
}

async function readInputArrayBuffer(input) {
  if (input instanceof ArrayBuffer) return input;
  if (ArrayBuffer.isView(input)) {
    const { buffer, byteOffset, byteLength } = input;
    return buffer.slice(byteOffset, byteOffset + byteLength);
  }
  if (isUrlLike(input)) {
    const response = await fetch(input);
    if (!response.ok) {
      throw new Error(`Audio fetch failed: ${response.status}`);
    }
    return await response.arrayBuffer();
  }
  if (input instanceof Blob) {
    return await input.arrayBuffer();
  }
  throw new Error(`Unsupported audio input: ${Object.prototype.toString.call(input)}`);
}

function getInputName(input) {
  if (typeof input === 'string') return input;
  if (input instanceof URL) return String(input);
  if (typeof input?.name === 'string') return input.name;
  return '';
}

function getInputMimeType(input) {
  return typeof input?.type === 'string' ? input.type.toLowerCase() : '';
}

function readAscii(view, offset, length) {
  let out = '';
  for (let i = 0; i < length; i += 1) {
    out += String.fromCharCode(view.getUint8(offset + i));
  }
  return out;
}

function detectWavInput(arrayBuffer, input) {
  const view = new DataView(arrayBuffer);
  const headerLooksLikeWav = arrayBuffer.byteLength >= WAV_HEADER_SIZE
    && readAscii(view, 0, 4) === 'RIFF'
    && readAscii(view, 8, 4) === 'WAVE';
  if (headerLooksLikeWav) return true;

  const mime = getInputMimeType(input);
  if (mime === 'audio/wav' || mime === 'audio/x-wav' || mime === 'audio/wave') return true;

  const name = getInputName(input).toLowerCase();
  return name.endsWith('.wav');
}

function sinc(x) {
  if (x === 0) return 1;
  const pix = Math.PI * x;
  return Math.sin(pix) / pix;
}

function lanczosKernel(x, a = LANCZOS_A) {
  const absX = Math.abs(x);
  if (absX >= a) return 0;
  return sinc(x) * sinc(x / a);
}

async function getLibsamplerateModule() {
  if (!libsamplerateModulePromise) {
    libsamplerateModulePromise = import('@alexanderolsen/libsamplerate-js');
  }
  return await libsamplerateModulePromise;
}

function normalizeCustomResamplerQuality(quality) {
  return CUSTOM_RESAMPLER_QUALITY_OPTIONS.some((option) => option.value === quality)
    ? quality
    : DEFAULT_CUSTOM_RESAMPLER_QUALITY;
}

function resampleLinear(audio, fromRate, toRate) {
  if (fromRate === toRate) return audio instanceof Float32Array ? audio : new Float32Array(audio);
  if (!Number.isFinite(fromRate) || !Number.isFinite(toRate) || fromRate <= 0 || toRate <= 0) {
    throw new Error(`Invalid resample rates: from=${fromRate}, to=${toRate}`);
  }

  const ratio = toRate / fromRate;
  const outLength = Math.max(1, Math.round(audio.length * ratio));
  const out = new Float32Array(outLength);
  const scale = fromRate / toRate;
  for (let i = 0; i < outLength; i += 1) {
    const pos = i * scale;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, audio.length - 1);
    const frac = pos - i0;
    out[i] = audio[i0] * (1 - frac) + audio[i1] * frac;
  }
  return out;
}

function resampleLanczos(audio, fromRate, toRate, a = LANCZOS_A) {
  if (fromRate === toRate) return audio instanceof Float32Array ? audio : new Float32Array(audio);
  if (!Number.isFinite(fromRate) || !Number.isFinite(toRate) || fromRate <= 0 || toRate <= 0) {
    throw new Error(`Invalid resample rates: from=${fromRate}, to=${toRate}`);
  }

  const ratio = toRate / fromRate;
  const outLength = Math.max(1, Math.round(audio.length * ratio));
  const out = new Float32Array(outLength);
  const scale = Math.min(1, ratio);
  const support = a / scale;

  for (let i = 0; i < outLength; i += 1) {
    const sourcePos = i / ratio;
    const left = Math.max(0, Math.ceil(sourcePos - support));
    const right = Math.min(audio.length - 1, Math.floor(sourcePos + support));
    let sum = 0;
    let norm = 0;

    for (let j = left; j <= right; j += 1) {
      const x = (sourcePos - j) * scale;
      const weight = lanczosKernel(x, a);
      if (weight === 0) continue;
      sum += audio[j] * weight;
      norm += weight;
    }

    out[i] = norm !== 0 ? sum / norm : audio[Math.max(0, Math.min(audio.length - 1, Math.round(sourcePos)))];
  }

  return out;
}

function getLibsamplerateConverterType(ConverterType, quality) {
  switch (quality) {
    case 'best':
      return ConverterType.SRC_SINC_BEST_QUALITY;
    case 'balanced':
      return ConverterType.SRC_SINC_MEDIUM_QUALITY;
    case 'fast':
      return ConverterType.SRC_SINC_FASTEST;
    case 'hold':
      return ConverterType.SRC_ZERO_ORDER_HOLD;
    default:
      return ConverterType.SRC_SINC_MEDIUM_QUALITY;
  }
}

async function resampleWithStrategy(audio, fromRate, toRate, quality = DEFAULT_CUSTOM_RESAMPLER_QUALITY) {
  if (fromRate === toRate) {
    return {
      audio: audio instanceof Float32Array ? audio : new Float32Array(audio),
      resampler: 'none',
      resamplerQuality: normalizeCustomResamplerQuality(quality),
    };
  }

  const normalizedQuality = normalizeCustomResamplerQuality(quality);
  if (normalizedQuality === 'linear') {
    return {
      audio: resampleLinear(audio, fromRate, toRate),
      resampler: 'linear-parity',
      resamplerQuality: normalizedQuality,
    };
  }

  const cacheKey = `${normalizedQuality}:1:${fromRate}:${toRate}`;
  let converterPromise = resamplerCache.get(cacheKey);
  if (!converterPromise) {
    const { ConverterType, create } = await getLibsamplerateModule();
    converterPromise = create(1, fromRate, toRate, {
      converterType: getLibsamplerateConverterType(ConverterType, normalizedQuality),
    });
    resamplerCache.set(cacheKey, converterPromise);
  }

  try {
    const converter = await converterPromise;
    return {
      audio: converter.simple(audio),
      resampler: 'libsamplerate-js',
      resamplerQuality: normalizedQuality,
    };
  } catch (error) {
    console.warn('[audio-prep] libsamplerate-js failed, falling back to Lanczos resampler:', error);
    resamplerCache.delete(cacheKey);
    return {
      audio: resampleLanczos(audio, fromRate, toRate),
      resampler: 'lanczos-fallback',
      resamplerQuality: normalizedQuality,
    };
  }
}

function resolveWaveFormatTag(view, fmtOffset, audioFormat, fmtChunkSize) {
  if (audioFormat !== 0xfffe) return audioFormat;
  if (fmtChunkSize < 40) {
    throw new Error('Unsupported WAV extensible format: fmt chunk too small.');
  }
  return view.getUint16(fmtOffset + 24, true);
}

function pcmSampleToFloat(view, offset, bitsPerSample) {
  switch (bitsPerSample) {
    case 8:
      return (view.getUint8(offset) - 128) / 128;
    case 16:
      return view.getInt16(offset, true) / 32768;
    case 24: {
      const b0 = view.getUint8(offset);
      const b1 = view.getUint8(offset + 1);
      const b2 = view.getUint8(offset + 2);
      let value = b0 | (b1 << 8) | (b2 << 16);
      if (value & 0x800000) value |= ~0xffffff;
      return value / 8388608;
    }
    case 32:
      return view.getInt32(offset, true) / 2147483648;
    default:
      throw new Error(`Unsupported PCM WAV bits per sample: ${bitsPerSample}`);
  }
}

function floatSampleToFloat(view, offset, bitsPerSample) {
  if (bitsPerSample === 32) return view.getFloat32(offset, true);
  if (bitsPerSample === 64) return view.getFloat64(offset, true);
  throw new Error(`Unsupported float WAV bits per sample: ${bitsPerSample}`);
}

function downmixFrame(samples) {
  if (samples.length === 1) return samples[0];
  if (samples.length === 2) {
    return (Math.SQRT2 * (samples[0] + samples[1])) / 2;
  }
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) sum += samples[i];
  return sum / samples.length;
}

function decodeWavArrayBuffer(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  if (arrayBuffer.byteLength < WAV_HEADER_SIZE) {
    throw new Error('Invalid WAV: file too small.');
  }
  if (readAscii(view, 0, 4) !== 'RIFF' || readAscii(view, 8, 4) !== 'WAVE') {
    throw new Error('Invalid WAV: expected RIFF/WAVE header.');
  }

  let offset = WAV_HEADER_SIZE;
  let fmt = null;
  let dataOffset = null;
  let dataSize = null;

  while (offset + 8 <= arrayBuffer.byteLength) {
    const chunkId = readAscii(view, offset, 4);
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataStart = offset + 8;
    const nextOffset = chunkDataStart + chunkSize + (chunkSize % 2);

    if (chunkId === 'fmt ') {
      fmt = {
        audioFormat: view.getUint16(chunkDataStart, true),
        numChannels: view.getUint16(chunkDataStart + 2, true),
        sampleRate: view.getUint32(chunkDataStart + 4, true),
        bitsPerSample: view.getUint16(chunkDataStart + 14, true),
        chunkSize,
        chunkOffset: chunkDataStart,
      };
    } else if (chunkId === 'data') {
      dataOffset = chunkDataStart;
      dataSize = chunkSize;
    }

    offset = nextOffset;
  }

  if (!fmt || dataOffset == null || dataSize == null) {
    throw new Error('Invalid WAV: missing fmt or data chunk.');
  }

  const audioFormat = resolveWaveFormatTag(view, fmt.chunkOffset, fmt.audioFormat, fmt.chunkSize);
  const bytesPerSample = fmt.bitsPerSample / 8;
  const totalSamples = Math.floor(dataSize / bytesPerSample);
  const totalFrames = Math.floor(totalSamples / fmt.numChannels);
  const mono = new Float32Array(totalFrames);
  const frameSamples = new Array(fmt.numChannels);

  let sampleOffset = dataOffset;
  for (let frame = 0; frame < totalFrames; frame += 1) {
    for (let channel = 0; channel < fmt.numChannels; channel += 1) {
      frameSamples[channel] = audioFormat === 3
        ? floatSampleToFloat(view, sampleOffset, fmt.bitsPerSample)
        : pcmSampleToFloat(view, sampleOffset, fmt.bitsPerSample);
      sampleOffset += bytesPerSample;
    }
    mono[frame] = downmixFrame(frameSamples);
  }

  return {
    audio: mono,
    sampleRate: fmt.sampleRate,
    numChannels: fmt.numChannels,
    strategy: 'wav-parser',
  };
}

function audioBufferToMono(audioBuffer) {
  const channels = audioBuffer.numberOfChannels;
  const mono = new Float32Array(audioBuffer.length);

  if (channels === 1) {
    mono.set(audioBuffer.getChannelData(0));
    return mono;
  }

  if (channels === 2) {
    const left = audioBuffer.getChannelData(0);
    const right = audioBuffer.getChannelData(1);
    for (let i = 0; i < audioBuffer.length; i += 1) {
      mono[i] = (Math.SQRT2 * (left[i] + right[i])) / 2;
    }
    return mono;
  }

  for (let channel = 0; channel < channels; channel += 1) {
    const data = audioBuffer.getChannelData(channel);
    for (let i = 0; i < audioBuffer.length; i += 1) {
      mono[i] += data[i] / channels;
    }
  }
  return mono;
}

async function decodeAudioAtContextRate(arrayBuffer, sampleRate = undefined) {
  const Ctx = globalThis.AudioContext || globalThis.webkitAudioContext;
  if (!Ctx) throw new Error('AudioContext is not available in this environment.');
  const ctx = sampleRate ? new Ctx({ sampleRate }) : new Ctx();
  try {
    return await ctx.decodeAudioData(arrayBuffer.slice(0));
  } finally {
    await ctx.close();
  }
}

async function prepareViaDemo(arrayBuffer, targetSampleRate) {
  const decodeStart = nowMs();
  const audioBuffer = await decodeAudioAtContextRate(arrayBuffer, targetSampleRate);
  const decodeMs = nowMs() - decodeStart;

  const downmixStart = nowMs();
  const audio = audioBufferToMono(audioBuffer);
  const downmixMs = nowMs() - downmixStart;

  return {
    audio,
    profile: {
      backend: 'demo',
      strategy: 'audiocontext-target-rate',
      inputSampleRate: audioBuffer.sampleRate,
      outputSampleRate: targetSampleRate,
      decodeMs,
      downmixMs,
      resampleMs: 0,
      resampler: 'browser',
      resamplerQuality: null,
      totalMs: decodeMs + downmixMs,
    },
  };
}

async function prepareViaCustomJs(arrayBuffer, input, targetSampleRate, resamplerQuality) {
  let decoded;
  let decodeMs = 0;
  let downmixMs = 0;

  if (detectWavInput(arrayBuffer, input)) {
    const decodeStart = nowMs();
    decoded = decodeWavArrayBuffer(arrayBuffer);
    decodeMs = nowMs() - decodeStart;
  } else {
    const decodeStart = nowMs();
    const audioBuffer = await decodeAudioAtContextRate(arrayBuffer);
    decodeMs = nowMs() - decodeStart;

    const downmixStart = nowMs();
    const audio = audioBufferToMono(audioBuffer);
    downmixMs = nowMs() - downmixStart;
    decoded = {
      audio,
      sampleRate: audioBuffer.sampleRate,
      strategy: 'browser-codec-native-rate',
    };
  }

  const resampleStart = nowMs();
  const { audio, resampler, resamplerQuality: resolvedResamplerQuality } = await resampleWithStrategy(
    decoded.audio,
    decoded.sampleRate,
    targetSampleRate,
    resamplerQuality,
  );
  const resampleMs = nowMs() - resampleStart;

  return {
    audio,
    profile: {
      backend: 'custom-js',
      strategy: decoded.strategy,
      inputSampleRate: decoded.sampleRate,
      outputSampleRate: targetSampleRate,
      decodeMs,
      downmixMs,
      resampleMs,
      resampler,
      resamplerQuality: resolvedResamplerQuality,
      totalMs: decodeMs + downmixMs + resampleMs,
    },
  };
}

async function prepareViaTransformers(input, targetSampleRate, readAudio) {
  if (typeof readAudio !== 'function') {
    throw new Error('transformers.js read_audio() is not available in this build.');
  }

  let source = input;
  let revokeUrl = null;

  if (!isUrlLike(input)) {
    if (!(input instanceof Blob)) {
      throw new Error(`Unsupported audio input for transformers.js audio prep: ${Object.prototype.toString.call(input)}`);
    }
    revokeUrl = URL.createObjectURL(input);
    source = revokeUrl;
  }

  const start = nowMs();
  try {
    const audio = await readAudio(source, targetSampleRate);
    const totalMs = nowMs() - start;
    return {
      audio,
      profile: {
        backend: 'transformers',
        strategy: 'transformers-read_audio',
        inputSampleRate: null,
        outputSampleRate: targetSampleRate,
        decodeMs: totalMs,
        downmixMs: null,
        resampleMs: null,
        resampler: 'browser',
        resamplerQuality: null,
        totalMs,
      },
    };
  } finally {
    if (revokeUrl) {
      URL.revokeObjectURL(revokeUrl);
    }
  }
}

export async function prepareBrowserAudioInput(input, {
  targetSampleRate,
  backend = DEFAULT_BACKEND,
  readAudio = null,
  customResamplerQuality = DEFAULT_CUSTOM_RESAMPLER_QUALITY,
} = {}) {
  if (input instanceof Float32Array) {
    return {
      audio: input,
      profile: {
        backend: 'pcm',
        strategy: 'precomputed-float32',
        inputSampleRate: targetSampleRate,
        outputSampleRate: targetSampleRate,
        decodeMs: 0,
        downmixMs: 0,
        resampleMs: 0,
        resampler: 'none',
        resamplerQuality: null,
        totalMs: 0,
      },
    };
  }

  if (!Number.isFinite(targetSampleRate) || targetSampleRate <= 0) {
    throw new Error(`Invalid target sample rate: ${targetSampleRate}`);
  }

  if (backend === 'transformers') {
    return await prepareViaTransformers(input, targetSampleRate, readAudio);
  }

  const arrayBuffer = await readInputArrayBuffer(input);
  if (backend === 'custom-js') {
    return await prepareViaCustomJs(arrayBuffer, input, targetSampleRate, customResamplerQuality);
  }

  return await prepareViaDemo(arrayBuffer, targetSampleRate);
}

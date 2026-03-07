const roundValue = (value, digits = 2) => {
  if (!Number.isFinite(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
};

const average = (values) => {
  if (!Array.isArray(values) || values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
};

const median = (values) => {
  if (!Array.isArray(values) || values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
};

const quantile = (values, ratio) => {
  if (!Array.isArray(values) || values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * ratio;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
};

const distributionSummary = (values, digits = 2) => {
  const valid = values.filter(Number.isFinite);
  if (valid.length === 0) return null;
  return {
    min: roundValue(Math.min(...valid), digits),
    q1: roundValue(quantile(valid, 0.25), digits),
    median: roundValue(quantile(valid, 0.5), digits),
    q3: roundValue(quantile(valid, 0.75), digits),
    max: roundValue(Math.max(...valid), digits),
  };
};

const effectiveRtfxForOverlap = (rawRtfx, durationSec, overlapSec) => {
  if (!Number.isFinite(rawRtfx) || !Number.isFinite(durationSec) || durationSec <= 0 || !Number.isFinite(overlapSec)) {
    return null;
  }
  const effectiveProgressSec = Math.max(0, durationSec - Math.max(0, overlapSec));
  if (effectiveProgressSec <= 0) return null;
  return rawRtfx * (effectiveProgressSec / durationSec);
};

export function simulateOverlappedWindowInference({
  totalAudioSec,
  windowSec,
  overlapSec,
  rawRtfx,
}) {
  const safeTotalAudioSec = Math.max(0, Number(totalAudioSec) || 0);
  const safeWindowSec = Number(windowSec);
  const safeOverlapSec = Math.max(0, Number(overlapSec) || 0);
  const safeRawRtfx = Number(rawRtfx);

  if (!Number.isFinite(safeWindowSec) || safeWindowSec <= 0 || !Number.isFinite(safeRawRtfx) || safeRawRtfx <= 0) {
    return null;
  }

  const advanceSec = safeWindowSec - safeOverlapSec;
  if (!Number.isFinite(advanceSec) || advanceSec <= 0) return null;

  let coveredSec = 0;
  let processedSec = 0;
  let steps = 0;

  while (coveredSec < safeTotalAudioSec) {
    processedSec += safeWindowSec;
    coveredSec += advanceSec;
    steps += 1;
  }

  const totalTimeSec = processedSec / safeRawRtfx;
  const effectiveRtfx = totalTimeSec > 0 ? safeTotalAudioSec / totalTimeSec : null;
  const overheadSec = Math.max(0, processedSec - safeTotalAudioSec);
  const overheadRatio = safeTotalAudioSec > 0 ? overheadSec / safeTotalAudioSec : 0;

  return {
    totalAudioSec: roundValue(safeTotalAudioSec, 2),
    windowSec: roundValue(safeWindowSec, 2),
    overlapSec: roundValue(safeOverlapSec, 2),
    advanceSec: roundValue(advanceSec, 2),
    steps,
    processedSec: roundValue(processedSec, 2),
    overheadSec: roundValue(overheadSec, 2),
    overheadRatio: roundValue(overheadRatio, 6),
    totalTimeSec: roundValue(totalTimeSec, 4),
    rawRtfx: roundValue(safeRawRtfx, 4),
    effectiveRtfx: roundValue(effectiveRtfx, 4),
  };
}

const escapeCsv = (value) => {
  const text = value == null ? '' : String(value);
  if (!/[",\n]/.test(text)) return text;
  return `"${text.replace(/"/g, '""')}"`;
};

export function formatSeconds(value) {
  if (!Number.isFinite(value)) return '-';
  if (value < 60) return `${value.toFixed(1)}s`;
  const minutes = Math.floor(value / 60);
  const seconds = value % 60;
  return `${minutes}m ${seconds.toFixed(1)}s`;
}

export function getDurationBucket(seconds, bucketSize = 30) {
  const safeBucket = Math.max(5, Number(bucketSize) || 30);
  const start = Math.floor(Math.max(seconds, 0) / safeBucket) * safeBucket;
  const end = start + safeBucket;
  return {
    start,
    end,
    label: `${formatSeconds(start)} - ${formatSeconds(end)}`,
  };
}

export function parseDurationTargets(value) {
  return Array.from(
    new Set(
      String(value ?? '')
        .split(/[\s,;]+/)
        .map((part) => Number(part.trim()))
        .filter((part) => Number.isFinite(part) && part > 0)
        .map((part) => Math.round(part * 1000) / 1000)
    )
  ).sort((a, b) => a - b);
}

export function buildTargetSamplePlan(catalog, targets, samplesPerTarget = 1) {
  const requestedTargets = Array.isArray(targets) ? targets : [];
  const safeSamplesPerTarget = Math.max(1, Math.floor(Number(samplesPerTarget) || 1));
  const used = new Set();
  const plan = [];

  for (const targetSec of requestedTargets) {
    const ranked = catalog
      .filter((entry) => Number.isFinite(entry?.durationSec))
      .map((entry) => ({
        ...entry,
        deltaSec: Math.abs(entry.durationSec - targetSec),
      }))
      .sort((a, b) => {
        if (a.deltaSec !== b.deltaSec) return a.deltaSec - b.deltaSec;
        return a.durationSec - b.durationSec;
      });

    let picked = 0;
    for (const candidate of ranked) {
      if (used.has(candidate.path)) continue;
      used.add(candidate.path);
      plan.push({
        targetSec,
        ...candidate,
      });
      picked += 1;
      if (picked >= safeSamplesPerTarget) break;
    }
  }

  return plan;
}

function createSeededRng(seed) {
  let state = Math.floor(Number(seed) || Date.now()) % 2147483647;
  if (state <= 0) state += 2147483646;
  return () => {
    state = (state * 16807) % 2147483647;
    return (state - 1) / 2147483646;
  };
}

function distributeQuota(total, buckets) {
  const quotas = new Map(buckets.map((bucket) => [bucket.key, bucket.minimum]));
  let remaining = Math.max(0, total - buckets.reduce((sum, bucket) => sum + bucket.minimum, 0));
  let eligible = buckets
    .filter((bucket) => bucket.capacity > bucket.minimum)
    .map((bucket) => ({
      ...bucket,
      extraCapacity: bucket.capacity - bucket.minimum,
    }));

  while (remaining > 0 && eligible.length > 0) {
    const weightSum = eligible.reduce((sum, bucket) => sum + bucket.weight, 0) || eligible.length;
    let assignedThisRound = 0;

    for (const bucket of eligible) {
      const rawShare = remaining * (bucket.weight / weightSum);
      const extra = Math.min(bucket.extraCapacity, Math.floor(rawShare));
      if (extra <= 0) continue;
      quotas.set(bucket.key, (quotas.get(bucket.key) ?? 0) + extra);
      bucket.extraCapacity -= extra;
      assignedThisRound += extra;
    }

    remaining -= assignedThisRound;
    eligible = eligible.filter((bucket) => bucket.extraCapacity > 0);
    if (remaining <= 0 || eligible.length === 0) break;

    eligible.sort((a, b) => {
      const aRemainder = (remaining * (a.weight / weightSum)) % 1;
      const bRemainder = (remaining * (b.weight / weightSum)) % 1;
      if (aRemainder !== bRemainder) return bRemainder - aRemainder;
      if (a.weight !== b.weight) return b.weight - a.weight;
      return a.start - b.start;
    });

    for (const bucket of eligible) {
      if (remaining <= 0) break;
      if (bucket.extraCapacity <= 0) continue;
      quotas.set(bucket.key, (quotas.get(bucket.key) ?? 0) + 1);
      bucket.extraCapacity -= 1;
      remaining -= 1;
    }

    eligible = eligible.filter((bucket) => bucket.extraCapacity > 0);
  }

  return quotas;
}

export function buildRandomSamplePlan(catalog, sampleCount = 12, seed = Date.now(), bucketSizeSec = 30) {
  const safeCount = Math.max(1, Math.floor(Number(sampleCount) || 1));
  const entries = catalog.filter((entry) => Number.isFinite(entry?.durationSec) && entry.durationSec > 0);
  if (entries.length === 0) return [];
  const sorted = [...entries].sort((a, b) => a.durationSec - b.durationSec);
  const rng = createSeededRng(seed);
  const picked = [];
  const used = new Set();
  const selectedCountsByBucket = new Map();

  const pushEntry = (entry) => {
    if (!entry || used.has(entry.path)) return false;
    used.add(entry.path);
    picked.push({
      ...entry,
      duplicateOfPath: null,
      duplicateOrdinal: null,
    });
    const bucket = getDurationBucket(entry.durationSec, bucketSizeSec);
    const bucketKey = `${bucket.start}-${bucket.end}`;
    selectedCountsByBucket.set(bucketKey, (selectedCountsByBucket.get(bucketKey) ?? 0) + 1);
    return true;
  };

  // Always cover both edges so the random run preserves the true duration range.
  pushEntry(sorted[0]);
  if (safeCount > 1) pushEntry(sorted[sorted.length - 1]);
  if (safeCount > 2) pushEntry(sorted[1]);
  if (safeCount > 3) pushEntry(sorted[sorted.length - 2]);

  const remaining = Math.max(0, Math.min(safeCount, sorted.length) - picked.length);
  if (remaining > 0) {
    // Balance against the same fixed-second bucket scheme used by the UI summaries.
    const bucketsByKey = new Map();
    for (const entry of sorted) {
      const bucket = getDurationBucket(entry.durationSec, bucketSizeSec);
      const key = `${bucket.start}-${bucket.end}`;
      if (!bucketsByKey.has(key)) {
        bucketsByKey.set(key, []);
      }
      bucketsByKey.get(key).push(entry);
    }
    const buckets = Array.from(bucketsByKey.entries())
      .sort((a, b) => {
        const [aStart] = a[0].split('-').map(Number);
        const [bStart] = b[0].split('-').map(Number);
        return aStart - bStart;
      })
      .map(([key, bucket]) => ({
        key,
        start: Number(key.split('-')[0]),
        entries: bucket,
      }));

    for (const bucket of buckets) {
      for (let i = bucket.entries.length - 1; i > 0; i -= 1) {
        const j = Math.floor(rng() * (i + 1));
        [bucket.entries[i], bucket.entries[j]] = [bucket.entries[j], bucket.entries[i]];
      }
    }

    const anchorCountsByBucket = new Map();
    for (const entry of picked) {
      const bucket = getDurationBucket(entry.durationSec, bucketSizeSec);
      const key = `${bucket.start}-${bucket.end}`;
      anchorCountsByBucket.set(key, (anchorCountsByBucket.get(key) ?? 0) + 1);
    }

    const lastIndex = Math.max(1, buckets.length - 1);
    const weightedBuckets = buckets.map((bucket, index) => {
      const progress = index / lastIndex;
      const frontBias = progress <= 0.12
        ? 0.84 + (progress / 0.12) * 0.12
        : 1;
      const tailBias = progress <= 0.65
        ? 1
        : 1 - ((progress - 0.65) / 0.35) * 0.5;
      return {
        ...bucket,
        capacity: bucket.entries.length,
        minimum: anchorCountsByBucket.get(bucket.key) ?? 0,
        weight: Math.max(0.45, frontBias * tailBias),
      };
    });

    const quotas = distributeQuota(Math.min(safeCount, sorted.length), weightedBuckets);

    for (const bucket of weightedBuckets) {
      const quota = quotas.get(bucket.key) ?? 0;
      while ((selectedCountsByBucket.get(bucket.key) ?? 0) < quota) {
        const nextUniqueEntry = bucket.entries.find((entry) => !used.has(entry.path));
        if (!nextUniqueEntry) break;
        pushEntry(nextUniqueEntry);
      }
    }

    while (picked.length < safeCount) {
      const fallbackBucket = weightedBuckets
        .map((bucket) => ({
          ...bucket,
          selected: selectedCountsByBucket.get(bucket.key) ?? 0,
          available: bucket.entries.filter((entry) => !used.has(entry.path)).length,
          quota: quotas.get(bucket.key) ?? 0,
        }))
        .filter((bucket) => bucket.available > 0)
        .sort((a, b) => {
          const aDeficit = a.quota - a.selected;
          const bDeficit = b.quota - b.selected;
          if (aDeficit !== bDeficit) return bDeficit - aDeficit;
          if (a.weight !== b.weight) return b.weight - a.weight;
          return a.start - b.start;
        })[0];

      if (!fallbackBucket) break;
      const nextEntry = fallbackBucket.entries.find((entry) => !used.has(entry.path));
      if (!nextEntry) break;
      pushEntry(nextEntry);
    }
  }

  return picked
    .sort((a, b) => a.durationSec - b.durationSec)
    .map((entry, index) => ({
      ...entry,
      targetSec: null,
      deltaSec: 0,
      randomIndex: index + 1,
    }));
}

function roundTarget(value) {
  if (!Number.isFinite(value) || value <= 0) return null;
  if (value < 12) return Math.round(value);
  if (value < 60) return Math.round(value / 2) * 2;
  if (value < 180) return Math.round(value / 5) * 5;
  return Math.round(value);
}

export function mergeTargetDurations(...targetLists) {
  return Array.from(
    new Set(
      targetLists
        .flat()
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value) && value > 0)
        .map((value) => Math.round(value * 1000) / 1000)
    )
  ).sort((a, b) => a - b);
}

export function suggestDurationTargets(catalog, {
  minSec = 2,
  maxPoints = 24,
  growthFactor = Math.sqrt(1.8),
} = {}) {
  const safeMaxPoints = Math.max(1, Math.floor(Number(maxPoints) || 1));
  const durations = catalog
    .map((entry) => entry?.durationSec)
    .filter((value) => Number.isFinite(value) && value > 0)
    .sort((a, b) => a - b);

  if (durations.length === 0) return [];

  const maxSec = durations[durations.length - 1];
  const targets = [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64].filter((value) => value >= minSec && value <= maxSec);

  const quantiles = [0.08, 0.16, 0.25, 0.35, 0.5, 0.62, 0.72, 0.8, 0.88, 0.93, 0.97, 1];
  for (const q of quantiles) {
    const index = Math.min(durations.length - 1, Math.max(0, Math.floor((durations.length - 1) * q)));
    const rounded = roundTarget(durations[index]);
    if (rounded != null) {
      targets.push(rounded);
    }
  }

  let current = minSec;
  while (current < maxSec && targets.length < safeMaxPoints - 1) {
    const rounded = roundTarget(current);
    if (rounded != null) {
      targets.push(rounded);
    }
    current *= growthFactor;
  }
  targets.push(roundTarget(maxSec));

  // Densify the long tail explicitly so sparse long-form coverage does not
  // collapse to only a couple of far-apart targets near the max duration.
  if (maxSec >= 180) {
    const tailStart = Math.max(180, maxSec * 0.45);
    const tailFractions = [0, 0.18, 0.36, 0.54, 0.72, 0.86, 1];
    for (const fraction of tailFractions) {
      const candidate = roundTarget(tailStart + (maxSec - tailStart) * fraction);
      if (candidate != null) {
        targets.push(candidate);
      }
    }
  }

  const merged = mergeTargetDurations(targets);
  if (merged.length <= safeMaxPoints) return merged;

  // Uniformly downsample to safeMaxPoints, always keeping first and last.
  if (safeMaxPoints === 1) return [merged[0]];
  const lastIndex = merged.length - 1;
  return Array.from({ length: safeMaxPoints }, (_, i) => {
    const sourceIndex = Math.round((i / (safeMaxPoints - 1)) * lastIndex);
    return merged[sourceIndex];
  }).filter((value, i, arr) => i === 0 || value !== arr[i - 1]);
}

export function refineTargetsFromRuns(runs, {
  minGapSec = 12,
  maxInsertions = 8,
} = {}) {
  const validRuns = runs
    .filter((run) => !run?.error && Number.isFinite(run?.durationSec))
    .sort((a, b) => a.durationSec - b.durationSec);

  if (validRuns.length < 2) return [];

  const candidates = [];
  for (let i = 1; i < validRuns.length; i += 1) {
    const left = validRuns[i - 1];
    const right = validRuns[i];
    const gapSec = right.durationSec - left.durationSec;
    if (gapSec < minGapSec) continue;

    const leftRtfx = left.modelRtfx ?? left.wallRtfx ?? null;
    const rightRtfx = right.modelRtfx ?? right.wallRtfx ?? null;
    const dropRatio =
      Number.isFinite(leftRtfx) && Number.isFinite(rightRtfx) && leftRtfx > 0
        ? (leftRtfx - rightRtfx) / leftRtfx
        : 0;

    candidates.push({
      score: gapSec + Math.max(0, dropRatio) * 100,
      midpoint: roundTarget((left.durationSec + right.durationSec) / 2),
    });
  }

  return mergeTargetDurations(
    candidates
      .sort((a, b) => b.score - a.score)
      .slice(0, maxInsertions)
      .map((candidate) => candidate.midpoint)
  );
}

export function summarizeBenchmarkRuns(runs, bucketSize = 30, overlapSec = 6) {
  const safeOverlapSec = Math.max(0, Number(overlapSec) || 0);
  const simulatedReferenceAudioSec = 90 * 60;
  const grouped = new Map();

  for (const run of runs) {
    if (run?.error || !Number.isFinite(run?.durationSec)) continue;
    const bucket = getDurationBucket(run.durationSec, bucketSize);
    const key = `${bucket.start}-${bucket.end}`;
    if (!grouped.has(key)) {
      grouped.set(key, {
        ...bucket,
        runs: [],
      });
    }
    grouped.get(key).runs.push(run);
  }

  return Array.from(grouped.values())
    .map((bucket) => {
      const wallRtfx = bucket.runs.map((run) => run.wallRtfx).filter(Number.isFinite);
      const modelRtfx = bucket.runs.map((run) => run.modelRtfx).filter(Number.isFinite);
      const encodeMs = bucket.runs.map((run) => run.encodeMs).filter(Number.isFinite);
      const decodeMs = bucket.runs.map((run) => run.decodeMs).filter(Number.isFinite);
      const totalModelMs = bucket.runs.map((run) => run.modelTotalMs).filter(Number.isFinite);
      const totalAudioSec = bucket.runs.reduce((sum, run) => sum + run.durationSec, 0);
      const wallRtfxDistribution = distributionSummary(wallRtfx, 2);
      const modelRtfxDistribution = distributionSummary(modelRtfx, 2);
      const avgModelRtfx = average(modelRtfx);
      const avgAudioSec = bucket.runs.length > 0 ? totalAudioSec / bucket.runs.length : null;
      const recommendedWindowSec = avgAudioSec;
      const effectiveModelRtfx = effectiveRtfxForOverlap(avgModelRtfx, recommendedWindowSec, safeOverlapSec);
      const simulation = simulateOverlappedWindowInference({
        totalAudioSec: simulatedReferenceAudioSec,
        windowSec: recommendedWindowSec,
        overlapSec: safeOverlapSec,
        rawRtfx: avgModelRtfx,
      });

      return {
        label: bucket.label,
        start: bucket.start,
        end: bucket.end,
        count: bucket.runs.length,
        totalAudioSec: roundValue(totalAudioSec, 2),
        avgAudioSec: roundValue(avgAudioSec, 2),
        avgWallRtfx: roundValue(average(wallRtfx), 2),
        medianWallRtfx: roundValue(median(wallRtfx), 2),
        wallRtfxDistribution,
        avgModelRtfx: roundValue(avgModelRtfx, 2),
        medianModelRtfx: roundValue(median(modelRtfx), 2),
        modelRtfxDistribution,
        effectiveOverlapSec: roundValue(safeOverlapSec, 2),
        recommendedWindowSec,
        effectiveModelRtfx: roundValue(effectiveModelRtfx, 2),
        simulatedReferenceAudioSec,
        simulatedSteps: simulation?.steps ?? null,
        simulatedProcessedSec: simulation?.processedSec ?? null,
        simulatedOverheadSec: simulation?.overheadSec ?? null,
        simulatedOverheadRatio: simulation?.overheadRatio != null ? roundValue(simulation.overheadRatio * 100, 2) : null,
        simulatedEffectiveModelRtfx: simulation?.effectiveRtfx != null ? roundValue(simulation.effectiveRtfx, 2) : null,
        avgEncodeMs: roundValue(average(encodeMs), 1),
        avgDecodeMs: roundValue(average(decodeMs), 1),
        avgModelTotalMs: roundValue(average(totalModelMs), 1),
      };
    })
    .sort((a, b) => a.start - b.start);
}

export function findEffectiveRtfxSweetSpot(summary, overlapSec = 6) {
  const safeOverlapSec = Math.max(0, Number(overlapSec) || 0);
  const buckets = Array.isArray(summary) ? summary : [];
  const ranked = buckets
    .map((bucket) => {
      const durationSec = bucket.recommendedWindowSec ?? bucket.avgAudioSec;
      const rawRtfx = bucket.avgModelRtfx;
      const simulation = simulateOverlappedWindowInference({
        totalAudioSec: bucket.simulatedReferenceAudioSec ?? (90 * 60),
        windowSec: durationSec,
        overlapSec: safeOverlapSec,
        rawRtfx,
      });
      if (!Number.isFinite(simulation?.effectiveRtfx)) return null;
      return {
        ...bucket,
        effectiveOverlapSec: roundValue(safeOverlapSec, 2),
        recommendedWindowSec: durationSec,
        effectiveModelRtfx: roundValue(effectiveRtfxForOverlap(rawRtfx, durationSec, safeOverlapSec), 2),
        simulatedSteps: simulation.steps,
        simulatedProcessedSec: simulation.processedSec,
        simulatedOverheadSec: simulation.overheadSec,
        simulatedOverheadRatio: simulation.overheadRatio != null ? roundValue(simulation.overheadRatio * 100, 2) : null,
        simulatedEffectiveModelRtfx: roundValue(simulation.effectiveRtfx, 2),
      };
    })
    .filter(Boolean)
    .sort((a, b) => {
      if (b.simulatedEffectiveModelRtfx !== a.simulatedEffectiveModelRtfx) {
        return b.simulatedEffectiveModelRtfx - a.simulatedEffectiveModelRtfx;
      }
      return a.start - b.start;
    });

  return ranked[0] ?? null;
}

export function summarizeBenchmarkOverall(runs) {
  const validRuns = runs.filter((run) => !run?.error && Number.isFinite(run?.durationSec));
  const wallRtfx = validRuns.map((run) => run.wallRtfx).filter(Number.isFinite);
  const modelRtfx = validRuns.map((run) => run.modelRtfx).filter(Number.isFinite);
  const encodeMs = validRuns.map((run) => run.encodeMs).filter(Number.isFinite);
  const decodeMs = validRuns.map((run) => run.decodeMs).filter(Number.isFinite);
  const modelTotalMs = validRuns.map((run) => run.modelTotalMs).filter(Number.isFinite);

  return {
    count: validRuns.length,
    failed: runs.length - validRuns.length,
    totalAudioSec: roundValue(validRuns.reduce((sum, run) => sum + run.durationSec, 0), 2),
    avgWallRtfx: roundValue(average(wallRtfx), 2),
    medianWallRtfx: roundValue(median(wallRtfx), 2),
    avgModelRtfx: roundValue(average(modelRtfx), 2),
    medianModelRtfx: roundValue(median(modelRtfx), 2),
    avgEncodeMs: roundValue(average(encodeMs), 1),
    avgDecodeMs: roundValue(average(decodeMs), 1),
    avgModelTotalMs: roundValue(average(modelTotalMs), 1),
  };
}

export function benchmarkSummaryToCsv(summary) {
  const headers = [
    'bucket_label',
    'start_sec',
    'end_sec',
    'count',
    'total_audio_sec',
    'avg_audio_sec',
    'avg_wall_rtfx',
    'median_wall_rtfx',
    'avg_model_rtfx',
    'effective_overlap_sec',
    'recommended_window_sec',
    'effective_model_rtfx',
    'simulated_reference_audio_sec',
    'simulated_steps',
    'simulated_processed_sec',
    'simulated_overhead_sec',
    'simulated_overhead_ratio_pct',
    'simulated_effective_model_rtfx',
    'median_model_rtfx',
    'avg_encode_ms',
    'avg_decode_ms',
    'avg_model_total_ms',
  ];

  const rows = (Array.isArray(summary) ? summary : []).map((bucket) => [
    bucket.label,
    bucket.start,
    bucket.end,
    bucket.count,
    bucket.totalAudioSec,
    bucket.avgAudioSec,
    bucket.avgWallRtfx,
    bucket.medianWallRtfx,
    bucket.avgModelRtfx,
    bucket.effectiveOverlapSec,
    bucket.recommendedWindowSec,
    bucket.effectiveModelRtfx,
    bucket.simulatedReferenceAudioSec,
    bucket.simulatedSteps,
    bucket.simulatedProcessedSec,
    bucket.simulatedOverheadSec,
    bucket.simulatedOverheadRatio,
    bucket.simulatedEffectiveModelRtfx,
    bucket.medianModelRtfx,
    bucket.avgEncodeMs,
    bucket.avgDecodeMs,
    bucket.avgModelTotalMs,
  ]);

  return [headers, ...rows]
    .map((row) => row.map(escapeCsv).join(','))
    .join('\n');
}

export function benchmarkRunsToCsv(runs) {
  const headers = [
    'name',
    'source_path',
    'size_bytes',
    'target_sec',
    'repeat_count',
    'base_repeat_count',
    'boosted_repeat_count',
    'repeat_boost_applied',
    'sampled_duration_sec',
    'sampled_delta_sec',
    'duration_sec',
    'mode',
    'text_len',
    'word_count',
    'token_count',
    'audio_decode_ms',
    'wall_ms',
    'wall_rtfx',
    'wall_rtfx_stddev',
    'model_total_ms',
    'model_rtfx',
    'model_rtfx_stddev',
    'encode_ms',
    'decode_ms',
    'feature_shape',
    'feature_frames',
    'feature_bins',
    'feature_mib',
    'attention_mask_mib',
    'encoder_payload_mib',
    'audio_pcm_mib',
    'tokenize_ms',
    'error',
  ];

  const rows = runs.map((run) => [
    run.name,
    run.sourcePath,
    run.sizeBytes,
    run.targetSec,
    run.repeatCount,
    run.baseRepeatCount,
    run.boostedRepeatCount,
    run.repeatBoostApplied,
    run.sampledDurationSec,
    run.sampledDeltaSec,
    run.durationSec,
    run.mode,
    run.textLen,
    run.wordCount,
    run.tokenCount,
    run.audioDecodeMs,
    run.wallMs,
    run.wallRtfx,
    run.wallRtfxStddev,
    run.modelTotalMs,
    run.modelRtfx,
    run.modelRtfxStddev,
    run.encodeMs,
    run.decodeMs,
    run.featureShape,
    run.featureFrames,
    run.featureBins,
    run.featureMiB,
    run.attentionMaskMiB,
    run.encoderPayloadMiB,
    run.audioPcmMiB,
    run.tokenizeMs,
    run.error ?? '',
  ]);

  return [headers, ...rows]
    .map((row) => row.map(escapeCsv).join(','))
    .join('\n');
}

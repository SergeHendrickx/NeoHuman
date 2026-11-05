// Small cross-device benchmark: fixed clean-room crowd scenario, sampled
// per-frame, with localStorage history + JSON export/import so runs from a
// phone or Quest can be compared against desktop runs.
import {
  clearAgents,
  isAgentSimulationActive,
  setAgentLodPin,
  setAutoLodEnabled,
  simulation,
  spawnAgents,
  toggleAgentSimulation,
} from "./crowd.js";
import * as THREE from "three";
import {
  controls,
  fitCamera,
  isFollowing,
  orthographicCamera,
  perspectiveCamera,
  readRenderExecMs,
  renderer,
  setFollow,
  syncCameraInputs,
} from "./scene.js";
import { state } from "./state.js";

const STORE_KEY = "mhr.bench.v1";
const MAX_RUNS = 50;
const MIN_WARMUP_MS = 2000;
const MAX_WARMUP_MS = 20000;
const RAMP_SETTLE_MS = 800;

let cancelRequested = false;

export function cancelBenchmark() {
  cancelRequested = true;
}

function percentile(sorted, q) {
  if (sorted.length === 0) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.ceil((q / 100) * sorted.length) - 1));
  return sorted[i];
}

function avg(nums) {
  if (nums.length === 0) return 0;
  let s = 0;
  for (const n of nums) s += n;
  return s / nums.length;
}

/** Pure summary over per-frame samples (unit-tested in node). */
export function summarizeSamples(samples) {
  const frames = samples.map((s) => s.frameMs).sort((a, b) => a - b);
  const mean = avg(frames);
  const nonNull = (pick) => samples.map(pick).filter((v) => v !== null && v !== undefined);
  return {
    frames: samples.length,
    fpsAvg: mean > 0 ? 1000 / mean : 0,
    frameAvg: mean,
    frameP50: percentile(frames, 50),
    frameP95: percentile(frames, 95),
    agentsAvg: avg(samples.map((s) => s.agentMs)),
    postAvg: avg(samples.map((s) => s.postMs)),
    uiAvg: avg(samples.map((s) => s.uiMs)),
    tickAvg: avg(samples.map((s) => s.agentMs + s.postMs + s.uiMs)),
    callsAvg: avg(samples.map((s) => s.calls)),
    trisAvg: avg(samples.map((s) => s.tris)),
    gpuExecAvg: avg(nonNull((s) => s.gpuExecMs)),
    gpuExecSamples: nonNull((s) => s.gpuExecMs).length,
    renderExecAvg: avg(nonNull((s) => s.renderExecMs)),
    renderExecSamples: nonNull((s) => s.renderExecMs).length,
  };
}

function shortGpu(full) {
  if (!full) return "unknown";
  const m = /ANGLE \(([^,)]+)/.exec(full) || /Adreno[^,)]*/.exec(full);
  if (m) return m[1] || m[0];
  const parts = full.split(",").map((p) => p.trim()).filter(Boolean);
  return (parts[0] || full).slice(0, 48);
}

/** Best-effort GPU identification: WebGPU adapter info, else WebGL debug ext. */
export function detectEngine() {
  const engine = state.isWebGPU ? "WebGPU" : "WebGL";
  let gpu = "";
  try {
    const info = renderer?.backend?.adapter?.info;
    if (info) gpu = info.description || info.device || "";
  } catch { /* adapter info unavailable */ }
  if (!gpu) {
    try {
      const canvas = document.createElement("canvas");
      const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
      const ext = gl?.getExtension("WEBGL_debug_renderer_info");
      gpu = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "";
    } catch { /* webgl probe unavailable */ }
  }
  return { engine, gpu: gpu || "unknown", gpuShort: shortGpu(gpu) };
}

function settled(targetCount) {
  const t = simulation.timings;
  return t.pairCount >= targetCount && (simulation.gpuStatus().pendingMigrations || 0) === 0;
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

// Benchmark framing: the landing shot is a face close-up, so the crowd
// spawns out of frame (and frustum culling then measures an empty scene).
// Frame the wander arena (+/-5.5m) for the run, restore everything after.
const CROWD_FRAME = { center: [0, 1.0, 0], size: 7 };

function frameCrowdForBenchmark() {
  const saved = {
    view: state.lastView,
    target: controls.target.clone(),
    follow: isFollowing(),
    persp: {
      pos: perspectiveCamera.position.clone(),
      near: perspectiveCamera.near,
      far: perspectiveCamera.far,
    },
    ortho: {
      pos: orthographicCamera.position.clone(),
      near: orthographicCamera.near,
      far: orthographicCamera.far,
      box: [
        orthographicCamera.left,
        orthographicCamera.right,
        orthographicCamera.top,
        orthographicCamera.bottom,
      ],
    },
  };
  setFollow(false); // A followed wanderer would drag the camera mid-run.
  fitCamera("front", {
    center: new THREE.Vector3(...CROWD_FRAME.center),
    maxDimension: CROWD_FRAME.size,
  });
  return saved;
}

function restoreCameraAfterBenchmark(saved) {
  if (!saved) return;
  state.lastView = saved.view;
  perspectiveCamera.position.copy(saved.persp.pos);
  perspectiveCamera.near = saved.persp.near;
  perspectiveCamera.far = saved.persp.far;
  perspectiveCamera.updateProjectionMatrix();
  orthographicCamera.position.copy(saved.ortho.pos);
  orthographicCamera.near = saved.ortho.near;
  orthographicCamera.far = saved.ortho.far;
  [orthographicCamera.left, orthographicCamera.right,
    orthographicCamera.top, orthographicCamera.bottom] = saved.ortho.box;
  orthographicCamera.updateProjectionMatrix();
  controls.target.copy(saved.target);
  controls.update();
  syncCameraInputs();
  setFollow(saved.follow);
}

function sampleFrame(prev) {
  const now = performance.now();
  const t = simulation.timings;
  const info = renderer?.info?.render;
  return {
    sample: {
      frameMs: prev === null ? 0 : now - prev,
      agentMs: t.agentMs,
      postMs: t.gpuMs,
      uiMs: t.uiMs,
      calls: info ? (info.drawCalls ?? info.calls ?? 0) : 0,
      tris: info ? info.triangles : 0,
      gpuExecMs: t.gpuExecMs,
      renderExecMs: readRenderExecMs(),
    },
    now,
  };
}

/**
 * Run one benchmark: clear scene, spawn `agents`, pin LOD unless "auto",
 * warm up until the pools settle, sample `seconds`, then restore.
 * Calls onPhase(text) for status; returns the run record (or null if cancelled).
 */
export async function runBenchmark({ agents, seconds, lod, device }, onPhase = () => {}) {
  cancelRequested = false;
  const count = Math.max(1, Math.min(2000, Math.floor(Number(agents) || 0)));
  const measureMs = Math.max(3000, Math.min(60000, (Number(seconds) || 10) * 1000));
  const pin = lod === "auto" ? null : Number(lod);
  const phase = (text) => onPhase(text);

  const preCount = simulation.agents.length;
  const preAutoLod = simulation.autoLod;
  const simWasRunning = isAgentSimulationActive();
  let savedCam = null;
  try {
    savedCam = frameCrowdForBenchmark();
    phase("clearing scene…");
    clearAgents();
    if (!isAgentSimulationActive()) toggleAgentSimulation();
    setAutoLodEnabled(pin === null);
    phase(`spawning ${count} agents…`);
    const spawned = await spawnAgents(count);
    if (cancelRequested) return null;
    if (pin !== null) {
      for (const agent of simulation.agents) setAgentLodPin(agent.id, pin);
    }
    const target = spawned?.added ?? count;
    phase("warming up…");
    const warmStart = performance.now();
    for (;;) {
      await nextFrame();
      if (cancelRequested) return null;
      const elapsed = performance.now() - warmStart;
      if (elapsed >= MIN_WARMUP_MS && settled(target)) break;
      if (elapsed >= MAX_WARMUP_MS) break;
      if (Math.floor(elapsed / 500) !== Math.floor((elapsed - 16) / 500)) {
        phase(`warming up… ${Math.floor(elapsed / 1000)}s`);
      }
    }
    phase(`measuring… 0s / ${Math.round(measureMs / 1000)}s`);
    const samples = [];
    let prev = null;
    const measureStart = performance.now();
    let frameSum = 0;
    let lastTick = 0;
    for (;;) {
      await nextFrame();
      if (cancelRequested) return null;
      const { sample, now } = sampleFrame(prev);
      prev = now;
      if (sample.frameMs > 0) {
        samples.push(sample);
        frameSum += sample.frameMs;
      }
      const elapsed = now - measureStart;
      const tick = Math.floor(elapsed / 1000);
      if (tick !== lastTick) {
        lastTick = tick;
        const fps = frameSum > 0 ? (1000 * samples.length) / frameSum : 0;
        phase(`measuring… ${tick}s / ${Math.round(measureMs / 1000)}s · ${fps.toFixed(0)} fps · ${simulation.agents.length} agents`);
      }
      if (elapsed >= measureMs) break;
    }
    const summary = summarizeSamples(samples);
    const lodMix = {};
    let cpu = 0;
    for (const agent of simulation.agents) {
      if (!agent.isGpuAgent) { cpu++; continue; }
      lodMix[agent.lod] = (lodMix[agent.lod] || 0) + 1;
    }
    const { engine, gpu, gpuShort } = detectEngine();
    const canvas = renderer?.domElement;
    return {
      id: `run-${Date.now().toString(36)}`,
      date: new Date().toISOString(),
      device: String(device || gpuShort || "device").slice(0, 32),
      engine,
      gpu,
      gpuShort,
      canvas: canvas ? { w: canvas.width, h: canvas.height, dpr: window.devicePixelRatio || 1 } : null,
      ua: navigator.userAgent.slice(0, 120),
      params: { agents: count, seconds: Math.round(measureMs / 1000), lod: pin === null ? "auto" : pin },
      ...summary,
      lodMix,
      cpu,
      gpuAgents: simulation.agents.length - cpu,
    };
  } finally {
    restoreCameraAfterBenchmark(savedCam);
    phase("restoring…");
    try {
      clearAgents();
      setAutoLodEnabled(preAutoLod);
      if (isAgentSimulationActive() !== simWasRunning) toggleAgentSimulation();
      if (preCount > 0 && !cancelRequested) {
        phase(`restoring ${preCount} agents…`);
        await spawnAgents(preCount);
      }
    } catch (err) {
      console.warn("[bench] restore failed:", err);
    }
  }
}

/**
 * Capacity ramp: grow the crowd by `step` agents every `intervalMs` until the
 * interval's average fps drops below `minFps` (or maxAgents/cancel hits).
 * Spawn-hitch frames are excluded via a settle window after each step, so the
 * verdict reflects sustainable load. Reports the highest count that held
 * minFps; the failing step stays spawned and is counted in finalAgents.
 */
export async function runRampBenchmark(
  { step, intervalMs, minFps, maxAgents, lod, device },
  onPhase = () => {},
) {
  cancelRequested = false;
  const perStep = Math.max(1, Math.min(500, Math.floor(Number(step) || 5)));
  const interval = Math.max(1000, Math.min(30000, Number(intervalMs) || 2500));
  const floor = Math.max(1, Math.min(120, Number(minFps) || 20));
  const cap = Math.max(1, Math.min(2000, Math.floor(Number(maxAgents) || 1000)));
  const pin = lod === "auto" ? null : Number(lod);
  const phase = (text) => onPhase(text);

  const preCount = simulation.agents.length;
  const preAutoLod = simulation.autoLod;
  const simWasRunning = isAgentSimulationActive();
  let savedCam = null;
  try {
    savedCam = frameCrowdForBenchmark();
    phase("clearing scene…");
    clearAgents();
    if (!isAgentSimulationActive()) toggleAgentSimulation();
    setAutoLodEnabled(pin === null);
    let capacity = 0;
    let capacitySamples = [];
    let lastFps = 0;
    let lastSamples = [];
    const steps = [];
    for (;;) {
      const before = simulation.agents.length;
      if (before >= cap) {
        phase(`agent cap ${cap} reached.`);
        break;
      }
      phase(`spawning ${Math.min(perStep, cap - before)} agents (${before} live)…`);
      const spawned = await spawnAgents(Math.min(perStep, cap - before));
      if (cancelRequested) return null;
      if ((spawned?.added ?? 0) === 0) break; // Engine cap hit.
      const count = simulation.agents.length;
      if (pin !== null) {
        for (const agent of simulation.agents) setAgentLodPin(agent.id, pin);
      }
      // Settle: spawn-hitch frames don't count toward the verdict.
      const settleUntil = performance.now() + RAMP_SETTLE_MS;
      for (;;) {
        await nextFrame();
        if (cancelRequested) return null;
        if (performance.now() >= settleUntil) break;
      }
      const samples = [];
      let prev = null;
      const measureStart = performance.now();
      const measureUntil = measureStart + Math.max(500, interval - RAMP_SETTLE_MS);
      let stepTick = -1;
      for (;;) {
        await nextFrame();
        if (cancelRequested) return null;
        const { sample, now } = sampleFrame(prev);
        prev = now;
        if (sample.frameMs > 0) samples.push(sample);
        const tick = Math.floor((now - measureStart) / 1000);
        if (tick !== stepTick) {
          stepTick = tick;
          phase(`${count} agents · measuring… ${tick + 1}s`);
        }
        if (now >= measureUntil) break;
      }
      const fps = summarizeSamples(samples).fpsAvg;
      lastFps = fps;
      lastSamples = samples;
      steps.push({ agents: count, fps: +fps.toFixed(1) });
      if (fps < floor) {
        phase(`below ${floor} fps at ${count} agents — capacity ${capacity}.`);
        break;
      }
      capacity = count;
      capacitySamples = samples;
      phase(`${count} agents · ${fps.toFixed(0)} fps (floor ${floor})`);
    }
    if (cancelRequested) return null;
    const summarySamples = capacitySamples.length ? capacitySamples
      : lastSamples.length ? lastSamples : [{ frameMs: 0 }];
    const summary = summarizeSamples(summarySamples);
    const lodMix = {};
    let cpu = 0;
    for (const agent of simulation.agents) {
      if (!agent.isGpuAgent) { cpu++; continue; }
      lodMix[agent.lod] = (lodMix[agent.lod] || 0) + 1;
    }
    const { engine, gpu, gpuShort } = detectEngine();
    const canvas = renderer?.domElement;
    return {
      id: `run-${Date.now().toString(36)}`,
      date: new Date().toISOString(),
      device: String(device || gpuShort || "device").slice(0, 32),
      engine,
      gpu,
      gpuShort,
      canvas: canvas ? { w: canvas.width, h: canvas.height, dpr: window.devicePixelRatio || 1 } : null,
      ua: navigator.userAgent.slice(0, 120),
      params: {
        mode: "ramp",
        agents: capacity,
        step: perStep,
        intervalS: +(interval / 1000).toFixed(1),
        minFps: floor,
        lod: pin === null ? "auto" : pin,
      },
      ...summary,
      steps,
      finalAgents: simulation.agents.length,
      lastFps: +lastFps.toFixed(1),
      lodMix,
      cpu,
      gpuAgents: simulation.agents.length - cpu,
    };
  } finally {
    restoreCameraAfterBenchmark(savedCam);
    phase("restoring…");
    try {
      clearAgents();
      setAutoLodEnabled(preAutoLod);
      if (isAgentSimulationActive() !== simWasRunning) toggleAgentSimulation();
      if (preCount > 0 && !cancelRequested) {
        phase(`restoring ${preCount} agents…`);
        await spawnAgents(preCount);
      }
    } catch (err) {
      console.warn("[bench] restore failed:", err);
    }
  }
}

export function loadRuns() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    const runs = raw ? JSON.parse(raw) : [];
    return Array.isArray(runs) ? runs : [];
  } catch {
    return [];
  }
}

export function saveRun(run) {
  const runs = [run, ...loadRuns().filter((r) => r && r.id !== run.id)].slice(0, MAX_RUNS);
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(runs));
  } catch { /* storage full/blocked: history is best-effort */ }
  return runs;
}

export function clearRuns() {
  try {
    localStorage.removeItem(STORE_KEY);
  } catch { /* ignore */ }
  return [];
}

export function exportRuns() {
  const blob = new Blob([JSON.stringify({ version: 1, runs: loadRuns() }, null, 1)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `mhr-bench-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function importRuns(file) {
  const text = await file.text();
  const data = JSON.parse(text);
  const incoming = Array.isArray(data) ? data : data.runs;
  if (!Array.isArray(incoming)) throw new Error("no runs array in file");
  const seen = new Set(loadRuns().map((r) => r.id));
  let added = 0;
  for (const run of incoming) {
    if (!run || typeof run !== "object" || !run.id || seen.has(run.id)) continue;
    seen.add(run.id);
    saveRun(run);
    added++;
  }
  return { added, total: loadRuns().length };
}

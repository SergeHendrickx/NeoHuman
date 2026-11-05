import { elements, state } from "./state.js";
import { computeVramBytes } from "./vram.js";

const PERF_SAMPLES = 80;

export const perfData = {
  frametimes: new Float32Array(PERF_SAMPLES).fill(16.6),
  evalTimes: new Float32Array(PERF_SAMPLES).fill(0.08),
  head: 0,
  count: 0,
  totalBandwidthSavedBytes: 0,
  lastEvalTime: 0.08,
  activeTab: "live",
  isIdle: true,
};

export function setPerfIdle(idle) {
  perfData.isIdle = idle;
}

export function updateRuntimeStats(renderer, scene) {
  if (!renderer || !elements.perfCard) return;

  // 1. Draw calls & Triangles from renderer.info
  const infoRender = renderer.info?.render;
  // WebGPU splits the counters: `calls` counts render() invocations (scene pass
  // + output quad, never reset), per-object draws live in `drawCalls`. The
  // WebGL fallback only has classic `calls`, hence the order.
  const drawCalls = infoRender?.drawCalls ?? infoRender?.calls ?? 0;
  const triangles = infoRender?.triangles ?? 0;

  if (elements.perfDrawCalls) {
    elements.perfDrawCalls.textContent = `${drawCalls}`;
  }
  if (elements.perfTrianglesBadge) {
    elements.perfTrianglesBadge.textContent =
      triangles >= 1000
        ? `${(triangles / 1000).toFixed(1)}K tris`
        : `${triangles} tris`;
  }

  // 2. Count active meshes and crowd instances in scene
  let totalMeshes = 0;
  let skinnedCount = 0;
  let instancedCount = 0;

  scene.traverse((obj) => {
    if (obj.isMesh && obj.visible) {
      totalMeshes++;
      if (obj.isSkinnedMesh) skinnedCount++;
      if (obj.isInstancedMesh) {
        instancedCount += obj.count;
      }
    }
  });

  if (elements.perfMeshCount) {
    elements.perfMeshCount.textContent = `${totalMeshes}`;
  }
  if (elements.perfDrawSub) {
    // Plain counts: skinned meshes include garments, so "root + crowd"
    // math would miscount (crowd size lives in the crowd stats anyway).
    elements.perfDrawSub.textContent = `${skinnedCount} skinned · ${instancedCount} instanced`;
  }

  // 3. VRAM calculation (GPU Buffer attributes & storage buffers)
  const { bytes: vramBytes, geometries: geomCount } = computeVramBytes(
    scene,
    state.blendshapes,
    state.correctiveDeltas,
  );

  const vramMb = (vramBytes / (1024 * 1024)).toFixed(1);
  if (elements.perfVramVal) {
    elements.perfVramVal.textContent = `${vramMb} MB`;
  }
  if (elements.perfVramSub) {
    // Ground truth is the traverse: renderer.info's geometry counter only
    // decrements on dispose(), which pooled agent shells deliberately skip
    // (their GPU buffers are provider-owned; see _releaseRecord).
    elements.perfVramSub.textContent = `${geomCount} geoms · GPU buffers`;
  }

  // 4. Host / JS Memory
  let heapMb = 0;
  if (typeof performance !== "undefined" && performance.memory && performance.memory.usedJSHeapSize) {
    heapMb = (performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(0);
  }

  let weightsBytes = 0;
  if (state.blendshapes) {
    weightsBytes += (state.blendshapes.neutralVertices?.byteLength || 0) +
                    (state.blendshapes.blendshapeDeltas?.byteLength || 0);
  }
  if (state.poseCorrectives) {
    weightsBytes += (state.poseCorrectives.l0Weights?.byteLength || 0) +
                    (state.poseCorrectives.l2Data?.byteLength || 0) +
                    (state.poseCorrectives.l2Indptr?.byteLength || 0) +
                    (state.poseCorrectives.l2Indices?.byteLength || 0);
  }
  if (state.rigMapping) {
    weightsBytes += state.rigMapping.TpMatrix?.byteLength || 0;
  }
  if (state.unposedBase) {
    weightsBytes += state.unposedBase.byteLength;
  }
  if (state.correctiveDeltas) {
    weightsBytes += state.correctiveDeltas.byteLength;
  }
  if (state.poseFeatures) {
    weightsBytes += state.poseFeatures.byteLength;
  }
  if (state.poseHidden) {
    weightsBytes += state.poseHidden.byteLength;
  }
  if (state.jointPositions) {
    weightsBytes += state.jointPositions.byteLength;
  }
  const weightsMb = (weightsBytes / (1024 * 1024)).toFixed(1);

  if (elements.perfMemVal) {
    elements.perfMemVal.textContent = heapMb > 0 ? `${heapMb} MB` : `${weightsMb} MB`;
  }
  if (elements.perfMemSub) {
    elements.perfMemSub.textContent = heapMb > 0
      ? `Heap (${weightsMb} MB weights)`
      : `CSR & Rig weights in RAM`;
  }
}

export function recordPerfMetric(evalMs, kind = "pose") {
  perfData.lastEvalTime = Math.max(0.01, evalMs);
  if (elements.perfEvalLatency) {
    elements.perfEvalLatency.textContent =
      evalMs < 0.1 ? "< 0.1 ms" : `${evalMs.toFixed(1)} ms`;
  }
  if (elements.perfPipelineTotal) {
    const label =
      kind === "pose"
        ? (state.metadata?.model?.applyCorrectives ? "Active: FK + CSR Neural Correctives" : "Active: FK Kinematics")
        : "Active: Blendshapes";
    const timeStr = evalMs < 0.1 ? "< 0.1 ms" : `${evalMs.toFixed(1)} ms`;
    elements.perfPipelineTotal.textContent = `${label} (${timeStr})`;
  }
}

const STAGE_ORDER = ["fk", "blend", "corr", "mesh", "render"];
const STAGE_LABELS = {
  fk: "Forward Kinematics",
  blend: "Blendshapes",
  corr: "CSR Pose Correctives",
  mesh: "Normals, bounds and LOD transfer",
  render: "Three.js render",
};
const STAGE_ELEMENTS = {
  fk: { seg: "perfSegFk", val: "perfValFk" },
  blend: { seg: "perfSegBlend", val: "perfValBlend" },
  corr: { seg: "perfSegCorr", val: "perfValCorr" },
  mesh: { seg: "perfSegMesh", val: "perfValMesh" },
  render: { seg: "perfSegRender", val: "perfValRender" },
};
// Last measured milliseconds per stage; each evaluator overwrites only its
// own entries, so values from the other parameter group go stale, not zero.
const stageTimes = { fk: 0, blend: 0, corr: 0, mesh: 0, render: 0 };
const stageDetail = { fk: "", blend: "", corr: "", mesh: "", render: "" };

export function formatStageMs(ms) {
  return ms < 0.1 ? "< 0.1 ms" : `${ms.toFixed(1)} ms`;
}

function updatePipelineBar() {
  const total = STAGE_ORDER.reduce((sum, key) => sum + stageTimes[key], 0);
  if (!(total > 0)) return; // keep placeholder widths until first measurement
  for (const key of STAGE_ORDER) {
    const ms = stageTimes[key];
    const label = formatStageMs(ms);
    const segEl = elements[STAGE_ELEMENTS[key].seg];
    const valEl = elements[STAGE_ELEMENTS[key].val];
    if (segEl) {
      segEl.style.width = `${((100 * ms) / total).toFixed(1)}%`;
      segEl.title = stageDetail[key]
        ? `${STAGE_LABELS[key]} (${label}) — ${stageDetail[key]}`
        : `${STAGE_LABELS[key]} (${label})`;
    }
    if (valEl) valEl.textContent = label;
  }
}

export function recordStageTimes(partial, detail) {
  let changed = false;
  for (const [key, ms] of Object.entries(partial)) {
    if (key in stageTimes && Number.isFinite(ms) && ms >= 0) {
      stageTimes[key] = ms;
      changed = true;
    }
  }
  if (detail) {
    for (const [key, text] of Object.entries(detail)) {
      if (key in stageDetail) stageDetail[key] = text;
    }
  }
  if (changed) updatePipelineBar();
}

export function readStageTimes() {
  return { ...stageTimes };
}

export function recordFrameSample(frametimeMs) {
  perfData.frametimes[perfData.head] = frametimeMs;
  perfData.evalTimes[perfData.head] = perfData.lastEvalTime;
  perfData.head = (perfData.head + 1) % PERF_SAMPLES;
}

export function recordBandwidthSaved(bytes = 226611) {
  perfData.totalBandwidthSavedBytes += bytes;
  if (elements.perfSavedBandwidth) {
    const mb = (perfData.totalBandwidthSavedBytes / (1024 * 1024)).toFixed(1);
    elements.perfSavedBandwidth.textContent = `Saved: ${mb} MB session`;
  }
}

export function drawPerfGraph() {
  const canvas = elements.perfGraphCanvas;
  if (!canvas || perfData.activeTab !== "live") return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  const dpr = window.devicePixelRatio || 1;
  const width = Math.floor(rect.width);
  const height = Math.floor(rect.height);

  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }

  ctx.save();
  ctx.scale(dpr, dpr);

  // Background
  ctx.fillStyle = "#0c1017";
  ctx.fillRect(0, 0, width, height);

  // Grid setup: 0 to 33.3 ms range (30 FPS boundary)
  const maxMs = 33.3;
  const padBottom = 16;
  const padTop = 10;
  const plotH = height - padBottom - padTop;

  function getY(ms) {
    const clamped = Math.max(0, Math.min(ms, maxMs));
    return padTop + plotH * (1.0 - clamped / maxMs);
  }

  // Draw grid lines
  ctx.lineWidth = 1;

  // 16.6 ms (60 FPS) line
  const y60 = getY(16.6);
  ctx.strokeStyle = "rgba(98, 209, 154, 0.25)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(35, y60);
  ctx.lineTo(width - 8, y60);
  ctx.stroke();

  // 8.3 ms (120 FPS) line
  const y120 = getY(8.33);
  ctx.strokeStyle = "rgba(127, 156, 255, 0.2)";
  ctx.beginPath();
  ctx.moveTo(35, y120);
  ctx.lineTo(width - 8, y120);
  ctx.stroke();

  ctx.setLineDash([]);

  // Axis labels
  ctx.fillStyle = "#6b788c";
  ctx.font = "9px Inter, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText("60 FPS", 32, y60);
  ctx.fillText("120 FPS", 32, y120);
  ctx.fillText("0 ms", 32, height - padBottom);

  // Plot Frametime (Emerald Green)
  const count = PERF_SAMPLES;
  const stepX = (width - 45) / (count - 1);
  const startX = 38;

  // Frametime Area Fill
  const gradFt = ctx.createLinearGradient(0, padTop, 0, height - padBottom);
  gradFt.addColorStop(0, "rgba(98, 209, 154, 0.22)");
  gradFt.addColorStop(1, "rgba(98, 209, 154, 0.0)");

  ctx.beginPath();
  ctx.moveTo(startX, height - padBottom);
  for (let i = 0; i < count; i++) {
    const idx = (perfData.head + i) % count;
    const ft = perfData.frametimes[idx];
    const x = startX + i * stepX;
    const y = getY(ft);
    ctx.lineTo(x, y);
  }
  ctx.lineTo(startX + (count - 1) * stepX, height - padBottom);
  ctx.closePath();
  ctx.fillStyle = gradFt;
  ctx.fill();

  // Frametime Line
  ctx.beginPath();
  for (let i = 0; i < count; i++) {
    const idx = (perfData.head + i) % count;
    const ft = perfData.frametimes[idx];
    const x = startX + i * stepX;
    const y = getY(ft);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#62d19a";
  ctx.lineWidth = 1.75;
  ctx.stroke();

  // Eval Latency Line & Fill (Purple / Neon Magenta)
  const gradEval = ctx.createLinearGradient(0, padTop, 0, height - padBottom);
  gradEval.addColorStop(0, "rgba(192, 132, 252, 0.35)");
  gradEval.addColorStop(1, "rgba(192, 132, 252, 0.0)");

  ctx.beginPath();
  ctx.moveTo(startX, height - padBottom);
  for (let i = 0; i < count; i++) {
    const idx = (perfData.head + i) % count;
    const et = perfData.evalTimes[idx];
    const x = startX + i * stepX;
    const y = getY(et);
    ctx.lineTo(x, y);
  }
  ctx.lineTo(startX + (count - 1) * stepX, height - padBottom);
  ctx.closePath();
  ctx.fillStyle = gradEval;
  ctx.fill();

  ctx.beginPath();
  for (let i = 0; i < count; i++) {
    const idx = (perfData.head + i) % count;
    const et = perfData.evalTimes[idx];
    const x = startX + i * stepX;
    const y = getY(et);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = "#c084fc";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  if (perfData.isIdle) {
    ctx.fillStyle = "rgba(141, 153, 171, 0.45)";
    ctx.font = "bold 9px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText("ON-DEMAND (IDLE)", width - 10, padTop + 12);
  }

  ctx.restore();
}

elements.perfTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.perfTab;
    perfData.activeTab = target;
    elements.perfTabs.forEach((t) => t.classList.toggle("is-active", t === tab));
    Object.entries(elements.perfPanes).forEach(([key, pane]) => {
      if (pane) {
        const active = key === target;
        pane.classList.toggle("is-active", active);
        pane.hidden = !active;
      }
    });
    if (target === "live") {
      drawPerfGraph();
    }
  });
});

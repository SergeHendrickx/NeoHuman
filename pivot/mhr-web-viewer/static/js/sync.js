import { state } from "./state.js";
import { updateInferenceTime } from "./ui.js";
import { evaluatePoseClientSide } from "./rig.js";

let activeBlendshapeSyncController = null;
let activeBlendshapeSyncPromise = null;
let activeSyncController = null;
let activeSyncPromise = null;

// Resolves once every in-flight background sync has settled. Used to flush
// server state before export so the export reflects the current UI parameters.
export function pendingSyncs() {
  return Promise.allSettled(
    [activeSyncPromise, activeBlendshapeSyncPromise].filter(Boolean),
  );
}

async function postPoseParam(index, value, signal) {
  const response = await fetch(
    "/api/deform?nogeometry=1",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        action: "set",
        group: "pose",
        index,
        value,
      }),
      signal,
    },
  );
  if (!response.ok) return null;
  updateInferenceTime(response);
  return response;
}

function applySnappedRootTy(response) {
  const rootTy = Number(response.headers.get("X-MHR-Root-TY"));
  const rootParameter = state.metadata.groups.pose.find(
    (item) => item.name === "root_ty",
  );
  if (rootParameter && Number.isFinite(rootTy) && rootParameter.value !== rootTy) {
    // Snap-to-ground adjusted root_ty on the server; re-evaluate so the
    // client mesh stays grounded instead of lagging one interaction behind.
    rootParameter.value = rootTy;
    evaluatePoseClientSide();
  }
}

function trackPoseTask(task, controller) {
  activeSyncPromise = task;
  return task.finally(() => {
    if (activeSyncPromise === task) activeSyncPromise = null;
    if (activeSyncController === controller) activeSyncController = null;
  });
}

function trackBlendshapeTask(task, controller) {
  activeBlendshapeSyncPromise = task;
  return task.finally(() => {
    if (activeBlendshapeSyncPromise === task) activeBlendshapeSyncPromise = null;
    if (activeBlendshapeSyncController === controller) {
      activeBlendshapeSyncController = null;
    }
  });
}

export async function syncBlendshapeServer(group, index, value) {
  if (activeBlendshapeSyncController) {
    activeBlendshapeSyncController.abort();
  }
  const controller = new AbortController();
  activeBlendshapeSyncController = controller;
  const task = (async () => {
    try {
      const response = await fetch(
        "/api/deform?nogeometry=1",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "set",
            group,
            index,
            value,
          }),
          signal: controller.signal,
        },
      );
      if (!response.ok) return;
      updateInferenceTime(response);
    } catch (err) {
      if (err.name !== "AbortError") {
        console.warn("Background blendshape sync:", err);
      }
    }
  })();
  await trackBlendshapeTask(task, controller);
}

export async function syncPoseServer(index, value) {
  if (activeSyncController) {
    activeSyncController.abort();
  }
  const controller = new AbortController();
  activeSyncController = controller;
  const task = (async () => {
    try {
      const response = await postPoseParam(index, value, controller.signal);
      if (!response) return;
      applySnappedRootTy(response);
    } catch (err) {
      if (err.name !== "AbortError") {
        console.warn("Background server sync:", err);
      }
    }
  })();
  await trackPoseTask(task, controller);
}

// Whole-group vector write (avatar codes, multi-param applies). One POST,
// so unlike a per-index loop it cannot abort itself halfway.
export async function syncGroupParams(group, values) {
  const pose = group === "pose";
  const active = pose ? activeSyncController : activeBlendshapeSyncController;
  if (active) {
    active.abort();
  }
  const controller = new AbortController();
  if (pose) {
    activeSyncController = controller;
  } else {
    activeBlendshapeSyncController = controller;
  }
  const task = (async () => {
    try {
      const response = await fetch(
        "/api/deform?nogeometry=1",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            action: "set_group",
            group,
            index: 0,
            values: [...values].map((v) => {
              const n = Number(v);
              return Number.isFinite(n) ? n : 0;
            }),
          }),
          signal: controller.signal,
        },
      );
      if (!response.ok) return;
      updateInferenceTime(response);
      if (pose) applySnappedRootTy(response);
    } catch (err) {
      if (err.name !== "AbortError") {
        console.warn("Background server sync:", err);
      }
    }
  })();
  await (pose ? trackPoseTask(task, controller) : trackBlendshapeTask(task, controller));
}

// Batch variant for joint-gizmo drags, which can touch several pose
// parameters at once. Posts them in order, then applies the final snap state.
export async function syncPoseParams(indices) {
  if (activeSyncController) {
    activeSyncController.abort();
  }
  const controller = new AbortController();
  activeSyncController = controller;
  const task = (async () => {
    try {
      let last = null;
      for (const index of indices) {
        const param = state.metadata.groups.pose[index];
        last = await postPoseParam(index, param.value, controller.signal);
        if (!last) return;
      }
      if (last) applySnappedRootTy(last);
    } catch (err) {
      if (err.name !== "AbortError") {
        console.warn("Background server sync:", err);
      }
    }
  })();
  await trackPoseTask(task, controller);
}

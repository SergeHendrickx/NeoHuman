import { addMeshReplaceListener } from "./rig.js";
import { requestRender } from "./scene.js";
import { simulation } from "./sim/AgentSimulation.js";
import { POOL_LODS } from "./sim/LodPolicy.js";

// The LOD loader replaces the shared mesh (boot only, now that the LOD UI
// is gone): the sim re-arms its topology guard. Clone/instance tiers were
// removed; agents are the only crowd system left.
addMeshReplaceListener(() => {
  simulation.handleLodSwitch();
});

export function spawnAgents(count) {
  return simulation.spawn(count);
}

export function clearAgents() {
  simulation.clear();
}

export function agentCount() {
  return simulation.count;
}

export function isAgentSimulationActive() {
  return simulation.isRunning;
}

export function toggleAgentSimulation() {
  simulation.toggle();
  return simulation.isRunning;
}

export function startSimulation() {
  simulation.start();
  return simulation.isRunning;
}

export function setAgentBehavior(mode) {
  simulation.setBehaviorMode(mode);
}

export function setAutoLodEnabled(enabled) {
  simulation.autoLod = Boolean(enabled);
  simulation._notifyChange();
  requestRender();
}

export function setSelectionRingVisible(on) {
  simulation.setSelectionRingVisible(on);
  requestRender();
}

export function setShowLodColors(enabled) {
  simulation.setShowLodColors(enabled);
}

export function setFrustumCullingEnabled(enabled) {
  simulation.setFrustumCulling(enabled);
}

export function setAgentLodPin(id, pin) {
  const agent = simulation.agents.find((a) => a.id === Number(id));
  if (agent && agent.isGpuAgent) {
    const n = pin === null || pin === undefined || pin === "auto" ? null : Number(pin);
    agent.lodPin = n !== null && POOL_LODS.includes(n) ? n : null;
    simulation._notifyChange();
  }
}

export function getAgentBehavior() {
  return simulation.behaviorMode;
}

export function listAgents() {
  return simulation.agents.map((a) => ({
    id: a.id,
    behavior: a.behavior ?? simulation.behaviorMode,
    x: a.position.x,
    z: a.position.z,
    lod: a.lod ?? null,
    gpu: Boolean(a.isGpuAgent),
  }));
}

export function getShowLodColors() {
  return simulation.showLodColors;
}

export function setOneAgentBehavior(id, mode) {
  const agent = simulation.agents.find((a) => a.id === Number(id));
  if (!agent) throw new Error(`no agent ${id}`);
  agent.setBehavior(mode);
  simulation._notifyChange();
  return true;
}

export function addSimulationChangeListener(fn) {
  return simulation.addChangeListener(fn);
}

export function selectAgent(id) {
  simulation.selectAgent(id);
}

export function getSelectedAgent() {
  return simulation.getSelectedAgent();
}

export function getSelectedAgentId() {
  return simulation.selectedAgentId;
}

export function removeAgent(id) {
  simulation.removeAgent(id);
}

export function focusSelectedAgent() {
  simulation.focusSelectedAgent();
}

export function stepSimulation(frames = 1, dt = 1 / 60) {
  simulation.step(frames, dt);
}

export function setSimulationTimeScale(scale) {
  simulation.setTimeScale(scale);
}

export function getSimulationTimeScale() {
  return simulation.timeScale;
}

export function pickAgentAtNdc(ndcPoint, camera) {
  return simulation.pickAgentAtNdc(ndcPoint, camera);
}

export function prewarmGpuAgents() {
  simulation.prewarmGpu();
}

export function refreshGpuAgents() {
  simulation.refreshGpuAgents();
}

export function getSimulationStats() {
  return {
    frameIndex: simulation.frameIndex,
    simTime: simulation.simTime,
    isRunning: simulation.isRunning,
    timeScale: simulation.timeScale,
    count: simulation.count,
    selectedId: simulation.selectedAgentId,
    gpu: simulation.gpuStatus(),
    timings: simulation.timings,
  };
}

export { simulation };

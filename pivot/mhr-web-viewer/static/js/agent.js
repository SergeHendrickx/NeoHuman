/**
 * Agent loop: Meta Model API (OpenAI-compatible) tool calling over the action catalog.
 *
 * No DOM here: runAgentTurn() takes messages + a state summary, drives the
 * dispatcher, and reports via onEvent. Slice C binds the chat panel.
 */

import { TOOL_SCHEMAS } from "./actions.js";

export const DEFAULT_BASE_URL = "https://api.meta.ai/v1";
export const DEFAULT_MODEL = "muse-spark-1.3";
export const MAX_ITERS = 6;

export const SYSTEM_PROMPT = [
  "You operate NeoHuman, a 3D human-rig and crowd simulator. Use tools for every state change; never claim an action without calling it.",
  "Actor #0 is the initial actor; spawned agents have ids >= 1. setParam and gaze act on the active actor: selectActor first when the user names someone else.",
  "setParam groups: pose (joint rotations), identity (body shape), expression (72 blendshapes, index 0..71). Values outside a parameter's range are clamped.",
  "benchmark is slow (seconds plus spawn time); prefer it only when asked to measure. clearAgents/removeActor need user approval, which is handled automatically.",
  "Keep replies short: one line per action plus the outcome. If a tool reports ok:false, say what failed and stop that branch.",
].join(" ");

/**
 * snapshot: { agents, gpu, cpu, selectedId, selected, behavior, lodColors,
 *   simRunning, view } — fields optional; missing ones are skipped.
 * selected: { behavior, x, z } when an agent is selected.
 */
export function formatSnapshot(s = {}) {
  const lines = [];
  if (s.agents !== undefined) {
    const mix = s.gpu !== undefined || s.cpu !== undefined ? ` (${s.gpu ?? "?"} GPU, ${s.cpu ?? "?"} CPU)` : "";
    lines.push(`agents: ${s.agents}${mix}`);
  }
  if (s.selectedId !== undefined) {
    if (s.selectedId === 0 || s.selectedId === null) {
      lines.push("selected: actor #0 (0)");
    } else if (s.selected) {
      lines.push(
        `selected: agent ${s.selectedId} (behavior ${s.selected.behavior ?? "?"}, pos ${s.selected.x ?? "?"}, ${s.selected.z ?? "?"})`
      );
    } else {
      lines.push(`selected: agent ${s.selectedId}`);
    }
  }
  if (s.behavior !== undefined) lines.push(`default behavior: ${s.behavior}`);
  if (s.lodColors !== undefined) lines.push(`lodColors: ${s.lodColors ? "on" : "off"}`);
  if (s.simRunning !== undefined) lines.push(`sim: ${s.simRunning ? "running" : "paused"}`);
  if (s.view !== undefined) lines.push(`view: ${s.view}`);
  return lines.length ? lines.join("\n") : "(empty scene)";
}

export function buildBody(model, messages) {
  return {
    model,
    messages,
    tools: TOOL_SCHEMAS,
    tool_choice: "auto",
    temperature: 0.2,
  };
}

export async function chatCompletion({ baseUrl, key, model, messages }, fetchImpl = globalThis.fetch) {
  const url = `${String(baseUrl).replace(/\/+$/, "")}/chat/completions`;
  let res;
  try {
    res = await fetchImpl(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
      body: JSON.stringify(buildBody(model, messages)),
    });
  } catch (err) {
    throw new Error(`request failed: ${err?.message ?? err}`);
  }
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 300) || res.statusText}`);
  }
  const data = await res.json();
  const message = data?.choices?.[0]?.message;
  if (!message) throw new Error("response had no message");
  return message;
}

/** Normalize one assistant message's tool calls; bad JSON becomes a dispatchable error. */
export function extractToolCalls(message) {
  const out = [];
  for (const call of message?.tool_calls ?? []) {
    const name = call?.function?.name;
    let args = {};
    try {
      args = JSON.parse(call?.function?.arguments ?? "{}") ?? {};
    } catch {
      out.push({ id: call.id, name, args: null, argsError: `bad arguments JSON: ${call?.function?.arguments}` });
      continue;
    }
    out.push({ id: call.id, name, args, argsError: null });
  }
  return out;
}

/**
 * Run one user turn to completion (possibly several model<->tool round trips).
 * - dispatcher: from actions.js createDispatcher
 * - messages: prior OpenAI message objects (user/assistant/tool), latest user last
 * - snapshotText: formatSnapshot() output, injected beside the system prompt
 * - config: { baseUrl, key, model }
 * - approve(name, args): resolves true to run a destructive action
 * - onEvent({ type, ... }): text | tool-call | tool-result | approval | error
 * Returns { messages } with this turn's traffic appended.
 */
export async function runAgentTurn({
  dispatcher,
  messages,
  snapshotText,
  config,
  fetchImpl = globalThis.fetch,
  maxIters = MAX_ITERS,
  approve = async () => true,
  onEvent = () => {},
}) {
  const convo = [...messages];
  const system = { role: "system", content: `${SYSTEM_PROMPT}\n\nCurrent state:\n${snapshotText}` };
  for (let iter = 0; iter < maxIters; iter++) {
    let assistant;
    try {
      assistant = await chatCompletion({ ...config, messages: [system, ...convo] }, fetchImpl);
    } catch (err) {
      onEvent({ type: "error", error: err.message });
      return { messages: convo, error: err.message };
    }
    convo.push(assistant);
    const calls = extractToolCalls(assistant);
    if (calls.length === 0) {
      if (assistant.content) onEvent({ type: "text", text: assistant.content });
      return { messages: convo };
    }
    for (const call of calls) {
      onEvent({ type: "tool-call", id: call.id, name: call.name, args: call.args });
      let result;
      if (call.argsError) {
        result = { ok: false, error: call.argsError };
      } else if (dispatcher.isDestructive(call.name)) {
        onEvent({ type: "approval", id: call.id, name: call.name, args: call.args });
        const ok = await approve(call.name, call.args);
        result = ok
          ? await dispatcher.dispatch(call.name, call.args, { source: "chat" })
          : { ok: false, error: "denied by user" };
      } else {
        result = await dispatcher.dispatch(call.name, call.args, { source: "chat" });
      }
      onEvent({ type: "tool-result", id: call.id, name: call.name, result });
      convo.push({ role: "tool", tool_call_id: call.id, content: JSON.stringify(result) });
    }
  }
  onEvent({ type: "error", error: `stopped after ${maxIters} tool rounds` });
  return { messages: convo, error: `stopped after ${maxIters} tool rounds` };
}

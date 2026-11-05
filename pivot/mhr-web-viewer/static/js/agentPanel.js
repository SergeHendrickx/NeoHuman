/**
 * Floating agent console: chat UI over the tool loop (agent.js).
 * Self-wires its DOM; app.js only passes { dispatcher, getSnapshot }.
 */

import { runAgentTurn, formatSnapshot, DEFAULT_BASE_URL, DEFAULT_MODEL } from "./agent.js";

const LS_KEY = "mhr.agentKey";
const LS_MODEL = "mhr.agentModel.v2"; // v2: reseed onto muse-spark-1.3.
const LS_URL = "mhr.agentUrl.v2"; // v2: reseed everyone onto the Meta default.
const LS_COLLAPSED = "mhr.agentCollapsed";
const LS_POS = "mhr.agentPos";
const LS_LOG_OPEN = "mhr.agentLogOpen";
const LS_SETTINGS_OPEN = "mhr.agentSettingsOpen";
const MAX_HISTORY = 40;

export function initAgentPanel({ dispatcher, getSnapshot, log = null }) {
  const $ = (id) => document.getElementById(id);
  const root = $("agent-panel");
  if (!root || typeof dispatcher?.dispatch !== "function") return null;
  const els = {
    root,
    head: $("agent-head"),
    status: $("agent-status"),
    collapse: $("agent-collapse"),
    body: $("agent-body"),
    key: $("agent-key"),
    model: $("agent-model"),
    url: $("agent-url"),
    log: $("agent-log"),
    form: $("agent-form"),
    input: $("agent-input"),
    send: $("agent-send"),
  };
  if (Object.values(els).some((el) => !el)) return null;

  let messages = [];
  let busy = false;

  // Settings persist per browser; the key never leaves it except to the model API.
  els.key.value = localStorage.getItem(LS_KEY) ?? "";
  els.model.value = localStorage.getItem(LS_MODEL) ?? DEFAULT_MODEL;
  els.url.value = localStorage.getItem(LS_URL) ?? DEFAULT_BASE_URL;
  for (const [el, lsKey] of [[els.key, LS_KEY], [els.model, LS_MODEL], [els.url, LS_URL]]) {
    el.addEventListener("change", () => localStorage.setItem(lsKey, el.value.trim()));
  }

  const setCollapsed = (collapsed) => {
    els.body.hidden = collapsed;
    els.collapse.textContent = collapsed ? "+" : "–";
    localStorage.setItem(LS_COLLAPSED, collapsed ? "1" : "0");
  };
  setCollapsed(localStorage.getItem(LS_COLLAPSED) === "1");
  els.collapse.addEventListener("click", () => setCollapsed(!els.body.hidden));

  // Drag by the header (collapse button stays clickable); position persists.
  try {
    const saved = JSON.parse(localStorage.getItem(LS_POS) ?? "null");
    if (saved && Number.isFinite(saved.left) && Number.isFinite(saved.top)) {
      const maxLeft = Math.max(0, window.innerWidth - 120);
      const maxTop = Math.max(0, window.innerHeight - 60);
      els.root.style.left = `${Math.min(Math.max(0, saved.left), maxLeft)}px`;
      els.root.style.top = `${Math.min(Math.max(0, saved.top), maxTop)}px`;
      els.root.style.bottom = "auto";
    }
  } catch {
    // Corrupt or unavailable storage: keep the default corner.
  }
  els.head.addEventListener("pointerdown", (event) => {
    if (event.target.closest("button")) return;
    const startX = event.clientX;
    const startY = event.clientY;
    const rect = els.root.getBoundingClientRect();
    els.root.style.left = `${rect.left}px`;
    els.root.style.top = `${rect.top}px`;
    els.root.style.bottom = "auto";
    const move = (ev) => {
      els.root.style.left = `${Math.max(0, rect.left + ev.clientX - startX)}px`;
      els.root.style.top = `${Math.max(0, rect.top + ev.clientY - startY)}px`;
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      try {
        const at = els.root.getBoundingClientRect();
        localStorage.setItem(LS_POS, JSON.stringify({ left: at.left, top: at.top }));
      } catch {
        // Private mode etc: stay ephemeral.
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  });

  // Disclosure memory: the action log defaults to open (it is the reason
  // the console exists); settings default to closed.
  const persistToggle = (id, lsKey, defOpen) => {
    const details = $(id);
    if (!details) return;
    const saved = localStorage.getItem(lsKey);
    details.open = saved === null ? defOpen : saved === "1";
    details.addEventListener("toggle", () => {
      try {
        localStorage.setItem(lsKey, details.open ? "1" : "0");
      } catch {
        // Private mode etc: stay ephemeral.
      }
    });
  };
  persistToggle("agent-actions", LS_LOG_OPEN, true);
  persistToggle("agent-settings", LS_SETTINGS_OPEN, false);

  const setStatus = (state) => {
    els.status.dataset.state = state;
    els.status.title = `Agent ${state}`;
  };

  function addLine(kind, text) {
    const div = document.createElement("div");
    div.className = `agent-msg agent-${kind}`;
    div.textContent = text;
    els.log.appendChild(div);
    els.log.scrollTop = els.log.scrollHeight;
    return div;
  }

  function requestApproval(name, args) {
    setStatus("approval");
    return new Promise((resolve) => {
      const div = document.createElement("div");
      div.className = "agent-msg agent-approval";
      const label = document.createElement("span");
      label.textContent = `Allow ${name} ${JSON.stringify(args ?? {})}? `;
      const yes = document.createElement("button");
      yes.type = "button";
      yes.textContent = "Approve";
      const no = document.createElement("button");
      no.type = "button";
      no.textContent = "Deny";
      const done = (value) => {
        yes.disabled = true;
        no.disabled = true;
        label.textContent += value ? "(approved)" : "(denied)";
        setStatus("working");
        resolve(value);
      };
      yes.addEventListener("click", () => done(true));
      no.addEventListener("click", () => done(false));
      div.append(label, yes, no);
      els.log.appendChild(div);
      els.log.scrollTop = els.log.scrollHeight;
    });
  }

  function onEvent(ev) {
    if (ev.type === "text") addLine("asst", ev.text);
    else if (ev.type === "tool-call") addLine("call", `call: ${ev.name} ${JSON.stringify(ev.args ?? {})}`);
    else if (ev.type === "tool-result") {
      const r = ev.result ?? {};
      addLine(
        r.ok ? "result" : "error",
        r.ok ? `${ev.name} ok: ${JSON.stringify(r.result ?? null)}` : `${ev.name} failed: ${r.error ?? "?"}`
      );
    } else if (ev.type === "error") addLine("error", ev.error);
  }

  async function send(text) {
    const key = els.key.value.trim();
    if (!key) {
      addLine("error", "Paste a Meta API key in Settings first.");
      return;
    }
    if (busy || !text) return;
    busy = true;
    els.send.disabled = true;
    setStatus("working");
    messages.push({ role: "user", content: text });
    addLine("user", text);
    try {
      const out = await runAgentTurn({
        dispatcher,
        messages,
        snapshotText: formatSnapshot(getSnapshot()),
        config: {
          baseUrl: els.url.value.trim() || DEFAULT_BASE_URL,
          key,
          model: els.model.value.trim() || DEFAULT_MODEL,
        },
        approve: requestApproval,
        onEvent,
      });
      messages = out.messages.slice(-MAX_HISTORY);
      setStatus(out.error ? "error" : "idle");
    } catch (err) {
      addLine("error", String(err?.message ?? err));
      setStatus("error");
    }
    busy = false;
    els.send.disabled = false;
  }

  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = els.input.value.trim();
    els.input.value = "";
    void send(text);
  });

  // Settings form exists only to give the password field a form owner;
  // Enter there must not navigate.
  $("agent-settings-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
  });

  // Universal audit trail: chat AND console dispatches render here.
  const actionsLog = $("agent-actions-log");
  if (log && actionsLog) {
    const renderAction = (e) => {
      const div = document.createElement("div");
      div.className = `agent-msg agent-action${e.ok ? "" : " agent-error"}`;
      const t = String(e.time ?? "").slice(11, 19);
      div.textContent = e.ok
        ? `#${e.seq} ${t} [${e.source}] ${e.name} ${JSON.stringify(e.args ?? {})} ok ${e.ms}ms`
        : `#${e.seq} ${t} [${e.source}] ${e.name} FAIL ${e.error ?? "?"}`;
      actionsLog.appendChild(div);
      actionsLog.scrollTop = actionsLog.scrollHeight;
    };
    for (const entry of log.entries) renderAction(entry);
    log.subscribe(renderAction);
  }

  addLine("info", "Agent console. Paste a Meta API key in Settings, then ask — e.g. “spawn 20 agents and focus agent 3”.");
  setStatus("idle");
  return { send };
}

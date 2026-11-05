/**
 * Transparency views: full POST + full return for API calls, plus run logs.
 * Pure HTML string builders (no DOM) shared by the in-card Result tabs and
 * the inspector API tab. Matches the backend contract: complete bodies with
 * only inline base64 redacted (see redactBlobs, applied at capture time).
 */

import { escapeHtml, formatRunLogTime } from "./util.js";

export const META_API_URL = "https://api.meta.ai/v1/responses";

function codeBlock(title, obj, open = false) {
  const text = escapeHtml(JSON.stringify(obj ?? null, null, 2));
  return `
    <details class="flow-api-details"${open ? " open" : ""}>
      <summary class="flow-api-summary">${escapeHtml(title)}</summary>
      <div class="reui-code-box">
        <pre class="reui-code-box-pre">${text}</pre>
      </div>
    </details>
  `;
}

function metaGrid(meta) {
  const rows = [
    ["model", meta.model],
    ["status", meta.status],
    ["latency", meta.latency_s != null ? `${meta.latency_s}s` : null],
    ["request", meta.request_id],
    ["response", meta.response_id],
    ["usage", meta.usage && Object.keys(meta.usage).length ? JSON.stringify(meta.usage) : null],
  ].filter(([, v]) => v !== null && v !== undefined && v !== "");
  if (rows.length === 0) return "";
  return `
    <div class="flow-api-meta">
      ${rows.map(([k, v]) => `
        <div class="flow-api-meta-row">
          <span class="flow-api-meta-key">${escapeHtml(k)}</span>
          <span class="flow-api-meta-val">${escapeHtml(String(v))}</span>
        </div>`).join("")}
    </div>
  `;
}

export function renderApiTraceHtml(apiCall) {
  if (!apiCall || !apiCall.hop1) {
    return `<div class="flow-api-empty">No API trace captured yet — run this card.</div>`;
  }
  const h1 = apiCall.hop1;
  const statusLine = h1.status ? `HTTP ${h1.status}` : (h1.error ? "no response" : "awaiting response…");
  const errHtml = h1.error
    ? `<div class="flow-api-error">Error: ${escapeHtml(h1.error)}</div>`
    : "";

  let hop2Html = "";
  const trace = apiCall.hop2;
  if (trace) {
    const upstreamUrl = apiCall.hop2url || META_API_URL;
    hop2Html = `
      <div class="flow-api-hop-title">Hop 2 · chat_server → Meta Model API</div>
      <div class="flow-api-hop-sub">${escapeHtml(`POST ${upstreamUrl}`)}</div>
      ${metaGrid(trace.meta || {})}
      ${codeBlock("Upstream POST (full body, images redacted)", trace.post)}
      ${codeBlock("Upstream response (full return)", trace.response)}
    `;
  } else if (h1.status) {
    hop2Html = apiCall.local
      ? `<div class="flow-api-empty">Local GPU compute — no upstream API call.</div>`
      : `<div class="flow-api-empty">No upstream trace — the backend returns none on errors.</div>`;
  }

  return `
    <div class="flow-api-hop-title">Hop 1 · viewer → chat_server</div>
    <div class="flow-api-hop-sub">${escapeHtml(`${h1.method || "POST"} ${h1.url || ""}`)} · ${escapeHtml(statusLine)}</div>
    ${errHtml}
    ${codeBlock("Request (full body, images redacted)", h1.request)}
    ${codeBlock("Response (full return)", h1.response, true)}
    ${hop2Html}
  `;
}

export function renderRunLogHtml(runLog) {
  if (!runLog || runLog.length === 0) {
    return `<div class="flow-api-empty">No runs yet.</div>`;
  }
  const rows = [...runLog].reverse().map(e => `
    <div class="flow-runlog-row${e.level === "error" ? " is-error" : ""}">
      <span class="flow-runlog-time">${formatRunLogTime(e.t)}</span>
      <span class="flow-runlog-msg">${escapeHtml(e.msg)}</span>
    </div>`).join("");
  return `<div class="flow-runlog">${rows}</div>`;
}

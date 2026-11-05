/**
 * Step Inspector & Diagnostics Drawer
 * Renders tabs for Visuals & Artifacts, Hardware & ModelSlot memory residency, and API Payloads.
 */

import { escapeHtml, resolveAssetImageUrl, sameHtml } from "./util.js";
import { flowAssetChronology, formatFlowTime } from "./persist.js";
import { renderApiTraceHtml, renderRunLogHtml } from "./trace-view.js";
import { retryCardImage } from "./node-renderer.js";

export function selectNode(app, nodeId) {
  app.selectedNodeId = nodeId;
  const node = app.nodes.find(n => n.id === nodeId);
  if (!node) return;

  document.querySelectorAll(".flow-node").forEach(el => {
    el.classList.toggle("selected", el.id === nodeId);
  });

  app.inspector.classList.remove("collapsed");
  document.getElementById("inspector-node-title").textContent = node.title;
  const kindLabel = node.nodeType === "action"
    ? (node.styleKind === "meta-api" ? "API" : "LOCAL")
    : node.nodeType.toUpperCase();
  document.getElementById("inspector-node-subtitle").textContent = `${node.subtitle} · ${kindLabel}`;
  document.getElementById("inspector-node-badge").textContent = node.category;
  document.getElementById("inspector-node-badge").dataset.variant = node.badgeVariant;

  retryCardImage(app, nodeId);
  renderInspectorContent(app);
}

export function renderInspectorContent(app) {
  const node = app.nodes.find(n => n.id === app.selectedNodeId) || app.nodes[0];
  if (!node) return;

  const container = document.getElementById("inspector-dynamic-content");

  let html = "";
  let apiCopyText = "";
  if (app.activeTab === "overview") {
    const artUrl = (node.nodeType === "asset" ? resolveAssetImageUrl(app, node) : null) || node.imgUrl || node.resultImage;
    const artFallback = node.imgFallback || node.resultImageFallback;
    const artBadge = node.imgBadge || node.resultImageBadge;
    let visualBlock = "";
    if (artUrl) {
      visualBlock = `
        <div class="inspector-prop-row">
          <span class="inspector-prop-label">Visual Asset Payload</span>
          <img class="inspector-artifact-img" src="${escapeHtml(artUrl)}" onerror="this.onerror=null;this.src='${escapeHtml(artFallback || "")}'" alt="${escapeHtml(node.title)}">
          <span style="font-size:10.5px;font-family:var(--font-mono);color:var(--text-subtle)">${escapeHtml(artBadge || "")}</span>
        </div>
      `;
    }

    const isDemoRun = String(node.status || "").includes("Demo");
    let runNoteBlock = "";
    if (node.runError && isDemoRun) {
      runNoteBlock = `
        <div class="inspector-prop-row">
          <span class="inspector-prop-label">Run Note</span>
          <div style="font-size:11.5px;color:var(--warning);background:var(--warning-light);padding:8px 10px;border-radius:var(--radius-sm);border:1px solid var(--warning-border)">
            Backend unreachable — showing demo outputs. Cause: ${escapeHtml(node.runError)}
          </div>
        </div>
      `;
    } else if (node.runError) {
      runNoteBlock = `
        <div class="inspector-prop-row">
          <span class="inspector-prop-label">Run Note</span>
          <div style="font-size:11.5px;color:var(--destructive);background:var(--destructive-light);padding:8px 10px;border-radius:var(--radius-sm);border:1px solid var(--destructive-border)">
            Run failed — no outputs produced. Cause: ${escapeHtml(node.runError)}
          </div>
        </div>
      `;
    } else if (node.runNote) {
      runNoteBlock = `
        <div class="inspector-prop-row">
          <span class="inspector-prop-label">Run Note</span>
          <div style="font-size:11.5px;color:var(--info);background:var(--info-light);padding:8px 10px;border-radius:var(--radius-sm);border:1px solid var(--info-border)">
            ${escapeHtml(node.runNote)}
          </div>
        </div>
      `;
    } else if (node.demo) {
      runNoteBlock = `
        <div class="inspector-prop-row">
          <span class="inspector-prop-label">Run Note</span>
          <div style="font-size:11.5px;color:var(--warning);background:var(--warning-light);padding:8px 10px;border-radius:var(--radius-sm);border:1px solid var(--warning-border)">
            Demo artifact — produced while the backend was unreachable.
          </div>
        </div>
      `;
    }

    let ioBlock = "";
    if (node.consumes || node.produces) {
      ioBlock = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
          ${node.consumes ? `
            <div class="inspector-prop-row">
              <span class="inspector-prop-label">Consumes</span>
              <div class="inspector-prop-val">${escapeHtml(node.consumes.join(", "))}</div>
            </div>` : ""}
          ${node.produces ? `
            <div class="inspector-prop-row">
              <span class="inspector-prop-label">Produces</span>
              <div class="inspector-prop-val">${escapeHtml(node.produces.join(", "))}</div>
            </div>` : ""}
        </div>
      `;
    }

    html = `
      ${visualBlock}
      ${runNoteBlock}

      <div class="inspector-prop-row">
        <span class="inspector-prop-label">Description & Role</span>
        <div style="font-size:12px;color:var(--text-muted);line-height:1.5">${escapeHtml(node.description)}</div>
      </div>

      ${ioBlock}

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div class="inspector-prop-row">
          <span class="inspector-prop-label">Entity Type</span>
          <div class="inspector-prop-val">${node.nodeType === "action" ? "Action / Model" : node.nodeType === "note" ? "Board Note" : "Asset Artifact"}</div>
        </div>
        <div class="inspector-prop-row">
          <span class="inspector-prop-label">Format / Tensor</span>
          <div class="inspector-prop-val">${escapeHtml(node.tensorShape || node.format || "Standard")}</div>
        </div>
      </div>
    `;

  } else if (app.activeTab === "hardware") {
    html = `
      <div class="inspector-prop-row">
        <span class="inspector-prop-label">Model Family & Origin</span>
        <div class="inspector-prop-val">${escapeHtml(node.familyLabel || "Core Foundation Utility")}</div>
      </div>
      <div class="inspector-prop-row">
        <span class="inspector-prop-label">Execution Environment</span>
        <div class="inspector-prop-val">${escapeHtml(node.runtime || "CPU In-Memory")}</div>
      </div>
      <div class="inspector-prop-row">
        <span class="inspector-prop-label">Residency Policy</span>
        <div class="inspector-prop-val">${escapeHtml(node.residency || node.status)}</div>
      </div>
      <div class="inspector-prop-row">
        <span class="inspector-prop-label">Execution Latency</span>
        <div class="inspector-prop-val">${escapeHtml(node.latency)}</div>
      </div>
      <div class="inspector-prop-row">
        <span class="inspector-prop-label">Sequential ModelSlot Policy</span>
        <div style="font-size:11.5px;color:var(--text-muted);background:var(--surface-elevated);padding:8px 10px;border-radius:var(--radius-sm);border:1px solid var(--line)">
          Sapiens models share an isolated <code>ModelSlot</code> on CUDA. Each model is loaded, evaluated, and immediately evicted via <code>del + gc.collect() + torch.cuda.empty_cache()</code>, keeping VRAM strictly under the 95% threshold.
        </div>
      </div>
    `;
  } else if (app.activeTab === "assets") {
    const assets = flowAssetChronology(app.nodes);
    html = assets.length === 0
      ? `<div class="flow-api-empty">No assets yet — run a card to produce some.</div>`
      : `<div class="flow-asset-list">${assets.map(a => {
        const thumb = a.imgUrl || a.resultImage;
        const sub = [a.imgBadge || a.assetKind || "", a.status || ""].filter(Boolean).join(" · ");
        return `
          <button type="button" class="flow-asset-row${a.id === app.selectedNodeId ? " selected" : ""}" data-node-id="${escapeHtml(a.id)}" title="Select this card">
            ${thumb ? `<img class="flow-asset-thumb" src="${escapeHtml(thumb)}" alt="" loading="lazy" onerror="this.style.display='none'">` : `<span class="flow-asset-thumb flow-asset-thumb-none"></span>`}
            <span class="flow-asset-row-info">
              <span class="flow-asset-row-title">${escapeHtml(a.title || a.id)}</span>
              <span class="flow-asset-row-sub">${escapeHtml(sub)}${a.createdAt ? ` · ${escapeHtml(formatFlowTime(a.createdAt))}` : ""}</span>
            </span>
          </button>`;
      }).join("")}</div>`;
  } else if (app.activeTab === "api") {
    const payload = {
      node_id: node.id,
      node_type: node.nodeType,
      family: node.family || null,
      title: node.title,
      latency: node.latency,
      status: node.status,
      consumes: node.consumes || null,
      produces: node.produces || null,
      tensor_shape: node.tensorShape || null,
      list_items: node.resultData ? node.resultData.count : null,
      run_id: app.activeRunId
    };
    const codeText = JSON.stringify(payload, null, 2);

    const fullTrace = { api_call: node.apiCall || null, run_log: node.runLog || [], node: payload };
    const fullTraceText = JSON.stringify(fullTrace, null, 2);

    html = `
      <div class="reui-code-box">
        <div class="reui-code-box-header">
          <span>API Trace — full POST + full return</span>
          <button id="btn-copy-code" class="reui-btn sm">Copy</button>
        </div>
      </div>
      ${renderApiTraceHtml(node.apiCall)}
      <div class="flow-api-hop-title" style="margin-top:12px">Run log</div>
      ${renderRunLogHtml(node.runLog)}
      <details class="flow-api-details">
        <summary class="flow-api-summary">Entity JSON signature</summary>
        <div class="reui-code-box">
          <pre class="reui-code-box-pre">${escapeHtml(codeText)}</pre>
        </div>
      </details>
    `;

    apiCopyText = fullTraceText;
  }

  // Skip identical repaints: selection-heavy flows re-render constantly, and
  // re-parsing a large trace plus re-decoding the artifact image is the
  // slowest part of a pointer interaction.
  if (!html || sameHtml(container, `${app.activeTab}\n${html}`)) return;
  container.innerHTML = html;
  if (app.activeTab === "api") {
    container.querySelector("#btn-copy-code").addEventListener("click", () => {
      navigator.clipboard.writeText(apiCopyText);
      const b = container.querySelector("#btn-copy-code");
      b.textContent = "Copied!";
      setTimeout(() => b.textContent = "Copy", 1500);
    });
  }
  if (app.activeTab === "assets") {
    container.querySelectorAll(".flow-asset-row").forEach(row => {
      row.addEventListener("click", () => selectNode(app, row.dataset.nodeId));
    });
  }
}

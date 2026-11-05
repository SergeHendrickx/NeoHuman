// Regression tests for the transparency views (card tabs + inspector).
// Run from the flow-viewer folder:  node tests/trace-view.test.mjs
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { renderApiTraceHtml, renderRunLogHtml } from "../js/trace-view.js";

const TRACE = {
  at: 1700000000000,
  hop1: {
    method: "POST",
    url: "/api/debug/detect",
    status: 200,
    request: { chat_id: "flow-session", concept: "people", image_b64: "<3 base64 chars redacted>" },
    response: { count: 2, concept: "people" },
    error: null,
  },
  hop2: {
    post: { model: "sam-3.1", input: [] },
    response: { output_text: "<sam tokens>" },
    meta: { status: 200, latency_s: 1.85, request_id: "req-1", model: "sam-3.1",
      response_id: "resp-1", usage: { input_tokens: 10 } },
  },
};

describe("renderApiTraceHtml", () => {
  it("renders both hops with meta grid and full bodies", () => {
    const html = renderApiTraceHtml(TRACE);
    assert.match(html, /Hop 1 · viewer → chat_server/);
    assert.match(html, /POST \/api\/debug\/detect/);
    assert.match(html, /HTTP 200/);
    assert.match(html, /Hop 2 · chat_server → Meta Model API/);
    assert.match(html, /https:\/\/api\.meta\.ai\/v1\/responses/);
    assert.match(html, /req-1/);
    assert.match(html, /resp-1/);
    assert.match(html, /1\.85s/);
    assert.match(html, /&quot;concept&quot;/);
  });

  it("escapes hostile payload content", () => {
    const evil = structuredClone(TRACE);
    evil.hop1.request.concept = `"><img src=x onerror=alert(1)>`;
    evil.hop2.meta.model = `<b>sam</b>`;
    const html = renderApiTraceHtml(evil);
    assert.doesNotMatch(html, /<img src=x/);
    assert.match(html, /&lt;img src=x/);
    assert.doesNotMatch(html, /<b>sam<\/b>/);
  });

  it("renders a pending request while awaiting the response", () => {
    const pending = {
      at: 1700000000000,
      hop1: {
        method: "POST",
        url: "/api/debug/muse",
        status: null,
        request: { chat_id: "flow-session", prompt: "a cat" },
        response: null,
        error: null,
      },
      hop2: null,
    };
    const html = renderApiTraceHtml(pending);
    assert.match(html, /POST \/api\/debug\/muse/);
    assert.match(html, /awaiting response/);
    assert.match(html, /&quot;prompt&quot;/);
    assert.doesNotMatch(html, /Hop 2/);
    assert.doesNotMatch(html, /returns none on errors/);
  });

  it("prefers an explicit hop 2 URL over the responses default", () => {
    const muse = structuredClone(TRACE);
    muse.hop2url = "https://api.meta.ai/v1/images/edits";
    const html = renderApiTraceHtml(muse);
    assert.match(html, /https:\/\/api\.meta\.ai\/v1\/images\/edits/);
    assert.doesNotMatch(html, /v1\/responses/);
  });

  it("labels local compute instead of blaming errors", () => {
    const local = structuredClone(TRACE);
    local.hop2 = null;
    local.local = true;
    const html = renderApiTraceHtml(local);
    assert.match(html, /Local GPU compute/);
    assert.doesNotMatch(html, /returns none on errors/);
  });

  it("handles missing trace, errors and missing hop 2", () => {
    assert.match(renderApiTraceHtml(null), /No API trace captured yet/);
    const failed = structuredClone(TRACE);
    failed.hop1.status = 502;
    failed.hop1.error = "boom";
    failed.hop2 = null;
    const html = renderApiTraceHtml(failed);
    assert.match(html, /HTTP 502/);
    assert.match(html, /Error: boom/);
    assert.match(html, /returns none on errors/);
  });
});

describe("renderRunLogHtml", () => {
  it("renders newest-first rows with error styling and escaping", () => {
    const html = renderRunLogHtml([
      { t: 1700000000000, level: "info", msg: "Run started" },
      { t: 1700000001000, level: "error", msg: "Failed: <x>" },
    ]);
    assert.match(html, /is-error/);
    assert.match(html, /Failed: &lt;x&gt;/);
    assert.ok(html.indexOf("Failed") < html.indexOf("Run started"));
  });

  it("handles empty logs", () => {
    assert.match(renderRunLogHtml(null), /No runs yet/);
    assert.match(renderRunLogHtml([]), /No runs yet/);
  });
});

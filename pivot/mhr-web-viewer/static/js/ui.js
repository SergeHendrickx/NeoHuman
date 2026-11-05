import { currentParameter, elements, isLocked, state } from "./state.js";
import { formatCount, formatNumber } from "./utils.js";

export function updateInferenceTime(response) {
  const inferenceMs = Number(response.headers.get("X-MHR-Inference-Ms"));
  elements.renderTime.textContent = Number.isFinite(inferenceMs)
    ? `${formatNumber(inferenceMs, 1)} ms`
    : "—";
}

export function setStatus(message, kind = "ready") {
  elements.statusText.textContent = message;
  elements.status.classList.toggle("is-busy", kind === "busy");
  elements.status.classList.toggle("is-error", kind === "error");
}

export function showLoading(message) {
  elements.loadingText.textContent = message;
  elements.loading.classList.remove("is-hidden");
}

export function hideLoading() {
  elements.loading.classList.add("is-hidden");
}

export function setInputsDisabled(disabled) {
  state.requestPending = disabled;
  [
    elements.resetCurrent,
    elements.resetGroup,
    elements.resetAll,
    elements.correctives,
  ].forEach((element) => {
    element.disabled = disabled;
  });
  elements.parameter.setAttribute("aria-disabled", String(disabled));
  elements.parameter.querySelectorAll(".parameter-item").forEach((item) => {
    item.disabled = disabled;
  });
  const parameter = state.metadata ? currentParameter() : null;
  const locked = parameter ? isLocked(parameter) : false;
  elements.slider.disabled = disabled || locked;
  elements.value.disabled = disabled || locked;
  elements.resetCurrent.disabled = disabled || locked;
}

export function updateModelStatistics() {
  const model = state.metadata.model;
  elements.vertexCount.textContent = formatCount(model.vertexCount);
  elements.faceCount.textContent = formatCount(model.faceCount);
  elements.jointCount.textContent = formatCount(model.jointCount);
}

export function updateModelControls() {
  const model = state.metadata.model;
  elements.correctives.classList.toggle("is-on", model.applyCorrectives);
  elements.correctives.setAttribute(
    "aria-checked",
    String(model.applyCorrectives),
  );
  elements.correctivesState.textContent = model.applyCorrectives ? "On" : "Off";
}

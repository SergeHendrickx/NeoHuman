import * as THREE from "three";

export function formatNumber(value, digits = 4) {
  return Number(value).toFixed(digits);
}

export function formatCount(value) {
  return new Intl.NumberFormat().format(value);
}

export function createDotTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 32;
  canvas.height = 32;
  const context = canvas.getContext("2d");
  context.fillStyle = "#ffffff";
  context.beginPath();
  context.arc(16, 16, 14, 0, Math.PI * 2);
  context.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export async function fetchChecked(resource, options, fallbackMessage) {
  const response = await fetch(resource, options);
  if (response.ok) return response;

  let message = fallbackMessage;
  try {
    const payload = await response.json();
    message = payload.error || message;
  } catch {
    // The backend may return an HTML or empty error response.
  }
  throw new Error(message);
}

export function downloadFilename(response, fallback) {
  const disposition = response.headers.get("Content-Disposition") || "";
  const match = disposition.match(/filename="?([^";]+)"?/i);
  return match ? match[1] : fallback;
}

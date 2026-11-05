/**
 * Shareable NeoHuman codes: avatar JSON -> deflate -> base64url text.
 * Pure module (CompressionStream + TextEncoder only) so node can test it.
 */
export const AVATAR_CODE_VERSION = 1;

const B64U = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

function bytesToB64url(bytes) {
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const n = (bytes[i] << 16) | ((bytes[i + 1] ?? 0) << 8) | (bytes[i + 2] ?? 0);
    out += B64U[(n >> 18) & 63] + B64U[(n >> 12) & 63];
    if (i + 1 < bytes.length) out += B64U[(n >> 6) & 63];
    if (i + 2 < bytes.length) out += B64U[n & 63];
  }
  return out;
}

function b64urlToBytes(text) {
  if (typeof text !== "string" || text.length % 4 === 1 || /[^A-Za-z0-9_-]/.test(text)) {
    throw new Error("bad characters");
  }
  const vals = [...text].map((ch) => B64U.indexOf(ch));
  const out = [];
  for (let i = 0; i < vals.length; i += 4) {
    const n = (vals[i] << 18) | ((vals[i + 1] ?? 0) << 12)
      | ((vals[i + 2] ?? 0) << 6) | (vals[i + 3] ?? 0);
    out.push((n >> 16) & 255);
    if (i + 2 < vals.length) out.push((n >> 8) & 255);
    if (i + 3 < vals.length) out.push(n & 255);
  }
  return new Uint8Array(out);
}

async function deflateJson(data) {
  const raw = new TextEncoder().encode(JSON.stringify(data));
  const stream = new CompressionStream("deflate");
  const writer = stream.writable.getWriter();
  await writer.write(raw);
  await writer.close();
  return new Uint8Array(await new Response(stream.readable).arrayBuffer());
}

async function inflateJson(bytes) {
  const stream = new DecompressionStream("deflate");
  const writer = stream.writable.getWriter();
  await writer.write(bytes);
  await writer.close();
  const buf = await new Response(stream.readable).arrayBuffer();
  return JSON.parse(new TextDecoder().decode(buf));
}

export async function packAvatarData(data) {
  return bytesToB64url(await deflateJson(data));
}

export async function unpackAvatarCode(code) {
  let bytes;
  try {
    bytes = b64urlToBytes(String(code ?? ""));
  } catch {
    throw new Error("Not an avatar code (bad characters).");
  }
  if (!bytes.length) throw new Error("Not an avatar code (empty).");
  try {
    return await inflateJson(bytes);
  } catch {
    throw new Error("Not an avatar code (unreadable).");
  }
}

/** Structural check against live group lengths; throws with a human reason. */
export function validateAvatarData(data, counts) {
  if (!data || typeof data !== "object") throw new Error("Not an avatar code.");
  if (data.v !== AVATAR_CODE_VERSION) {
    throw new Error(`Unsupported avatar code version ${data.v ?? "?"} (want ${AVATAR_CODE_VERSION}).`);
  }
  for (const [group, n] of Object.entries(counts)) {
    const arr = data.params?.[group];
    if (!Array.isArray(arr) || arr.length !== n || !arr.every(Number.isFinite)) {
      throw new Error(`Avatar code has a bad ${group} group.`);
    }
  }
  return data;
}

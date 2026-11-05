import { fetchChecked } from "./utils.js";

const DB_NAME = "MHR_CACHE_V3";
const STORE_NAME = "binary_assets";
const DB_VERSION = 1;

function openDB() {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") {
      reject(new Error("IndexedDB not supported"));
      return;
    }
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (e) => {
      const db = e.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = (e) => resolve(e.target.result);
    request.onerror = (e) => reject(e.target.error);
  });
}

async function getCachedAsset(key) {
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

async function setCachedAsset(key, arrayBuffer) {
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readwrite");
      const store = tx.objectStore(STORE_NAME);
      store.put(arrayBuffer, key);
      tx.oncomplete = () => resolve(true);
      tx.onerror = () => resolve(false);
    });
  } catch {
    // Non-fatal if storage fails or quota is exceeded
  }
}

export async function fetchBinaryAsset(url, fallbackMessage) {
  const cached = await getCachedAsset(url);
  if (cached) {
    return cached;
  }
  const response = await fetchChecked(url, { cache: "force-cache" }, fallbackMessage);
  const buffer = await response.arrayBuffer();
  try {
    setCachedAsset(url, buffer.slice(0));
  } catch {
    // Non-fatal
  }
  return buffer;
}

/*
 * The API, in one place.
 *
 * The page is rendered by `backend/api/flask_app.py` from the same
 * origin as `/api/*`, so there is nothing to configure and no CORS to
 * arrange. That template also injects `window.__VERILENS__` before this
 * bundle loads — the API base and the limits the server enforces — so
 * the client can refuse an over-long message without a round trip
 * instead of keeping its own copy of numbers that live in Python.
 *
 * Two cases arrive without it: `npm run dev` serves the page from Vite
 * and proxies `/api` to Flask (still same-origin to the browser), and a
 * bundle opened straight off disk has no origin to borrow, so it points
 * at the default dev address. Hence the fallbacks.
 */
const CONFIG = window.__VERILENS__ || {}

const ORIGIN =
  window.location.protocol === 'file:'
    ? 'http://127.0.0.1:5000'
    : window.location.origin

const API = ORIGIN + (CONFIG.apiBase || '/api')

/** What the server will refuse, so the page can say so first. */
const LIMITS = {
  maxTextChars: CONFIG.maxTextChars || 10000,
  maxUploadBytes: CONFIG.maxUploadBytes || 64 * 1024 * 1024,
}

export { API, LIMITS }

/**
 * POST to an endpoint and return the parsed body.
 *
 * A FormData payload is sent as a multipart form (that is how a file
 * gets there); anything else goes as JSON. A failed *fetch* means the
 * server is not up, which is a different problem from a failed
 * analysis, so it gets its own message.
 */
export async function check(path, payload) {
  const options =
    payload instanceof FormData
      ? { method: 'POST', body: payload }
      : {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }

  let response
  try {
    response = await fetch(API + path, options)
  } catch {
    throw new Error(
      `Cannot reach the API at ${API}. Is the Flask server running?`
    )
  }

  const data = await response.json().catch(() => ({}))

  if (!response.ok) {
    throw new Error(data.detail || `Request failed (${response.status})`)
  }

  return data
}

/** Liveness, plus which optional retrievers and models are loaded. */
export async function getHealth() {
  const response = await fetch(API + '/health')

  if (!response.ok) throw new Error(`Health check failed (${response.status})`)

  return response.json()
}

/* ---- small shared helpers the views all need ---- */

const LABEL_CLASS = {
  true: 'green',
  supported: 'green',
  false: 'red',
  fabricated: 'red',
  misleading: 'amber',
  disputed: 'amber',
  unverified: 'amber',
}

/** A verdict label mapped to the colour class the stylesheet defines. */
export function labelClass(label) {
  return LABEL_CLASS[String(label).toLowerCase()] || 'amber'
}

/** A 0..1 confidence as a whole percentage. */
export function pct(x) {
  return Math.round((Number(x) || 0) * 100)
}

// Default to same-origin (""), so Vite's dev server proxy can forward /api/* to the backend.
// If you deploy frontend and backend separately, set VITE_API_BASE (e.g. http://127.0.0.1:8000).
const API_BASE = import.meta.env.VITE_API_BASE || ''

async function httpJson(path, options = {}) {
  const { timeoutMs = 0, ...fetchOptions } = options || {}
  const controller = timeoutMs > 0 ? new AbortController() : null
  let timer = null
  if (controller) {
    timer = window.setTimeout(() => controller.abort(), timeoutMs)
  }

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      cache: 'no-store',
      headers: { 'Content-Type': 'application/json', ...(fetchOptions.headers || {}) },
      ...fetchOptions,
      ...(controller ? { signal: controller.signal } : {})
    })
    if (!res.ok) {
      const txt = await res.text()
      throw new Error(`HTTP ${res.status}: ${txt}`)
    }
    return await res.json()
  } catch (e) {
    // Fetch throws a TypeError on network/proxy/CORS failures. AbortError means
    // the backend did not respond within the explicit timeout for this action.
    const msg = String(e?.message || e)
    if (e?.name === 'AbortError') {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s: ${path}`)
    }
    if (msg.toLowerCase().includes('failed to fetch') || msg.toLowerCase().includes('networkerror')) {
      throw new Error(
        `${msg}. Backend might be offline. Make sure FastAPI is running (default: http://127.0.0.1:8000) and that the frontend proxy/CORS is configured.`
      )
    }
    throw e
  } finally {
    if (timer) window.clearTimeout(timer)
  }
}

export async function createRun({ project_path, prompt, copy_project_to_workspace = false, input_mode = 'text', backend = null }) {
  return await httpJson('/api/runs', {
    method: 'POST',
    body: JSON.stringify({ project_path, prompt, copy_project_to_workspace, input_mode, backend })
  })
}



// ------------------------------
// Runtime LLM backend switching
// ------------------------------

export async function getModelBackend() {
  return await httpJson('/api/model/backend', { timeoutMs: 5000 })
}

export async function setModelBackend(backend) {
  return await httpJson('/api/model/backend', {
    method: 'POST',
    body: JSON.stringify({ backend }),
    timeoutMs: 0
  })
}

// ------------------------------
// Model feedback / RLHF
// ------------------------------

export async function submitModelFeedback(payload = {}) {
  return await httpJson('/api/feedback', {
    method: 'POST',
    body: JSON.stringify(payload)
  })
}

export async function getFeedbackStats() {
  return await httpJson('/api/feedback/stats', { timeoutMs: 10000 })
}

export async function getLocalStatus({ deep = false } = {}) {
  const qs = deep ? '?deep=true' : ''
  return await httpJson(`/api/local/status${qs}`, { timeoutMs: deep ? 30000 : 5000 })
}

export async function askQA({
  question,
  orchestrator = null,
  project_path = null,
  use_web = true,
  use_local_refs = true,
  input_mode = 'text',
  backend = null,
  // multi-turn
  conversation_id = null,
  conversation_title = null
}) {
  return await httpJson('/api/qa/ask', {
    method: 'POST',
    body: JSON.stringify({
      question,
      orchestrator,
      project_path,
      use_web,
      use_local_refs,
      input_mode,
      backend,
      conversation_id,
      conversation_title
    })
  })
}

export async function createQAConversation({ orchestrator = null, title = null } = {}) {
  return await httpJson('/api/qa/conversations', {
    method: 'POST',
    body: JSON.stringify({ orchestrator, title })
  })
}

export async function listQAConversations({ orchestrator = null, limit = 25 } = {}) {
  const qp = new URLSearchParams({ limit: String(limit) })
  if (orchestrator) qp.set('orchestrator', orchestrator)
  const res = await fetch(`${API_BASE}/api/qa/conversations?${qp.toString()}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

export async function getQAConversation(conversation_id, limit = 100) {
  const qp = new URLSearchParams({ limit: String(limit) })
  const res = await fetch(`${API_BASE}/api/qa/conversations/${encodeURIComponent(conversation_id)}?${qp.toString()}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

// ------------------------------
// Trace viewer API
// ------------------------------

export async function listConversationTraces(conversation_id, { turn_id = null, orchestrator = null, limit = 50 } = {}) {
  const qp = new URLSearchParams({ limit: String(limit) })
  if (turn_id) qp.set('turn_id', turn_id)
  if (orchestrator) qp.set('orchestrator', orchestrator)
  const res = await fetch(
    `${API_BASE}/api/qa/conversations/${encodeURIComponent(conversation_id)}/traces?${qp.toString()}`
  )
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

export async function getConversationTrace(trace_id) {
  const res = await fetch(`${API_BASE}/api/qa/traces/${encodeURIComponent(String(trace_id))}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

// Code pipeline run trace viewer
export async function listRunTraces(run_id, { orchestrator = null, limit = 50 } = {}) {
  const qp = new URLSearchParams({ limit: String(limit) })
  if (orchestrator) qp.set('orchestrator', orchestrator)
  const res = await fetch(
    `${API_BASE}/api/runs/${encodeURIComponent(String(run_id))}/traces?${qp.toString()}`
  )
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

export async function getRunTrace(trace_id) {
  const res = await fetch(`${API_BASE}/api/runs/traces/${encodeURIComponent(String(trace_id))}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}


export async function getRun(run_id) {
  return await httpJson(`/api/runs/${run_id}`)
}

export async function searchHiveMemory(q) {
  const res = await fetch(`${API_BASE}/api/memory/hive/search?q=${encodeURIComponent(q)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

export async function searchTypeMemory(agent_type, q) {
  const res = await fetch(
    `${API_BASE}/api/memory/type/${encodeURIComponent(agent_type)}/search?q=${encodeURIComponent(q)}`
  )
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

export async function searchAllMemory(q, scopes = null) {
  const qp = new URLSearchParams({ q })
  if (scopes) qp.set('scopes', scopes)
  const res = await fetch(`${API_BASE}/api/memory/all/search?${qp.toString()}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}


// ------------------------------
// Uploads (documents/images)
// ------------------------------

export async function uploadFile(file, { train = true, addToHive = true } = {}) {
  const fd = new FormData()
  fd.append('file', file)

  const qs = new URLSearchParams({
    train: String(!!train),
    add_to_hive: String(!!addToHive),
  })

  const res = await fetch(`${API_BASE}/api/uploads?${qs.toString()}`, {
    method: 'POST',
    body: fd,
  })

  if (!res.ok) {
    const txt = await res.text().catch(() => '')
    throw new Error(txt || `Upload failed (${res.status})`)
  }

  return res.json()
}

export async function listUploads({ limit = 50 } = {}) {
  const qs = new URLSearchParams({ limit: String(limit) })
  const res = await fetch(`${API_BASE}/api/uploads?${qs.toString()}`)
  if (!res.ok) {
    const txt = await res.text().catch(() => '')
    throw new Error(txt || `List uploads failed (${res.status})`)
  }
  return res.json()
}

export async function getUpload(uploadId) {
  const res = await fetch(`${API_BASE}/api/uploads/${encodeURIComponent(uploadId)}`)
  if (!res.ok) {
    const txt = await res.text().catch(() => '')
    throw new Error(txt || `Get upload failed (${res.status})`)
  }
  return res.json()
}


// ------------------------------
// Speech formatting + feedback
// ------------------------------

export async function formatSpeech({ text, mode = 'qa' } = {}) {
  return await httpJson('/api/speech/format', {
    method: 'POST',
    body: JSON.stringify({ text, mode })
  })
}

export async function submitSpeechFeedback({ mode = 'qa', raw_text, formatted_text, final_text } = {}) {
  return await httpJson('/api/speech/feedback', {
    method: 'POST',
    body: JSON.stringify({ mode, raw_text, formatted_text, final_text })
  })
}

export async function listSpeechReplacements({ mode = 'qa', limit = 100 } = {}) {
  const qp = new URLSearchParams({ mode: String(mode || 'qa'), limit: String(limit) })
  const res = await fetch(`${API_BASE}/api/speech/replacements?${qp.toString()}`)
  if (!res.ok) {
    const txt = await res.text().catch(() => '')
    throw new Error(txt || `List speech replacements failed (${res.status})`)
  }
  return await res.json()
}

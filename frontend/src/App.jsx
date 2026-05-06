import React, { useEffect, useMemo, useRef, useState } from 'react'
import { askQA, createRun, getRun, uploadFile, listUploads, getModelBackend, setModelBackend, submitModelFeedback, getFeedbackStats, getLocalStatus } from './api/client.js'
import PromptPanel from './components/PromptPanel.jsx'
import RunsPanel from './components/RunsPanel.jsx'
import LogsPanel from './components/LogsPanel.jsx'
import AgentPanel from './components/AgentPanel.jsx'
import TraceViewer from './components/TraceViewer.jsx'


const SELECTED_BACKEND_STORAGE_KEY = 'agenticHiveSelectedBackend'

function normalizeDashboardBackend(value, fallback = 'local') {
  const v = String(value || '').trim().toLowerCase()
  return ['local', 'gemini', 'openai'].includes(v) ? v : fallback
}

function readStoredBackend() {
  try {
    const raw = window.localStorage.getItem(SELECTED_BACKEND_STORAGE_KEY)
    if (!raw) return null
    return normalizeDashboardBackend(raw, null)
  } catch (_) {
    return null
  }
}

function hasStoredBackendSelection() {
  return !!readStoredBackend()
}

function writeStoredBackend(backend) {
  try {
    window.localStorage.setItem(SELECTED_BACKEND_STORAGE_KEY, normalizeDashboardBackend(backend))
  } catch (_) {}
}

function FeedbackControls({
  targetType,
  targetId,
  conversationId = null,
  runId = null,
  promptText = '',
  responseText = '',
  backend = 'local',
  projectPath = '',
  mode = 'qa',
  inputMode = 'text',
  onSubmitted,
  disabled = false
}) {
  const [open, setOpen] = useState(false)
  const [comment, setComment] = useState('')
  const [correction, setCorrection] = useState('')
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  async function send(score, opts = {}) {
    setSaving(true)
    setError('')
    setMessage('')
    try {
      const corrected = opts.corrected_response ?? correction
      const notes = opts.comment ?? comment
      const res = await submitModelFeedback({
        target_type: targetType,
        target_id: targetId,
        conversation_id: conversationId,
        run_id: runId,
        score,
        prompt: promptText,
        response: responseText,
        corrected_response: corrected,
        comment: notes,
        backend,
        project_path: projectPath,
        mode,
        input_mode: inputMode,
        add_to_training: true,
        meta: { source: 'dashboard_feedback_controls' }
      })
      setMessage(res?.message || 'Feedback saved.')
      if (corrected) setOpen(false)
      onSubmitted?.(res, { score, corrected_response: corrected, comment: notes })
    } catch (e) {
      setError(e?.message || String(e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="feedback-box">
      <div className="chat-actions">
        <button type="button" className="chat-chip" disabled={disabled || saving} onClick={() => send(1, { corrected_response: '', comment: comment || 'Good response' })}>
          👍 Good
        </button>
        <button type="button" className="chat-chip" disabled={disabled || saving} onClick={() => send(-1, { corrected_response: '', comment: comment || 'Bad response' })}>
          👎 Bad
        </button>
        <button type="button" className="chat-chip" disabled={disabled || saving} onClick={() => setOpen((v) => !v)}>
          ✍ Add correction
        </button>
      </div>

      {open ? (
        <div style={{ marginTop: 8 }}>
          <label className="small">Correction / preferred response</label>
          <textarea
            value={correction}
            onChange={(e) => setCorrection(e.target.value)}
            placeholder="Write the answer or patch the model should have produced. This is queued for the manual RLHF/LoRA trainer."
            style={{ minHeight: 90 }}
          />
          <label className="small" style={{ marginTop: 8 }}>Optional feedback note</label>
          <input
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="What was wrong or what should be preferred?"
          />
          <div className="chat-actions" style={{ marginTop: 8 }}>
            <button type="button" disabled={disabled || saving || !correction.trim()} onClick={() => send(0)}>
              {saving ? 'Saving…' : 'Save correction for RL'}
            </button>
            <button type="button" className="chat-chip" disabled={saving} onClick={() => setOpen(false)}>
              Cancel
            </button>
          </div>
        </div>
      ) : null}

      {message ? <div className="small" style={{ marginTop: 6 }}>{message}</div> : null}
      {error ? <div className="small" style={{ marginTop: 6 }}><b>Feedback error:</b> {error}</div> : null}
    </div>
  )
}

function ChatMessage({ msg, conversationId, onViewTrace, onFollowup, llmBackend, projectPath, promptText, promptInputMode, onFeedbackSubmitted }) {
  const isUser = msg.role === 'user'
  const title = isUser ? 'You' : (msg.orchestrator || 'assistant')
  const subtitle = isUser
    ? (msg.input_mode ? `input: ${msg.input_mode}` : '')
    : (msg.meta?.kind ? msg.meta.kind : '')

  const turnId = msg.meta?.turn_id
  const canTrace = Boolean(conversationId && turnId)
  const traceOrchestrator = isUser ? null : (msg.orchestrator || null)

  const result = msg.meta?.result || null
  const keyPoints = Array.isArray(result?.key_points) ? result.key_points : []
  const sources = Array.isArray(result?.sources_used) ? result.sources_used : []
  const followups = Array.isArray(result?.followups) ? result.followups : []

  return (
    <div className={`chat-row ${isUser ? 'user' : 'assistant'}`}>
      <div className="chat-bubble">
        <div className="chat-meta">
          <b>{title}</b>
          {subtitle ? <span style={{ opacity: 0.75 }}>{subtitle}</span> : null}
          <span style={{ opacity: 0.6 }}>{msg.created_at}</span>
        </div>

        <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>

        {!isUser && msg.meta?.domain ? (
          <div className="small" style={{ marginTop: 8, opacity: 0.85 }}>
            domain: {msg.meta.domain}
          </div>
        ) : null}

        {!isUser && keyPoints.length ? (
          <div style={{ marginTop: 10 }}>
            <div className="small" style={{ opacity: 0.9, marginBottom: 6 }}><b>Key points</b></div>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {keyPoints.slice(0, 12).map((kp, idx) => (
                <li key={idx} style={{ marginBottom: 4 }}>{kp}</li>
              ))}
            </ul>
          </div>
        ) : null}

        {!isUser && sources.length ? (
          <div style={{ marginTop: 10 }}>
            <div className="small" style={{ opacity: 0.9, marginBottom: 6 }}><b>Sources</b></div>
            <ul style={{ margin: 0, paddingLeft: 18 }}>
              {sources.slice(0, 8).map((u, idx) => (
                <li key={idx} style={{ marginBottom: 4 }}>
                  <a href={u} target="_blank" rel="noreferrer">{u}</a>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {!isUser && followups.length ? (
          <div style={{ marginTop: 10 }}>
            <div className="small" style={{ opacity: 0.9, marginBottom: 6 }}><b>Suggested follow-ups</b></div>
            <div className="chat-actions">
              {followups.slice(0, 6).map((fu, idx) => (
                <button
                  key={idx}
                  type="button"
                  className="chat-chip"
                  onClick={() => onFollowup?.(fu)}
                  title="Send follow-up"
                >
                  {fu}
                </button>
              ))}
            </div>
          </div>
        ) : null}

        {!isUser && !msg.meta?.pending ? (
          <FeedbackControls
            targetType="qa"
            targetId={turnId}
            conversationId={conversationId}
            promptText={promptText || ''}
            responseText={msg.content || ''}
            backend={llmBackend}
            projectPath={projectPath}
            mode="qa"
            inputMode={promptInputMode || 'text'}
            onSubmitted={onFeedbackSubmitted}
          />
        ) : null}

        {canTrace ? (
          <div className="chat-actions" style={{ marginTop: 10 }}>
            <button
              type="button"
              onClick={() => onViewTrace({ conversationId, turnId, initialOrchestrator: traceOrchestrator })}
              style={{ fontSize: 12 }}
            >
              View trace
            </button>
          </div>
        ) : null}
      </div>
    </div>
  )
}

export default function App() {
  const [mode, setMode] = useState('code') // 'code' | 'qa'
  const [qaOrchestrator, setQaOrchestrator] = useState('interactive')
  const [inputMode, setInputMode] = useState('text') // 'text' | 'voice'
  const [speakVoiceResponses, setSpeakVoiceResponses] = useState(() => {
    try {
      const raw = window.localStorage.getItem('speakVoiceResponses')
      return raw == null ? true : raw === 'true'
    } catch (_) {
      return true
    }
  })

  const [projectPath, setProjectPath] = useState('sample_projects/hello_py')
  const [codePrompt, setCodePrompt] = useState('Add a new function greet(name) and tests for it.')
  const [qaDraft, setQaDraft] = useState('')
  const [copyToWorkspace, setCopyToWorkspace] = useState(false)

  const [llmBackend, setLlmBackend] = useState(() => readStoredBackend() || 'local')
  const [llmStatus, setLlmStatus] = useState(null)
  const [backendStatusLoaded, setBackendStatusLoaded] = useState(false)
  const [modelSwitching, setModelSwitching] = useState(false)
  const [modelSwitchError, setModelSwitchError] = useState(null)
  const selectedBackendRef = useRef(llmBackend || 'local')

  const [feedbackStats, setFeedbackStats] = useState(null)
  const [feedbackStatsError, setFeedbackStatsError] = useState(null)
  const [feedbackNotice, setFeedbackNotice] = useState(null)
  const [localStatus, setLocalStatus] = useState(null)
  const [localStatusError, setLocalStatusError] = useState(null)

  const [runId, setRunId] = useState(null)
  const [run, setRun] = useState(null)

  // QA multi-turn state
  const [qa, setQa] = useState(null) // last turn response
  const [qaConversationId, setQaConversationId] = useState(null)
  const [qaMessages, setQaMessages] = useState([])
  const [qaUseWeb, setQaUseWeb] = useState(true)
  const [qaUseLocalRefs, setQaUseLocalRefs] = useState(true)
  const [qaError, setQaError] = useState(null)
  const [qaShowInternal, setQaShowInternal] = useState(false)

  // Uploads (documents/images) state
  const [uploads, setUploads] = useState([])
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [uploadTrain, setUploadTrain] = useState(true)
  const [uploadAddToHive, setUploadAddToHive] = useState(true)

  const qaBottomRef = useRef(null)

  useEffect(() => {
    const normalized = normalizeDashboardBackend(llmBackend, 'local')
    selectedBackendRef.current = normalized
    writeStoredBackend(normalized)
  }, [llmBackend])

  useEffect(() => {
    try {
      window.localStorage.setItem('speakVoiceResponses', String(!!speakVoiceResponses))
      if (!speakVoiceResponses) {
        window.speechSynthesis?.cancel()
      }
    } catch (_) {}
  }, [speakVoiceResponses])

  useEffect(() => {
    let cancelled = false
    getModelBackend()
      .then((status) => {
        if (cancelled) return
        setLlmStatus(status)
        // Do not let a slow/stale server default snap the dropdown back to
        // local. The dashboard selection is the source of truth for new
        // Q&A/code requests because every request sends backend explicitly.
        if (!hasStoredBackendSelection()) {
          setLlmBackend(normalizeDashboardBackend(status?.active_backend, 'local'))
        }
        setBackendStatusLoaded(true)
        setModelSwitchError(null)
      })
      .catch((e) => {
        if (cancelled) return
        setBackendStatusLoaded(true)
        setModelSwitchError(e?.message || String(e))
      })
    return () => {
      cancelled = true
    }
  }, [])

  async function refreshFeedbackStats() {
    try {
      const stats = await getFeedbackStats()
      setFeedbackStats(stats)
      setFeedbackStatsError(null)
      return stats
    } catch (e) {
      setFeedbackStatsError(e?.message || String(e))
      return null
    }
  }

  async function refreshLocalStatus(deep = false) {
    try {
      const status = await getLocalStatus({ deep })
      setLocalStatus(status)
      setLocalStatusError(null)
    } catch (e) {
      setLocalStatusError(e?.message || String(e))
    }
  }

  function handleFeedbackSubmitted(res, payload = {}) {
    setFeedbackNotice(res?.message || 'Feedback saved.')
    // Optimistically update the visible counters immediately, then reconcile
    // against the backend DB stats.
    setFeedbackStats((prev) => {
      const base = prev || { total: 0, positive: 0, negative: 0, corrections: 0, max_id: 0, by_backend: {}, by_target_type: {} }
      const reward = Number(res?.reward ?? payload?.score ?? 0)
      const hasCorrection = Boolean(String(payload?.corrected_response || '').trim())
      return {
        ...base,
        total: Number(base.total || 0) + 1,
        positive: Number(base.positive || 0) + (reward > 0 ? 1 : 0),
        negative: Number(base.negative || 0) + (reward < 0 ? 1 : 0),
        corrections: Number(base.corrections || 0) + (hasCorrection ? 1 : 0),
        max_id: Math.max(Number(base.max_id || 0), Number(res?.feedback_id || 0))
      }
    })
    refreshFeedbackStats()
  }

  useEffect(() => {
    refreshFeedbackStats()
    const iv = window.setInterval(() => refreshFeedbackStats(), 5000)
    return () => window.clearInterval(iv)
  }, [])

  useEffect(() => {
    // Shallow status only: do not auto-import torch/CUDA from the dashboard.
    // Deep CUDA checks are manual via the Check CUDA/RTX button.
    if (backendStatusLoaded && llmBackend === 'local') refreshLocalStatus(false)
  }, [backendStatusLoaded, llmBackend])


// ------------------------------
// Voice output (Text-to-Speech)
// ------------------------------
const ttsStateRef = useRef({
  runStatus: null,
  agentStates: {},
  logsLen: 0,
  lastError: null,
})

function speak(text, { interrupt = false } = {}) {
  try {
    if (typeof window === 'undefined') return
    if (!('speechSynthesis' in window) || !('SpeechSynthesisUtterance' in window)) return
    const msg = prepareSpeechText(text)
    if (!msg) return
    if (interrupt) window.speechSynthesis.cancel()
    const u = new SpeechSynthesisUtterance(msg)
    u.lang = 'en-US'
    u.rate = 1
    u.pitch = 1
    u.volume = 1
    window.speechSynthesis.speak(u)
  } catch (_) {}
}

function _firstLine(s) {
  const t = String(s || '').trim()
  if (!t) return ''
  const line = t.split(/\r?\n/)[0]
  return line.length > 180 ? line.slice(0, 177) + '…' : line
}

function prepareSpeechText(text) {
  let t = String(text || '').trim()
  if (!t) return ''
  t = t.replace(/```[\s\S]*?```/g, ' code omitted. ')
  t = t.replace(/`([^`]+)`/g, '$1')
  t = t.replace(/^#{1,6}\s*/gm, '')
  t = t.replace(/\*\*([^*]+)\*\*/g, '$1')
  t = t.replace(/\*([^*]+)\*/g, '$1')
  t = t.replace(/^\s*[-*+]\s+/gm, '')
  t = t.replace(/\[(.*?)\]\((.*?)\)/g, '$1')
  t = t.replace(/\s+/g, ' ').trim()
  return t
}

  // Trace viewer UI
  const [traceOpen, setTraceOpen] = useState(false)
  const [traceContext, setTraceContext] = useState({ conversationId: null, turnId: null, runId: null, initialOrchestrator: null })

  const [starting, setStarting] = useState(false)

  function resetConversation() {
    setQaConversationId(null)
    setQaMessages([])
    setQa(null)
    setQaError(null)
    setQaShowInternal(false)
    setTraceOpen(false)
    setTraceContext({ conversationId: null, turnId: null, runId: null, initialOrchestrator: null })
  }

  function openTrace({ conversationId = null, turnId = null, runId = null, initialOrchestrator = null }) {
    setTraceContext({ conversationId, turnId, runId, initialOrchestrator })
    setTraceOpen(true)
  }

  async function refreshBackendStatus() {
    try {
      const status = await getModelBackend()
      setLlmStatus(status)
      // Keep the user-selected routing backend. The server active backend is
      // displayed through llmStatus but should not override the dropdown.
      setBackendStatusLoaded(true)
      setModelSwitchError(null)
      return status
    } catch (e) {
      setBackendStatusLoaded(true)
      setModelSwitchError(e?.message || String(e))
      return null
    }
  }

  function handleModelBackendChange(nextBackend) {
    const backend = normalizeDashboardBackend(nextBackend, llmBackend || 'local')
    if (!backend) return

    // The selected dashboard backend is the routing source of truth. New Q&A
    // and code requests include this backend in the request body, so switching
    // remains usable even if the server's global default persistence call is
    // delayed by a local CUDA/model-load thread.
    selectedBackendRef.current = backend
    writeStoredBackend(backend)
    setLlmBackend(backend)
    setBackendStatusLoaded(true)
    setModelSwitching(false)
    setModelSwitchError(null)
    if (backend !== 'local') {
      setLocalStatus(null)
      setLocalStatusError(null)
    } else {
      refreshLocalStatus(false)
    }

    // Best-effort server default update. This no longer blocks or reverts the
    // dashboard. If it fails, explicit per-request routing still uses backend.
    Promise.resolve()
      .then(() => setModelBackend(backend))
      .then((status) => {
        setLlmStatus(status)
        setBackendStatusLoaded(true)
        if (normalizeDashboardBackend(status?.active_backend, backend) !== backend) {
          setModelSwitchError(`Dashboard is routing new requests to ${backend}; server default still reports ${status?.active_backend || 'unknown'}.`)
        } else {
          setModelSwitchError(null)
        }
      })
      .catch((e) => {
        setModelSwitchError(`${e?.message || String(e)}. Dashboard will still route new requests to ${backend}.`)
      })
  }

  async function onStart(overrideQuestion = null) {
    setStarting(true)
    try {
      if (mode === 'qa') {
        setQaError(null)
        // React passes a SyntheticEvent to onClick handlers when you pass the
        // function reference directly (e.g. onClick={onStart}). If we treat
        // that as a question, calling .trim() crashes. Only treat explicit
        // strings as an override question.
        const overrideText = typeof overrideQuestion === 'string' ? overrideQuestion : null
        const draftText = typeof qaDraft === 'string' ? qaDraft : ''
        const q = (overrideText ?? draftText ?? '').trim()
        if (!q) return

        // Chat-like UX: optimistic user bubble + thinking bubble
        const optimisticTurnId = `local-${Date.now()}`
        const now = new Date().toISOString()
        const optimisticUser = {
          id: optimisticTurnId + '-u',
          conversation_id: qaConversationId,
          role: 'user',
          content: q,
          orchestrator: qaOrchestrator,
          input_mode: inputMode,
          meta: { turn_id: optimisticTurnId },
          created_at: now
        }
        const optimisticAssistant = {
          id: optimisticTurnId + '-a',
          conversation_id: qaConversationId,
          role: 'assistant',
          content: (llmBackend === 'local'
            ? 'Local model is working… first request may download/load weights. Logs appear after the Q&A turn completes; backend terminal shows live HF loading/generation.'
            : '…'),
          orchestrator: 'hive_master_orchestrator',
          input_mode: null,
          meta: { turn_id: optimisticTurnId, pending: true },
          created_at: now
        }
        setQaMessages((prev) => [...(prev || []), optimisticUser, optimisticAssistant])
        setQaDraft('')

        const resp = await askQA({
          question: q,
          orchestrator: qaOrchestrator,
          project_path: projectPath,
          use_web: qaUseWeb,
          use_local_refs: qaUseLocalRefs,
          input_mode: inputMode,
          backend: selectedBackendRef.current || llmBackend,
          conversation_id: qaConversationId
        })
        setQa(resp)
        setQaConversationId(resp.conversation_id)
        setQaMessages(resp.messages || [])

// Read the final answer aloud by default unless voice answers are turned off.
if (speakVoiceResponses) {
  speak(resp.answer, { interrupt: true })
}
        setRunId(null)
        setRun(null)
      } else {
        const resp = await createRun({
          project_path: projectPath,
          prompt: codePrompt,
          copy_project_to_workspace: copyToWorkspace,
          input_mode: inputMode,
          backend: selectedBackendRef.current || llmBackend
        })
        setRunId(resp.run_id)
        setRun(null)
        resetConversation()
      }
    } catch (e) {
      if (mode === 'qa') {
        setQaError(e.message)
      } else {
        alert(e.message)
      }
    } finally {
      setStarting(false)
    }
  }

  async function refreshUploads() {
    if (mode !== 'qa') return
    try {
      const r = await listUploads({ limit: 25 })
      setUploads(r.uploads || [])
      setUploadError(null)
    } catch (e) {
      console.error(e)
      setUploadError(e?.message || String(e))
    }
  }

  async function handleUploadSelected(e) {
    const files = Array.from(e?.target?.files || [])
    if (!files.length) return

    setUploading(true)
    setUploadError(null)

    try {
      // Upload sequentially to keep UI simple/predictable.
      for (const f of files) {
        await uploadFile(f, { train: uploadTrain, addToHive: uploadAddToHive })
      }
      await refreshUploads()
    } catch (err) {
      console.error(err)
      setUploadError(err?.message || String(err))
    } finally {
      setUploading(false)
      try {
        // allow re-selecting the same file
        e.target.value = ''
      } catch (_) {}
    }
  }

  useEffect(() => {
    if (mode !== 'qa') return
    refreshUploads()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode])

  // Auto-scroll chat on new messages
  useEffect(() => {
    if (mode !== 'qa') return
    try {
      qaBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    } catch (_) {}
  }, [mode, qaMessages])

  useEffect(() => {
    if (!runId) return

    let cancelled = false
    let missingCount = 0
    let intervalId = null
    async function tick() {
      try {
        const r = await getRun(runId)
        if (cancelled) return
        missingCount = 0
        setRun(r)
        if (r && r.status && r.status !== 'running' && intervalId) {
          window.clearInterval(intervalId)
          intervalId = null
        }
      } catch (e) {
        if (cancelled) return
        const msg = e?.message || String(e)
        console.error(e)
        if (msg.includes('HTTP 404') || msg.includes('run_id not found')) {
          missingCount += 1
          if (missingCount >= 2) {
            setRun({
              run_id: runId,
              status: 'missing',
              created_at: '',
              updated_at: new Date().toISOString(),
              project_root: projectPath,
              prompt: codePrompt,
              input_mode: inputMode,
              logs: [`Run ${runId} is no longer available on the backend. This usually happens after a backend restart/reload. Polling stopped.`],
              agent_statuses: [],
              result: null,
              error: 'Run not found on backend. Start a new run.'
            })
            setRunId(null)
            if (intervalId) window.clearInterval(intervalId)
            intervalId = null
          }
        }
      }
    }

    tick()
    intervalId = window.setInterval(() => tick(), 1000)
    return () => {
      cancelled = true
      if (intervalId) window.clearInterval(intervalId)
    }
  }, [runId, projectPath, codePrompt, inputMode])

  const agentStatuses = run?.agent_statuses || []
  const logs = run?.logs || []


// Narrate code runs whenever voice answers are enabled, regardless of typed or spoken input.
useEffect(() => {
  if (mode !== 'code') return
  if (!run) return
  if (!speakVoiceResponses) return

  const st = ttsStateRef.current

  // run status change
  if (run.status && run.status !== st.runStatus) {
    speak(`Run status: ${run.status}`, { interrupt: false })
    st.runStatus = run.status
  }

  // agent state changes
  const agents = run.agent_statuses || []
  for (const a of agents) {
    const key = a.agent_id || `${a.agent_type || 'agent'}`
    const prev = st.agentStates[key]
    if (a.state && a.state !== prev) {
      const label = a.agent_type || 'agent'
      speak(`${label}: ${a.state}`, { interrupt: false })
      st.agentStates[key] = a.state
    }
  }

  // blockers / errors
  if (run.error && run.error !== st.lastError) {
    speak(`Blocker: ${_firstLine(run.error)}`, { interrupt: false })
    st.lastError = run.error
  }

  // optionally speak important log lines only (avoid spam)
  const lines = run.logs || []
  const startIdx = Math.max(0, st.logsLen || 0)
  const newLines = lines.slice(startIdx)
  st.logsLen = lines.length

  for (const ln of newLines.slice(-10)) {
    const s = String(ln || '')
    if (!s) continue
    if (/(\berror\b|\bfailed\b|\bexception\b|\binstall\b|\bretrying\b|\bfallback\b)/i.test(s)) {
      speak(_firstLine(s.replace(/^\[[0-9:]+\]\s*/, '')), { interrupt: false })
    }
  }
}, [run, mode, speakVoiceResponses])

  const displayMessages = useMemo(() => {
    if (!qaMessages || qaMessages.length === 0) return []
    if (qaShowInternal) return qaMessages

    // Chatbot-like: show only user messages + the final assistant message from the master orchestrator.
    return qaMessages.filter((m) => {
      if (m.role === 'user') return true
      if (m.role === 'assistant' && (m.orchestrator || '') === 'hive_master_orchestrator') return true
      return false
    })
  }, [qaMessages, qaShowInternal])

  function sendFollowup(text) {
    if (!text) return
    const q = String(text)
    setQaDraft(q)
    setInputMode('text')
    // Auto-send followups for chatbot feel
    if (!starting) onStart(q)
  }

  function qaPromptForTurn(turnId) {
    if (!turnId || !qaMessages) return { content: '', input_mode: 'text' }
    const found = [...qaMessages].reverse().find((m) => m.role === 'user' && m.meta?.turn_id === turnId)
    return { content: found?.content || '', input_mode: found?.input_mode || 'text' }
  }

  const localCudaSummary = (() => {
    if (!localStatus) return 'Not checked yet.'
    if (localStatus.cuda_available == null || localStatus.deep === false) return 'Shallow status only; CUDA not probed yet.'
    if (!localStatus.cuda_available) return 'CUDA not visible; local model/trainer will use CPU unless PyTorch CUDA is fixed.'
    const resolved = String(localStatus.resolved_device || 'cuda:0')
    const idx = resolved.includes(':') ? Number(resolved.split(':').pop()) : 0
    const devices = Array.isArray(localStatus.cuda_devices) ? localStatus.cuda_devices : []
    const dev = devices.find((d) => Number(d.index) === idx) || devices[0]
    const name = dev?.name || resolved
    const used = dev?.reserved_mb ?? dev?.allocated_mb ?? 0
    const total = dev?.total_memory_mb ?? '?'
    return `${localStatus.loaded ? 'loaded' : 'not loaded yet'} • ${resolved} • ${name} • VRAM reserved ${used}/${total} MB`
  })()

  return (
    <div className="container">
      <div className="header">
        <h1 style={{ margin: 0 }}>Agentic Hive Studio</h1>
        <div className="small">React UI • FastAPI backend • Multi-agent orchestration • selected model: <b>{llmBackend}</b>{llmStatus?.active_backend && llmStatus.active_backend !== llmBackend ? <> • server default: <b>{llmStatus.active_backend}</b></> : null}</div>
      </div>

      <PromptPanel
        mode={mode}
        setMode={setMode}
        qaOrchestrator={qaOrchestrator}
        setQaOrchestrator={(v) => {
          // If switching orchestrator mid-chat, start a new conversation
          if (mode === 'qa' && qaConversationId) {
            resetConversation()
          }
          setQaOrchestrator(v)
        }}
        inputMode={inputMode}
        setInputMode={setInputMode}

        projectPath={projectPath}
        setProjectPath={setProjectPath}
        // PromptPanel owns the voice dictation logic; we bind it to the
        // correct input depending on mode.
        prompt={mode === 'qa' ? qaDraft : codePrompt}
        setPrompt={mode === 'qa' ? setQaDraft : setCodePrompt}
        copyToWorkspace={copyToWorkspace}
        setCopyToWorkspace={setCopyToWorkspace}
        onStart={onStart}
        speakVoiceResponses={speakVoiceResponses}
        setSpeakVoiceResponses={setSpeakVoiceResponses}
        llmBackend={llmBackend}
        llmStatus={llmStatus}
        modelSwitching={modelSwitching}
        modelSwitchError={modelSwitchError}
        onLlmBackendChange={handleModelBackendChange}
        disabled={starting}
      />

      <div className="card status-card">
        <div className="small">
          <b>RL feedback:</b> total {feedbackStats?.total ?? 0} • positive {feedbackStats?.positive ?? 0} • negative {feedbackStats?.negative ?? 0} • corrections {feedbackStats?.corrections ?? 0}
          {feedbackNotice ? <> • {feedbackNotice}</> : null}
          {feedbackStatsError ? <> • stats error: {feedbackStatsError}</> : null}
          <button type="button" className="chat-chip" style={{ marginLeft: 8 }} onClick={refreshFeedbackStats}>Refresh RL stats</button>
          <button type="button" className="chat-chip" style={{ marginLeft: 8 }} onClick={refreshBackendStatus}>Refresh backend</button>
        </div>
        {llmBackend === 'local' ? (
          <div className="small" style={{ marginTop: 6 }}>
            <b>Local GPU status:</b> {localCudaSummary}
            {localStatus?.training ? <> • trainer device: {localStatus.training.device_config} • feedback weight: {localStatus.training.feedback_weight}</> : null}
            {localStatusError ? <> • status error: {localStatusError}</> : null}
            <button type="button" className="chat-chip" style={{ marginLeft: 8 }} onClick={() => refreshLocalStatus(false)}>Refresh shallow</button>
            <button type="button" className="chat-chip" style={{ marginLeft: 8 }} onClick={() => refreshLocalStatus(true)}>Check CUDA/RTX</button>
          </div>
        ) : null}
      </div>

      {mode === 'code' ? (
        <>
          <RunsPanel run={run} onViewTrace={(rid) => openTrace({ runId: rid, initialOrchestrator: "orchestrator" })} />
          <div className="grid">
            <AgentPanel agentStatuses={agentStatuses} />
            <LogsPanel logs={logs} />
          </div>

          {run && run.status !== 'running' ? (
            <div className="card">
              <h3 style={{ marginTop: 0 }}>{run.result ? 'Result (JSON)' : 'Run result / error'}</h3>
              {run.result ? (
                <pre>{JSON.stringify(run.result, null, 2)}</pre>
              ) : (
                <pre>{run.error || (run.logs || []).slice(-120).join('\n') || 'No result payload was produced.'}</pre>
              )}
              <FeedbackControls
                targetType="code"
                targetId={run.run_id}
                runId={run.run_id}
                promptText={run.result?.prompt || run.prompt || codePrompt}
                responseText={run.result ? JSON.stringify(run.result, null, 2) : (run.error || (run.logs || []).slice(-120).join('\n') || '')}
                backend={llmBackend}
                projectPath={run.result?.project_root || run.project_root || projectPath}
                mode="code"
                inputMode={run.input_mode || inputMode}
                onSubmitted={handleFeedbackSubmitted}
              />
            </div>
          ) : null}
        </>
      ) : (
        <>
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ marginTop: 0, marginBottom: 0 }}>Conversation</h3>
              <button type="button" onClick={resetConversation} disabled={starting}>
                New conversation
              </button>
            </div>
            <div className="small" style={{ marginTop: 8 }}>
              orchestrator: <b>{qaOrchestrator}</b>
              {qaConversationId ? (
                <>
                  {' '}• conversation_id: <b>{qaConversationId}</b>
                </>
              ) : (
                <> • (no conversation yet)</>
              )}
            </div>

            <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              <label style={{ display: 'flex', gap: 8, alignItems: 'center', margin: 0 }}>
                <input
                  type="checkbox"
                  checked={qaShowInternal}
                  onChange={(e) => setQaShowInternal(e.target.checked)}
                />
                Show internal orchestrator messages
              </label>
              <div className="small" style={{ opacity: 0.75 }}>
                (Traces are recorded regardless)
              </div>
            </div>

            <div style={{ marginTop: 14 }}>
              {displayMessages && displayMessages.length ? (
                <div className="chat-thread">
                  {displayMessages.map((m) => (
                    <ChatMessage
                      key={`${m.id}-${m.role}`}
                      msg={m}
                      conversationId={qaConversationId}
                      onViewTrace={openTrace}
                      onFollowup={sendFollowup}
                      llmBackend={llmBackend}
                      projectPath={projectPath}
                      promptText={qaPromptForTurn(m.meta?.turn_id).content}
                      promptInputMode={qaPromptForTurn(m.meta?.turn_id).input_mode}
                      onFeedbackSubmitted={handleFeedbackSubmitted}
                    />
                  ))}

                  {/* Auto-scroll anchor */}
                  <div ref={qaBottomRef} />
                </div>
              ) : (
                <div className="small">No messages yet. Ask something to start the thread.</div>
              )}
            </div>

            <div style={{ marginTop: 14 }}>
              <div style={{ marginBottom: 12, padding: 12, border: '1px dashed #ddd', borderRadius: 12 }}>
                <div style={{ display: 'flex', gap: 12, alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}>
                  <div>
                    <b>Upload documents / images</b>
                    <div className="small" style={{ opacity: 0.75 }}>
                      Ingested into hive memory (usable by local + Gemini + OpenAI) and optionally added to the background training set.
                    </div>
                  </div>
                  <input type="file" multiple onChange={handleUploadSelected} disabled={starting || uploading} />
                </div>

                <div style={{ marginTop: 10, display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
                  <label style={{ display: 'flex', gap: 8, alignItems: 'center', margin: 0 }}>
                    <input type="checkbox" checked={uploadAddToHive} onChange={(e) => setUploadAddToHive(e.target.checked)} />
                    Add to hive memory
                  </label>
                  <label style={{ display: 'flex', gap: 8, alignItems: 'center', margin: 0 }}>
                    <input type="checkbox" checked={uploadTrain} onChange={(e) => setUploadTrain(e.target.checked)} />
                    Use for training
                  </label>
                  <div style={{ flex: 1 }} />
                  <button type="button" onClick={refreshUploads} disabled={starting || uploading}>
                    Refresh
                  </button>
                </div>

                {uploading ? <div className="small" style={{ marginTop: 10 }}>Uploading…</div> : null}
                {uploadError ? (
                  <div className="small" style={{ marginTop: 10 }}>
                    <b>Upload error:</b> {uploadError}
                  </div>
                ) : null}

                {uploads && uploads.length ? (
                  <details style={{ marginTop: 10 }}>
                    <summary>Recently ingested files ({uploads.length})</summary>
                    <ul className="small" style={{ marginTop: 8 }}>
                      {uploads.map((u) => (
                        <li key={u.upload_id}>
                          <b>{u.filename}</b> — {u.content_type || 'unknown'} — {u.text_chars} chars — {u.chunks_added} chunks — {u.training_examples_added} training ex
                        </li>
                      ))}
                    </ul>
                  </details>
                ) : (
                  <div className="small" style={{ marginTop: 10, opacity: 0.75 }}>No uploads yet.</div>
                )}
              </div>

              <label>Chat</label>
              <textarea
                value={qaDraft}
                className="chat-input"
                onChange={(e) => {
                  setQaDraft(e.target.value)
                  setInputMode('text')
                }}
                placeholder="Type your question here… (Enter = send, Shift+Enter = newline)"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    if (!starting && qaDraft.trim()) onStart()
                  }
                }}
              />

              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginTop: 10 }}>
                <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={qaUseWeb}
                    onChange={(e) => setQaUseWeb(e.target.checked)}
                  />
                  Use web research
                </label>
                <label style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <input
                    type="checkbox"
                    checked={qaUseLocalRefs}
                    onChange={(e) => setQaUseLocalRefs(e.target.checked)}
                  />
                  Use local references
                </label>
                <div style={{ flex: 1 }} />
                <button type="button" onClick={() => onStart()} disabled={starting || !qaDraft.trim()}>
                  {starting ? 'Sending…' : 'Send'}
                </button>
              </div>

              {qaError ? (
                <div style={{ marginTop: 10, padding: 10, border: '1px solid #f0caca', borderRadius: 10 }}>
                  <div className="small" style={{ opacity: 0.9 }}>
                    <b>Error:</b> {qaError}
                  </div>
                </div>
              ) : null}

              {qa && qa.answer && (!qaMessages || qaMessages.length === 0) ? (
                <div className="small" style={{ marginTop: 10, opacity: 0.85 }}>
                  (Fallback) Answer: {qa.answer}
                </div>
              ) : null}
            </div>
          </div>


          {qa && (
            <div className="card">
              <h3 style={{ marginTop: 0 }}>Last turn details</h3>
              <div className="small" style={{ marginBottom: 10 }}>
                turn_id: {qa.turn_id} • model: {llmBackend} • log_file: {qa.log_file}
              </div>
              <details style={{ marginTop: 12 }}>
                <summary>Show raw result JSON</summary>
                <pre>{JSON.stringify(qa.result, null, 2)}</pre>
              </details>
              <details style={{ marginTop: 12 }}>
                <summary>Show logs (last turn)</summary>
                <pre>{(qa.logs || []).join('\n')}</pre>
              </details>
            </div>
          )}
        </>
      )}

      <TraceViewer
        open={traceOpen}
        onClose={() => setTraceOpen(false)}
        conversationId={traceContext.conversationId}
        turnId={traceContext.turnId}
        runId={traceContext.runId}
        initialOrchestrator={traceContext.initialOrchestrator}
      />

    </div>
  )
}

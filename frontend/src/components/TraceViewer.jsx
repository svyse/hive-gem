import React, { useEffect, useMemo, useState } from 'react'
import { getConversationTrace, listConversationTraces, getRunTrace, listRunTraces } from '../api/client.js'

function Tag({ children }) {
  return (
    <span
      style={{
        display: 'inline-block',
        padding: '2px 8px',
        border: '1px solid rgba(255,255,255,0.18)',
        borderRadius: 999,
        fontSize: 12,
        opacity: 0.9,
        marginRight: 6,
        marginBottom: 6,
        background: 'rgba(255,255,255,0.06)'
      }}
    >
      {children}
    </span>
  )
}

function prettyJson(obj) {
  try {
    return JSON.stringify(obj, null, 2)
  } catch (e) {
    return String(obj)
  }
}

function EventRow({ ev }) {
  const kind = ev.kind || 'event'
  const ts = ev.ts || ''
  const rest = { ...ev }
  delete rest.kind
  delete rest.ts
  return (
    <div style={{ borderBottom: '1px solid rgba(255,255,255,0.12)', padding: '8px 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12 }}>
        <div>
          <b>{kind}</b>
        </div>
        <div style={{ opacity: 0.7, fontSize: 12 }}>{ts}</div>
      </div>
      {Object.keys(rest).length ? (
        <pre style={{ margin: '6px 0 0', whiteSpace: 'pre-wrap', fontSize: 12 }}>{prettyJson(rest)}</pre>
      ) : null}
    </div>
  )
}

export default function TraceViewer({
  open,
  onClose,
  conversationId,
  turnId,
  runId,
  initialOrchestrator = null
}) {
  const [orchestratorFilter, setOrchestratorFilter] = useState(initialOrchestrator)
  const [loadingList, setLoadingList] = useState(false)
  const [listResp, setListResp] = useState(null)
  const [listError, setListError] = useState(null)

  const [selectedTraceId, setSelectedTraceId] = useState(null)
  const [loadingTrace, setLoadingTrace] = useState(false)
  const [traceResp, setTraceResp] = useState(null)
  const [traceError, setTraceError] = useState(null)

  const isRun = Boolean(runId)

  // Reset filter when opening for a new message
  useEffect(() => {
    if (!open) return
    setOrchestratorFilter(initialOrchestrator)
    setSelectedTraceId(null)
    setTraceResp(null)
    setTraceError(null)
  }, [open, initialOrchestrator])

  async function loadList() {
    if (isRun) {
      if (!runId) return
    } else {
      if (!conversationId || !turnId) return
    }

    setLoadingList(true)
    setListError(null)
    try {
      const resp = isRun
        ? await listRunTraces(runId, {
            orchestrator: orchestratorFilter,
            limit: 100
          })
        : await listConversationTraces(conversationId, {
            turn_id: turnId,
            orchestrator: orchestratorFilter,
            limit: 100
          })

      setListResp(resp)
      // auto-select
      const first = resp?.traces?.[0]?.id
      if (first) {
        setSelectedTraceId(first)
      }
    } catch (e) {
      setListResp(null)
      setListError(e.message)
    } finally {
      setLoadingList(false)
    }
  }

  async function loadTrace(traceId) {
    if (!traceId) return
    setLoadingTrace(true)
    setTraceError(null)
    try {
      const resp = isRun ? await getRunTrace(traceId) : await getConversationTrace(traceId)
      setTraceResp(resp)
    } catch (e) {
      setTraceResp(null)
      setTraceError(e.message)
    } finally {
      setLoadingTrace(false)
    }
  }

  useEffect(() => {
    if (!open) return
    loadList()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, conversationId, turnId, runId, orchestratorFilter])

  useEffect(() => {
    if (!open) return
    if (!selectedTraceId) return
    loadTrace(selectedTraceId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, selectedTraceId])

  const traces = listResp?.traces || []

  const trace = traceResp?.trace || null
  const link = traceResp?.link || null

  const keyEvents = useMemo(() => {
    if (!trace || !Array.isArray(trace.events)) return []
    return trace.events.filter((e) => ['sequence_learning', 'agent_selection', 'orchestrator_selection', 'query_end', 'error'].includes(e.kind))
  }, [trace])

  if (!open) return null

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.55)',
        display: 'flex',
        alignItems: 'stretch',
        justifyContent: 'flex-end',
        zIndex: 9999
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: 'min(900px, 92vw)',
          background: 'rgba(15,16,32,0.98)',
          color: '#e9e9f2',
          height: '100%',
          padding: 16,
          overflow: 'auto',
          boxShadow: '-8px 0 30px rgba(0,0,0,0.15)'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
          <div>
            <h3 style={{ margin: 0 }}>Trace viewer</h3>
            <div style={{ fontSize: 12, opacity: 0.8, marginTop: 4 }}>
{isRun ? (
                <>run_id: <b>{runId}</b></>
              ) : (
                <>conversation_id: <b>{conversationId}</b> • turn_id: <b>{turnId}</b></>
              )}
            </div>
          </div>
          <button type="button" onClick={onClose}>Close</button>
        </div>

        <div style={{ marginTop: 12, display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
          <Tag>filter: {orchestratorFilter || 'all orchestrators'}</Tag>
          {initialOrchestrator ? (
            orchestratorFilter ? (
              <button type="button" onClick={() => setOrchestratorFilter(null)}>
                Show all orchestrators
              </button>
            ) : (
              <button type="button" onClick={() => setOrchestratorFilter(initialOrchestrator)}>
                Show only {initialOrchestrator}
              </button>
            )
          ) : null}
          <button type="button" onClick={loadList} disabled={loadingList}>Reload</button>
        </div>

        <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: '320px 1fr', gap: 14 }}>
          <div style={{ border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10, padding: 12, overflow: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <b>Traces</b>
              <span style={{ fontSize: 12, opacity: 0.7 }}>{traces.length}</span>
            </div>

            {loadingList ? <div style={{ marginTop: 10, fontSize: 12 }}>Loading…</div> : null}
            {listError ? <div style={{ marginTop: 10, color: 'crimson', fontSize: 12 }}>{listError}</div> : null}
            {!loadingList && !listError && traces.length === 0 ? (
              <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>
                No traces found yet for this item. (Q&A traces are recorded per turn; code-pipeline traces are recorded when the run finishes.)
              </div>
            ) : null}

            <div style={{ marginTop: 10 }}>
              {traces.map((t) => {
                const isSel = String(t.id) === String(selectedTraceId)
                const s = t.summary || {}
                return (
                  <div
                    key={t.id}
                    onClick={() => setSelectedTraceId(t.id)}
                    style={{
                      cursor: 'pointer',
                      border: '1px solid rgba(255,255,255,0.18)',
                      borderRadius: 10,
                      padding: 10,
                      marginBottom: 10,
                      background: isSel ? 'rgba(51,92,255,0.16)' : 'rgba(255,255,255,0.04)'
                    }}
                  >
                    <div style={{ fontSize: 12, opacity: 0.75 }}>{t.created_at}</div>
                    <div style={{ marginTop: 6 }}>
                      <b>{t.orchestrator}</b>
                    </div>
                    <div style={{ fontSize: 12, opacity: 0.8, marginTop: 4 }}>
                      domain: {s.domain || '—'} • success: {String(s.success)}
                    </div>
                    {s.query ? (
                      <div style={{ fontSize: 12, opacity: 0.75, marginTop: 6, whiteSpace: 'pre-wrap' }}>
                        {String(s.query).slice(0, 140)}{String(s.query).length > 140 ? '…' : ''}
                      </div>
                    ) : null}
                  </div>
                )
              })}
            </div>
          </div>

          <div style={{ border: '1px solid rgba(255,255,255,0.12)', borderRadius: 10, padding: 12, overflow: 'auto' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
              <b>Details</b>
              {link ? (
                <span style={{ fontSize: 12, opacity: 0.7 }}>
                  trace_id: {link.id} • hive_memory_id: {link.hive_memory_id ?? '—'}
                </span>
              ) : null}
            </div>

            {loadingTrace ? <div style={{ marginTop: 10, fontSize: 12 }}>Loading…</div> : null}
            {traceError ? <div style={{ marginTop: 10, color: 'crimson', fontSize: 12 }}>{traceError}</div> : null}
            {!loadingTrace && !traceError && !trace ? (
              <div style={{ marginTop: 10, fontSize: 12, opacity: 0.8 }}>Select a trace to view details.</div>
            ) : null}

            {trace ? (
              <>
                <div style={{ marginTop: 10 }}>
                  <Tag>orchestrator: {trace.orchestrator}</Tag>
                  {trace.domain ? <Tag>domain: {trace.domain}</Tag> : null}
                  <Tag>success: {String(trace.success)}</Tag>
                </div>

                {trace.agent_sequence?.length ? (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 12, opacity: 0.75, marginBottom: 6 }}><b>Agent call order</b></div>
                    <div style={{ fontSize: 12, whiteSpace: 'pre-wrap' }}>{trace.agent_sequence.join('  →  ')}</div>
                  </div>
                ) : null}

                {trace.memory_sequence?.length ? (
                  <div style={{ marginTop: 10 }}>
                    <div style={{ fontSize: 12, opacity: 0.75, marginBottom: 6 }}><b>Memory access order</b></div>
                    <div style={{ fontSize: 12, whiteSpace: 'pre-wrap' }}>{trace.memory_sequence.join('  →  ')}</div>
                  </div>
                ) : null}

                {keyEvents.length ? (
                  <div style={{ marginTop: 14 }}>
                    <details open>
                      <summary><b>Key events</b> (learning + selection)</summary>
                      <div style={{ marginTop: 8 }}>
                        {keyEvents.map((e, idx) => (
                          <EventRow key={idx} ev={e} />
                        ))}
                      </div>
                    </details>
                  </div>
                ) : null}

                {Array.isArray(trace.events) ? (
                  <div style={{ marginTop: 14 }}>
                    <details>
                      <summary><b>Full event timeline</b> ({trace.events.length})</summary>
                      <div style={{ marginTop: 8 }}>
                        {trace.events.map((e, idx) => (
                          <EventRow key={idx} ev={e} />
                        ))}
                      </div>
                    </details>
                  </div>
                ) : null}

                <div style={{ marginTop: 14 }}>
                  <details>
                    <summary><b>Raw trace JSON</b></summary>
                    <pre style={{ marginTop: 10, whiteSpace: 'pre-wrap', fontSize: 12 }}>{prettyJson(trace)}</pre>
                  </details>
                </div>
              </>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}

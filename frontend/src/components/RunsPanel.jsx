import React from 'react'

export default function RunsPanel({ run, onViewTrace }) {
  if (!run) return null

  return (
    <div className="card">
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
        <div><strong>Run:</strong> <span className="badge">{run.run_id}</span></div>
        <div><strong>Status:</strong> <span className="badge">{run.status}</span></div>
        {onViewTrace ? (
          <button type="button" onClick={() => onViewTrace(run.run_id)} style={{ marginLeft: 6 }}>
            View trace
          </button>
        ) : null}
        <div className="small">Updated: {run.updated_at}</div>
      </div>

      <div className="small" style={{ marginTop: 10 }}>
        Project root in run: <code>{run.project_root}</code>
      </div>

      {run.error && (
        <div style={{ marginTop: 10 }}>
          <strong>Error:</strong>
          <pre>{run.error}</pre>
        </div>
      )}
    </div>
  )
}

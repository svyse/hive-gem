import React from 'react'

export default function LogsPanel({ logs }) {
  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <h3 style={{ margin: 0 }}>Logs</h3>
        <div className="small">{logs?.length || 0} lines</div>
      </div>
      <pre style={{ marginTop: 12 }}>{(logs || []).join('\n')}</pre>
    </div>
  )
}

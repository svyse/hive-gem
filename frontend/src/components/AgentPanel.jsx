import React from 'react'

export default function AgentPanel({ agentStatuses }) {
  return (
    <div className="card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <h3 style={{ margin: 0 }}>Agents</h3>
        <div className="small">{agentStatuses?.length || 0} active/seen</div>
      </div>
      <div style={{ marginTop: 12 }}>
        {(!agentStatuses || agentStatuses.length === 0) && <div className="small">No agent status yet.</div>}
        {agentStatuses && agentStatuses.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {agentStatuses.map((a) => (
              <div key={a.agent_id} className="card" style={{ marginTop: 0 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                  <div><strong>{a.agent_type}</strong></div>
                  <div className="badge">{a.state}</div>
                </div>
                <div className="small" style={{ marginTop: 8 }}>
                  <div>ID: <code>{a.agent_id}</code></div>
                  <div>Last: {a.last_update || '-'}</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

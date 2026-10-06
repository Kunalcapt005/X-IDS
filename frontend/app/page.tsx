const metrics = [
  ["Flows / sec", "0"],
  ["Active alerts", "0"],
  ["Macro-F1", "—"],
  ["Model", "Not trained"],
];

export default function DashboardPage() {
  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">X-IDS</div>
        <nav className="nav">
          <div className="active">Overview</div>
          <div>Sessions</div>
          <div>Alerts</div>
          <div>Models</div>
        </nav>
      </aside>

      <section className="content">
        <header className="header">
          <div>
            <div className="eyebrow">Security Operations Center</div>
            <h1>Detection overview</h1>
          </div>
          <div className="muted">Replay engine offline</div>
        </header>

        <section className="grid">
          {metrics.map(([label, value]) => (
            <div className="card" key={label}>
              <div className="label">{label}</div>
              <div className="metric">{value}</div>
            </div>
          ))}
        </section>

        <section className="section card">
          <div className="header" style={{ marginBottom: 8 }}>
            <div>
              <div className="eyebrow">Live feed</div>
              <h2 style={{ margin: "4px 0 0" }}>Recent alerts</h2>
            </div>
            <span className="badge">Waiting for session</span>
          </div>

          <table className="alerts">
            <thead>
              <tr><th>Time</th><th>Attack type</th><th>Confidence</th><th>Severity</th></tr>
            </thead>
            <tbody>
              <tr><td colSpan={4} className="muted">No alerts yet. Start a replay session after the backend model artifact is available.</td></tr>
            </tbody>
          </table>
        </section>
      </section>
    </main>
  );
}

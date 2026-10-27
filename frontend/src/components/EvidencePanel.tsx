import { useEffect, useState } from "react";
import { FileCheck2, RefreshCw } from "lucide-react";
import { api } from "../api";
import type { BenchmarkEvidence } from "../types";

export function EvidencePanel() {
  const [evidence, setEvidence] = useState<BenchmarkEvidence | null>(null);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    let active = true;
    setError("");
    api.evidence().then(result => { if (active) setEvidence(result); }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [revision]);
  return <section className="evidence-panel" aria-label="Benchmark evidence and model integrity">
    <header><div><p className="eyebrow">MEASURED PERFORMANCE</p><h2>Evidence you can inspect.</h2><p>Every figure has a source, population and evaluation method.</p></div><button className="button-quiet" onClick={() => setRevision(r => r + 1)} aria-label="Refresh evidence"><RefreshCw size={15} />Refresh</button></header>
    {error && <p className="timeline-error" role="alert">{error}</p>}
    {!evidence && !error && <p className="timeline-muted" role="status">Loading repository evidence…</p>}
    {evidence && <><p className="evidence-notice">{evidence.notice}</p><div className="evidence-grid">{evidence.modules.map(module => <article key={module.id}>
      <div className="evidence-card-heading"><FileCheck2 size={18} /><h3>{module.name}</h3></div>
      <span className={`evidence-status ${module.artifact_integrity === false ? "unavailable" : ""}`}>{module.artifact_integrity === null ? "Local document pipeline" : module.artifact_integrity ? "Model checksums verified" : "Model checkpoints unavailable"}</span>
      <dl>{module.metrics.map(metric => <div key={metric.label}><dt>{metric.label}{metric.lower_is_better ? " · lower is better" : ""}</dt><dd>{(metric.value * 100).toFixed(2)}<span>%</span></dd></div>)}</dl>
      {!module.metrics.length && <p className="timeline-muted">No verified report values are available.</p>}
      <p className="evidence-cohort">{module.cohort}</p><p className="evidence-protocol">{module.protocol}</p>
      <p className="evidence-scope">{module.evaluation_scope}</p>
      {module.evidence_status === "reported" && <a href={module.source_url} target="_blank" rel="noreferrer">Inspect {module.source_filename}<span>Report SHA-256: {module.source_sha256?.slice(0, 12)}</span></a>}
    </article>)}</div></>}
  </section>;
}

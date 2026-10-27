import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { Activity, ArrowDownRight, ArrowUpRight, CheckCircle2, Download, FileText, Image as ImageIcon, Plus, RefreshCw, ShieldCheck, Upload, X } from "lucide-react";
import { api } from "../api";
import type { AuditEvent, BatchOutcome, LabItemSchema, Patient, PatientTimeline, ReportedSex, TimelineRecord, TrendSeries } from "../types";

const directions: Record<string, string> = {
  toward_range: "Toward reference range", away_from_range: "Away from reference range", within_range: "Within reference range",
  unchanged_distance: "Unchanged distance", numeric_change_only: "Numeric change only", insufficient_data: "Awaiting another date",
  not_comparable: "Ambiguous dates", score_change_only: "Model score change",
};
const modalities: Record<string, string> = {
  chest_xray: "Pediatric chest X-ray", chest_multilabel: "Chest X-ray · 14 labels", skin_lesion: "Dermatoscopy",
  blood_smear: "Blood smear", retina: "Retina · experimental",
};
const formatDate = (value: string) => new Date(`${value}T12:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
const dateValue = (d: Date) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
const previewUrl = (id: string) => `/api/uploads/${encodeURIComponent(id)}/preview`;
const downloadUrl = (id: string) => `/api/uploads/${encodeURIComponent(id)}/download`;
interface QueueItem { file: File; date: string; series: string }

export function PatientTimelinePage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientId, setPatientId] = useState("");
  const [timeline, setTimeline] = useState<PatientTimeline | null>(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [alias, setAlias] = useState("");
  const [sex, setSex] = useState<ReportedSex>("unknown");
  const [kind, setKind] = useState<"lab_report" | "medical_image">("lab_report");
  const [imageMode, setImageMode] = useState<"screen" | "archive_only">("archive_only");
  const [modality, setModality] = useState("chest_xray");
  const [pediatric, setPediatric] = useState(false);
  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [outcomes, setOutcomes] = useState<BatchOutcome[]>([]);
  const [tab, setTab] = useState<"labs" | "images" | "records" | "audit">("labs");
  const [review, setReview] = useState<TimelineRecord | null>(null);
  const [audit, setAudit] = useState<AuditEvent[]>([]);
  const [metric, setMetric] = useState("");
  const [existing, setExisting] = useState<{ id: string; filename: string }[]>([]);
  const [existingId, setExistingId] = useState("");
  const [linkDate, setLinkDate] = useState("");
  const [linkSeries, setLinkSeries] = useState("");

  const refresh = useCallback(async (id: string) => {
    const [list, data] = await Promise.all([api.patients(), api.timeline(id)]);
    setPatients(list); setTimeline(data);
  }, []);

  useEffect(() => {
    let alive = true;
    api.patients().then(list => { if (alive) { setPatients(list); setPatientId(list[0]?.id ?? ""); } }).catch(e => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    let alive = true;
    setTimeline(null); setMetric(""); setReview(null); setOutcomes([]);
    if (!patientId) return;
    setLoading(true); setError(null);
    api.timeline(patientId).then(data => { if (alive) setTimeline(data); }).catch(e => { if (alive) setError(e.message); }).finally(() => { if (alive) setLoading(false); });
    return () => { alive = false; };
  }, [patientId]);

  useEffect(() => {
    if (tab !== "audit") return;
    let alive = true;
    api.audit().then(events => { if (alive) setAudit(events); }).catch(e => { if (alive) setError(e.message); });
    return () => { alive = false; };
  }, [tab, timeline]);

  const createPatient = async (event: FormEvent) => {
    event.preventDefault(); setBusy(true); setError(null);
    try {
      const patient = await api.createPatient(alias, sex);
      setPatients(await api.patients()); setPatientId(patient.id); setCreating(false); setAlias("");
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const addFiles = (files: FileList | null) => {
    if (!files) return;
    const selected = Array.from(files);
    if (queue.length + selected.length > 12 || selected.some(f => f.size > 10 * 1024 * 1024)) {
      setError("Choose up to 12 files, each no larger than 10 MB."); return;
    }
    const next = [...queue, ...selected.map(file => ({ file, date: "", series: "" }))];
    if (next.reduce((sum, item) => sum + item.file.size, 0) > 60 * 1024 * 1024) { setError("The batch must be no larger than 60 MB."); return; }
    setQueue(next); setOutcomes([]); setError(null);
  };

  const runBatch = async () => {
    setBusy(true); setError(null); setOutcomes([]);
    try {
      const result = await api.batch(patientId, queue.map(q => q.file), queue.map(q => ({ observed_on: q.date, series_label: q.series })), kind, modality, pediatric, imageMode);
      setOutcomes(result.outcomes);
      setQueue(items => items.filter((_, i) => result.outcomes[i]?.status === "error"));
      await refresh(patientId); setTab("records");
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const loadExampleCase = async () => {
    setBusy(true); setError(null);
    try {
      let patient = patients.find(p => p.alias === "Example case · generated records");
      if (!patient) patient = await api.createPatient("Example case · generated records", "female");
      const dates = [90, 45, 0].map(days => { const d = new Date(); d.setDate(d.getDate() - days); return dateValue(d); });
      const files = [220, 170, 130].map((value, i) => new File([
        `GENERATED EXAMPLE DATA — NOT A PATIENT RECORD\nTotal Cholesterol: ${value} mg/dL (Reference: 100-200)\nHemoglobin: ${[10.2, 11.4, 12.8][i]} g/dL (Reference: 12-16)\n`,
      ], `example-visit-${i + 1}.txt`, { type: "text/plain" }));
      const result = await api.batch(patient.id, files, dates.map(observed_on => ({ observed_on, series_label: "", data_origin: "generated_example" as const })), "lab_report", "chest_xray", false);
      setPatients(await api.patients()); setPatientId(patient.id);
      await refresh(patient.id); setTab("records"); setOutcomes(result.outcomes);
    } catch (e) { setError((e as Error).message); } finally { setBusy(false); }
  };

  const series = tab === "images" ? timeline?.image_series ?? [] : timeline?.lab_series ?? [];
  const activeSeries = series.find(s => s.id === metric) ?? series[0];
  const reviewed = timeline?.records.filter(r => r.review_status === "reviewed").length ?? 0;
  const canUpload = patientId && queue.length > 0 && queue.every(q => q.date && (kind === "lab_report" || q.series.trim())) && !(kind === "medical_image" && imageMode === "screen" && modality === "chest_xray" && !pediatric);

  return <div className="timeline-page">
    <header className="timeline-heading"><div><p className="eyebrow">LONGITUDINAL INTELLIGENCE</p><h1>See the whole journey.</h1><p>One patient workspace. Every observation, in context.</p></div><span className="timeline-local"><ShieldCheck size={15} /> LOCAL RESEARCH WORKSPACE</span></header>
    <div className="timeline-toolbar">
      <label>Patient workspace<select aria-label="Patient workspace" value={patientId} disabled={busy} onChange={e => setPatientId(e.target.value)}><option value="">Select a patient</option>{patients.map(p => <option key={p.id} value={p.id}>{p.alias}</option>)}</select></label>
      <button className="button-secondary" disabled={busy} onClick={() => setCreating(v => !v)}><Plus size={15} /> New patient</button>
      <button className="button-quiet" disabled={busy} onClick={() => void loadExampleCase()}>Explore an example case</button><small className="example-origin-note">Generated data · not patient records</small>
      {patientId && <a className="button-secondary timeline-export" href={`/api/patients/${patientId}/export`} download><Download size={15} /> Export timeline</a>}
    </div>
    {creating && <form className="content-panel timeline-create" onSubmit={createPatient}><label>Patient alias<input required maxLength={80} value={alias} onChange={e => setAlias(e.target.value)} placeholder="e.g. Research case 024" /></label><label>Reported biological sex<select value={sex} onChange={e => setSex(e.target.value as ReportedSex)}><option value="unknown">Unknown / not provided</option><option value="female">Female</option><option value="male">Male</option><option value="intersex">Intersex</option></select></label><button className="button-primary" disabled={busy}>Create workspace</button><p>Use de-identified aliases or clearly marked generated examples. Storage is local and not encrypted at rest.</p></form>}
    {error && <div className="timeline-error" role="alert">{error}<button aria-label="Dismiss error" onClick={() => setError(null)}><X size={15} /></button></div>}
    {loading && <p className="timeline-loading" role="status"><RefreshCw size={16} className="animate-spin" /> Loading patient observations…</p>}
    {!patientId && !loading && <section className="timeline-empty content-panel"><Activity size={36} /><h2>A single result is a snapshot.<br />A timeline reveals the pattern.</h2><p>Create a patient alias, add dated reports, then verify the extracted values to begin comparing visits.</p><button className="button-primary" onClick={() => setCreating(true)}><Plus size={16} /> Create your first patient</button></section>}
    {patientId && <>
      <div className="journey-stats"><div><span>OBSERVATIONS</span><strong>{timeline?.records.length ?? 0}</strong><small>Across reports and images</small></div><div><span>REVIEWED</span><strong>{reviewed}<em> / {timeline?.records.length ?? 0}</em></strong><small>Source checked by you</small></div><div><span>COMPARABLE LAB SERIES</span><strong>{timeline?.lab_series.filter(s => s.change !== null).length ?? 0}</strong><small>At least two distinct dates</small></div><div><span>REPORTED SEX</span><strong className="journey-sex">{timeline?.patient.reported_sex ?? "—"}</strong><small>Explicit metadata, never inferred</small></div></div>
      <section className="content-panel batch-panel">
        <div className="panel-heading"><div><p className="eyebrow">BUILD THE PATIENT JOURNEY</p><h2>Add observations in one batch</h2></div><span className="micro-tag">12 FILES · 10 MB EACH · 60 MB TOTAL</span></div>
        <div className="batch-controls"><label>Observation type<select value={kind} disabled={busy || queue.length > 0} onChange={e => { setKind(e.target.value as typeof kind); setOutcomes([]); }}><option value="lab_report">Lab reports</option><option value="medical_image">Medical images</option></select></label>{kind === "medical_image" && <label>Imaging modality<select value={modality} disabled={busy} onChange={e => { setModality(e.target.value); setPediatric(false); }}>{Object.entries(modalities).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>}{kind === "medical_image" && <label>Processing workflow<select value={imageMode} disabled={busy} onChange={e => { setImageMode(e.target.value as typeof imageMode); setPediatric(false); }}><option value="archive_only">Store originals · no AI screening</option><option value="screen">Run AI screening · checkpoints required</option></select></label>}<label className={`batch-drop ${busy ? "disabled" : ""}`}><Upload size={18} /><strong>Choose multiple {kind === "lab_report" ? "reports" : "images"}</strong><span>{kind === "lab_report" ? "PDF, TXT or scanned images" : "PNG, JPG, TIFF, BMP, WEBP"}</span><input aria-label="Choose multiple observations" type="file" multiple disabled={busy} accept={kind === "lab_report" ? ".pdf,.txt,.png,.jpg,.jpeg,.tiff,.bmp,.webp" : ".png,.jpg,.jpeg,.tiff,.bmp,.webp"} onChange={e => { addFiles(e.target.files); e.target.value = ""; }} /></label></div>
        {kind === "medical_image" && <p className="timeline-notice">{imageMode === "archive_only" ? "Originals will be stored for side-by-side visual review. No AI prediction or score is generated. " : "Screening requires the matching local model checkpoints. "}Use the same site, view and acquisition protocol for each named series (e.g. “left forearm lesion · dermatoscopy”). Scores measure model behavior; they do not measure severity.</p>}
        {kind === "medical_image" && imageMode === "screen" && modality === "chest_xray" && <label className="timeline-confirm"><input type="checkbox" checked={pediatric} onChange={e => setPediatric(e.target.checked)} /> I confirm these images are from a child aged 1–5, matching the pediatric training cohort.</label>}
        {queue.length > 0 && <div className="batch-queue">{queue.map((item, i) => <div key={`${item.file.name}-${i}`} className="batch-queue-row"><span><FileText size={15} /><strong>{item.file.name}</strong><small>{(item.file.size / 1024).toFixed(0)} KB</small></span><label>Report / acquisition date<input aria-label={`Date for ${item.file.name}`} type="date" max={dateValue(new Date())} required value={item.date} disabled={busy} onChange={e => setQueue(items => items.map((q, j) => j === i ? { ...q, date: e.target.value } : q))} /></label>{kind === "medical_image" && <label>Acquisition series<input aria-label={`Series for ${item.file.name}`} placeholder="Same site and view" maxLength={80} value={item.series} disabled={busy} onChange={e => setQueue(items => items.map((q, j) => j === i ? { ...q, series: e.target.value } : q))} /></label>}<button className="icon-button" disabled={busy} aria-label={`Remove ${item.file.name}`} onClick={() => setQueue(items => items.filter((_, j) => j !== i))}><X size={15} /></button></div>)}<div className="batch-submit"><span>Dates come from the original report. Each file is saved and reviewed separately.</span><button className="button-primary" disabled={!canUpload || busy} onClick={() => void runBatch()}>{busy ? <><RefreshCw size={15} className="animate-spin" /> Analyzing batch…</> : <><Upload size={15} /> Analyze {queue.length} files</>}</button></div></div>}
        {outcomes.length > 0 && <div className="batch-outcomes" role="status">{outcomes.map((item, i) => <div key={i} className={`outcome-${item.status}`}><span>{item.status === "saved" ? "Saved · review required" : item.status === "duplicate" ? "Already saved" : "Needs attention"}</span><strong>{item.filename}</strong><small>{item.message ?? item.summary}</small></div>)}</div>}
      </section>
      <div className="journey-tabs" role="tablist" aria-label="Timeline views">{([['labs', 'Lab evolution'], ['images', 'Image comparison'], ['records', 'Source records'], ['audit', 'Activity log']] as const).map(([id, name]) => <button key={id} role="tab" aria-selected={tab === id} onClick={() => { setTab(id); setMetric(""); }}>{name}{id === "records" && <span>{timeline?.records.length ?? 0}</span>}</button>)}</div>
      {(tab === "labs" || tab === "images") && <>
        <div className="timeline-notice"><ShieldCheck size={17} /><span>{timeline?.notice ?? "Only reviewed, dated observations are compared."}{timeline?.excluded_unreviewed ? ` ${timeline.excluded_unreviewed} observations await review or a date.` : ""}</span></div>
        {series.length ? <div className="trend-layout"><aside className="trend-metrics">{series.map(s => <button className={activeSeries?.id === s.id ? "selected" : ""} key={s.id} onClick={() => setMetric(s.id)}><span>{s.name}</span><strong>{s.points.at(-1)?.value.toFixed(2)} <small>{s.unit}</small></strong><em>{s.series_label ?? directions[s.direction]}</em></button>)}</aside>{activeSeries && <TrendChart series={activeSeries} />}</div> : <div className="content-panel timeline-empty compact"><Activity size={28} /><h2>{tab === "labs" ? "Review the source. Reveal the trend." : "Build a comparable image series."}</h2><p>{tab === "labs" ? "Reviewed lab values with a known unit will appear here. Add a second dated report to see change." : "Use reviewed, accepted scans from the same acquisition series. A verified model artifact is required; experimental retinal staging is excluded."}</p><button className="button-secondary" onClick={() => setTab("records")}>Review source records</button></div>}
        {tab === "images" && timeline && <ImageComparison records={timeline.records.filter(r => r.kind === "medical_image" && r.has_file)} />}
      </>}
      {tab === "records" && <section className="content-panel source-panel"><div className="panel-heading"><div><p className="eyebrow">TRACEABLE TO THE SOURCE</p><h2>Observations & review queue</h2></div><button className="button-quiet" disabled={busy} onClick={async () => { try { setExisting((await api.uploads()).filter(r => ["lab_report", "medical_image"].includes(r.kind))); } catch (e) { setError((e as Error).message); } }}>Link an earlier upload</button></div>
        {existing.length > 0 && <div className="link-existing"><label>Saved upload<select value={existingId} onChange={e => setExistingId(e.target.value)}><option value="">Choose a saved file</option>{existing.map(r => <option key={r.id} value={r.id}>{r.filename}</option>)}</select></label><label>Observation date<input type="date" max={dateValue(new Date())} value={linkDate} onChange={e => setLinkDate(e.target.value)} /></label><label>Image series, if applicable<input maxLength={80} value={linkSeries} onChange={e => setLinkSeries(e.target.value)} /></label><button className="button-secondary" disabled={busy || !existingId || !linkDate} onClick={async () => { setBusy(true); try { await api.linkRecord(existingId, patientId, linkDate, linkSeries); setExisting([]); await refresh(patientId); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } }}>Link to patient</button></div>}
        {timeline?.records.length ? <div className="source-list">{[...timeline.records].reverse().map(record => <article key={record.id}><div className={`source-icon ${record.review_status === "reviewed" ? "reviewed" : ""}`}>{record.kind === "medical_image" ? <ImageIcon size={20} /> : <FileText size={20} />}</div><div className="source-copy"><time>{formatDate(record.observed_on)}</time><h3>{record.filename}</h3><p>{record.summary}{record.series_label ? ` · ${record.series_label}` : ""}</p><span className="data-origin-badge">{record.result.data_origin === "generated_example" ? "Generated example" : "Uploaded source · unverified"}</span><span className={`review-badge ${record.review_status}`}>{record.review_status === "reviewed" ? "Source reviewed" : "Review required"}</span></div><div className="source-actions">{record.has_file && <a className="button-quiet" href={downloadUrl(record.id)} download aria-label={`Download ${record.filename}`}><Download size={15} /></a>}<button className="button-secondary" onClick={() => setReview(record)}>{record.review_status === "reviewed" ? "View review" : "Review source"}</button></div></article>)}</div> : <p className="timeline-muted">No observations yet. Add your first batch above.</p>}
      </section>}
      {tab === "audit" && <section className="content-panel"><h2>Account activity</h2><p className="timeline-muted">Latest 100 workspace events. Values and document contents are excluded from this log.</p><div className="audit-list">{audit.map((event, i) => <div key={i}><CheckCircle2 size={15} /><strong>{event.action.replaceAll(".", " · ")}</strong><code>{event.resource_id.slice(0, 10)}</code><time>{new Date(event.occurred_at).toLocaleString()}</time></div>)}</div></section>}
    </>}
    {review && <ReviewDialog record={review} onClose={() => setReview(null)} onSave={async corrections => { await api.review(review.id, corrections); await refresh(patientId); setReview(null); }} />}
  </div>;
}

function TrendChart({ series }: { series: TrendSeries }) {
  const [selected, setSelected] = useState(0);
  useEffect(() => setSelected(series.points.length - 1), [series.id, series.points.length]);
  const points = series.points;
  const low = Math.min(...points.map(p => p.value));
  const high = Math.max(...points.map(p => p.value));
  const padding = Math.max((high - low) * 0.2, Math.abs(high) * 0.04, 1);
  const minimum = Math.max(0, low - padding), maximum = high + padding;
  const times = points.map(p => new Date(p.date).getTime());
  const start = Math.min(...times), end = Math.max(...times);
  const coords = points.map((p, i) => ({ x: end === start ? 380 : 65 + (times[i] - start) / (end - start) * 625, y: 210 - (p.value - minimum) / (maximum - minimum) * 160 }));
  const detail = points[selected] ?? points.at(-1)!;
  const positive = series.direction === "toward_range" || series.direction === "within_range";
  return <section className="content-panel trend-chart"><div className="panel-heading"><div><p className="eyebrow">{series.series_label ?? "REVIEWED LAB OBSERVATIONS"}</p><h2>{series.name}</h2></div><span className={`trend-direction ${positive ? "positive" : series.direction === "away_from_range" ? "negative" : ""}`}>{positive ? <ArrowDownRight size={16} /> : <ArrowUpRight size={16} />}{directions[series.direction]}</span></div><div className="trend-value"><strong>{points.at(-1)?.value.toFixed(2)}</strong><span>{series.unit}</span>{series.change !== null && <em>{series.change > 0 ? "+" : ""}{series.change.toFixed(2)} {series.modality ? "percentage points" : series.unit} since baseline</em>}</div>
    <svg viewBox="0 0 740 260" role="img" aria-label={`${series.name} over time, ${points.length} observations. Values listed below.`}>
      {[0, 1, 2, 3].map(i => <g key={i}><line x1="65" x2="690" y1={50 + i * 53.33} y2={50 + i * 53.33} stroke="#e5eeeb" /><text x="53" y={54 + i * 53.33} textAnchor="end" fontSize="11" fill="#81958f">{(maximum - i / 3 * (maximum - minimum)).toFixed(1)}</text></g>)}
      {coords.length > 1 && <polyline points={coords.map(p => `${p.x},${p.y}`).join(" ")} fill="none" stroke="#358c78" strokeWidth="3" />}
      {coords.map((p, i) => <g key={i} onClick={() => setSelected(i)}><circle cx={p.x} cy={p.y} r={selected === i ? 8 : 5} fill={selected === i ? "#184e44" : "#3e9b84"} stroke="white" strokeWidth="3" /><title>{formatDate(points[i].date)}: {points[i].value} {series.unit}</title></g>)}
      <text x="65" y="245" fontSize="11" fill="#81958f">{formatDate(points[0].date)}</text><text x="690" y="245" textAnchor="end" fontSize="11" fill="#81958f">{points.length > 1 ? formatDate(points.at(-1)!.date) : ""}</text>
    </svg><div className="trend-observation"><span>{formatDate(detail.date)}</span><strong>{detail.value} {series.unit}</strong><small>{detail.reference_range ? `Reference: ${detail.reference_range} · ` : ""}{detail.filename}{detail.corrected ? " · corrected during review" : ""}</small></div><p className="timeline-muted">{series.reason}</p><details><summary>View all observations</summary><table className="trend-table"><thead><tr><th>Date</th><th>Value</th><th>Source</th></tr></thead><tbody>{points.map((point, i) => <tr key={point.record_id}><td><button onClick={() => setSelected(i)}>{formatDate(point.date)}</button></td><td>{point.value} {series.unit}</td><td>{point.filename}</td></tr>)}</tbody></table></details>
  </section>;
}

function ImageComparison({ records }: { records: TimelineRecord[] }) {
  const [first, setFirst] = useState("");
  const [last, setLast] = useState("");
  const options = useMemo(() => [...records].sort((a, b) => a.observed_on.localeCompare(b.observed_on)), [records]);
  const a = options.find(r => r.id === first) ?? options[0];
  const b = options.find(r => r.id === last) ?? options.at(-1);
  if (!a || !b) return null;
  return <section className="content-panel image-comparison"><h2>Source images, side by side</h2><p className="timeline-muted">Visual review of originals. Display previews show the first frame and may be downscaled. Images are not registered or aligned; acquisition differences may affect appearance. Download originals for detail.</p><div>{[[a, setFirst, "Earlier observation"], [b, setLast, "Later observation"]].map(([record, setter, label], i) => { const item = record as TimelineRecord; const set = setter as (id: string) => void; return <figure key={i}><label>{label as string}<select value={item.id} onChange={e => set(e.target.value)}>{options.map(r => <option key={r.id} value={r.id}>{r.observed_on} · {r.filename}</option>)}</select></label><img src={previewUrl(item.id)} alt={`Original ${item.filename}`} /><figcaption>{item.series_label || "No acquisition series"} · {item.review_status} · {String(item.result.status ?? "unknown")}</figcaption></figure>; })}</div></section>;
}

function ReviewDialog({ record, onClose, onSave }: { record: TimelineRecord; onClose: () => void; onSave: (values: Record<string, number>) => Promise<void> }) {
  const dialogRef = useRef<HTMLElement>(null);
  const tests = (record.result.extracted_tests ?? []) as LabItemSchema[];
  const reviewedValues = (record.result.reviewed_values ?? {}) as Record<string, number>;
  const [values, setValues] = useState<Record<string, string>>(() => Object.fromEntries(tests.map(t => [t.canonical_name, String(reviewedValues[t.canonical_name] ?? t.value)])));
  const [confirmed, setConfirmed] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const invalid = tests.some(t => values[t.canonical_name]?.trim() === "" || !Number.isFinite(Number(values[t.canonical_name])) || Number(values[t.canonical_name]) < 0);
  useEffect(() => {
    const prior = document.activeElement as HTMLElement | null;
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !saving) onClose();
      if (e.key !== "Tab") return;
      const elements = dialogRef.current?.querySelectorAll<HTMLElement>('button:not(:disabled),a[href],input:not(:disabled),select:not(:disabled)');
      if (!elements?.length) return;
      const first = elements[0], last = elements[elements.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("keydown", onKey); document.body.style.overflow = overflow; prior?.focus(); };
  }, [onClose, saving]);
  return <div className="review-overlay"><section ref={dialogRef} className="review-dialog" role="dialog" aria-modal="true" aria-labelledby="review-title"><header><div><p className="eyebrow">SOURCE VERIFICATION</p><h2 id="review-title">{record.filename}</h2></div><button className="icon-button" disabled={saving} aria-label="Close review" onClick={onClose} autoFocus><X size={20} /></button></header><p className="data-origin-badge">{record.result.data_origin === "generated_example" ? "Generated example data · not a patient record" : "User-uploaded source · authenticity unverified"}</p><p>Compare with the original document. Correct any extraction errors before including these values in the timeline.</p><a className="button-secondary" href={downloadUrl(record.id)} target="_blank" rel="noreferrer">Open original source <Download size={15} /></a>
    {record.kind === "lab_report" ? tests.length ? <div className="review-tests"><table><thead><tr><th>Analyte</th><th>Reviewed value</th><th>Unit / reference</th></tr></thead><tbody>{tests.map(test => <tr key={test.canonical_name}><td><strong>{test.display_name}</strong><small>Extracted: {test.value}</small></td><td><input aria-label={`Reviewed ${test.display_name}`} type="number" min="0" step="any" value={values[test.canonical_name]} onChange={e => { setValues(v => ({ ...v, [test.canonical_name]: e.target.value })); setConfirmed(false); }} /></td><td>{test.unit}<small>{test.reference_range}</small></td></tr>)}</tbody></table></div> : <p className="timeline-error">No values were extracted. Re-scan this report before review.</p> : <div className="review-image"><img src={previewUrl(record.id)} alt="Original uploaded medical image" /><p>{record.summary}</p><p className="timeline-notice">Reviewing a source does not turn a rejected or low-confidence screening into a comparable observation.</p></div>}
    <label className="timeline-confirm"><input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} /> I checked the original source, values, units, report date and patient assignment.</label>{error && <p role="alert" className="timeline-error">{error}</p>}<footer><button className="button-quiet" disabled={saving} onClick={onClose}>Cancel</button><button className="button-primary" disabled={!confirmed || invalid || saving || (record.kind === "lab_report" && !tests.length)} onClick={async () => { setSaving(true); try { await onSave(Object.fromEntries(Object.entries(values).map(([key, value]) => [key, Number(value)]))); } catch (e) { setError((e as Error).message); } finally { setSaving(false); } }}><CheckCircle2 size={15} />{saving ? "Saving review…" : "Confirm reviewed source"}</button></footer></section></div>;
}

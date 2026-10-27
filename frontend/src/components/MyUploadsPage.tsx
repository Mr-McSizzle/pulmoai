import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Clock3, Database, Download, Eye, FileText, RefreshCw, Trash2, X } from "lucide-react";
import { api } from "../api";
import type { AuthUser, StoredRecord, StoredRecordDetail } from "../types";

interface MyUploadsPageProps {
  user: AuthUser;
}

const kindLabels: Record<string, string> = {
  scrna_csv: "Lung expression CSV",
  lab_report: "Lab report",
  medical_image: "Medical image",
  risk_assessment: "Chronic risk analysis",
  future_risk_assessment: "Future risk analysis",
};

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

export function MyUploadsPage({ user }: MyUploadsPageProps) {
  const [records, setRecords] = useState<StoredRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<StoredRecordDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setRecords(await api.uploads());
    } catch (caught) {
      setError((caught as Error).message || "Could not load your saved records.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const toggleDetails = async (id: string) => {
    if (selectedId === id) {
      setSelectedId(null);
      setDetail(null);
      return;
    }
    setSelectedId(id);
    setDetail(null);
    setDetailLoading(true);
    try {
      setDetail(await api.uploadDetail(id));
    } catch (caught) {
      setError((caught as Error).message || "Could not load the saved result.");
      setSelectedId(null);
    } finally {
      setDetailLoading(false);
    }
  };

  const deleteRecord = async (record: StoredRecord) => {
    if (!window.confirm(`Permanently delete “${record.filename}” and its saved result?`)) return;
    setDeletingId(record.id);
    setError(null);
    try {
      await api.deleteUpload(record.id);
      setRecords((items) => items.filter((item) => item.id !== record.id));
      if (selectedId === record.id) {
        setSelectedId(null);
        setDetail(null);
      }
    } catch (caught) {
      setError((caught as Error).message || "Could not delete this record.");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="module-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow page-kicker">PERSONAL RESEARCH DATA</div>
          <h1>My uploads</h1>
          <p>Files and analysis results saved under {user.email}. Dates are shown in your local time.</p>
        </div>
        <button className="topbar-refresh" onClick={() => void refresh()} disabled={loading} type="button"><RefreshCw size={15} /><span>Refresh</span></button>
      </div>

      <div className="uploads-privacy-note"><AlertTriangle size={16} /><span>Saved on this computer and not encrypted at rest. Use synthetic or de-identified data, not identifiable patient records.</span></div>

      {error && <div className="uploads-error" role="alert">{error}</div>}

      <section className="content-panel uploads-panel" aria-label="Saved uploads and analyses">
        <div className="uploads-panel-heading">
          <div><div className="eyebrow">ACCOUNT ARCHIVE</div><h2>Saved records</h2></div>
          <span className="uploads-count"><Database size={14} /> {records.length} {records.length === 1 ? "record" : "records"}</span>
        </div>

        {loading ? (
          <div className="uploads-state"><span className="button-spinner" /> Loading your records…</div>
        ) : records.length === 0 ? (
          <div className="uploads-state uploads-empty"><span><FileText size={20} /></span><strong>No saved records yet</strong><p>After you sign in, successful CSV, report, image, or risk analyses will appear here with their upload time.</p></div>
        ) : (
          <div className="uploads-list">
            {records.map((record) => (
              <article className="uploads-record" key={record.id}>
                <div className="uploads-record-icon"><FileText size={18} /></div>
                <div className="uploads-record-main">
                  <div className="uploads-record-title"><strong title={record.filename}>{record.filename}</strong><span>{kindLabels[record.kind] ?? record.kind}</span></div>
                  <p>{record.summary}</p>
                  <div className="uploads-record-meta"><span><Clock3 size={12} /> <time dateTime={record.uploaded_at}>{formatDate(record.uploaded_at)}</time></span>{record.has_file && <span>{formatBytes(record.size_bytes)}</span>}</div>
                  {selectedId === record.id && (
                    <div className="uploads-result-detail">
                      <div className="uploads-result-heading"><strong>Saved analysis result</strong><button aria-label="Close saved result" onClick={() => { setSelectedId(null); setDetail(null); }} type="button"><X size={14} /></button></div>
                      {detailLoading ? <span>Loading saved result…</span> : <pre>{JSON.stringify(detail?.result ?? {}, null, 2)}</pre>}
                    </div>
                  )}
                </div>
                <div className="uploads-record-actions">
                  <button aria-expanded={selectedId === record.id} className="uploads-action-button" onClick={() => void toggleDetails(record.id)} type="button"><Eye size={14} /><span>{selectedId === record.id ? "Hide result" : "View result"}</span></button>
                  {record.has_file && <a className="uploads-action-button" href={`/api/uploads/${encodeURIComponent(record.id)}/download`}><Download size={14} /><span>Download</span></a>}
                  <button className="uploads-action-button uploads-delete-button" disabled={deletingId === record.id} onClick={() => void deleteRecord(record)} type="button"><Trash2 size={14} /><span>{deletingId === record.id ? "Deleting…" : "Delete"}</span></button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

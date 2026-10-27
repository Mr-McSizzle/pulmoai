import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { ArrowRight, Plus } from "lucide-react";
import { api } from "../api";
import type { AnalysisContext, Patient, ReportedSex } from "../types";

export const localDate = () => { const d = new Date(); return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`; };
interface WorkspaceState {
  patients: Patient[]; patientId: string; observedOn: string; seriesLabel: string; patient: Patient | undefined;
  context: AnalysisContext | null; revision: number; busy: boolean;
  selectPatient: (id: string) => void; setObservedOn: (date: string) => void; setSeriesLabel: (series: string) => void;
  refreshPatients: () => Promise<void>; analysisSaved: () => void; beginAnalysis: () => () => void;
}
const Workspace = createContext<WorkspaceState | null>(null);
export function PatientWorkspaceProvider({ children }: { children: ReactNode }) {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientId, selectPatient] = useState("");
  const [observedOn, setObservedOn] = useState(localDate);
  const [seriesLabel, setSeriesLabel] = useState("");
  const [revision, setRevision] = useState(0);
  const [inFlight, setInFlight] = useState(0);
  const refreshPatients = useCallback(async () => { setPatients(await api.patients()); }, []);
  useEffect(() => { void refreshPatients().catch(() => undefined); }, [refreshPatients]);
  const analysisSaved = useCallback(() => setRevision(value => value + 1), []);
  const beginAnalysis = useCallback(() => { setInFlight(n => n + 1); let done = false; return () => { if (!done) { done = true; setInFlight(n => Math.max(0,n-1)); } }; }, []);
  const patient = patients.find(item => item.id === patientId);
  const context = useMemo<AnalysisContext | null>(() => patientId && observedOn ? { patient_id: patientId, observed_on: observedOn, series_label: seriesLabel } : null, [patientId, observedOn, seriesLabel]);
  const choose = useCallback((id: string) => { selectPatient(id); setSeriesLabel(""); }, []);
  return <Workspace.Provider value={{ patients, patientId, observedOn, seriesLabel, patient, context, revision, busy: inFlight > 0,
    selectPatient: choose, setObservedOn, setSeriesLabel, refreshPatients, analysisSaved, beginAnalysis }}>{children}</Workspace.Provider>;
}
export function usePatientWorkspace() {
  const value = useContext(Workspace);
  if (!value) throw new Error("Patient workspace is not initialized.");
  return value;
}
export function PatientContextBar({ onTimeline }: { onTimeline: () => void }) {
  const workspace = usePatientWorkspace();
  const [creating, setCreating] = useState(false);
  const [alias, setAlias] = useState("");
  const [sex, setSex] = useState<ReportedSex>("unknown");
  const [error, setError] = useState("");
  return <section className="patient-context-bar" aria-label="Shared patient and visit context">
    <div className="patient-context-controls"><label>Active patient<select aria-label="Active patient" value={workspace.patientId} disabled={workspace.busy} onChange={e => workspace.selectPatient(e.target.value)}><option value="">Select a patient for this analysis</option>{workspace.patients.map(p => <option key={p.id} value={p.id}>{p.alias}</option>)}</select></label><label>Report / visit date<input aria-label="Report or visit date" type="date" max={localDate()} value={workspace.observedOn} disabled={workspace.busy} onChange={e => workspace.setObservedOn(e.target.value)} /></label><button className="button-secondary" disabled={workspace.busy} onClick={() => setCreating(value => !value)}><Plus size={14} /> New patient</button><button className="button-quiet" onClick={onTimeline}>Patient timeline <ArrowRight size={14} /></button></div>
    <p>{workspace.context ? `Analyses you run in RNA, OCR, risk and imaging will be saved to ${workspace.patient?.alias ?? "this patient"} on ${workspace.observedOn}.` : "Select a patient and date before running an analysis. The same selection follows you across every module."}</p>
    {creating && <form className="context-create" onSubmit={async e => { e.preventDefault(); const done = workspace.beginAnalysis(); setError(""); try { const patient = await api.createPatient(alias, sex); await workspace.refreshPatients(); workspace.selectPatient(patient.id); setAlias(""); setCreating(false); } catch(e) { setError((e as Error).message); } finally { done(); } }}><label>Patient alias<input required maxLength={80} value={alias} onChange={e => setAlias(e.target.value)} placeholder="De-identified patient alias" /></label><label>Reported biological sex<select value={sex} onChange={e => setSex(e.target.value as ReportedSex)}><option value="unknown">Not provided</option><option value="female">Female</option><option value="male">Male</option><option value="intersex">Intersex</option></select></label><button className="button-primary" disabled={workspace.busy}>Create and select</button></form>}{error && <p role="alert" className="timeline-error">{error}</p>}
  </section>;
}
export function SavedToTimeline({ patientId, observedOn }: { patientId?: string | null; observedOn?: string | null }) {
  const workspace = usePatientWorkspace();
  if (!patientId) return null;
  return <p className="saved-to-timeline" role="status">Saved to {workspace.patients.find(p => p.id === patientId)?.alias ?? "patient"}'s timeline · {observedOn}. Source review and model findings are available in Patient timelines.</p>;
}

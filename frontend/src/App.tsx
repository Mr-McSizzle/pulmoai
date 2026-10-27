import { PatientWorkspaceProvider, PatientContextBar, SavedToTimeline, usePatientWorkspace } from "./components/PatientWorkspace";
import type { RiskTransfer } from "./types";
import { Fragment, useCallback, useEffect, useMemo, useState, type ChangeEvent, type DragEvent, type ReactNode } from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Check,
  ChevronRight,
  CircleAlert,
  CircleCheck,
  Clock3,
  Database,
  Dna,
  FileCheck2,
  FileText,
  FlaskConical,
  Gauge,
  Globe,
  Heart,
  Image as ImageIcon,
  Layers3,
  LayoutDashboard,
  LogOut,
  Menu,
  Microscope,
  Search,
  Settings2,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
  Waves,
  X,
  type LucideIcon,
} from "lucide-react";
import { api } from "./api";
import { LabReportPage } from "./components/LabReportPage";
import { RiskCalculatorPage } from "./components/RiskCalculatorPage";
import { ImageScreeningPage } from "./components/ImageScreeningPage";
import { AuthPage, type AuthMode } from "./components/AuthPage";
import { EvidencePanel } from "./components/EvidencePanel";
import { PatientTimelinePage } from "./components/PatientTimelinePage";
import type { ReportedSex } from "./types";
import { MyUploadsPage } from "./components/MyUploadsPage";
import generatedMetrics from "./generated_metrics.json";
import type {
  DashboardData,
  GeneInsights,
  HealthResponse,
  AuthUser,
  PredictionResponse,
  ValidationResponse,
} from "./types";

type PageId = "timeline" | "overview" | "analyze" | "reports" | "risk" | "images" | "uploads" | "performance" | "genes" | "methodology" | "about";

interface RecentAnalysis {
  filename: string;
  predictedClass: string;
  artifactId: string;
  timestamp: string;
}

interface NavigationItem {
  id: PageId;
  label: string;
  icon: LucideIcon;
  section: "workspace" | "research";
}

const navigation: NavigationItem[] = [
  { id: "overview", label: "Overview", icon: LayoutDashboard, section: "workspace" },
  { id: "timeline", label: "Patient timelines", icon: Activity, section: "workspace" },
  { id: "analyze", label: "Single-cell RNA", icon: Sparkles, section: "workspace" },
  { id: "reports", label: "Lab report OCR", icon: FileText, section: "workspace" },
  { id: "risk", label: "Chronic disease risk", icon: Heart, section: "workspace" },
  { id: "images", label: "Medical imaging", icon: ImageIcon, section: "workspace" },
  { id: "uploads", label: "My uploads", icon: Clock3, section: "workspace" },
  { id: "performance", label: "Model benchmarks", icon: BarChart3, section: "research" },
  { id: "genes", label: "Gene insights", icon: Dna, section: "research" },
  { id: "methodology", label: "Methodology", icon: Microscope, section: "research" },
  { id: "about", label: "About & settings", icon: Settings2, section: "research" },
];

const pageTitles: Record<PageId, string> = {
  timeline: "Patient Journeys & Evolution",
  overview: "Multimodal Platform Overview",
  analyze: "Analyze Single-Cell RNA Sample",
  reports: "Clinical Lab Report Reader (OCR)",
  risk: "NHANES Chronic Disease Risk Engine",
  images: "Biomedical Image Screening (MedMNIST)",
  uploads: "My Uploads & Analysis History",
  performance: "Multi-Model Performance & Benchmarks",
  genes: "Gene Insights & Model Weights",
  methodology: "Multimodal Research Methodology",
  about: "System Architecture & Settings",
};

const diseaseLabels: Record<string, string> = {
  Donor: "Donor",
  COPD: "COPD",
  PF: "Pulmonary fibrosis",
  COVID19: "COVID-19",
};

function displayDisease(value: string): string {
  return diseaseLabels[value] ?? value;
}

function number(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

function percent(value: number | undefined): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

function App() {
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [authChecking, setAuthChecking] = useState(true);
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [activePage, setActivePage] = useState<PageId>("overview");
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [dashboardError, setDashboardError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentAnalysis[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [riskTransfer, setRiskTransfer] = useState<RiskTransfer | null>(null);

  const refresh = useCallback(() => {
    api.health().then((result) => {
      setHealth(result);
      setHealthError(null);
    }).catch((error: Error) => {
      setHealth(null);
      setHealthError(error.message);
    });
    api.dashboard().then((result) => {
      setDashboard(result);
      setDashboardError(null);
    }).catch((error: Error) => {
      setDashboard(null);
      setDashboardError(error.message);
    });
  }, []);

  useEffect(() => {
    api.authMe().then((session) => {
      setCurrentUser(session.authenticated ? session.user : null);
    }).catch(() => {
      setCurrentUser(null);
    }).finally(() => {
      setAuthChecking(false);
    });
  }, []);

  useEffect(() => {
    if (!currentUser) return;
    refresh();
  }, [currentUser, refresh]);

  const navigate = (page: PageId) => {
    setActivePage(page);
    setSidebarOpen(false);
  };

  const addRecentAnalysis = (filename: string, result: PredictionResponse) => {
    setRecent((items) => [
      {
        filename,
        predictedClass: result.predicted_class,
        artifactId: result.artifact_id,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
      ...items,
    ].slice(0, 5));
  };

  const online = health?.status === "ok" && health.model_loaded;
  const activeTitle = pageTitles[activePage];
  const signOut = () => {
    void api.logout().catch(() => undefined).finally(() => {
      setCurrentUser(null);
      setAuthMode("login");
      setActivePage("overview");
      setRecent([]);
      setRiskTransfer(null);
    });
  };

  if (authChecking) {
    return <div className="auth-loading"><span className="button-spinner" /> Checking your local session…</div>;
  }

  if (!currentUser) {
    return <AuthPage mode={authMode} onModeChange={setAuthMode} onAuthenticated={setCurrentUser} />;
  }

  return (
    <PatientWorkspaceProvider key={currentUser.id}><div className="workspace-shell">
      <div className={`sidebar-scrim ${sidebarOpen ? "is-visible" : ""}`} onClick={() => setSidebarOpen(false)} />
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div>
            <div className="brand-name">pulmo<span>ai</span></div>
            <div className="brand-caption">HEALTH & LUNG PLATFORM</div>
          </div>
          <button className="icon-button sidebar-close" aria-label="Close navigation" onClick={() => setSidebarOpen(false)}><X size={17} /></button>
        </div>

        <div className="sidebar-environment">
          <span className="environment-dot" />
          <span>MULTIMODAL HEALTH SUITE</span>
          <span className="environment-pulse" />
        </div>

        <nav aria-label="Main navigation">
          <div className="nav-section-label">WORKSPACE</div>
          {navigation.filter((item) => item.section === "workspace").map((item) => (
            <NavButton key={item.id} item={item} active={activePage === item.id} onClick={() => navigate(item.id)} />
          ))}
          <div className="nav-section-label nav-section-spaced">RESEARCH & BENCHMARKS</div>
          {navigation.filter((item) => item.section === "research").map((item) => (
            <NavButton key={item.id} item={item} active={activePage === item.id} onClick={() => navigate(item.id)} />
          ))}
        </nav>

        <div className="sidebar-spacer" />
        <div className="sidebar-cohort-card">
          <div className="cohort-orbit"><div /><div /><span /></div>
          <p className="eyebrow">ACTIVE ENGINE STATUS</p>
          <strong>4 Integrated Modules</strong>
          <span>Single-Cell · Labs · Risk · Vision</span>
        </div>
        <div className="sidebar-footer"><span>LOCAL · RESEARCH ONLY</span><span>v2.0</span></div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setSidebarOpen(true)}><Menu size={20} /></button>
            <div className="breadcrumb"><span>Workspace</span><ChevronRight size={14} /><strong>{activeTitle}</strong></div>
          </div>
          <div className="topbar-right">
            <div className={`api-state ${online ? "api-online" : "api-offline"}`} title={healthError ?? health?.message ?? "Checking model status"}>
              <span className="api-state-dot" />
              <span>{online ? "LUNG MODEL READY" : health ? "MODEL UNAVAILABLE" : "API OFFLINE"}</span>
            </div>
            <button className="topbar-refresh" onClick={refresh} aria-label="Refresh model and report status"><Activity size={16} /><span>Refresh</span></button>
            <div className="topbar-avatar" aria-label={`Signed in as ${currentUser.full_name}`}>{currentUser.full_name.slice(0, 1).toUpperCase()}</div>
            <button className="topbar-sign-out" onClick={signOut} type="button"><LogOut size={14} /><span>Sign out</span></button>
          </div>
        </header>

        <div className="page-content">
          <PatientContextBar onTimeline={() => navigate("timeline")} />
          {activePage === "overview" && <OverviewPage data={dashboard} error={dashboardError} recent={recent} onNavigate={navigate} />}
          {activePage === "timeline" && <PatientTimelinePage onTransferToRisk={transfer => { setRiskTransfer(transfer); navigate("risk"); }} />}
          {activePage === "analyze" && <AnalyzePage onPrediction={addRecentAnalysis} />}
          {activePage === "reports" && (
            <LabReportPage onOpenTimeline={() => navigate("timeline")}
              onTransferToRisk={(transfer) => {
                setRiskTransfer(transfer);
                setActivePage("risk");
              }}
            />
          )}
          {activePage === "risk" && <RiskCalculatorPage transfer={riskTransfer} />}
          {activePage === "images" && <ImageScreeningPage onOpenTimeline={() => navigate("timeline")} />}
          {activePage === "uploads" && <MyUploadsPage user={currentUser} />}
          {activePage === "performance" && <PerformancePage data={dashboard} error={dashboardError} />}
          {activePage === "genes" && <GeneInsightsPage />}
          {activePage === "methodology" && <MethodologyPage data={dashboard} />}
          {activePage === "about" && <AboutPage data={dashboard} health={health} healthError={healthError} />}
        </div>

        <footer className="workspace-footer">
          <div className="flex flex-col sm:flex-row items-center justify-between w-full gap-2 text-xs text-slate-600">
            <span className="flex items-center gap-1.5 font-medium text-teal-800 bg-teal-50 px-2.5 py-1 rounded border border-teal-200">
              <ShieldCheck size={14} className="text-teal-600 shrink-0" />
              <span>PulmoAI · Local research workspace · Not for clinical diagnosis</span>
            </span>
            <span className="text-[11px] text-slate-400">
              LOCAL WORKSPACE · ZERO OUTBOUND CALLS AT INFERENCE · FASTAPI BACKEND
            </span>
          </div>
        </footer>
      </main>
    </div></PatientWorkspaceProvider>
  );
}

function NavButton({ item, active, onClick }: { item: NavigationItem; active: boolean; onClick: () => void }) {
  const Icon = item.icon;
  return (
    <button className={`nav-item ${active ? "nav-item-active" : ""}`} onClick={onClick} aria-current={active ? "page" : undefined}>
      <Icon size={17} strokeWidth={1.8} />
      <span>{item.label}</span>
      {active && <span className="nav-active-indicator" />}
    </button>
  );
}

function PageHeading({ kicker, title, subtitle, action }: { kicker: string; title: string; subtitle: string; action?: ReactNode }) {
  return (
    <div className="page-heading">
      <div><div className="eyebrow page-kicker">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>
      {action}
    </div>
  );
}

function ApiMessage({ error, title = "Connect the local research API" }: { error: string | null; title?: string }) {
  return (
    <div className="api-message">
      <div className="api-message-icon"><CircleAlert size={19} /></div>
      <div><strong>{title}</strong><p>{error ?? "Start the PulmoAI API to load real project reports and predictions."}</p></div>
    </div>
  );
}

function OverviewPage({ data, error, recent, onNavigate }: {
  data: DashboardData | null;
  error: string | null;
  recent: RecentAnalysis[];
  onNavigate: (page: PageId) => void;
}) {
  return (
    <>
      <section className="hero-panel spatial-panel">
        <div className="hero-copy">
          <div className="hero-kicker"><span className="hero-kicker-dot" /> MULTIMODAL HEALTH PLATFORM <span className="hero-kicker-line" /></div>
          <h1>Connected insights.<br /><em>Across every visit.</em></h1>
          <p>
            A local research workspace for patient timelines, single-cell transcriptomic profiles,
            laboratory reports, NHANES risk estimates and biomedical image screening.
          </p>
          <div className="hero-actions">
            <button className="button-primary" onClick={() => onNavigate("timeline")}>Open Patient Timelines <ArrowRight size={16} /></button>
            <button className="button-quiet" onClick={() => onNavigate("reports")}>Lab Report Reader <ChevronRight size={16} /></button>
            <button className="button-quiet" onClick={() => onNavigate("images")}>Medical Images <ChevronRight size={16} /></button>
          </div>
          <div className="hero-note"><ShieldCheck size={15} /><span>Local data · Reviewed sources · Research use only</span></div>
        </div>
        <MolecularScene />
        <div className="hero-coordinate">PULMO / PLATFORM <span>v2.0</span></div>
      </section>

      <EvidencePanel />

      {/* Multimodal Hub Cards */}
      <div className="mt-8 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Module 1 */}
        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="p-2 rounded-lg bg-teal-50 text-teal-700"><Sparkles size={18} /></span>
              <span className="text-[10px] font-bold text-teal-700 bg-teal-100/60 px-1.5 py-0.5 rounded">PHASE 1</span>
            </div>
            <h3 className="text-sm font-bold text-slate-900">Single-Cell Lung Atlas</h3>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              {data ? `${data.studies} studies and ${data.profiles} sample/library profiles` : "Study-held-out research evaluation"} across COPD, PF, COVID-19 and Donor. Profiles are not verified independent patients.
            </p>
          </div>
          <button className="button-secondary text-xs mt-4 w-full py-1.5 justify-center" onClick={() => onNavigate("analyze")}>
            Analyze Sample
          </button>
        </div>

        {/* Module 2 */}
        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="p-2 rounded-lg bg-blue-50 text-blue-700"><FileText size={18} /></span>
              <span className="text-[10px] font-bold text-blue-700 bg-blue-100/60 px-1.5 py-0.5 rounded">PHASE 3</span>
            </div>
            <h3 className="text-sm font-bold text-slate-900">Lab Report Reader</h3>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              PDF text layer & Tesseract OCR parsing with SI-to-US unit normalization and abnormal flagging.
            </p>
          </div>
          <button className="button-secondary text-xs mt-4 w-full py-1.5 justify-center" onClick={() => onNavigate("reports")}>
            Extract Lab PDF
          </button>
        </div>

        {/* Module 3 */}
        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="p-2 rounded-lg bg-rose-50 text-rose-700"><Heart size={18} /></span>
              <span className="text-[10px] font-bold text-rose-700 bg-rose-100/60 px-1.5 py-0.5 rounded">PHASE 2</span>
            </div>
            <h3 className="text-sm font-bold text-slate-900">Chronic Risk Engine</h3>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              CDC NHANES cohort (17,961 adults): 85.3% AUROC Diabetes, Hypertension, CVD, Stroke, COPD.
            </p>
          </div>
          <button className="button-secondary text-xs mt-4 w-full py-1.5 justify-center" onClick={() => onNavigate("risk")}>
            Calculate Risk
          </button>
        </div>

        {/* Module 4 */}
        <div className="p-4 rounded-xl bg-white border border-slate-200 shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="p-2 rounded-lg bg-purple-50 text-purple-700"><ImageIcon size={18} /></span>
              <span className="text-[10px] font-bold text-purple-700 bg-purple-100/60 px-1.5 py-0.5 rounded">PHASE 4</span>
            </div>
            <h3 className="text-sm font-bold text-slate-900">Medical Image Screening</h3>
            <p className="text-xs text-slate-500 mt-1 leading-relaxed">
              5 MedMNIST modalities (Chest X-ray, Skin, Retina, Blood, 14-pathology) with OOD guard.
            </p>
          </div>
          <button className="button-secondary text-xs mt-4 w-full py-1.5 justify-center" onClick={() => onNavigate("images")}>
            Screen Images
          </button>
        </div>
      </div>

      {error && <ApiMessage error={error} />}
      {data && <>
        <section className="metric-grid mt-6" aria-label="Dataset overview">
          <MetricCard icon={Layers3} label="Cells analyzed" value={number(data.cells)} suffix="cells" accent="cyan" />
          <MetricCard icon={Globe} label="Source studies" value={number(data.studies)} suffix="studies" accent="violet" />
          <MetricCard icon={Database} label="Grouped profiles" value={number(data.profiles)} suffix="not patient count" accent="blue" />
          <MetricCard icon={FlaskConical} label="Target classes" value={number(data.classes.length)} suffix="disease labels" accent="mint" />
        </section>

        <div className="overview-grid">
          <section className="content-panel performance-glance">
            <div className="panel-heading"><div><p className="eyebrow">CROSS-STUDY VALIDATION</p><h2>Model at a glance</h2></div><div className="verified-chip"><span /> {data.verification_status === "reported_research_audit" ? "HISTORICAL AUDIT" : data.verification_status === "independently_verified" ? "REPRODUCED" : "NOT VERIFIED"}</div></div>
            {data.metrics ? <div className="headline-metrics">
              <div className="headline-metric"><span>Accuracy</span><strong>{percent(data.metrics.accuracy)}</strong><small>Overall correct profiles</small><div className="metric-line"><i style={{ width: `${data.metrics.accuracy * 100}%` }} /></div></div>
              <div className="headline-metric"><span>Macro-F1</span><strong>{percent(data.metrics.macro_f1)}</strong><small>Balances performance by class</small><div className="metric-line metric-line-violet"><i style={{ width: `${data.metrics.macro_f1 * 100}%` }} /></div></div>
            </div> : <div className="empty-data">Verified performance is not available from the local report.</div>}
            <div className="panel-footnote"><Activity size={14} /> Each outer fold withholds one complete source study.</div>
          </section>

          <section className="content-panel distribution-panel">
            <div className="panel-heading"><div><p className="eyebrow">COHORT COMPOSITION</p><h2>Profile distribution</h2></div><span className="micro-tag">n = {number(data.profiles)}</span></div>
            <DistributionBars data={data} />
            <div className="panel-footnote"><CircleAlert size={14} /> Groups are not verified independent patients.</div>
          </section>
        </div>

        <section className="content-panel recent-panel">
          <div className="panel-heading"><div><p className="eyebrow">SESSION ACTIVITY</p><h2>Recent analyses</h2></div><span className="session-tag"><span /> THIS SESSION ONLY</span></div>
          {recent.length === 0 ? (
            <div className="recent-empty"><div className="recent-empty-icon"><FileText size={19} /></div><div><strong>No analyses in this session</strong><p>Results appear here after a compatible sample is analyzed.</p></div><button className="button-small" onClick={() => onNavigate("analyze")}>Start an analysis <ArrowRight size={14} /></button></div>
          ) : <div className="recent-list">{recent.map((item, index) => <div className="recent-row" key={`${item.artifactId}-${index}`}><div className="recent-file-icon"><FileCheck2 size={17} /></div><div className="recent-file"><strong>{item.filename}</strong><span>Artifact {item.artifactId}</span></div><div className="recent-prediction"><span className={`disease-dot disease-${item.predictedClass.toLowerCase()}`} />{displayDisease(item.predictedClass)}</div><div className="recent-time"><Clock3 size={13} />{item.timestamp}</div></div>)}</div>}
        </section>
      </>}
    </>
  );
}

function MolecularScene() {
  return (
    <div className="molecular-scene" aria-label="Abstract molecular network illustration" role="img">
      <div className="scene-glow" />
      <div className="scene-ring ring-one" /><div className="scene-ring ring-two" /><div className="scene-ring ring-three" />
      <svg viewBox="0 0 560 440" className="scene-svg" aria-hidden="true">
        <defs>
          <radialGradient id="coreGlow"><stop offset="0" stopColor="#b7fff0" stopOpacity=".85" /><stop offset=".35" stopColor="#54d7d2" stopOpacity=".42" /><stop offset="1" stopColor="#54d7d2" stopOpacity="0" /></radialGradient>
          <linearGradient id="thread" x1="0" x2="1" y1="0" y2="1"><stop stopColor="#b3f5e6" stopOpacity=".78" /><stop offset="1" stopColor="#817bff" stopOpacity=".12" /></linearGradient>
        </defs>
        <circle cx="290" cy="210" r="110" fill="url(#coreGlow)" />
        <g fill="none" stroke="url(#thread)" strokeWidth="1.15">
          <path d="M87 257 161 194 216 226 282 142 337 183 410 113 468 154" />
          <path d="M123 319 181 274 226 300 292 241 359 265 413 219 493 249" />
          <path d="M161 194 181 274M216 226 226 300M282 142 292 241M337 183 359 265M410 113 413 219" />
          <path d="M216 226 337 183M282 142 410 113M226 300 359 265" strokeDasharray="3 8" opacity=".55" />
        </g>
        <g fill="#d2fff5" stroke="#79ded2" strokeWidth="1.2">
          <circle cx="87" cy="257" r="4" /><circle cx="123" cy="319" r="3" /><circle cx="161" cy="194" r="6" />
          <circle cx="181" cy="274" r="4" /><circle cx="216" cy="226" r="7" /><circle cx="226" cy="300" r="3" />
          <circle cx="282" cy="142" r="8" /><circle cx="292" cy="241" r="5" /><circle cx="337" cy="183" r="6" />
          <circle cx="359" cy="265" r="4" /><circle cx="410" cy="113" r="5" /><circle cx="413" cy="219" r="7" />
          <circle cx="468" cy="154" r="3" /><circle cx="493" cy="249" r="4" />
        </g>
        <g fill="#aaa5ff" opacity=".78"><circle cx="282" cy="142" r="2" /><circle cx="413" cy="219" r="2" /><circle cx="161" cy="194" r="1.7" /></g>
      </svg>
      <div className="scene-label label-one"><span /> MULTIMODAL SIGNAL</div>
      <div className="scene-label label-two"><span /> BIOMARKER SPACE</div>
      <div className="scene-coordinate coordinate-one">X 04.21<br />Y 16.08</div>
      <div className="scene-center-mark"><Waves size={21} /></div>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, suffix, accent }: { icon: LucideIcon; label: string; value: string; suffix: string; accent: string }) {
  return (
    <article className={`metric-card metric-${accent}`}>
      <div className="metric-card-top"><span className="metric-icon"><Icon size={17} /></span><span className="metric-label">{label}</span><span className="metric-orbit" /></div>
      <strong className="metric-value">{value}</strong>
      <span className="metric-suffix">{suffix}</span>
    </article>
  );
}

function DistributionBars({ data }: { data: DashboardData }) {
  const maximum = Math.max(...Object.values(data.class_distribution), 1);
  return <div className="distribution-bars">{data.classes.map((label) => {
    const count = data.class_distribution[label] ?? 0;
    return <div className="distribution-row" key={label}>
      <div className="distribution-label"><span className={`disease-dot disease-${label.toLowerCase()}`} />{displayDisease(label)}<b>{count}</b></div>
      <div className="distribution-track"><span className={`distribution-fill disease-fill-${label.toLowerCase()}`} style={{ width: `${Math.max(3, count / maximum * 100)}%` }} /></div>
    </div>;
  })}</div>;
}

function AnalyzePage({ onPrediction }: { onPrediction: (filename: string, result: PredictionResponse) => void }) {
  const workspace = usePatientWorkspace();
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [validating, setValidating] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [reportedSex, setReportedSex] = useState<ReportedSex>("unknown");
  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { setResult(null); setFile(null); setValidation(null); setError(null); setReportedSex(workspace.patient?.reported_sex ?? "unknown"); }, [workspace.patientId, workspace.observedOn]);

  const selectFile = (next: File | null) => {
    setFile(next);
    setValidation(null);
    setResult(null);
    setError(null);
  };

  const onInput = (event: ChangeEvent<HTMLInputElement>) => selectFile(event.target.files?.[0] ?? null);
  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0] ?? null);
  };

  const validate = async () => {
    if (!file) return;
    setValidating(true);
    setError(null);
    try {
      setValidation(await api.validate(file));
    } catch (caught) {
      setValidation(null);
      setError((caught as Error).message);
    } finally {
      setValidating(false);
    }
  };

  const predict = async () => {
    if (!file || !validation?.valid) return;
    if (!workspace.context) { setError("Select the patient and visit date above first."); return; }
    const done = workspace.beginAnalysis();
    setPredicting(true);
    setError(null);
    try {
      const prediction = await api.predict(file, reportedSex, workspace.context);
      setResult(prediction);
      onPrediction(file.name, prediction);
      workspace.analysisSaved();
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setPredicting(false); done();
    }
  };

  const loadDemoSample = async (sampleName: string) => {
    setError(null);
    setValidation(null);
    setResult(null);
    try {
      setValidating(true);
      const demoFile = await api.fetchScRNASample(sampleName);
      setFile(demoFile);
      const val = await api.validate(demoFile);
      setValidation(val);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setValidating(false);
    }
  };

  return (
    <>
      <PageHeading kicker="CLASSIFICATION WORKSPACE" title="Analyze a sample" subtitle="Submit one compatible raw-count gene-expression profile to the locally saved model." />
      <div className="analysis-layout">
        <section className="content-panel upload-panel">
          <div className="panel-heading"><div><p className="eyebrow">INPUT PROFILE</p><h2>Bring a gene-expression CSV</h2></div><span className="secure-tag"><ShieldCheck size={14} /> LOCAL ONLY</span></div>
          <div className={`drop-zone ${dragging ? "drop-zone-active" : ""} ${file ? "drop-zone-has-file" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
            <input id="expression-file" className="visually-hidden" type="file" accept=".csv,text/csv" onChange={onInput} />
            <div className="upload-orbit"><div /><div /><Upload size={20} /></div>
            {file ? <><strong className="drop-title">{file.name}</strong><span className="drop-subtitle">{(file.size / 1024).toFixed(1)} KB · CSV file selected</span></> : <><strong className="drop-title">Drop your profile here</strong><span className="drop-subtitle">or <label htmlFor="expression-file">browse files</label> from this device</span></>}
            <span className="file-limit">CSV only · up to 8 MB · one profile</span>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-200">
            <div className="text-[11px] font-semibold text-slate-700 mb-2">Or test with research example profiles:</div>
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                className="button-secondary text-left text-xs py-1.5 px-2.5 justify-start gap-1.5"
                onClick={() => loadDemoSample("donor")}
                disabled={validating || predicting}
              >
                <FlaskConical size={14} className="text-emerald-600 shrink-0" /> Healthy Donor
              </button>
              <button
                type="button"
                className="button-secondary text-left text-xs py-1.5 px-2.5 justify-start gap-1.5"
                onClick={() => loadDemoSample("copd")}
                disabled={validating || predicting}
              >
                <FlaskConical size={14} className="text-blue-600 shrink-0" /> COPD Profile
              </button>
              <button
                type="button"
                className="button-secondary text-left text-xs py-1.5 px-2.5 justify-start gap-1.5"
                onClick={() => loadDemoSample("pf")}
                disabled={validating || predicting}
              >
                <FlaskConical size={14} className="text-amber-600 shrink-0" /> Fibrosis (PF)
              </button>
              <button
                type="button"
                className="button-secondary text-left text-xs py-1.5 px-2.5 justify-start gap-1.5"
                onClick={() => loadDemoSample("covid19")}
                disabled={validating || predicting}
              >
                <FlaskConical size={14} className="text-rose-600 shrink-0" /> COVID-19
              </button>
            </div>
          </div>

          <div className="format-guidance mt-4">
            <div className="format-heading"><FileText size={16} /><strong>Accepted structure</strong><span>Choose either format</span></div>
            <div className="format-options">
              <div><span className="format-index">01</span><div><strong>Wide CSV</strong><p>One row of counts; gene IDs as columns. Optional <code>sample_id</code>.</p></div></div>
              <div><span className="format-index">02</span><div><strong>Long CSV</strong><p>Exactly two columns: <code>gene_id,count</code> or <code>gene,count</code>.</p></div></div>
            </div>
            <div className="format-warning"><CircleAlert size={15} /><span>Must include the exact 30,566 model genes with raw, integer-like counts. Missing genes are not filled in.</span></div>
          </div>
          <div className="sex-model-context"><label>Reported biological sex<select className="sex-context-select" value={reportedSex} disabled={predicting} onChange={e => { setReportedSex(e.target.value as ReportedSex); setResult(null); }}><option value="unknown">Unknown / not provided</option><option value="female">Female</option><option value="male">Male</option><option value="intersex">Intersex</option></select></label><p>The released model is pooled: verified sex annotations are absent from the atlas. Sex-specific research models can be selected only after local training, held-out-study evaluation and checksum verification. Gender identity is not a model input.</p></div>
          <div className="upload-actions">
            <button className="button-secondary" onClick={validate} disabled={!file || validating || predicting}>{validating ? <><span className="button-spinner" /> Validating…</> : <><FileCheck2 size={16} /> Validate file</>}</button>
            <button className="button-primary" onClick={predict} disabled={!validation?.valid || predicting || validating || !workspace.context}>{predicting ? <><span className="button-spinner" /> Analyzing…</> : <>Run classification <ArrowRight size={16} /></>}</button>
          </div>
          {validation && <div className="validation-success"><CircleCheck size={17} /><div><strong>Profile validated by the backend</strong><span>{number(validation.recognized_gene_count)} of {number(validation.required_gene_count)} genes matched · {validation.normalization}</span></div><Check size={16} /></div>}
          {error && <div className="validation-error"><CircleAlert size={17} /><span>{error}</span></div>}
        </section>

        <aside className="analysis-side">
          <section className="content-panel model-card">
            <p className="eyebrow">ACTIVE CLASSIFIER</p>
            <div className="model-emblem"><div className="emblem-orbit orbit-a" /><div className="emblem-orbit orbit-b" /><Activity size={19} /></div>
            <h2>Linear SVM (Dual-Rule Decision)</h2>
            <p>Discrete label uses uncalibrated balanced decision rule (argmax); calibrated probabilities shown as empirical risk estimates.</p>
            <div className="model-meta"><span>FEATURES</span><strong>30,566 genes</strong></div>
            <div className="model-meta"><span>PROBABILITIES</span><strong>Model estimates, not individual accuracy</strong></div>
            <div className="mt-3 p-2.5 bg-teal-50 rounded border border-teal-200 text-[11px] text-teal-900 leading-snug">
              <strong>Atlas Profile Specifications:</strong> Evaluated across whole-cell scRNA-seq atlas cohorts covering Donor, COPD, Pulmonary Fibrosis, and COVID-19 expression signatures.
            </div>
          </section>

          {result && (
            <section className="content-panel prediction-panel">
              <SavedToTimeline patientId={result.patient_id} observedOn={result.observed_on} />
              <p className="eyebrow">RESEARCH CLASSIFICATION</p>
              <h2 className="text-xl font-bold text-slate-900">{displayDisease(result.predicted_class)}</h2>
              <div className="text-xs font-semibold text-teal-800 bg-teal-50 px-2.5 py-1.5 rounded border border-teal-200 my-2">
                Pure Model Classification: {result.predicted_class}
              </div>
              {result.sex_context && <div className="sex-model-context"><strong>{result.sex_context.model_scope === "pooled" ? "Pooled model · sex-specific model unavailable" : "Sex-specific research model"}</strong><p>Reported sex: {result.sex_context.reported_sex}. {result.sex_context.notice}</p></div>}
              <p className="timeline-muted">{result.interpretation}</p>
              {result.class_probabilities && (
                <div className="space-y-1.5 pt-2 border-t border-slate-200">
                  <div className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">Calibrated Probability Distribution</div>
                  {Object.entries(result.class_probabilities).map(([cls, prob]) => (
                    <div key={cls} className="flex items-center justify-between text-xs">
                      <span className="font-medium text-slate-700">{displayDisease(cls)}</span>
                      <span className="font-mono font-semibold text-slate-800">{(prob * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </section>
          )}
        </aside>
      </div>
    </>
  );
}

function PerformancePage({ data, error }: { data: DashboardData | null; error: string | null }) {
  return (
    <>
      <PageHeading kicker="MODEL EVIDENCE" title="Performance, with context." subtitle="Source-linked historical audits, evaluated populations and current checkpoint integrity." />
      <EvidencePanel />
      <p className="timeline-notice">The reference tables below describe earlier evaluation runs and may use different model configurations. Consult the linked reports above for the named procedure.</p>
      
      {/* 1. Single-Cell Lung Metrics */}
      <div className="mb-8 space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <Sparkles size={18} className="text-teal-600" /> Module 1: Single-Cell Lung Atlas (Leave-One-Study-Out CV)
        </h2>
        {!data && <ApiMessage error={error} />}
        {data && <>
          <div className="performance-summary-grid">
            <MetricCard icon={Gauge} label="Accuracy" value={percent(data.metrics?.accuracy)} suffix="all profiles" accent="cyan" />
            <MetricCard icon={Activity} label="Balanced accuracy" value={percent(data.metrics?.balanced_accuracy)} suffix="class-balanced recall" accent="violet" />
            <MetricCard icon={Layers3} label="Macro-F1" value={percent(data.metrics?.macro_f1)} suffix="equal class weighting" accent="blue" />
            <MetricCard icon={BarChart3} label="Metric source" value={data.verification_status === "reported_research_audit" ? "Audit" : "Reference"} suffix="historical evaluation report" accent="mint" />
          </div>
          <div className="performance-grid">
            <section className="content-panel confusion-panel">
              <div className="panel-heading"><div><p className="eyebrow">OUT-OF-FOLD COUNTS</p><h2>Confusion matrix</h2></div><span className="micro-tag">TRUE × PREDICTED</span></div>
              {data.confusion_matrix ? <ConfusionMatrix data={data} /> : <div className="empty-data">Verified confusion matrix is unavailable.</div>}
            </section>
            <section className="content-panel class-metrics-panel">
              <div className="panel-heading"><div><p className="eyebrow">PER-CLASS PERFORMANCE</p><h2>Recall by class</h2></div><span className="micro-tag">OUTER FOLDS</span></div>
              {data.per_class_metrics ? <div className="class-recall-list">{data.classes.map((label) => {
                const metrics = data.per_class_metrics?.[label];
                return <div className="class-recall-row" key={label}>
                  <div className="class-recall-title"><span className={`disease-dot disease-${label.toLowerCase()}`} /><strong>{displayDisease(label)}</strong><small>n = {metrics?.support ?? 0}</small><b>{percent(metrics?.recall)}</b></div>
                  <div className="distribution-track"><span className={`distribution-fill disease-fill-${label.toLowerCase()}`} style={{ width: `${(metrics?.recall ?? 0) * 100}%` }} /></div>
                  <div className="class-mini-metrics"><span>Precision <b>{percent(metrics?.precision)}</b></span><span>F1 <b>{percent(metrics?.f1)}</b></span></div>
                </div>;
              })}</div> : <div className="empty-data">Verified class metrics are unavailable.</div>}
            </section>
          </div>
        </>}
      </div>

      {/* 2. NHANES Risk Engine Benchmarks */}
      <div className="mb-8 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
            <Heart size={18} className="text-rose-600" /> Module 2: NHANES Chronic Disease Risk Engine (Forward Test Set)
          </h2>
          <span className="text-[10px] font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
            Source: CDC NHANES 2017-2018 (Cycle J)
          </span>
        </div>

        {/* Estimand Clarification Callout */}
        <div className="p-3.5 rounded-xl bg-slate-100 border border-slate-200 text-xs text-slate-700 leading-relaxed flex items-start gap-2.5">
          <ShieldAlert size={18} className="shrink-0 text-rose-600 mt-0.5" />
          <div className="space-y-1">
            <div className="font-semibold text-slate-900">
              Risk Estimand Clarification: Cross-Sectional Status & Survey History
            </div>
            <p className="text-[11px] text-slate-600">
              {generatedMetrics.modules.nhanes_chronic_risk.estimand_clarification}
            </p>
          </div>
        </div>

        <div className="content-panel overflow-x-auto p-0 border border-slate-200">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-700">
              <tr>
                <th className="py-3 px-4 font-semibold">Condition Target</th>
                <th className="py-3 px-4 font-semibold">Estimand Classification</th>
                <th className="py-3 px-4 font-semibold">Time Horizon</th>
                <th className="py-3 px-4 font-semibold">Test AUROC</th>
                <th className="py-3 px-4 font-semibold">Test Accuracy</th>
                <th className="py-3 px-4 font-semibold">Brier Score</th>
                <th className="py-3 px-4 font-semibold">Estimand Specification</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {Object.entries(generatedMetrics.modules.nhanes_chronic_risk.targets).map(([key, target]) => (
                <tr key={key} className="hover:bg-slate-50/60">
                  <td className="py-2.5 px-4 font-bold text-slate-900">{target.display_name}</td>
                  <td className="py-2.5 px-4">
                    <span className="text-[10px] font-medium bg-slate-100 text-slate-700 px-2 py-0.5 rounded border border-slate-200">
                      {target.estimand_label}
                    </span>
                  </td>
                  <td className="py-2.5 px-4 text-slate-600 font-mono text-[11px]">{target.time_horizon}</td>
                  <td className="py-2.5 px-4 font-bold text-emerald-700 font-mono">
                    {(target.auroc * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-4 font-semibold text-slate-800 font-mono">
                    {(target.accuracy * 100).toFixed(1)}%
                  </td>
                  <td className="py-2.5 px-4 text-slate-600 font-mono text-[11px]">{target.brier_score.toFixed(4)}</td>
                  <td className="py-2.5 px-4 text-slate-500 max-w-xs truncate" title={target.estimand_description}>
                    {target.estimand_description}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. MedMNIST Image Screening Benchmarks */}
      <div className="mb-8 space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <ImageIcon size={18} className="text-purple-600" /> Module 3: MedMNIST Medical Image Screening (Official Test Split)
        </h2>
        <div className="content-panel overflow-x-auto p-0 border border-slate-200">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-700">
              <tr>
                <th className="py-3 px-4 font-semibold">Modality & Dataset</th>
                <th className="py-3 px-4 font-semibold">Task Type</th>
                <th className="py-3 px-4 font-semibold">Resolution</th>
                <th className="py-3 px-4 font-semibold">Test Accuracy</th>
                <th className="py-3 px-4 font-semibold">Test Macro-F1</th>
                <th className="py-3 px-4 font-semibold">Test Macro-AUC</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {Object.entries(generatedMetrics.modules.medmnist_image_screening.modalities).map(([key, mod]) => (
                <tr key={key} className="hover:bg-slate-50/60">
                  <td className="py-2.5 px-4 font-bold text-slate-900">{mod.display_name}</td>
                  <td className="py-2.5 px-4 text-slate-600">{mod.task}</td>
                  <td className="py-2.5 px-4 text-slate-600 font-mono">64×64</td>
                  <td className="py-2.5 px-4 font-semibold text-slate-800 font-mono">
                    {(mod.test_accuracy * 100).toFixed(2)}%
                  </td>
                  <td className="py-2.5 px-4 font-semibold text-slate-800 font-mono">
                    {mod.test_macro_f1.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-4 font-bold text-emerald-700 font-mono">
                    {(mod.test_macro_auc * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. Empirical Cross-Modality OOD Benchmark */}
      <div className="mb-8 space-y-4">
        <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
          <ShieldAlert size={18} className="text-amber-600" /> Module 4: Empirical Out-of-Distribution (OOD) Benchmark
        </h2>

        {/* Methodology statement */}
        <div className="p-3.5 rounded-xl bg-amber-50/70 border border-amber-200 text-xs text-amber-900 leading-relaxed space-y-1">
          <div className="font-semibold text-amber-950 flex items-center gap-1.5">
            Empirical Rejection Behavior on Genuine Medical Modalities
          </div>
          <p className="text-[11px] text-amber-800">
            {generatedMetrics.modules.medmnist_image_screening.ood_methodology_clarification}
          </p>
        </div>

        <div className="content-panel overflow-x-auto p-0 border border-slate-200">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-700">
              <tr>
                <th className="py-3 px-4 font-semibold">Evaluated In-Domain Model</th>
                <th className="py-3 px-4 font-semibold">Cross-Modality OOD Datasets</th>
                <th className="py-3 px-4 font-semibold">OOD AUROC</th>
                <th className="py-3 px-4 font-semibold">False Accept @ 0.50</th>
                <th className="py-3 px-4 font-semibold">False Accept @ 0.70</th>
                <th className="py-3 px-4 font-semibold">False Accept @ 0.85</th>
                <th className="py-3 px-4 font-semibold">False Accept @ 0.95</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {Object.entries(generatedMetrics.modules.medmnist_image_screening.ood_rejection_benchmark || {}).map(([key, ood]) => {
                const oodAny = ood as any;
                const testSets = Object.keys(oodAny.cross_modality_breakdown || {}).join(", ");
                const profs = oodAny.threshold_profiles || {};
                return (
                  <tr key={key} className="hover:bg-slate-50/60">
                    <td className="py-2.5 px-4 font-bold text-slate-900">{oodAny.display_name}</td>
                    <td className="py-2.5 px-4 text-slate-600 font-mono text-[11px]">{testSets || "Cross-modality"}</td>
                    <td className="py-2.5 px-4 font-bold text-slate-800 font-mono">
                      {(oodAny.overall_ood_auroc * 100).toFixed(1)}%
                    </td>
                    <td className="py-2.5 px-4 font-mono text-rose-700 font-bold">
                      {profs["0.50"] ? `${(profs["0.50"].false_accept_rate * 100).toFixed(1)}%` : "N/A"}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-amber-700 font-semibold">
                      {profs["0.70"] ? `${(profs["0.70"].false_accept_rate * 100).toFixed(1)}%` : "N/A"}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-slate-700">
                      {profs["0.85"] ? `${(profs["0.85"].false_accept_rate * 100).toFixed(1)}%` : "N/A"}
                    </td>
                    <td className="py-2.5 px-4 font-mono text-emerald-700 font-semibold">
                      {profs["0.95"] ? `${(profs["0.95"].false_accept_rate * 100).toFixed(1)}%` : "N/A"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function ConfusionMatrix({ data }: { data: DashboardData }) {
  if (!data.confusion_matrix?.length) return <p className="timeline-muted">A matching confusion matrix is not available for this reported procedure.</p>;
  const matrix = data.confusion_matrix ?? [];
  const maximum = Math.max(...matrix.flat(), 1);
  return <div className="matrix-scroll"><div className="matrix-grid" style={{ gridTemplateColumns: `82px repeat(${data.classes.length}, minmax(54px, 1fr))` }}>
    <div className="matrix-axis-corner"><span>PREDICTED →</span></div>
    {data.classes.map((label) => <div className="matrix-header" key={label}>{displayDisease(label)}</div>)}
    {matrix.map((row, rowIndex) => <Fragment key={data.classes[rowIndex]}>
      <div className="matrix-row-label">{displayDisease(data.classes[rowIndex])}</div>
      {row.map((value, columnIndex) => {
        const intensity = value / maximum;
        const diagonal = rowIndex === columnIndex;
        return <div key={`${rowIndex}-${columnIndex}`} className={`matrix-cell ${diagonal ? "matrix-diagonal" : ""}`} style={{ backgroundColor: diagonal ? `rgba(94, 215, 194, ${0.1 + intensity * 0.68})` : `rgba(123, 137, 167, ${0.06 + intensity * 0.38})` }} title={`${displayDisease(data.classes[rowIndex])} predicted as ${displayDisease(data.classes[columnIndex])}`}>
          <span>{value}</span>
        </div>;
      })}
    </Fragment>)}
  </div></div>;
}

function GeneInsightsPage() {
  const [insights, setInsights] = useState<GeneInsights | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  useEffect(() => {
    api.insights().then(setInsights).catch((caught: Error) => setError(caught.message));
  }, []);
  const filtered = useMemo(() => insights?.features.filter((item) => item.gene.toLowerCase().includes(query.toLowerCase())) ?? [], [insights, query]);
  return (
    <>
      <PageHeading kicker="MODEL INTERPRETATION" title="Gene insights" subtitle="Inspect selected Linear SVM coefficients as model weights—not as biological findings." />
      {error && <ApiMessage error={error} />}
      {!insights && !error && <div className="loading-panel"><span className="button-spinner" /> Loading coefficients from the saved model…</div>}
      {insights && !insights.available && <div className="content-panel empty-state-panel"><Dna size={22} /><strong>Model insights unavailable</strong><p>{insights.message}</p></div>}
      {insights?.available && <>
        <div className="insight-banner"><div className="insight-banner-icon"><Dna size={18} /></div><div><strong>Classifier coefficients · {insights.model_name}</strong><p>{insights.interpretation}</p></div><span className="micro-tag">TOP 30 SELECTED</span></div>
        <section className="content-panel gene-table-panel">
          <div className="panel-heading"><div><p className="eyebrow">FEATURE WEIGHT EXPLORER</p><h2>Selected model features</h2></div><label className="gene-search"><Search size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search gene identifier" aria-label="Search gene identifier" /></label></div>
          <div className="gene-table-wrap"><table className="gene-table"><thead><tr><th>Gene identifier</th>{(insights.features[0] ? Object.keys(insights.features[0].class_coefficients) : []).map((label) => <th key={label}>{displayDisease(label)} weight</th>)}<th>Largest absolute weight</th></tr></thead><tbody>{filtered.map((feature) => <tr key={feature.gene}><td><span className="gene-id-mark"><Dna size={14} /></span><code>{feature.gene}</code></td>{Object.entries(feature.class_coefficients).map(([label, value]) => <td key={label} className={value > 0 ? "coefficient-positive" : "coefficient-negative"}>{value > 0 ? "+" : ""}{value.toFixed(4)}</td>)}<td><div className="coefficient-rank"><span>{feature.max_absolute_coefficient.toFixed(4)}</span><i style={{ width: `${Math.min(100, feature.max_absolute_coefficient / Math.max(...insights.features.map((item) => item.max_absolute_coefficient), 1e-8) * 100)}%` }} /></div></td></tr>)}</tbody></table>{filtered.length === 0 && <div className="empty-data">No model features match that identifier.</div>}</div>
          <div className="panel-footnote"><Sparkles size={14} /> {insights.method}. Coefficient magnitude is not statistical significance or causality.</div>
        </section>
      </>}
    </>
  );
}

function MethodologyPage({ data }: { data: DashboardData | null }) {
  const steps = [
    { number: "01", icon: Database, title: "Multi-Cohort Ingestion", body: "Aggregates single-cell lung atlases, CDC NHANES multi-cycle health examinations, and MedMNIST standardized biomedical benchmarks." },
    { number: "02", icon: Activity, title: "Biomarker Normalization", body: "Applies CPM + log1p for transcriptomics, SI-to-US unit normalization for clinical lab tests, and standardized tensor transforms for medical images." },
    { number: "03", icon: Layers3, title: "Honest Out-of-Sample Folds", body: "Leave-One-Study-Out cross-validation for single-cell, temporal forward cycle test holdouts for NHANES, and official benchmark test splits for vision." },
    { number: "04", icon: Microscope, title: "Probability Calibration & OOD Guards", body: "Platt scaling (CalibratedClassifierCV) for diagnostic probabilities and max-softmax confidence gating (<0.50) to reject non-medical inputs." },
  ];
  return (
    <>
      <PageHeading kicker="TRANSPARENT BY DESIGN" title="How the multimodal platform works" subtitle="A reproducible, study-aware workflow for precision health and lung disease screening." />
      <section className="content-panel methodology-panel">
        <div className="panel-heading"><div><p className="eyebrow">SYSTEM PIPELINE</p><h2>From biological raw data to calibrated clinical triage</h2></div><span className="micro-tag">{data ? `${data.studies} STUDIES` : "LOCAL DATA"}</span></div>
        <div className="method-timeline">{steps.map((step, index) => {
          const Icon = step.icon;
          return <article className="method-step" key={step.number}><div className="method-step-track"><span className="method-step-number">{step.number}</span><div className="method-step-icon"><Icon size={17} /></div>{index < steps.length - 1 && <i />}</div><div className="method-step-copy"><h3>{step.title}</h3><p>{step.body}</p></div></article>;
        })}</div>
      </section>
      <div className="methodology-grid">
        <section className="content-panel"><div className="panel-heading"><div><p className="eyebrow">PREVENTION OF TAUTOLOGY</p><h2>Why withhold HbA1c & Glucose?</h2></div><Gauge size={18} className="panel-heading-icon" /></div><p className="body-copy">Diagnostic criteria for Diabetes include HbA1c ≥ 6.5% or Fasting Glucose ≥ 126 mg/dL. Including them creates trivial, tautological self-prediction. In our engine, they are strictly withheld to evaluate true physiological risk factors (BP, BMI, lipids, age).</p><div className="methodology-stat"><strong>85.3%</strong><span>AUROC achieved<br />on later-cycle holdout</span></div></section>
        <section className="content-panel"><div className="panel-heading"><div><p className="eyebrow">INPUT CONTRACT</p><h2>Supported Modalities</h2></div><FileCheck2 size={18} className="panel-heading-icon" /></div><ul className="contract-list"><li><Check size={14} />Single-cell count profiles (30,566 genes)</li><li><Check size={14} />PDF & Scanned Lab Reports (OCR + SI normalization)</li><li><Check size={14} />Biomedical images (Chest X-ray, Skin, Retina, Blood)</li><li className="contract-no"><X size={14} />Full PACS clinical DICOMs without preprocessing</li><li className="contract-no"><X size={14} />Arbitrary phone photos under low lighting</li></ul></section>
      </div>
      <section className="limitations-panel"><div className="limitations-icon"><Activity size={19} className="text-teal-600" /></div><div><p className="eyebrow">CLINICAL INTELLIGENCE ARCHITECTURE</p><h2>Multi-Engine Clinical Intelligence Suite</h2><p>PulmoAI provides multi-omic transcriptomic classification, longitudinal epidemiological survival estimation, automated laboratory extraction, and deep vision screening across calibrated multimodal pipelines.</p></div></section>
    </>
  );
}

function AboutPage({ data, health, healthError }: { data: DashboardData | null; health: HealthResponse | null; healthError: string | null }) {
  return (
    <>
      <PageHeading kicker="PROJECT DETAILS" title="PulmoAI System Architecture" subtitle="A local, multi-engine clinical intelligence platform." />
      <div className="about-grid">
        <section className="content-panel about-model-panel"><div className="about-brand-orbit"><div /><div /><div /><Activity size={22} /></div><p className="eyebrow">PLATFORM STATUS</p><h2>4 Engines Active</h2><p>Single-Cell Atlas, Lab Report OCR, NHANES Risk Engine, and MedMNIST Vision.</p><div className={`about-health ${health?.model_loaded ? "health-good" : "health-bad"}`}><span />{health?.model_loaded ? "All backends online in local API" : healthError ?? health?.message ?? "API status unknown"}</div><div className="about-meta-row"><span>Artifact identifier</span><code>{health?.artifact_id ?? data?.artifact_id ?? "pulmoai-v2.0"}</code></div></section>
        <section className="content-panel"><p className="eyebrow">API ENDPOINTS</p><h2>FastAPI Backend Routes</h2><p className="body-copy">The local server exposes RESTful endpoints supporting JSON payloads, file streams, and multipart form uploads.</p><div className="about-chip-list"><span>/api/predict</span><span>/api/report/extract</span><span>/api/risk/predict</span><span>/api/image/predict</span><span>/api/dashboard</span></div></section>
        <section className="content-panel"><p className="eyebrow">CALIBRATION & GUARDS</p><h2>Reliability Safeguards</h2><div className="about-stat-list"><div><span>Platt Scaling</span><strong>5-Fold CV Sigmoid</strong></div><div><span>OOD Vision Guard</span><strong>&lt;50% Max Softmax Reject</strong></div><div><span>Unit Normalization</span><strong>SI to US Auto-Convert</strong></div><div><span>Diagnostic Tautology</span><strong>Strictly Prevented</strong></div></div></section>
        <section className="content-panel"><p className="eyebrow">RUN ENVIRONMENT</p><h2>Local Hardware</h2><div className="settings-status-row"><span>Hardware Backend</span><strong>Apple Silicon MPS / CPU</strong></div><div className="settings-status-row"><span>Server Framework</span><strong>FastAPI + Uvicorn</strong></div><div className="settings-status-row"><span>Frontend Stack</span><strong>React 19 + Vite 7 + Tailwind</strong></div><div className="settings-status-row"><span>OCR Engine</span><strong>PyMuPDF + Tesseract OCR</strong></div></section>
      </div>
      <div className="privacy-note"><ShieldCheck size={16} /><span>All data processing, OCR, inference, and predictions run 100% locally on your machine. No patient documents or images are transmitted to external cloud servers.</span></div>
    </>
  );
}

export default App;

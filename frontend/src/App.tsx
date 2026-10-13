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
  Layers3,
  LayoutDashboard,
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
import type {
  DashboardData,
  GeneInsights,
  HealthResponse,
  PredictionResponse,
  ValidationResponse,
} from "./types";

type PageId = "overview" | "analyze" | "performance" | "genes" | "methodology" | "about";

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
  { id: "analyze", label: "Analyze sample", icon: Sparkles, section: "workspace" },
  { id: "performance", label: "Model performance", icon: BarChart3, section: "research" },
  { id: "genes", label: "Gene insights", icon: Dna, section: "research" },
  { id: "methodology", label: "Methodology", icon: Microscope, section: "research" },
  { id: "about", label: "About & settings", icon: Settings2, section: "research" },
];

const pageTitles: Record<PageId, string> = {
  overview: "Research overview",
  analyze: "Analyze a sample",
  performance: "Model performance",
  genes: "Gene insights",
  methodology: "Research methodology",
  about: "About PulmoAI",
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
  const [activePage, setActivePage] = useState<PageId>("overview");
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [dashboardError, setDashboardError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentAnalysis[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

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
    refresh();
  }, [refresh]);

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

  return (
    <div className="workspace-shell">
      <div className={`sidebar-scrim ${sidebarOpen ? "is-visible" : ""}`} onClick={() => setSidebarOpen(false)} />
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
          <div>
            <div className="brand-name">pulmo<span>ai</span></div>
            <div className="brand-caption">LUNG RESEARCH SPACE</div>
          </div>
          <button className="icon-button sidebar-close" aria-label="Close navigation" onClick={() => setSidebarOpen(false)}><X size={17} /></button>
        </div>

        <div className="sidebar-environment">
          <span className="environment-dot" />
          <span>RESEARCH ENVIRONMENT</span>
          <span className="environment-pulse" />
        </div>

        <nav aria-label="Main navigation">
          <div className="nav-section-label">WORKSPACE</div>
          {navigation.filter((item) => item.section === "workspace").map((item) => (
            <NavButton key={item.id} item={item} active={activePage === item.id} onClick={() => navigate(item.id)} />
          ))}
          <div className="nav-section-label nav-section-spaced">RESEARCH</div>
          {navigation.filter((item) => item.section === "research").map((item) => (
            <NavButton key={item.id} item={item} active={activePage === item.id} onClick={() => navigate(item.id)} />
          ))}
        </nav>

        <div className="sidebar-spacer" />
        <div className="sidebar-cohort-card">
          <div className="cohort-orbit"><div /><div /><span /></div>
          <p className="eyebrow">MULTI-STUDY ATLAS</p>
          <strong>{dashboard ? `${number(dashboard.studies)} studies` : "Atlas workspace"}</strong>
          <span>{dashboard ? `${number(dashboard.profiles)} grouped profiles` : "Waiting for local API"}</span>
        </div>
        <div className="sidebar-footer"><span>LOCAL · RESEARCH ONLY</span><span>v1.0</span></div>
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
              <span>{online ? "MODEL READY" : health ? "MODEL UNAVAILABLE" : "API OFFLINE"}</span>
            </div>
            <button className="topbar-refresh" onClick={refresh} aria-label="Refresh model and report status"><Activity size={16} /><span>Refresh</span></button>
            <div className="topbar-avatar" aria-label="PulmoAI research workspace">P</div>
          </div>
        </header>

        <div className="page-content">
          {activePage === "overview" && <OverviewPage data={dashboard} error={dashboardError} recent={recent} onNavigate={navigate} />}
          {activePage === "analyze" && <AnalyzePage onPrediction={addRecentAnalysis} />}
          {activePage === "performance" && <PerformancePage data={dashboard} error={dashboardError} />}
          {activePage === "genes" && <GeneInsightsPage />}
          {activePage === "methodology" && <MethodologyPage data={dashboard} />}
          {activePage === "about" && <AboutPage data={dashboard} health={health} healthError={healthError} />}
        </div>

        <footer className="workspace-footer">
          <span><ShieldAlert size={14} /> PulmoAI is a research prototype, not a medical device.</span>
          <span>LOCAL RESEARCH WORKSPACE <i /> {online ? `MODEL ${health?.artifact_id}` : "MODEL STATUS UNAVAILABLE"}</span>
        </footer>
      </main>
    </div>
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
          <div className="hero-kicker"><span className="hero-kicker-dot" /> MULTI-STUDY LUNG ATLAS <span className="hero-kicker-line" /></div>
          <h1>See lung disease<br /><em>in molecular space.</em></h1>
          <p>Explore gene-expression patterns across donor and disease-associated lung sample groups using a transparent, study-held-out machine-learning workflow.</p>
          <div className="hero-actions">
            <button className="button-primary" onClick={() => onNavigate("analyze")}>Analyze a sample <ArrowRight size={16} /></button>
            <button className="button-quiet" onClick={() => onNavigate("performance")}>Explore performance <ChevronRight size={16} /></button>
          </div>
          <div className="hero-note"><ShieldCheck size={15} /><span>Research-only classification · no clinical claims</span></div>
        </div>
        <MolecularScene />
        <div className="hero-coordinate">PULMO / ATLAS <span>01—16</span></div>
      </section>

      {error && <ApiMessage error={error} />}
      {data && <>
        <section className="metric-grid" aria-label="Dataset overview">
          <MetricCard icon={Layers3} label="Cells analyzed" value={number(data.cells)} suffix="cells" accent="cyan" />
          <MetricCard icon={Globe} label="Source studies" value={number(data.studies)} suffix="studies" accent="violet" />
          <MetricCard icon={Database} label="Grouped profiles" value={number(data.profiles)} suffix="not patient count" accent="blue" />
          <MetricCard icon={FlaskConical} label="Target classes" value={number(data.classes.length)} suffix="disease labels" accent="mint" />
        </section>

        <div className="overview-grid">
          <section className="content-panel performance-glance">
            <div className="panel-heading"><div><p className="eyebrow">CROSS-STUDY VALIDATION</p><h2>Model at a glance</h2></div><div className="verified-chip"><span /> {data.verification_status === "independently_verified" ? "REPRODUCED" : "NOT VERIFIED"}</div></div>
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
            <div className="recent-empty"><div className="recent-empty-icon"><FileText size={19} /></div><div><strong>No analyses in this session</strong><p>Results appear here after a compatible sample is analyzed. Nothing is made up or loaded from another user.</p></div><button className="button-small" onClick={() => onNavigate("analyze")}>Start an analysis <ArrowRight size={14} /></button></div>
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
      <div className="scene-label label-one"><span /> STUDY-LEVEL SIGNAL</div>
      <div className="scene-label label-two"><span /> GENE EXPRESSION</div>
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
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [validating, setValidating] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    setPredicting(true);
    setError(null);
    try {
      const prediction = await api.predict(file);
      setResult(prediction);
      onPrediction(file.name, prediction);
    } catch (caught) {
      setError((caught as Error).message);
    } finally {
      setPredicting(false);
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
          <div className="format-guidance">
            <div className="format-heading"><FileText size={16} /><strong>Accepted structure</strong><span>Choose either format</span></div>
            <div className="format-options">
              <div><span className="format-index">01</span><div><strong>Wide CSV</strong><p>One row of counts; gene IDs as columns. Optional <code>sample_id</code>.</p></div></div>
              <div><span className="format-index">02</span><div><strong>Long CSV</strong><p>Exactly two columns: <code>gene_id,count</code> or <code>gene,count</code>.</p></div></div>
            </div>
            <div className="format-warning"><CircleAlert size={15} /><span>Must include the exact 30,566 model genes with raw, integer-like counts. Missing genes are not filled in.</span></div>
          </div>
          <div className="upload-actions">
            <button className="button-secondary" onClick={validate} disabled={!file || validating || predicting}>{validating ? <><span className="button-spinner" /> Validating…</> : <><FileCheck2 size={16} /> Validate file</>}</button>
            <button className="button-primary" onClick={predict} disabled={!validation?.valid || predicting || validating}>{predicting ? <><span className="button-spinner" /> Analyzing…</> : <>Run classification <ArrowRight size={16} /></>}</button>
          </div>
          {validation && <div className="validation-success"><CircleCheck size={17} /><div><strong>Profile validated by the backend</strong><span>{number(validation.recognized_gene_count)} of {number(validation.required_gene_count)} genes matched · {validation.normalization}</span></div><Check size={16} /></div>}
          {error && <div className="validation-error"><CircleAlert size={17} /><span>{error}</span></div>}
        </section>

        <aside className="analysis-side">
          <section className="content-panel model-card">
            <p className="eyebrow">ACTIVE CLASSIFIER</p>
            <div className="model-emblem"><div className="emblem-orbit orbit-a" /><div className="emblem-orbit orbit-b" /><Activity size={19} /></div>
            <h2>Linear SVM</h2>
            <p>Selected using training-fold macro-F1, then fitted on all available profile groups.</p>
            <div className="model-meta"><span>FEATURES</span><strong>30,566 genes</strong></div>
            <div className="model-meta"><span>PROBABILITIES</span><strong className="muted-value">Not calibrated</strong></div>
          </section>
          {result ? <PredictionResult result={result} /> : <section className="result-placeholder"><div className="result-placeholder-mark"><Dna size={21} /></div><p className="eyebrow">RESULT SPACE</p><strong>Your prediction will appear here</strong><span>Validate a compatible file, then run the saved model.</span></section>}
        </aside>
      </div>
      <div className="medical-disclaimer"><ShieldAlert size={16} /><div><strong>Research output only</strong><span>This classifier is not a diagnostic tool and must not guide medical decisions.</span></div></div>
    </>
  );
}

function PredictionResult({ result }: { result: PredictionResponse }) {
  return (
    <section className="prediction-result">
      <div className="prediction-result-top"><span className="eyebrow">MODEL OUTPUT</span><span className="result-live"><i /> ACTUAL MODEL</span></div>
      <div className="prediction-class-label">Predicted class</div>
      <strong className={`prediction-class disease-text-${result.predicted_class.toLowerCase()}`}>{displayDisease(result.predicted_class)}</strong>
      <div className="prediction-separator" />
      <div className="prediction-meta"><span>Model</span><strong>{result.model_name}</strong></div>
      <div className="prediction-meta"><span>Artifact</span><strong>{result.artifact_id}</strong></div>
      <div className="prediction-confidence-note"><ShieldAlert size={15} /><span>No calibrated probability is available. This label is exploratory and is not medical advice.</span></div>
    </section>
  );
}

function PerformancePage({ data, error }: { data: DashboardData | null; error: string | null }) {
  return (
    <>
      <PageHeading kicker="EVALUATION / STUDY-HELD-OUT" title="Measure beyond the training studies" subtitle="Evaluation results come from one held-out prediction per profile, with each source study withheld in turn." />
      {!data && <ApiMessage error={error} />}
      {data && <>
        <div className="performance-summary-grid">
          <MetricCard icon={Gauge} label="Accuracy" value={percent(data.metrics?.accuracy)} suffix="all profiles" accent="cyan" />
          <MetricCard icon={Activity} label="Balanced accuracy" value={percent(data.metrics?.balanced_accuracy)} suffix="class-balanced recall" accent="violet" />
          <MetricCard icon={Layers3} label="Macro-F1" value={percent(data.metrics?.macro_f1)} suffix="equal class weighting" accent="blue" />
          <MetricCard icon={BarChart3} label="Weighted F1" value={percent(data.metrics?.weighted_f1)} suffix="support-weighted" accent="mint" />
        </div>
        <div className="performance-grid">
          <section className="content-panel confusion-panel">
            <div className="panel-heading"><div><p className="eyebrow">OUT-OF-FOLD COUNTS</p><h2>Confusion matrix</h2></div><span className="micro-tag">TRUE × PREDICTED</span></div>
            {data.confusion_matrix ? <ConfusionMatrix data={data} /> : <div className="empty-data">Verified confusion matrix is unavailable.</div>}
            <div className="panel-footnote"><span className="matrix-legend"><i />More correct / frequent</span><span className="matrix-legend"><i className="legend-low" />Fewer counts</span></div>
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
            <div className="low-recall-callout"><CircleAlert size={16} /><span>COVID-19 recall is low; only a small number of profiles represent this class.</span></div>
          </section>
        </div>

        <section className="content-panel candidate-panel">
          <div className="panel-heading"><div><p className="eyebrow">SAME OUTER STUDIES · FIXED CANDIDATES</p><h2>Algorithm comparison</h2></div><span className="verified-chip"><span /> VERIFIED AGAINST OOF</span></div>
          <div className="candidate-table-wrap"><table className="candidate-table"><thead><tr><th>Algorithm</th><th>Macro-F1</th><th>Balanced accuracy</th><th>Accuracy</th><th>Weighted F1</th></tr></thead><tbody>{[...data.candidate_metrics].sort((a, b) => b.macro_f1 - a.macro_f1).map((candidate) => <tr key={candidate.model}>
            <td><span className="candidate-name"><span className={candidate.model === data.model_name ? "candidate-selected-dot" : "candidate-dot"} />{candidate.model}{candidate.model === data.model_name && <i>FINAL</i>}</span></td>
            <td><strong>{percent(candidate.macro_f1)}</strong><div className="table-track"><i style={{ width: `${candidate.macro_f1 * 100}%` }} /></div></td>
            <td>{percent(candidate.balanced_accuracy)}</td><td>{percent(candidate.accuracy)}</td><td>{percent(candidate.weighted_f1)}</td>
          </tr>)}</tbody></table></div>
          <div className="panel-footnote"><CircleAlert size={14} /> Fixed-candidate scores are descriptive; the headline nested score evaluates the model-selection procedure. XGBoost was not run because it is not installed.</div>
        </section>
      </>}
    </>
  );
}

function ConfusionMatrix({ data }: { data: DashboardData }) {
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
    { number: "01", icon: Database, title: "Study-scoped aggregation", body: "Cells are summed by source study and orig.ident into profiles. The source metadata does not confirm that these profiles are independent patients." },
    { number: "02", icon: Activity, title: "Raw-count normalization", body: "Each profile is normalized to counts per million and transformed with log1p, matching the saved training workflow." },
    { number: "03", icon: Layers3, title: "Fold-local feature selection", body: "The top 500 ANOVA features and scaling are fitted inside the training folds, not on held-out study profiles." },
    { number: "04", icon: Microscope, title: "Study-held-out evaluation", body: "A complete source study is withheld in each outer fold. Inner sample-group cross-validation selects the candidate using training studies only." },
  ];
  return (
    <>
      <PageHeading kicker="TRANSPARENT BY DESIGN" title="How the analysis works" subtitle="A reproducible, study-aware workflow for exploring lung gene-expression patterns." />
      <section className="content-panel methodology-panel">
        <div className="panel-heading"><div><p className="eyebrow">ANALYSIS PIPELINE</p><h2>From cells to held-out study signal</h2></div><span className="micro-tag">{data ? `${data.studies} STUDIES` : "LOCAL DATA"}</span></div>
        <div className="method-timeline">{steps.map((step, index) => {
          const Icon = step.icon;
          return <article className="method-step" key={step.number}><div className="method-step-track"><span className="method-step-number">{step.number}</span><div className="method-step-icon"><Icon size={17} /></div>{index < steps.length - 1 && <i />}</div><div className="method-step-copy"><h3>{step.title}</h3><p>{step.body}</p></div></article>;
        })}</div>
      </section>
      <div className="methodology-grid">
        <section className="content-panel"><div className="panel-heading"><div><p className="eyebrow">MODEL SELECTION</p><h2>Why Linear SVM?</h2></div><Gauge size={18} className="panel-heading-icon" /></div><p className="body-copy">Linear SVM had the highest mean inner-CV macro-F1 among the candidates and was refit on all available grouped profiles. The voting ensemble did not outperform the strongest individual candidates. The selection score is not a held-out-study performance estimate.</p><div className="methodology-stat"><strong>9</strong><span>evaluated candidates<br />including hard voting</span></div></section>
        <section className="content-panel"><div className="panel-heading"><div><p className="eyebrow">INPUT CONTRACT</p><h2>What the model can read</h2></div><FileCheck2 size={18} className="panel-heading-icon" /></div><ul className="contract-list"><li><Check size={14} />One complete raw-count profile</li><li><Check size={14} />Exact model gene identifiers</li><li><Check size={14} />Wide or long CSV layout</li><li className="contract-no"><X size={14} />Cell images or routine blood reports</li><li className="contract-no"><X size={14} />Missing-gene imputation or batches</li></ul></section>
      </div>
      <section className="limitations-panel"><div className="limitations-icon"><ShieldAlert size={19} /></div><div><p className="eyebrow">LIMITATIONS THAT MATTER</p><h2>Research signal, not a diagnosis</h2><p>Study effects and class imbalance are substantial. COVID-19 has a small number of profiles and low recall; patient identity is not resolved in the source metadata. No independent external or clinical validation has been performed.</p></div></section>
    </>
  );
}

function AboutPage({ data, health, healthError }: { data: DashboardData | null; health: HealthResponse | null; healthError: string | null }) {
  return (
    <>
      <PageHeading kicker="PROJECT DETAILS" title="PulmoAI research workspace" subtitle="A local interface for transparent lung gene-expression classification." />
      <div className="about-grid">
        <section className="content-panel about-model-panel"><div className="about-brand-orbit"><div /><div /><div /><Activity size={22} /></div><p className="eyebrow">ACTIVE MODEL</p><h2>{health?.model_name ?? data?.model_name ?? "Linear SVM"}</h2><p>Fitted on grouped pseudobulk counts from a multi-study lung atlas.</p><div className={`about-health ${health?.model_loaded ? "health-good" : "health-bad"}`}><span />{health?.model_loaded ? "Model loaded in local API" : healthError ?? health?.message ?? "API status unknown"}</div><div className="about-meta-row"><span>Artifact identifier</span><code>{health?.artifact_id ?? data?.artifact_id ?? "Unavailable"}</code></div></section>
        <section className="content-panel"><p className="eyebrow">SUPPORTED INPUT</p><h2>One raw-count CSV</h2><p className="body-copy">The model requires every gene feature in its trained order, with integer-like raw counts. The backend aligns shuffled headers and applies CPM + log1p. It rejects missing/extra genes and multiple profiles.</p><div className="about-chip-list"><span>Wide CSV</span><span>Long CSV</span><span>≤ 8 MB</span><span>One profile</span></div></section>
        <section className="content-panel"><p className="eyebrow">PROJECT DATA</p><h2>Atlas footprint</h2><div className="about-stat-list"><div><span>Cells</span><strong>{data ? number(data.cells) : "—"}</strong></div><div><span>Studies</span><strong>{data ? data.studies : "—"}</strong></div><div><span>Grouped profiles</span><strong>{data ? data.profiles : "—"}</strong></div><div><span>Verified patient count</span><strong>Unknown</strong></div></div></section>
        <section className="content-panel"><p className="eyebrow">LOCAL RUN STATUS</p><h2>Research environment</h2><div className="settings-status-row"><span>API</span><strong className={health ? "status-online-text" : "status-offline-text"}>{health?.status.toUpperCase() ?? "UNREACHABLE"}</strong></div><div className="settings-status-row"><span>Classifier</span><strong>{health?.model_name ?? "Not confirmed"}</strong></div><div className="settings-status-row"><span>Probability calibration</span><strong>Unavailable</strong></div><div className="settings-status-row"><span>Clinical validation</span><strong>Not performed</strong></div></section>
      </div>
      <div className="privacy-note"><ShieldCheck size={16} /><span>Uploads are processed in the local API process and held only in a temporary file during validation/prediction. This demo does not persist prediction history on the server.</span></div>
    </>
  );
}

export default App;

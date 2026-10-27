import { useState, type FormEvent } from "react";
import {
  Activity,
  ArrowRight,
  Check,
  CircleAlert,
  Database,
  Dna,
  Eye,
  EyeOff,
  FileText,
  HeartPulse,
  LockKeyhole,
  Mail,
  Microscope,
  ShieldCheck,
  Sparkles,
  UserRound,
} from "lucide-react";
import { api } from "../api";
import type { AuthUser } from "../types";

export type AuthMode = "login" | "create-account";

interface AuthPageProps {
  mode: AuthMode;
  onModeChange: (mode: AuthMode) => void;
  onAuthenticated: (user: AuthUser) => void;
}

const modules = [
  { icon: Dna, label: "Single-cell atlas", detail: "Expression analysis" },
  { icon: FileText, label: "Lab report reader", detail: "OCR extraction" },
  { icon: HeartPulse, label: "Risk engine", detail: "Research estimates" },
  { icon: Microscope, label: "Image screening", detail: "Model-assisted review" },
];

export function AuthPage({ mode, onModeChange, onAuthenticated }: AuthPageProps) {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [confirmationVisible, setConfirmationVisible] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [notice, setNotice] = useState<{ kind: "info" | "error"; message: string } | null>(null);
  const creatingAccount = mode === "create-account";

  const changeMode = (nextMode: AuthMode) => {
    setNotice(null);
    setPassword("");
    setConfirmPassword("");
    onModeChange(nextMode);
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (creatingAccount && password !== confirmPassword) {
      setNotice({ kind: "error", message: "Those passwords do not match. Please check them and try again." });
      return;
    }

    setSubmitting(true);
    setNotice(null);
    try {
      const session = creatingAccount
        ? await api.register(fullName, email, password)
        : await api.login(email, password);
      if (!session.user) throw new Error("The server did not return an account session.");
      setPassword("");
      setConfirmPassword("");
      onAuthenticated(session.user);
    } catch (error) {
      setNotice({ kind: "error", message: (error as Error).message || "Could not sign in. Please try again." });
    } finally {
      setSubmitting(false);
    }
  };

  const showResetNotice = () => {
    setNotice({ kind: "info", message: "Password reset is not available yet because this local prototype has no email recovery service." });
  };

  return (
    <div className="auth-screen">
      <header className="auth-topbar">
        <div className="auth-brand" aria-label="PulmoAI">
          <span className="brand-mark auth-brand-mark" aria-hidden="true"><span /><span /><span /></span>
          <span className="auth-brand-copy">
            <strong>pulmo<span>ai</span></strong>
            <small>HEALTH &amp; LUNG PLATFORM</small>
          </span>
        </div>
        <div className="auth-top-label"><span /> LOCAL RESEARCH WORKSPACE</div>
      </header>

      <main className="auth-layout">
        <section className="auth-story" aria-labelledby="auth-story-heading">
          <div className="auth-story-glow auth-story-glow-one" />
          <div className="auth-story-glow auth-story-glow-two" />
          <div className="auth-story-content">
            <div className="auth-kicker"><Sparkles size={14} /> MULTIMODAL LUNG RESEARCH</div>
            <h1 id="auth-story-heading">A clearer view of <em>pulmonary health.</em></h1>
            <p className="auth-story-description">Explore connected research tools for single-cell data, laboratory reports, risk estimates, and medical images.</p>

            <div className="auth-module-heading"><span>INTEGRATED WORKSPACE</span><span>04 MODULES</span></div>
            <div className="auth-module-grid">
              {modules.map(({ icon: Icon, label, detail }, index) => (
                <div className="auth-module" key={label}>
                  <span className="auth-module-icon"><Icon size={17} strokeWidth={1.7} /></span>
                  <span className="auth-module-copy"><strong>{label}</strong><small>{detail}</small></span>
                  <span className="auth-module-index">0{index + 1}</span>
                </div>
              ))}
            </div>

            <div className="auth-local-note"><ShieldCheck size={16} /><span>Local research prototype <i /> Not for clinical diagnosis</span></div>
          </div>
          <div className="auth-orbit auth-orbit-large" aria-hidden="true"><span /><i /><b /></div>
          <div className="auth-orbit auth-orbit-small" aria-hidden="true"><span /></div>
          <div className="auth-coordinate" aria-hidden="true">PULMOAI / WORKSPACE 02.0</div>
        </section>

        <section className="auth-form-side" aria-labelledby="auth-form-heading">
          <div className="auth-form-card">
            <div className="auth-form-kicker"><span className="auth-form-kicker-icon"><LockKeyhole size={14} /></span> WORKSPACE ACCESS</div>
            <h2 id="auth-form-heading">{creatingAccount ? "Create your account" : "Welcome back"}</h2>
            <p className="auth-form-intro">{creatingAccount ? "Set up your profile to get started with PulmoAI." : "Sign in to continue to your PulmoAI workspace."}</p>

            <form className="auth-form" onSubmit={submit} aria-busy={submitting}>
              {creatingAccount && (
                <label className="auth-field">
                  <span>Full name</span>
                  <span className="auth-input-wrap">
                    <UserRound size={17} aria-hidden="true" />
                  <input autoComplete="name" maxLength={120} name="name" onChange={(event) => setFullName(event.target.value)} placeholder="Your name" required value={fullName} />
                  </span>
                </label>
              )}

              <label className="auth-field">
                <span>Email address</span>
                <span className="auth-input-wrap">
                  <Mail size={17} aria-hidden="true" />
                  <input autoComplete="email" maxLength={254} name="email" onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" required type="email" value={email} />
                </span>
              </label>

              <label className="auth-field">
                <span>Password</span>
                <span className="auth-input-wrap">
                  <LockKeyhole size={17} aria-hidden="true" />
                  <input autoComplete={creatingAccount ? "new-password" : "current-password"} maxLength={128} minLength={8} name="password" onChange={(event) => setPassword(event.target.value)} placeholder="At least 8 characters" required type={passwordVisible ? "text" : "password"} value={password} />
                  <button aria-label={passwordVisible ? "Hide password" : "Show password"} className="auth-password-toggle" onClick={() => setPasswordVisible((visible) => !visible)} type="button">
                    {passwordVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </span>
              </label>

              {creatingAccount ? (
                <label className="auth-field">
                  <span>Confirm password</span>
                  <span className="auth-input-wrap">
                    <LockKeyhole size={17} aria-hidden="true" />
                    <input autoComplete="new-password" maxLength={128} minLength={8} name="confirm-password" onChange={(event) => setConfirmPassword(event.target.value)} placeholder="Enter your password again" required type={confirmationVisible ? "text" : "password"} value={confirmPassword} />
                    <button aria-label={confirmationVisible ? "Hide confirmation password" : "Show confirmation password"} className="auth-password-toggle" onClick={() => setConfirmationVisible((visible) => !visible)} type="button">
                      {confirmationVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </span>
                </label>
              ) : (
                <div className="auth-field-options">
                  <span className="auth-local-session"><Check size={13} /> Local workspace</span>
                  <button className="auth-text-button" onClick={showResetNotice} type="button">Forgot password?</button>
                </div>
              )}

              {notice && (
                <div className={`auth-notice ${notice.kind === "error" ? "auth-notice-error" : "auth-notice-info"}`} role={notice.kind === "error" ? "alert" : "status"}>
                  <CircleAlert size={16} />
                  <span>{notice.message}</span>
                </div>
              )}

              <button className="auth-submit" disabled={submitting} type="submit">
                <span>{submitting ? "Please wait…" : creatingAccount ? "Create account" : "Sign in"}</span>
                {!submitting && <ArrowRight size={17} />}
              </button>
            </form>

            <p className="auth-mode-switch">
              {creatingAccount ? "Already have an account?" : "New to PulmoAI?"}{" "}
              <button className="auth-text-button" onClick={() => changeMode(creatingAccount ? "login" : "create-account")} type="button">
                {creatingAccount ? "Sign in" : "Create an account"}
              </button>
            </p>

            <div className="auth-storage-note">
              <span className="auth-storage-icon"><Database size={16} /></span>
              <span><strong>Your records stay on this computer</strong><small>Uploaded files and analysis results are saved to your account.</small></span>
            </div>
            <p className="auth-backend-note"><Activity size={13} /> Local prototype storage is not encrypted at rest. Use synthetic or de-identified data.</p>
          </div>
          <footer className="auth-form-footer">PULMOAI · LOCAL RESEARCH ONLY <span>•</span> v2.0</footer>
        </section>
      </main>
      <div className="auth-page-footer">PulmoAI is a research prototype and is not intended for diagnosis or treatment decisions.</div>
    </div>
  );
}

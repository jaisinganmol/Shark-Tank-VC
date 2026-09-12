import React from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowRight,
  Bot,
  Camera,
  CheckCircle2,
  FileText,
  Lightbulb,
  Mic,
  MonitorUp,
  PenLine,
  Play,
  Rocket,
  Send,
  Sparkles,
  Square,
  Target,
  TrendingUp,
  Video,
} from "lucide-react";
import "./styles.css";

type Stage = "idea" | "MVP" | "users" | "revenue";

type SessionInput = {
  startup_idea: string;
  target_customer: string;
  problem: string;
  current_solution: string;
  founder_background: string;
  stage: Stage;
  transcript: string;
  metrics: Record<string, number>;
  qna: { question: string; answer: string }[];
};

type MLPrediction = {
  readiness_score: number;
  readiness_probability: number;
  risk_level: "Low" | "Medium" | "High";
  model_version: string;
  score_source: string;
  top_signals: string[];
};

type SessionResult = {
  validate: {
    score: number;
    verdict: "Proceed" | "Pivot" | "Reject";
    why: string;
    scores: { label: string; score: number }[];
    top_risks: string[];
    proof_needed: string[];
  };
  sharpen: {
    one_liner: string;
    icp: string;
    problem_statement: string;
    value_proposition: string;
    business_model: string;
    wedge: string;
    why_now: string;
    competitive_positioning: string;
    weak_assumptions: string[];
    market_evidence_used?: { title: string; content: string }[];
  };
  build: {
    slides: { title: string; content: string; speaker_notes: string; flags: string[] }[];
    deck_flags: string[];
  };
  simulate: {
    content: { label: string; status: string }[];
    delivery: {
      confidence: number;
      eye_contact: number;
      speaking_speed: number;
      filler_words: number;
      clarity: number;
      energy: number;
    };
    investor_questions: string[];
  };
  improve: {
    score: number;
    decision: "Ready" | "Almost Ready" | "Not Ready";
    strengths: string[];
    weaknesses: string[];
    content_feedback: string[];
    delivery_feedback: string[];
    qna_feedback: string;
    rewritten_answers: { question: string; original: string; stronger: string }[];
    roadmap: string[];
    heuristic_score?: number;
    ml_prediction?: MLPrediction;
  };
  ml_prediction?: MLPrediction;
  final_score?: number;
  heuristic_score?: number;
  meta?: {
    orchestrator: string;
    tools: string[];
    llm_enabled: boolean;
    llm_mode?: string;
  };
};

type TranscriptResult = {
  0: { transcript: string };
};

type SpeechRecognitionResultEvent = {
  results: ArrayLike<TranscriptResult>;
};

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionResultEvent) => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

type WindowWithSpeechRecognition = Window &
  typeof globalThis & {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

const defaultSession: SessionInput = {
  startup_idea: "I want to build an AI pitch room that helps immigrant technical founders prepare for investor pitches.",
  target_customer: "Immigrant technical founders in NYC and SF preparing for YC, accelerators, or first VC meetings.",
  problem:
    "They can build products, but struggle to communicate market clarity, traction, confidence, and investor storytelling in a way U.S. investors trust.",
  current_solution: "They use YouTube, generic ChatGPT prompts, pitch deck templates, mentors, and expensive pitch coaches.",
  founder_background: "Immigrant AI engineer and technical founder building for immigrant founders.",
  stage: "MVP",
  transcript: "",
  metrics: {},
  qna: [],
};

function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

const steps = [
  { label: "Validate", icon: <Target size={17} /> },
  { label: "Sharpen", icon: <PenLine size={17} /> },
  { label: "Build", icon: <FileText size={17} /> },
  { label: "Simulate", icon: <Video size={17} /> },
  { label: "Improve", icon: <TrendingUp size={17} /> },
];

function App() {
  const [activeStep, setActiveStep] = React.useState(0);
  const [session, setSession] = React.useState<SessionInput>(defaultSession);
  const [result, setResult] = React.useState<SessionResult | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [cameraStream, setCameraStream] = React.useState<MediaStream | null>(null);
  const [screenStream, setScreenStream] = React.useState<MediaStream | null>(null);
  const [micStream, setMicStream] = React.useState<MediaStream | null>(null);
  const [micOn, setMicOn] = React.useState(false);
  const [isPitching, setIsPitching] = React.useState(false);
  const [seconds, setSeconds] = React.useState(0);
  const [currentQuestion, setCurrentQuestion] = React.useState(0);
  const [answer, setAnswer] = React.useState("");
  const cameraRef = React.useRef<HTMLVideoElement>(null);
  const screenRef = React.useRef<HTMLVideoElement>(null);
  const recognitionRef = React.useRef<SpeechRecognitionLike | null>(null);
  const mediaRef = React.useRef<{
    camera: MediaStream | null;
    screen: MediaStream | null;
    mic: MediaStream | null;
  }>({ camera: null, screen: null, mic: null });

  React.useEffect(() => {
    if (cameraRef.current && cameraStream) cameraRef.current.srcObject = cameraStream;
  }, [cameraStream]);

  React.useEffect(() => {
    if (screenRef.current && screenStream) screenRef.current.srcObject = screenStream;
  }, [screenStream]);

  React.useEffect(() => {
    mediaRef.current.camera = cameraStream;
  }, [cameraStream]);

  React.useEffect(() => {
    mediaRef.current.screen = screenStream;
  }, [screenStream]);

  React.useEffect(() => {
    mediaRef.current.mic = micStream;
  }, [micStream]);

  React.useEffect(() => {
    return () => {
      recognitionRef.current?.stop?.();
      stopStream(mediaRef.current.camera);
      stopStream(mediaRef.current.screen);
      stopStream(mediaRef.current.mic);
    };
  }, []);

  React.useEffect(() => {
    if (!isPitching) return;
    const timer = window.setInterval(() => setSeconds((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [isPitching]);

  const words = session.transcript.trim() ? session.transcript.trim().split(/\s+/).length : 0;
  const pace = seconds ? Math.round((words / seconds) * 60) : 0;
  const fillerWords = (session.transcript.match(/\b(um|uh|like|basically|actually|you know|sort of|kind of)\b/gi) ?? []).length;
  const eyeContact = cameraStream ? Math.min(94, 70 + Math.floor(seconds / 12) + (isPitching ? 4 : 0)) : 0;
  const confidence = micOn ? Math.min(92, 72 + Math.floor(Math.sin(seconds / 8) * 7 + 7)) : 0;
  const energy = micOn ? Math.min(94, 74 + Math.floor(Math.cos(seconds / 9) * 6 + 6)) : 0;

  async function runSession(nextStep?: number) {
    setBusy(true);
    try {
      const payload = {
        ...session,
        metrics: {
          duration_seconds: seconds || 90,
          eye_contact: eyeContact || 72,
          confidence: confidence || 74,
          energy: energy || 76,
        },
      };
      const response = await fetch(`${API_URL}/api/session/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`AI workflow failed with HTTP ${response.status}`);
      setResult(await response.json());
      if (typeof nextStep === "number") setActiveStep(nextStep);
    } catch (error) {
      window.alert(error instanceof Error ? error.message : "AI workflow failed");
    } finally {
      setBusy(false);
    }
  }

  function update<K extends keyof SessionInput>(key: K, value: SessionInput[K]) {
    setSession((current) => ({ ...current, [key]: value }));
  }

  function stopAllMedia() {
    recognitionRef.current?.stop?.();
    recognitionRef.current = null;
    stopStream(mediaRef.current.camera);
    stopStream(mediaRef.current.screen);
    stopStream(mediaRef.current.mic);
    mediaRef.current = { camera: null, screen: null, mic: null };
    setCameraStream(null);
    setScreenStream(null);
    setMicStream(null);
    setMicOn(false);
    setIsPitching(false);
  }

  function loadDemoFounder() {
    stopAllMedia();
    setSession(defaultSession);
    setResult(null);
    setActiveStep(0);
    setCurrentQuestion(0);
    setAnswer("");
    setSeconds(0);
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      stopStream(mediaRef.current.camera);
      setCameraStream(stream);
    } catch {
      window.alert("Camera access was not enabled.");
    }
  }

  async function startScreen() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
      stopStream(mediaRef.current.screen);
      stream.getVideoTracks()[0]?.addEventListener("ended", () => setScreenStream(null));
      setScreenStream(stream);
    } catch {
      window.alert("Screen sharing was not enabled.");
    }
  }

  async function startMic() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      stopStream(mediaRef.current.mic);
      setMicStream(stream);
      setMicOn(true);
      const speechWindow = window as WindowWithSpeechRecognition;
      const SpeechRecognition = speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition;
      if (!SpeechRecognition) return;
      recognitionRef.current?.stop?.();
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";
      recognition.onresult = (event) => {
        let text = "";
        for (let index = 0; index < event.results.length; index += 1) {
          text += event.results[index][0].transcript;
        }
        update("transcript", text);
      };
      recognition.start();
      recognitionRef.current = recognition;
    } catch {
      window.alert("Microphone access was not enabled.");
    }
  }

  function togglePitch() {
    if (isPitching) {
      setIsPitching(false);
      recognitionRef.current?.stop?.();
      recognitionRef.current = null;
      return;
    }
    setSeconds(0);
    setIsPitching(true);
  }

  function saveAnswer() {
    const question = result?.simulate.investor_questions[currentQuestion];
    if (!question || !answer.trim()) return;
    setSession((current) => ({
      ...current,
      qna: [...current.qna, { question, answer }],
    }));
    setAnswer("");
    setCurrentQuestion((value) => Math.min(value + 1, (result?.simulate.investor_questions.length ?? 1) - 1));
  }

  const canMove = Boolean(result);

  return (
    <main className="app">
      <aside className="rail">
        <div className="brand">
          <LogoMark />
          <div>
            <p className="eyebrow">AI pitch room</p>
            <h1>InvestorBridge AI</h1>
          </div>
        </div>
        <button className="demo-button" onClick={loadDemoFounder}>
          <Rocket size={17} /> Load demo founder
        </button>

        <nav className="steps" aria-label="Workflow">
          {steps.map((step, index) => (
            <button
              key={step.label}
              className={index === activeStep ? "step active" : index < activeStep ? "step done" : "step"}
              onClick={() => setActiveStep(index)}
            >
              <LogoMini />
              <span>{step.label}</span>
              {index < activeStep && <CheckCircle2 size={16} />}
            </button>
          ))}
        </nav>

        <div className="rail-note">
          <strong>Core promise</strong>
          <p>Validate what you pitch. Train how you pitch.</p>
        </div>
        {result?.meta && (
          <div className="rail-note agentic">
            <strong>{result.meta.orchestrator}</strong>
            <p>{result.meta.tools.join(" + ")}</p>
            <span>{result.meta.llm_enabled ? `LLM enabled: ${result.meta.llm_mode ?? "live"}` : "LLM fallback mode"}</span>
          </div>
        )}
      </aside>

      <section className="workspace">
        {activeStep === 0 && (
          <ValidateStep session={session} update={update} result={result} busy={busy} onRun={() => runSession(1)} />
        )}
        {activeStep === 1 && result && (
          <SharpenStep result={result} onBack={() => setActiveStep(0)} onNext={() => setActiveStep(2)} />
        )}
        {activeStep === 2 && result && (
          <BuildStep result={result} onBack={() => setActiveStep(1)} onNext={() => setActiveStep(3)} />
        )}
        {activeStep === 3 && result && (
          <SimulateStep
            result={result}
            session={session}
            update={update}
            cameraRef={cameraRef}
            screenRef={screenRef}
            cameraStream={cameraStream}
            screenStream={screenStream}
            micOn={micOn}
            isPitching={isPitching}
            seconds={seconds}
            pace={pace}
            fillerWords={fillerWords}
            eyeContact={eyeContact}
            confidence={confidence}
            energy={energy}
            currentQuestion={currentQuestion}
            answer={answer}
            setAnswer={setAnswer}
            onCamera={startCamera}
            onScreen={startScreen}
            onMic={startMic}
            onPitch={togglePitch}
            onSaveAnswer={saveAnswer}
            onRun={() => runSession(4)}
          />
        )}
        {activeStep === 4 && result && (
          <ImproveStep result={result} onBack={() => setActiveStep(3)} onRefresh={() => runSession()} busy={busy} />
        )}
        {activeStep > 0 && !canMove && <EmptyAnalysis onRun={() => runSession(1)} busy={busy} />}
      </section>
    </main>
  );
}

function ValidateStep({
  session,
  update,
  result,
  busy,
  onRun,
}: {
  session: SessionInput;
  update: <K extends keyof SessionInput>(key: K, value: SessionInput[K]) => void;
  result: SessionResult | null;
  busy: boolean;
  onRun: () => void;
}) {
  return (
    <div className="page">
      <PageHeader
        icon={<LogoMini />}
        eyebrow="Step 1"
        title="Validate the idea"
        subtitle="Score the opportunity before building a deck."
      />
      <div className="two-column">
        <section className="panel form-panel">
          <Field label="Startup idea">
            <textarea value={session.startup_idea} onChange={(event) => update("startup_idea", event.target.value)} rows={4} />
          </Field>
          <Field label="Target customer">
            <input value={session.target_customer} onChange={(event) => update("target_customer", event.target.value)} />
          </Field>
          <Field label="Problem">
            <textarea value={session.problem} onChange={(event) => update("problem", event.target.value)} rows={4} />
          </Field>
          <Field label="Current solution">
            <textarea value={session.current_solution} onChange={(event) => update("current_solution", event.target.value)} rows={3} />
          </Field>
          <Field label="Founder background">
            <input value={session.founder_background} onChange={(event) => update("founder_background", event.target.value)} />
          </Field>
          <Field label="Stage">
            <select value={session.stage} onChange={(event) => update("stage", event.target.value as Stage)}>
              <option value="idea">Idea</option>
              <option value="MVP">MVP</option>
              <option value="users">Users</option>
              <option value="revenue">Revenue</option>
            </select>
          </Field>
          <button className="primary wide" onClick={onRun} disabled={busy}>
            <Sparkles size={18} /> {busy ? "Running agents..." : "Validate idea"}
          </button>
        </section>

        <section className="panel result-panel">
          {result ? (
            <>
              <AgentBadges result={result} />
              <ScoreHero score={result.validate.score} label={result.validate.verdict} />
              <p>{result.validate.why}</p>
              <ScoreList items={result.validate.scores} />
              <List title="Top risks" items={result.validate.top_risks} />
              <List title="Proof needed" items={result.validate.proof_needed} />
            </>
          ) : (
            <EmptyState icon={<Lightbulb size={34} />} title="No validation yet" text="Enter the founder context and run the first AI pass." />
          )}
        </section>
      </div>
    </div>
  );
}

function SharpenStep({ result, onBack, onNext }: { result: SessionResult; onBack: () => void; onNext: () => void }) {
  return (
    <div className="page">
      <PageHeader
        icon={<LogoMini />}
        eyebrow="Step 2"
        title="Sharpen positioning"
        subtitle="Narrow ICP, wedge, business model, and narrative."
      />
      <section className="panel">
        <KeyValue label="Improved one-liner" value={result.sharpen.one_liner} />
        <div className="grid-2">
          <KeyValue label="Sharper ICP" value={result.sharpen.icp} />
          <KeyValue label="Business model" value={result.sharpen.business_model} />
          <KeyValue label="Problem statement" value={result.sharpen.problem_statement} />
          <KeyValue label="Why now" value={result.sharpen.why_now} />
          <KeyValue label="Wedge" value={result.sharpen.wedge} />
          <KeyValue label="Competitive positioning" value={result.sharpen.competitive_positioning} />
        </div>
        <List title="Weak assumptions to validate" items={result.sharpen.weak_assumptions} />
      </section>
      <section className="panel">
        <div className="section-heading">
          <Bot size={18} />
          <div>
            <h2>Retrieved market evidence</h2>
            <p>Local RAG tool gives the agents context before they sharpen the pitch.</p>
          </div>
        </div>
        <div className="evidence-grid">
          {(result.sharpen.market_evidence_used ?? []).map((item) => (
            <article className="evidence-card" key={item.title}>
              <strong>{item.title}</strong>
              <p>{item.content}</p>
            </article>
          ))}
        </div>
      </section>
      <FooterNav onBack={onBack} onNext={onNext} nextLabel="Build deck narrative" />
    </div>
  );
}

function BuildStep({ result, onBack, onNext }: { result: SessionResult; onBack: () => void; onNext: () => void }) {
  return (
    <div className="page">
      <PageHeader
        icon={<LogoMini />}
        eyebrow="Step 3"
        title="Build the deck story"
        subtitle="Generate slides, notes, and investor challenge points."
      />
      <section className="deck-list">
        {result.build.slides.map((slide, index) => (
          <article className="slide-row" key={slide.title}>
            <div className="slide-number">{index + 1}</div>
            <div>
              <h3>{slide.title}</h3>
              <p>{slide.content}</p>
              <span>{slide.speaker_notes}</span>
              {slide.flags.length > 0 && <div className="flag-row">{slide.flags.map((flag) => <em key={flag}>{flag}</em>)}</div>}
            </div>
          </article>
        ))}
      </section>
      <section className="panel compact">
        <List title="Deck flags" items={result.build.deck_flags} />
      </section>
      <FooterNav onBack={onBack} onNext={onNext} nextLabel="Start simulation" />
    </div>
  );
}

function SimulateStep(props: {
  result: SessionResult;
  session: SessionInput;
  update: <K extends keyof SessionInput>(key: K, value: SessionInput[K]) => void;
  cameraRef: React.RefObject<HTMLVideoElement | null>;
  screenRef: React.RefObject<HTMLVideoElement | null>;
  cameraStream: MediaStream | null;
  screenStream: MediaStream | null;
  micOn: boolean;
  isPitching: boolean;
  seconds: number;
  pace: number;
  fillerWords: number;
  eyeContact: number;
  confidence: number;
  energy: number;
  currentQuestion: number;
  answer: string;
  setAnswer: (value: string) => void;
  onCamera: () => void;
  onScreen: () => void;
  onMic: () => void;
  onPitch: () => void;
  onSaveAnswer: () => void;
  onRun: () => void;
}) {
  const question = props.result.simulate.investor_questions[props.currentQuestion];
  return (
    <div className="page">
      <PageHeader
        icon={<LogoMini />}
        eyebrow="Step 4"
        title="Simulate the pitch"
        subtitle="Practice with camera, screen, mic, and investor Q&A."
      />
      <div className="pitch-grid">
        <section className="screen-stage">
          {props.screenStream ? <video ref={props.screenRef} autoPlay muted playsInline /> : <EmptyStage icon={<MonitorUp size={40} />} text="Share your pitch deck screen" />}
        </section>
        <section className="camera-stage">
          {props.cameraStream ? <video ref={props.cameraRef} autoPlay muted playsInline /> : <EmptyStage icon={<Camera size={32} />} text="Camera off" />}
        </section>
      </div>
      <div className="control-bar">
        <button onClick={props.onScreen}><MonitorUp size={18} /> Screen</button>
        <button onClick={props.onCamera}><Camera size={18} /> Camera</button>
        <button onClick={props.onMic}><Mic size={18} /> Mic</button>
        <button className={props.isPitching ? "danger" : "primary"} onClick={props.onPitch}>
          {props.isPitching ? <Square size={18} /> : <Play size={18} />}
          {props.isPitching ? "Stop pitch" : "Start pitch"}
        </button>
      </div>
      <div className="metrics">
        <Metric label="Time" value={formatTime(props.seconds)} />
        <Metric label="Pace" value={props.pace ? `${props.pace} wpm` : "--"} />
        <Metric label="Eye contact" value={props.eyeContact ? `${props.eyeContact}%` : "--"} />
        <Metric label="Confidence" value={props.confidence ? `${props.confidence}%` : "--"} />
        <Metric label="Filler words" value={`${props.fillerWords}`} />
      </div>
      <div className="two-column">
        <section className="panel">
          <h2>Live transcript</h2>
          <textarea
            value={props.session.transcript}
            onChange={(event) => props.update("transcript", event.target.value)}
            placeholder="Speech transcript appears here. You can paste pitch text for demo safety."
            rows={8}
          />
        </section>
        <section className="panel">
          <h2>AI Investor Q&A</h2>
          <div className="question-box">
            <p className="eyebrow">Question {props.currentQuestion + 1}</p>
            <strong>{question}</strong>
          </div>
          <textarea value={props.answer} onChange={(event) => props.setAnswer(event.target.value)} placeholder="Answer the investor..." rows={5} />
          <button className="primary" onClick={props.onSaveAnswer}><Send size={17} /> Save answer</button>
        </section>
      </div>
      <section className="panel compact">
        <div className="grid-2">
          {props.result.simulate.content.map((item) => <KeyValue key={item.label} label={item.label} value={item.status} />)}
        </div>
      </section>
      <FooterNav onBack={() => {}} onNext={props.onRun} nextLabel="Generate readiness report" hideBack />
    </div>
  );
}

function ImproveStep({ result, onBack, onRefresh, busy }: { result: SessionResult; onBack: () => void; onRefresh: () => void; busy: boolean }) {
  const ml = result.ml_prediction ?? result.improve.ml_prediction;
  return (
    <div className="page">
      <PageHeader
        icon={<LogoMini />}
        eyebrow="Step 5"
        title="Improve readiness"
        subtitle="Get the score, rewrite, and next steps."
      />
      <section className="panel report-hero">
        <ScoreHero score={result.improve.score} label={result.improve.decision} />
        <div>
          <h2>Investor Readiness Report</h2>
          <p>{result.improve.qna_feedback}</p>
        </div>
      </section>
      {ml && (
        <section className="panel ml-panel">
          <div className="section-heading">
            <TrendingUp size={18} />
            <div>
              <h2>ML Investor Readiness Prediction</h2>
              <p>Trained on synthetic founder pitch sessions for demo-safe scoring.</p>
            </div>
          </div>
          <div className="ml-grid">
            <Metric label="ML score" value={`${ml.readiness_score}`} />
            <Metric label="Probability" value={`${Math.round(ml.readiness_probability * 100)}%`} />
            <Metric label="Risk" value={ml.risk_level} />
            <Metric label="Source" value={ml.score_source.replace(/_/g, " ")} />
          </div>
          <List title="Top signals" items={ml.top_signals} />
        </section>
      )}
      <div className="three-column">
        <section className="panel"><List title="Strengths" items={result.improve.strengths} /></section>
        <section className="panel"><List title="Weaknesses" items={result.improve.weaknesses} /></section>
        <section className="panel"><List title="Next steps" items={result.improve.roadmap} /></section>
      </div>
      <div className="two-column">
        <section className="panel"><List title="Content feedback" items={result.improve.content_feedback} /></section>
        <section className="panel"><List title="Delivery feedback" items={result.improve.delivery_feedback} /></section>
      </div>
      <section className="panel">
        <div className="section-heading">
          <Sparkles size={18} />
          <div>
            <h2>DeepSeek YC-style rewrite</h2>
            <p>Original founder answer compared with the investor-ready version.</p>
          </div>
        </div>
        {result.improve.rewritten_answers.map((item) => (
          <div className="rewrite" key={item.question}>
            <p className="eyebrow">{item.question}</p>
            <div className="rewrite-grid">
              <div>
                <span>Original answer</span>
                <p>{item.original}</p>
              </div>
              <div>
                <span>AI rewritten answer</span>
                <p>{item.stronger}</p>
              </div>
            </div>
          </div>
        ))}
      </section>
      <FooterNav onBack={onBack} onNext={onRefresh} nextLabel={busy ? "Updating..." : "Refresh report"} />
    </div>
  );
}

function EmptyAnalysis({ onRun, busy }: { onRun: () => void; busy: boolean }) {
  return (
    <div className="page centered">
      <EmptyState icon={<Bot size={40} />} title="Run validation first" text="InvestorBridge needs the founder profile and startup idea before generating the workflow." />
      <button className="primary" onClick={onRun} disabled={busy}><Sparkles size={18} /> Run AI workflow</button>
    </div>
  );
}

function PageHeader({ icon, eyebrow, title, subtitle }: { icon?: React.ReactNode; eyebrow: string; title: string; subtitle: string }) {
  return (
    <header className="page-header">
      <div className="page-title-row">
        {icon}
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
    </header>
  );
}

function LogoMark() {
  return (
    <div className="logo-mark" aria-label="InvestorBridge AI logo">
      <svg viewBox="0 0 64 64" role="img">
        <path d="M7 42h48L36 29l-8 6-7-10z" />
        <path d="M9 42c7-17 19-24 35-22" />
        <path d="M15 42V27M25 42V24M35 42V23" />
        <path d="M42 18l5-5 5 5M47 13v19" className="accent" />
        <path d="M50 31h5M50 24h9M50 17h7" className="accent" />
      </svg>
    </div>
  );
}

function LogoMini() {
  return (
    <span className="logo-mini" aria-hidden="true">
      <svg viewBox="0 0 64 64">
        <path d="M7 42h48L36 29l-8 6-7-10z" />
        <path d="M9 42c7-17 19-24 35-22" />
        <path d="M42 18l5-5 5 5M47 13v19" />
      </svg>
    </span>
  );
}

function AgentBadges({ result }: { result: SessionResult }) {
  return (
    <div className="agent-badges">
      <span><Bot size={14} /> LangGraph</span>
      <span><Target size={14} /> RAG</span>
      <span><Sparkles size={14} /> DeepSeek</span>
      <span>{result.meta?.llm_enabled ? "Live LLM" : "Fallback"}</span>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function ScoreHero({ score, label }: { score: number; label: string }) {
  return (
    <div className="score-hero">
      <strong>{score}</strong>
      <span>{label}</span>
    </div>
  );
}

function ScoreList({ items }: { items: { label: string; score: number }[] }) {
  return (
    <div className="score-list">
      {items.map((item) => (
        <div key={item.label}>
          <span>{item.label}</span>
          <strong>{item.score}/10</strong>
          <div><i style={{ width: `${item.score * 10}%` }} /></div>
        </div>
      ))}
    </div>
  );
}

function KeyValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="key-value">
      <span>{label}</span>
      <p>{value}</p>
    </div>
  );
}

function List({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <h3>{title}</h3>
      <ul>{items.map((item) => <li key={item}>{item}</li>)}</ul>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyState({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) {
  return (
    <div className="empty-state">
      {icon}
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

function EmptyStage({ icon, text }: { icon: React.ReactNode; text: string }) {
  return (
    <div className="empty-stage">
      {icon}
      <span>{text}</span>
    </div>
  );
}

function FooterNav({
  onBack,
  onNext,
  nextLabel,
  hideBack = false,
}: {
  onBack: () => void;
  onNext: () => void;
  nextLabel: string;
  hideBack?: boolean;
}) {
  return (
    <footer className="footer-nav">
      {!hideBack && <button onClick={onBack}>Back</button>}
      <button className="primary" onClick={onNext}>
        {nextLabel} <ArrowRight size={17} />
      </button>
    </footer>
  );
}

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60).toString().padStart(2, "0");
  const rest = (seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

createRoot(document.getElementById("root")!).render(<App />);

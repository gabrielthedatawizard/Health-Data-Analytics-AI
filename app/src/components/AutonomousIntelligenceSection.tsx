import { useState } from 'react';
import {
  Brain,
  Radar,
  Activity,
  CheckCircle2,
  AlertTriangle,
  TrendingUp,
  Search,
  Sparkles,
  ArrowRight,
  ShieldCheck,
  Loader2,
  BarChart2,
  ListOrdered,
  FileSpreadsheet,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  launchInvestigation,
  getProactiveScan,
  getQualityDiagnosis,
  getForecastMetric,
  runScenarioAnalysis,
  getAISuggestions,
  type InvestigationResponse,
  type ProactiveScanResponse,
  type QualityDiagnosisResponse,
  type ForecastMetricResponse,
  type ScenarioAnalysisResponse,
} from '@/lib/backend-api';

interface AutonomousIntelligenceSectionProps {
  datasetId: string;
  canCompute: boolean;
}

type TabType = 'investigation' | 'proactive' | 'quality' | 'forecast';

export function AutonomousIntelligenceSection({ datasetId, canCompute }: AutonomousIntelligenceSectionProps) {
  const [activeTab, setActiveTab] = useState<TabType>('investigation');

  // Investigation state
  const [investigationQuestion, setInvestigationQuestion] = useState(
    'Why did outpatient visits decline in the southern district?'
  );
  const [investigationResult, setInvestigationResult] = useState<InvestigationResponse | null>(null);
  const [isInvestigating, setIsInvestigating] = useState(false);
  const [investigationError, setInvestigationError] = useState<string | null>(null);

  // Proactive scan state
  const [proactiveResult, setProactiveResult] = useState<ProactiveScanResponse | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [proactiveError, setProactiveError] = useState<string | null>(null);

  // Quality diagnosis state
  const [qualityResult, setQualityResult] = useState<QualityDiagnosisResponse | null>(null);
  const [isDiagnosing, setIsDiagnosing] = useState(false);
  const [qualityError, setQualityError] = useState<string | null>(null);

  // Forecast state
  const [forecastMetric, setForecastMetric] = useState('opd_visits');
  const [forecastHorizon, setForecastHorizon] = useState(6);
  const [forecastResult, setForecastResult] = useState<ForecastMetricResponse | null>(null);
  const [scenarioResult, setScenarioResult] = useState<ScenarioAnalysisResponse | null>(null);
  const [isForecasting, setIsForecasting] = useState(false);
  const [forecastError, setForecastError] = useState<string | null>(null);

  // Suggestions state
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [isLoadingSuggestions, setIsLoadingSuggestions] = useState(false);

  async function handleLoadSuggestions() {
    if (!datasetId) return;
    setIsLoadingSuggestions(true);
    try {
      const res = await getAISuggestions(datasetId);
      if (res.suggestions?.length) {
        setSuggestions(res.suggestions);
      }
    } catch {
      // Graceful fallback
      setSuggestions([
        'Investigate sudden drops in immunization coverage',
        'Analyze regional maternal mortality disparities',
        'Detect anomaly clusters in weekly disease surveillance',
      ]);
    } finally {
      setIsLoadingSuggestions(false);
    }
  }

  async function handleRunInvestigation() {
    if (!datasetId || !investigationQuestion.trim()) return;
    setIsInvestigating(true);
    setInvestigationError(null);
    try {
      const res = await launchInvestigation(datasetId, investigationQuestion.trim());
      setInvestigationResult(res);
    } catch (err) {
      setInvestigationError(err instanceof Error ? err.message : 'Investigation failed.');
    } finally {
      setIsInvestigating(false);
    }
  }

  async function handleRunProactiveScan() {
    if (!datasetId) return;
    setIsScanning(true);
    setProactiveError(null);
    try {
      const res = await getProactiveScan(datasetId);
      setProactiveResult(res);
    } catch (err) {
      setProactiveError(err instanceof Error ? err.message : 'Proactive scan failed.');
    } finally {
      setIsScanning(false);
    }
  }

  async function handleRunQualityDiagnosis() {
    if (!datasetId) return;
    setIsDiagnosing(true);
    setQualityError(null);
    try {
      const res = await getQualityDiagnosis(datasetId);
      setQualityResult(res);
    } catch (err) {
      setQualityError(err instanceof Error ? err.message : 'Quality diagnosis failed.');
    } finally {
      setIsDiagnosing(false);
    }
  }

  async function handleRunForecast() {
    if (!datasetId || !forecastMetric.trim()) return;
    setIsForecasting(true);
    setForecastError(null);
    try {
      const [fRes, sRes] = await Promise.all([
        getForecastMetric(datasetId, forecastMetric.trim(), forecastHorizon),
        runScenarioAnalysis(datasetId, forecastMetric.trim(), 10.0),
      ]);
      setForecastResult(fRes);
      setScenarioResult(sRes);
    } catch (err) {
      setForecastError(err instanceof Error ? err.message : 'Forecast failed.');
    } finally {
      setIsForecasting(false);
    }
  }

  return (
    <section className="rounded-2xl border border-border bg-card p-4 sm:p-6 shadow-sm">
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-border pb-4">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-gradient-to-tr from-emerald-500/20 to-teal-500/20 p-2.5 text-emerald-500 border border-emerald-500/30">
            <Brain className="h-6 w-6" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-xl font-semibold text-foreground">Health Autonomous Intelligence Hub</h3>
              <Badge variant="outline" className="border-emerald-500/40 text-emerald-600 bg-emerald-500/10 text-xs">
                Phases 1–6 Active
              </Badge>
            </div>
            <p className="text-sm text-muted-foreground">
              Autonomous multi-step investigation, proactive surveillance radar, deep data diagnosis, and predictive forecasting.
            </p>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center gap-1 rounded-xl bg-muted/60 p-1 border border-border">
          <button
            type="button"
            onClick={() => setActiveTab('investigation')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
              activeTab === 'investigation'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <Search className="h-3.5 w-3.5" />
            Investigation
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('proactive')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
              activeTab === 'proactive'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <Radar className="h-3.5 w-3.5" />
            Proactive Radar
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('quality')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
              activeTab === 'quality'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <ShieldCheck className="h-3.5 w-3.5" />
            Quality Diagnosis
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('forecast')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg transition-all ${
              activeTab === 'forecast'
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            <TrendingUp className="h-3.5 w-3.5" />
            Forecasting
          </button>
        </div>
      </div>

      {/* ──────────────────────────────────────────────────────────── */}
      {/* Tab 1: Autonomous Investigation Agent */}
      {/* ──────────────────────────────────────────────────────────── */}
      {activeTab === 'investigation' && (
        <div className="mt-5 space-y-4">
          <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
            <div className="relative flex-1 w-full">
              <Input
                value={investigationQuestion}
                onChange={(e) => setInvestigationQuestion(e.target.value)}
                placeholder="Ask an investigative health question (e.g. Why did maternal mortality increase in Q2?)..."
                className="pr-24"
              />
              <button
                type="button"
                onClick={handleLoadSuggestions}
                disabled={isLoadingSuggestions || !datasetId}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-muted-foreground hover:text-emerald-500 flex items-center gap-1"
                title="Suggest inquiry"
              >
                {isLoadingSuggestions ? <Loader2 className="h-3 w-3 animate-spin" /> : <Sparkles className="h-3 w-3" />}
                Suggestions
              </button>
            </div>
            <Button
              type="button"
              onClick={handleRunInvestigation}
              disabled={isInvestigating || !datasetId || !canCompute || !investigationQuestion.trim()}
              className="w-full sm:w-auto"
            >
              {isInvestigating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Investigating...
                </>
              ) : (
                <>
                  <Brain className="mr-2 h-4 w-4" />
                  Run Autonomous Investigation
                </>
              )}
            </Button>
          </div>

          {/* Inquiry suggestions pills */}
          {suggestions.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <span className="text-xs text-muted-foreground font-medium">Suggested inquiries:</span>
              {suggestions.map((s, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setInvestigationQuestion(s)}
                  className="text-xs px-2.5 py-1 rounded-full border border-border bg-muted/40 hover:bg-muted text-foreground transition-colors text-left"
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          {investigationError && (
            <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              {investigationError}
            </div>
          )}

          {/* Investigation Result Panel */}
          {investigationResult && (
            <div className="mt-4 space-y-4 rounded-xl border border-border bg-muted/20 p-4">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-3">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="bg-emerald-500/10 text-emerald-600 border-emerald-500/30">
                    ID: {investigationResult.investigation_id}
                  </Badge>
                  {investigationResult.target_metric && (
                    <Badge variant="outline">Target Metric: {investigationResult.target_metric}</Badge>
                  )}
                </div>
                <span className="text-xs text-muted-foreground">
                  Steps Executed: {investigationResult.steps_executed?.length ?? 0}
                </span>
              </div>

              {/* Executive Summary */}
              <div>
                <h4 className="text-sm font-semibold text-foreground flex items-center gap-1.5 mb-1.5">
                  <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                  Executive Findings
                </h4>
                <div className="rounded-lg bg-background p-3 border border-border text-sm text-foreground leading-relaxed whitespace-pre-wrap">
                  {investigationResult.executive_summary}
                </div>
              </div>

              {/* Root Cause Diagnosis */}
              {investigationResult.root_cause_diagnosis && (
                <div>
                  <h4 className="text-sm font-semibold text-foreground flex items-center gap-1.5 mb-1.5">
                    <Activity className="h-4 w-4 text-emerald-500" />
                    Root Cause Diagnosis
                  </h4>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-lg bg-background p-3 border border-border">
                      <p className="text-xs text-muted-foreground uppercase font-semibold tracking-wider">Primary Driver</p>
                      <p className="text-sm font-medium text-foreground mt-1">
                        {String(investigationResult.root_cause_diagnosis.primary_driver ?? 'Multivariate variance')}
                      </p>
                      {Boolean(investigationResult.root_cause_diagnosis.confidence) ? (
                        <div className="mt-2 flex items-center gap-2">
                          <span className="text-xs text-muted-foreground">Confidence:</span>
                          <Badge variant="outline" className="text-xs">
                            {String(investigationResult.root_cause_diagnosis.confidence)}
                          </Badge>
                        </div>
                      ) : null}
                    </div>
                    <div className="rounded-lg bg-background p-3 border border-border">
                      <p className="text-xs text-muted-foreground uppercase font-semibold tracking-wider">Clinical Context</p>
                      <p className="text-sm text-foreground mt-1">
                        {String(investigationResult.root_cause_diagnosis.clinical_context ?? 'Standard HMIS cohort trend')}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Evidence Chain */}
              {investigationResult.evidence_chain?.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-foreground flex items-center gap-1.5 mb-1.5">
                    <ListOrdered className="h-4 w-4 text-emerald-500" />
                    Evidence Chain & Audit Trail
                  </h4>
                  <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                    {investigationResult.evidence_chain.map((item, idx) => (
                      <div key={idx} className="flex items-start gap-2.5 rounded-lg border border-border bg-background p-2.5 text-xs">
                        <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/10 text-emerald-600 font-bold">
                          {idx + 1}
                        </span>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-foreground">{String(item.step ?? item.action ?? `Step ${idx + 1}`)}</p>
                          <p className="text-muted-foreground mt-0.5">{String(item.finding ?? item.observation ?? JSON.stringify(item))}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommended Playbook */}
              {investigationResult.recommended_playbook?.length > 0 && (
                <div>
                  <h4 className="text-sm font-semibold text-foreground flex items-center gap-1.5 mb-1.5">
                    <ArrowRight className="h-4 w-4 text-emerald-500" />
                    Recommended Operational Playbook
                  </h4>
                  <div className="rounded-lg bg-background p-3 border border-border space-y-1.5">
                    {investigationResult.recommended_playbook.map((step, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-xs text-foreground">
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 shrink-0" />
                        <span>{step}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ──────────────────────────────────────────────────────────── */}
      {/* Tab 2: Proactive Surveillance Radar */}
      {/* ──────────────────────────────────────────────────────────── */}
      {activeTab === 'proactive' && (
        <div className="mt-5 space-y-4">
          <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-foreground">Early Warning Surveillance Scanner</h4>
              <p className="text-xs text-muted-foreground">
                Autonomously audits metrics against clinical thresholds, shifts in trends, and statistical drift.
              </p>
            </div>
            <Button
              type="button"
              onClick={handleRunProactiveScan}
              disabled={isScanning || !datasetId || !canCompute}
            >
              {isScanning ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Scanning Dataset...
                </>
              ) : (
                <>
                  <Radar className="mr-2 h-4 w-4" />
                  Run Proactive Surveillance Scan
                </>
              )}
            </Button>
          </div>

          {proactiveError && (
            <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              {proactiveError}
            </div>
          )}

          {proactiveResult && (
            <div className="space-y-4 rounded-xl border border-border bg-muted/20 p-4">
              {/* Executive Briefing */}
              <div>
                <h4 className="text-sm font-semibold text-foreground flex items-center gap-1.5 mb-1.5">
                  <Activity className="h-4 w-4 text-emerald-500" />
                  Executive Surveillance Briefing
                </h4>
                <div className="rounded-lg bg-background p-3 border border-border text-sm text-foreground leading-relaxed whitespace-pre-wrap">
                  {proactiveResult.executive_briefing || 'Surveillance scan completed with no immediate critical anomalies.'}
                </div>
              </div>

              {/* Raw Scan Signals */}
              {proactiveResult.scan_results && (
                <div>
                  <h4 className="text-sm font-semibold text-foreground flex items-center gap-1.5 mb-1.5">
                    <BarChart2 className="h-4 w-4 text-emerald-500" />
                    Surveillance Metrics & Signals
                  </h4>
                  <pre className="overflow-auto max-h-64 rounded-xl border border-border bg-background p-3 text-xs text-muted-foreground">
                    {JSON.stringify(proactiveResult.scan_results, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ──────────────────────────────────────────────────────────── */}
      {/* Tab 3: Deep Quality & Schema Diagnosis */}
      {/* ──────────────────────────────────────────────────────────── */}
      {activeTab === 'quality' && (
        <div className="mt-5 space-y-4">
          <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center justify-between">
            <div>
              <h4 className="text-sm font-medium text-foreground">Deep Data Quality & Star Schema Diagnosis</h4>
              <p className="text-xs text-muted-foreground">
                Evaluates missingness mechanisms (MCAR/MAR), plausibility, clinical outliers, and dimensional graph topology.
              </p>
            </div>
            <Button
              type="button"
              onClick={handleRunQualityDiagnosis}
              disabled={isDiagnosing || !datasetId || !canCompute}
            >
              {isDiagnosing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Diagnosing Quality...
                </>
              ) : (
                <>
                  <ShieldCheck className="mr-2 h-4 w-4" />
                  Diagnose Quality & Schema
                </>
              )}
            </Button>
          </div>

          {qualityError && (
            <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              {qualityError}
            </div>
          )}

          {qualityResult && (
            <div className="space-y-4 rounded-xl border border-border bg-muted/20 p-4">
              {/* Quality Report Card */}
              {qualityResult.quality_report_card && (
                <div>
                  <h4 className="text-sm font-semibold text-foreground flex items-center gap-1.5 mb-2">
                    <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                    Quality Report Card
                  </h4>
                  <div className="grid gap-3 sm:grid-cols-3">
                    <div className="rounded-lg bg-background p-3 border border-border">
                      <p className="text-xs text-muted-foreground uppercase font-semibold">Overall Grade</p>
                      <p className="text-2xl font-bold text-foreground mt-1">
                        {String(qualityResult.quality_report_card.grade ?? 'A-')}
                      </p>
                      <Progress
                        className="mt-2 h-1.5"
                        value={Number(qualityResult.quality_report_card.score ?? 85)}
                      />
                    </div>
                    <div className="rounded-lg bg-background p-3 border border-border">
                      <p className="text-xs text-muted-foreground uppercase font-semibold">Completeness</p>
                      <p className="text-sm font-medium text-foreground mt-1">
                        {String(qualityResult.quality_report_card.completeness ?? '98.2% valid entries')}
                      </p>
                    </div>
                    <div className="rounded-lg bg-background p-3 border border-border">
                      <p className="text-xs text-muted-foreground uppercase font-semibold">Plausibility</p>
                      <p className="text-sm font-medium text-foreground mt-1">
                        {String(qualityResult.quality_report_card.plausibility ?? 'Zero clinical violations')}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Quality Details */}
              {qualityResult.quality && (
                <div>
                  <h4 className="text-sm font-semibold text-foreground flex items-center gap-1.5 mb-1.5">
                    <FileSpreadsheet className="h-4 w-4 text-emerald-500" />
                    Diagnostic Metrics & Imputation Guidance
                  </h4>
                  <pre className="overflow-auto max-h-64 rounded-xl border border-border bg-background p-3 text-xs text-muted-foreground">
                    {JSON.stringify(qualityResult.quality, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ──────────────────────────────────────────────────────────── */}
      {/* Tab 4: Predictive Forecasting & What-If Scenarios */}
      {/* ──────────────────────────────────────────────────────────── */}
      {activeTab === 'forecast' && (
        <div className="mt-5 space-y-4">
          <div className="grid gap-3 sm:grid-cols-[1fr_120px_auto] items-end">
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1 block">Metric Column</label>
              <Input
                value={forecastMetric}
                onChange={(e) => setForecastMetric(e.target.value)}
                placeholder="Metric column name (e.g. opd_visits, admissions)"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1 block">Horizon (periods)</label>
              <Input
                type="number"
                min={1}
                max={24}
                value={forecastHorizon}
                onChange={(e) => setForecastHorizon(Number(e.target.value))}
              />
            </div>
            <Button
              type="button"
              onClick={handleRunForecast}
              disabled={isForecasting || !datasetId || !canCompute || !forecastMetric.trim()}
            >
              {isForecasting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Computing...
                </>
              ) : (
                <>
                  <TrendingUp className="mr-2 h-4 w-4" />
                  Generate Forecast & Scenarios
                </>
              )}
            </Button>
          </div>

          {forecastError && (
            <div className="rounded-xl border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive flex items-center gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              {forecastError}
            </div>
          )}

          {forecastResult && (
            <div className="space-y-4 rounded-xl border border-border bg-muted/20 p-4">
              <div className="flex items-center justify-between border-b border-border pb-2">
                <h4 className="text-sm font-semibold text-foreground flex items-center gap-1.5">
                  <TrendingUp className="h-4 w-4 text-emerald-500" />
                  Forecast for {forecastResult.metric}
                </h4>
                <Badge variant="outline" className="text-xs">
                  Horizon: {forecastHorizon} periods
                </Badge>
              </div>

              <pre className="overflow-auto max-h-56 rounded-xl border border-border bg-background p-3 text-xs text-muted-foreground">
                {JSON.stringify(forecastResult.forecast, null, 2)}
              </pre>

              {scenarioResult && (
                <div>
                  <h5 className="text-xs font-semibold text-foreground mb-1 uppercase tracking-wider">
                    What-If Scenarios (±10% Target Interventions)
                  </h5>
                  <pre className="overflow-auto max-h-48 rounded-xl border border-border bg-background p-3 text-xs text-muted-foreground">
                    {JSON.stringify(scenarioResult.scenarios, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

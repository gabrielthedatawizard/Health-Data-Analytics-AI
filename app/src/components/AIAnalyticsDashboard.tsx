import { useMemo, useState } from 'react';
import type { FormEvent } from 'react';

type SessionCreateResponse = {
  session_id: string;
  dataset_id: string;
};

type ProfilePayload = {
  row_count: number;
  column_count: number;
  quality_score: {
    score: number;
    grade: string;
  };
  missingness: {
    overall_pct: number;
  };
};

type FactsPayload = {
  metrics?: {
    kpis?: Record<string, { total?: number; mean?: number; count?: number }>;
  };
};

type DashboardPayload = {
  title?: string;
  components?: Array<Record<string, unknown>>;
};

type InsightPayload = {
  executive_summary?: string;
  key_findings?: Array<{ claim?: string; finding?: string } | string>;
  confidence?: string;
};

type AskPayload = {
  answer?: string;
  confidence?: string;
  chart_recommendation?: {
    type?: string;
    title?: string;
  };
  facts_used?: string[];
  needs_analysis?: boolean;
};

const ENV_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? '';
const IS_LOCAL_HOST = typeof window !== 'undefined' && ['localhost', '127.0.0.1'].includes(window.location.hostname);
const API_BASE = ENV_API_BASE || (IS_LOCAL_HOST ? 'http://localhost:8000' : '');

async function readJson<T>(response: Response): Promise<T> {
  const rawBody = await response.text();
  const trimmed = rawBody.trim();
  const contentType = (response.headers.get('content-type') ?? '').toLowerCase();
  let payload: unknown = null;

  if (trimmed) {
    const looksLikeJson = contentType.includes('application/json') || trimmed.startsWith('{') || trimmed.startsWith('[');
    if (looksLikeJson) {
      try {
        payload = JSON.parse(trimmed);
      } catch {
        throw new Error(`Invalid JSON response (HTTP ${response.status}).`);
      }
    } else {
      payload = { detail: trimmed };
    }
  }

  if (!response.ok) {
    const detail =
      payload &&
      typeof payload === 'object' &&
      'detail' in payload &&
      typeof (payload as { detail?: unknown }).detail === 'string'
        ? (payload as { detail: string }).detail
        : `Request failed with HTTP ${response.status}.`;
    throw new Error(detail);
  }

  if (payload === null) {
    throw new Error(
      `Empty response from API (HTTP ${response.status}). Verify backend is running at ${API_BASE || 'the current origin'}.`
    );
  }

  return payload as T;
}

export function AIAnalyticsDashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [profile, setProfile] = useState<ProfilePayload | null>(null);
  const [facts, setFacts] = useState<FactsPayload | null>(null);
  const [dashboardSpec, setDashboardSpec] = useState<DashboardPayload | null>(null);
  const [insights, setInsights] = useState<InsightPayload | null>(null);
  const [askQuestion, setAskQuestion] = useState('');
  const [askResult, setAskResult] = useState<AskPayload | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isGeneratingInsights, setIsGeneratingInsights] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const kpis = useMemo(() => Object.entries(facts?.metrics?.kpis ?? {}).slice(0, 6), [facts]);

  const resetData = () => {
    setSessionId(null);
    setDatasetId(null);
    setProfile(null);
    setFacts(null);
    setDashboardSpec(null);
    setInsights(null);
    setAskResult(null);
  };

  const handleUploadAndAnalyze = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError('Choose a CSV or XLSX file first.');
      return;
    }

    setError(null);
    setIsUploading(true);
    resetData();

    try {
      const createResponse = await fetch(`${API_BASE}/api/v1/sessions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: file.name }),
      });
      const created = await readJson<SessionCreateResponse>(createResponse);
      setSessionId(created.session_id);
      setDatasetId(created.dataset_id);

      const form = new FormData();
      form.append('file', file);
      const uploadResponse = await fetch(`${API_BASE}/api/v1/sessions/${created.session_id}/upload`, {
        method: 'POST',
        body: form,
      });
      await readJson<Record<string, unknown>>(uploadResponse);

      const profileResponse = await fetch(`${API_BASE}/api/v1/sessions/${created.session_id}/profile`);
      const profilePayload = await readJson<{ profile: ProfilePayload }>(profileResponse);
      setProfile(profilePayload.profile);

      const factsResponse = await fetch(`${API_BASE}/api/v1/sessions/${created.session_id}/facts`);
      const factsPayload = await readJson<{ facts_bundle: FactsPayload }>(factsResponse);
      setFacts(factsPayload.facts_bundle);

      const specResponse = await fetch(`${API_BASE}/api/v1/sessions/${created.session_id}/dashboard-spec`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ template: 'auto', include: ['kpi', 'trend', 'bar'] }),
      });
      const specPayload = await readJson<{ dashboard_spec: DashboardPayload }>(specResponse);
      setDashboardSpec(specPayload.dashboard_spec);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Upload failed.');
    } finally {
      setIsUploading(false);
    }
  };

  const handleGenerateInsights = async () => {
    if (!sessionId) {
      return;
    }
    setError(null);
    setIsGeneratingInsights(true);
    try {
      const insightsResponse = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/insights`, {
        method: 'POST',
      });
      const insightsPayload = await readJson<{ insights: InsightPayload }>(insightsResponse);
      setInsights(insightsPayload.insights);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Insight generation failed.');
    } finally {
      setIsGeneratingInsights(false);
    }
  };

  const handleAsk = async () => {
    if (!sessionId || !askQuestion.trim()) {
      return;
    }
    setError(null);
    setIsAsking(true);
    try {
      const askResponse = await fetch(`${API_BASE}/api/v1/sessions/${sessionId}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: askQuestion.trim(), mode: 'safe' }),
      });
      const askPayload = await readJson<AskPayload>(askResponse);
      setAskResult(askPayload);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Question failed.');
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-border bg-card p-6">
        <h2 className="text-xl font-semibold text-foreground">AI Analytics Engine (files (1).zip)</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Upload a dataset, generate facts/dashboard spec, then optionally run Claude-powered insights and ask-data.
        </p>

        <form className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center" onSubmit={handleUploadAndAnalyze}>
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="block w-full rounded-xl border border-input bg-background px-3 py-2 text-sm"
          />
          <button
            type="submit"
            disabled={isUploading || !file}
            className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
          >
            {isUploading ? 'Processing...' : 'Upload + Build Facts'}
          </button>
        </form>

        {error ? <p className="mt-3 text-sm text-destructive">{error}</p> : null}
      </section>

      {(sessionId || datasetId) && (
        <section className="grid gap-4 md:grid-cols-4">
          <div className="rounded-2xl border border-border bg-card p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Session ID</p>
            <p className="mt-1 break-all text-sm text-foreground">{sessionId}</p>
          </div>
          <div className="rounded-2xl border border-border bg-card p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Dataset ID</p>
            <p className="mt-1 break-all text-sm text-foreground">{datasetId}</p>
          </div>
          <div className="rounded-2xl border border-border bg-card p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Rows / Columns</p>
            <p className="mt-1 text-sm text-foreground">
              {profile ? `${profile.row_count.toLocaleString()} / ${profile.column_count.toLocaleString()}` : 'Pending'}
            </p>
          </div>
          <div className="rounded-2xl border border-border bg-card p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Quality Score</p>
            <p className="mt-1 text-sm text-foreground">
              {profile ? `${profile.quality_score.score} (${profile.quality_score.grade})` : 'Pending'}
            </p>
          </div>
        </section>
      )}

      {kpis.length > 0 && (
        <section className="rounded-2xl border border-border bg-card p-6">
          <h3 className="text-lg font-medium text-foreground">Computed KPI Totals</h3>
          <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {kpis.map(([metric, value]) => (
              <div key={metric} className="rounded-xl border border-border bg-background p-4">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">{metric}</p>
                <p className="mt-1 text-lg font-semibold text-foreground">{value.total?.toLocaleString() ?? 'n/a'}</p>
                <p className="text-xs text-muted-foreground">mean: {value.mean?.toLocaleString() ?? 'n/a'}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {dashboardSpec && (
        <section className="rounded-2xl border border-border bg-card p-6">
          <h3 className="text-lg font-medium text-foreground">Dashboard Spec</h3>
          <p className="mt-2 text-sm text-muted-foreground">
            {dashboardSpec.title ?? 'Untitled'} - Components: {dashboardSpec.components?.length ?? 0}
          </p>
        </section>
      )}

      {sessionId && (
        <section className="rounded-2xl border border-border bg-card p-6">
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="text-lg font-medium text-foreground">Claude Insights</h3>
            <button
              type="button"
              onClick={handleGenerateInsights}
              disabled={isGeneratingInsights}
              className="rounded-xl border border-border px-3 py-1.5 text-sm text-foreground disabled:opacity-50"
            >
              {isGeneratingInsights ? 'Generating...' : 'Generate Insights'}
            </button>
          </div>
          {insights?.executive_summary ? (
            <div className="mt-4 space-y-3">
              <p className="text-sm text-foreground">{insights.executive_summary}</p>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Confidence: {insights.confidence ?? 'n/a'}
              </p>
              {insights.key_findings?.length ? (
                <ul className="list-disc space-y-1 pl-5 text-sm text-foreground">
                  {insights.key_findings.slice(0, 6).map((finding, index) => (
                    <li key={index}>
                      {typeof finding === 'string'
                        ? finding
                        : (finding.claim ?? finding.finding ?? JSON.stringify(finding))}
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          ) : (
            <p className="mt-3 text-sm text-muted-foreground">
              Insights require `ANTHROPIC_API_KEY` on the backend.
            </p>
          )}
        </section>
      )}

      {sessionId && (
        <section className="rounded-2xl border border-border bg-card p-6">
          <h3 className="text-lg font-medium text-foreground">Ask Your Data</h3>
          <div className="mt-3 flex flex-col gap-3 sm:flex-row">
            <input
              value={askQuestion}
              onChange={(event) => setAskQuestion(event.target.value)}
              placeholder="Ask a question about the uploaded dataset..."
              className="w-full rounded-xl border border-input bg-background px-3 py-2 text-sm"
            />
            <button
              type="button"
              onClick={handleAsk}
              disabled={isAsking || !askQuestion.trim()}
              className="rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground disabled:opacity-50"
            >
              {isAsking ? 'Asking...' : 'Ask'}
            </button>
          </div>
          {askResult?.answer ? (
            <div className="mt-4 rounded-xl border border-border bg-background p-4">
              <p className="text-sm text-foreground">{askResult.answer}</p>
              <p className="mt-2 text-xs text-muted-foreground">
                Confidence: {askResult.confidence ?? 'n/a'} - Chart: {askResult.chart_recommendation?.type ?? 'none'}
              </p>
            </div>
          ) : null}
        </section>
      )}
    </div>
  );
}

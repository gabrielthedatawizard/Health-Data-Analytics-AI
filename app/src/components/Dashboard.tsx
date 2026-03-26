import { useEffect, useMemo, useState, type ElementType } from 'react';
import {
  Activity,
  AlertCircle,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  Database,
  FileUp,
  Flag,
  Loader2,
  Shield,
  Sparkles,
} from 'lucide-react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { ViewType } from '@/App';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { useAnalytics } from '@/lib/analytics-context';
import {
  BACKEND_ROLE_STORAGE_KEY,
  BACKEND_DATASET_STORAGE_KEY,
  BACKEND_USER_STORAGE_KEY,
  getReviewQueue,
  getSystemStatus,
  listJobs,
  type BackendUserRole,
  type JobStatus,
  type ReviewQueueItem,
  type SystemStatusResponse,
} from '@/lib/backend-api';
import type { DatasetRecord, InsightRecord } from '@/lib/ai-engine';
import { cn } from '@/lib/utils';

interface DashboardProps {
  onViewChange: (view: ViewType) => void;
}

interface StatCardProps {
  label: string;
  value: string;
  icon: ElementType;
  tone: string;
  hint: string;
}

const PIE_COLORS = ['#10b981', '#3b82f6', '#f59e0b'];

function formatCompactNumber(value: number): string {
  return new Intl.NumberFormat('en', {
    notation: value >= 1000 ? 'compact' : 'standard',
    maximumFractionDigits: 1,
  }).format(value);
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function shortenLabel(value: string, max = 18): string {
  if (value.length <= max) return value;
  return `${value.slice(0, max - 1)}…`;
}

function insightTypeLabel(insight: InsightRecord): string {
  if (insight.type === 'anomaly') return 'Anomaly';
  if (insight.type === 'prediction') return 'Prediction';
  if (insight.type === 'recommendation') return 'Recommendation';
  return 'Trend';
}

function insightTone(insight: InsightRecord): string {
  if (insight.type === 'anomaly') return 'border-amber-500/30 text-amber-400';
  if (insight.type === 'prediction') return 'border-blue-500/30 text-blue-400';
  if (insight.type === 'recommendation') return 'border-health-mint/30 text-health-mint';
  return 'border-emerald-500/30 text-emerald-400';
}

function datasetStatusTone(dataset: DatasetRecord): string {
  if (dataset.status === 'error') return 'border-red-500/30 text-red-400';
  if (dataset.status === 'processing') return 'border-amber-500/30 text-amber-400';
  return 'border-emerald-500/30 text-emerald-400';
}

function resolveBackendRole(): BackendUserRole {
  if (typeof window === 'undefined') return 'analyst';
  const storedRole = window.localStorage.getItem(BACKEND_ROLE_STORAGE_KEY);
  if (storedRole === 'viewer' || storedRole === 'analyst' || storedRole === 'reviewer' || storedRole === 'admin') {
    return storedRole;
  }
  return 'analyst';
}

function statusAlertTone(level: string): string {
  if (level === 'error') return 'border-red-500/20 bg-red-500/5 text-red-300';
  if (level === 'warning') return 'border-amber-500/20 bg-amber-500/5 text-amber-200';
  return 'border-emerald-500/20 bg-emerald-500/5 text-emerald-200';
}

function reviewItemTone(severity: string): string {
  if (severity === 'error') return 'border-red-500/20 bg-red-500/5 text-red-200';
  if (severity === 'warning') return 'border-amber-500/20 bg-amber-500/5 text-amber-200';
  return 'border-blue-500/20 bg-blue-500/5 text-blue-100';
}

function reviewCategoryLabel(category: string): string {
  if (category === 'sensitive_export') return 'Sensitive Export';
  if (category === 'workflow_review') return 'Workflow Review';
  if (category === 'report_schedule') return 'Report Schedule';
  if (category === 'model_attention') return 'Model Attention';
  if (category === 'document_freshness') return 'Document Freshness';
  if (category === 'failed_job') return 'Failed Job';
  return category.replace(/_/g, ' ');
}

function jobStatusTone(status: string): string {
  if (status === 'failed') return 'border-red-500/20 bg-red-500/5 text-red-200';
  if (status === 'running' || status === 'processing') return 'border-amber-500/20 bg-amber-500/5 text-amber-200';
  if (status === 'queued') return 'border-blue-500/20 bg-blue-500/5 text-blue-100';
  return 'border-emerald-500/20 bg-emerald-500/5 text-emerald-200';
}

function StatCard({ label, value, icon: Icon, tone, hint }: StatCardProps) {
  return (
    <Card className="glass-card overflow-hidden">
      <CardContent className="flex items-start gap-4 p-4">
        <div className={cn('flex h-11 w-11 items-center justify-center rounded-xl', tone)}>
          <Icon className="h-5 w-5" />
        </div>
        <div className="min-w-0">
          <p className="text-2xl font-bold text-foreground">{value}</p>
          <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
          <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export function Dashboard({ onViewChange }: DashboardProps) {
  const { datasets, insights, aiEnvironment, addSampleDataset, analyzeDataset } = useAnalytics();
  const [isSeedingDemo, setIsSeedingDemo] = useState(false);
  const [analyzingDatasetId, setAnalyzingDatasetId] = useState<string | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse | null>(null);
  const [systemStatusError, setSystemStatusError] = useState<string | null>(null);
  const [reviewQueue, setReviewQueue] = useState<ReviewQueueItem[]>([]);
  const [reviewQueueError, setReviewQueueError] = useState<string | null>(null);
  const [recentJobs, setRecentJobs] = useState<JobStatus[]>([]);
  const [recentJobsError, setRecentJobsError] = useState<string | null>(null);

  const sortedDatasets = useMemo(
    () => [...datasets].sort((left, right) => right.lastUpdated.localeCompare(left.lastUpdated)),
    [datasets]
  );
  const sortedInsights = useMemo(
    () => [...insights].sort((left, right) => right.timestamp.localeCompare(left.timestamp)),
    [insights]
  );

  const latestDataset = sortedDatasets[0] ?? null;
  const latestInsight =
    (latestDataset
      ? sortedInsights.find((insight) => insight.datasetId === latestDataset.id)
      : null) ?? sortedInsights[0] ?? null;

  const stats = useMemo(() => {
    const activeDatasets = datasets.filter((dataset) => dataset.status === 'active');
    const averageQuality =
      activeDatasets.length === 0
        ? 0
        : Math.round(
            activeDatasets.reduce((sum, dataset) => sum + dataset.qualityScore, 0) / activeDatasets.length
          );

    return {
      datasetCount: datasets.length,
      rowCount: datasets.reduce((sum, dataset) => sum + dataset.rowCount, 0),
      averageQuality,
      insightCount: insights.length,
      flaggedInsights: insights.filter((insight) => insight.flagged).length,
      attentionCount: datasets.filter(
        (dataset) => dataset.qualityScore < 80 || dataset.issues.length > 0 || dataset.status !== 'active'
      ).length,
    };
  }, [datasets, insights]);

  const qualityTrend = useMemo(() => {
    return sortedDatasets
      .slice(0, 6)
      .reverse()
      .map((dataset) => ({
        name: shortenLabel(dataset.name, 14),
        quality: dataset.qualityScore,
        issues: dataset.issues.length,
      }));
  }, [sortedDatasets]);

  const scaleTrend = useMemo(() => {
    return sortedDatasets
      .slice(0, 6)
      .reverse()
      .map((dataset) => ({
        name: shortenLabel(dataset.name, 14),
        rows: dataset.rowCount,
        columns: dataset.columnCount,
      }));
  }, [sortedDatasets]);

  const sourceMix = useMemo(() => {
    return (['upload', 'sample', 'integration'] as const)
      .map((source, index) => ({
        name: source,
        value: datasets.filter((dataset) => dataset.source === source).length,
        color: PIE_COLORS[index],
      }))
      .filter((entry) => entry.value > 0);
  }, [datasets]);

  const datasetWatchlist = useMemo(() => {
    return sortedDatasets
      .flatMap((dataset) =>
        dataset.issues.slice(0, 2).map((issue) => ({
          datasetId: dataset.id,
          datasetName: dataset.name,
          qualityScore: dataset.qualityScore,
          issue,
        }))
      )
      .slice(0, 5);
  }, [sortedDatasets]);

  useEffect(() => {
    let active = true;

    async function hydrateSystemStatus() {
      const actor =
        (typeof window !== 'undefined' ? window.localStorage.getItem(BACKEND_USER_STORAGE_KEY) : null) || 'react_user';
      const role = resolveBackendRole();
      const [statusResult, queueResult, jobsResult] = await Promise.allSettled([
        getSystemStatus(actor, role),
        getReviewQueue(actor, role),
        listJobs(actor, role),
      ]);
      if (!active) return;

      if (statusResult.status === 'fulfilled') {
        setSystemStatus(statusResult.value);
        setSystemStatusError(null);
      } else {
        setSystemStatus(null);
        setSystemStatusError(
          statusResult.reason instanceof Error ? statusResult.reason.message : 'Could not load governed backend status.',
        );
      }

      if (queueResult.status === 'fulfilled') {
        setReviewQueue(queueResult.value.items ?? []);
        setReviewQueueError(null);
      } else {
        setReviewQueue([]);
        setReviewQueueError(
          queueResult.reason instanceof Error ? queueResult.reason.message : 'Could not load the governance queue.',
        );
      }

      if (jobsResult.status === 'fulfilled') {
        const jobs = [...(jobsResult.value.jobs ?? [])];
        jobs.sort((left, right) => {
          const leftTime = new Date(left.updated_at ?? left.created_at ?? 0).getTime();
          const rightTime = new Date(right.updated_at ?? right.created_at ?? 0).getTime();
          return rightTime - leftTime;
        });
        setRecentJobs(jobs.slice(0, 6));
        setRecentJobsError(null);
      } else {
        setRecentJobs([]);
        setRecentJobsError(jobsResult.reason instanceof Error ? jobsResult.reason.message : 'Could not load recent backend jobs.');
      }
    }

    void hydrateSystemStatus();
    return () => {
      active = false;
    };
  }, []);

  async function handleSeedDemo() {
    setIsSeedingDemo(true);
    try {
      await addSampleDataset('demo');
      onViewChange('datasets');
    } finally {
      setIsSeedingDemo(false);
    }
  }

  async function handleAnalyzeLatest() {
    if (!latestDataset) return;
    setAnalyzingDatasetId(latestDataset.id);
    try {
      await analyzeDataset(latestDataset.id);
      onViewChange('insights');
    } finally {
      setAnalyzingDatasetId(null);
    }
  }

  function handleOpenGovernedSession(datasetId: string) {
    if (typeof window !== 'undefined' && datasetId && datasetId !== 'documents' && datasetId !== 'system') {
      window.localStorage.setItem(BACKEND_DATASET_STORAGE_KEY, datasetId);
    }
    onViewChange('ai_analytics');
  }

  if (datasets.length === 0) {
    return (
      <div className="space-y-6">
        <section className="relative overflow-hidden rounded-3xl border border-health-mint/30 bg-card p-6 sm:p-8">
          <div className="absolute inset-y-0 right-0 w-1/2 bg-gradient-to-l from-health-mint/10 via-transparent to-transparent" />
          <div className="relative flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
            <div className="max-w-2xl">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline" className="border-health-mint/30 text-health-mint">
                  Governed backend
                </Badge>
                <Badge variant="outline">{aiEnvironment.status}</Badge>
                <Badge variant="outline">{aiEnvironment.mode}</Badge>
              </div>
              <h1 className="mt-4 text-2xl font-bold text-foreground sm:text-3xl">Build your governed analytics workspace</h1>
              <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground sm:text-base">
                Upload a real dataset or seed a demo one to start generating governed facts, explainable dashboard
                specs, inspectable AI insights, and auditable chart narratives.
              </p>
              <div className="mt-5 flex flex-wrap gap-3">
                <Button className="gap-2 gradient-mint text-background hover:opacity-90" onClick={() => onViewChange('upload')}>
                  <FileUp className="h-4 w-4" />
                  Upload Dataset
                </Button>
                <Button variant="outline" className="gap-2" onClick={() => void handleSeedDemo()} disabled={isSeedingDemo}>
                  {isSeedingDemo ? <Loader2 className="h-4 w-4 animate-spin" /> : <Database className="h-4 w-4" />}
                  Load Demo Dataset
                </Button>
                <Button variant="outline" className="gap-2" onClick={() => onViewChange('ai_analytics')}>
                  <Sparkles className="h-4 w-4" />
                  Open AI Analyst
                </Button>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-3 lg:w-[28rem]">
              <Card className="border-border bg-background/80">
                <CardContent className="p-4">
                  <Shield className="h-5 w-5 text-health-mint" />
                  <p className="mt-3 text-sm font-medium text-foreground">Governed reasoning</p>
                  <p className="mt-1 text-xs text-muted-foreground">Approved tools, semantic validation, and explicit evidence paths.</p>
                </CardContent>
              </Card>
              <Card className="border-border bg-background/80">
                <CardContent className="p-4">
                  <Activity className="h-5 w-5 text-blue-400" />
                  <p className="mt-3 text-sm font-medium text-foreground">Inspectable outputs</p>
                  <p className="mt-1 text-xs text-muted-foreground">Every answer can expose logic, chart payload, and result rows.</p>
                </CardContent>
              </Card>
              <Card className="border-border bg-background/80">
                <CardContent className="p-4">
                  <ArrowUpRight className="h-5 w-5 text-amber-400" />
                  <p className="mt-3 text-sm font-medium text-foreground">Stepwise buildout</p>
                  <p className="mt-1 text-xs text-muted-foreground">We are replacing mock surfaces with real governed workflows one slice at a time.</p>
                </CardContent>
              </Card>
            </div>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <section className="relative overflow-hidden rounded-3xl border border-health-mint/30 bg-card p-5 sm:p-6">
        <div className="absolute right-0 top-0 h-64 w-64 rounded-full bg-health-mint/10 blur-3xl" />
        <div className="relative flex flex-col gap-6 xl:flex-row xl:items-start xl:justify-between">
          <div className="max-w-2xl">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="border-health-mint/30 text-health-mint">
                Governed Workspace
              </Badge>
              <Badge variant="outline">{aiEnvironment.status}</Badge>
              <Badge variant="outline">{aiEnvironment.mode}</Badge>
              <Badge variant="outline">{aiEnvironment.version}</Badge>
            </div>
            <h1 className="mt-4 text-2xl font-bold text-foreground sm:text-3xl">
              Your analytics control plane is live
            </h1>
            <p className="mt-3 max-w-xl text-sm leading-6 text-muted-foreground sm:text-base">
              This dashboard now reflects governed workspace state instead of mock maternal-health samples. It
              surfaces the current dataset pipeline, recent AI outputs, and the fastest next actions for analysis.
            </p>

            <div className="mt-5 flex flex-wrap gap-3">
              <Button className="gap-2 gradient-mint text-background hover:opacity-90" onClick={() => onViewChange('upload')}>
                <FileUp className="h-4 w-4" />
                Upload Dataset
              </Button>
              <Button variant="outline" className="gap-2" onClick={() => onViewChange('ai_analytics')}>
                <Sparkles className="h-4 w-4" />
                Open AI Analyst
              </Button>
              <Button
                variant="outline"
                className="gap-2"
                onClick={() => void handleAnalyzeLatest()}
                disabled={!latestDataset || analyzingDatasetId === latestDataset?.id}
              >
                {analyzingDatasetId === latestDataset?.id ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <BarChart3 className="h-4 w-4" />
                )}
                Analyze Latest Dataset
              </Button>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 xl:w-[28rem]">
            <Card className="border-border bg-background/80">
              <CardContent className="p-4">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">Latest dataset</p>
                <p className="mt-2 truncate text-base font-semibold text-foreground">{latestDataset?.name ?? 'None'}</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  <Badge variant="outline">{latestDataset?.rowCount.toLocaleString() ?? 0} rows</Badge>
                  <Badge variant="outline">{latestDataset?.columnCount ?? 0} cols</Badge>
                  <Badge variant="outline" className={latestDataset ? datasetStatusTone(latestDataset) : ''}>
                    {latestDataset?.status ?? 'idle'}
                  </Badge>
                </div>
                <p className="mt-3 text-xs text-muted-foreground">
                  Updated {latestDataset ? formatTimestamp(latestDataset.lastUpdated) : 'not yet'}
                </p>
              </CardContent>
            </Card>

            <Card className="border-border bg-background/80">
              <CardContent className="p-4">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">Latest insight</p>
                <p className="mt-2 text-base font-semibold text-foreground">
                  {latestInsight?.title ?? 'No governed insight yet'}
                </p>
                <p className="mt-2 line-clamp-3 text-sm text-muted-foreground">
                  {latestInsight?.content ?? 'Run analysis on a dataset to generate your first governed insight.'}
                </p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {latestInsight ? (
                    <>
                      <Badge variant="outline" className={insightTone(latestInsight)}>
                        {insightTypeLabel(latestInsight)}
                      </Badge>
                      <Badge variant="outline">{latestInsight.confidence}% confidence</Badge>
                    </>
                  ) : (
                    <Badge variant="outline">Awaiting analysis</Badge>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Datasets"
          value={stats.datasetCount.toLocaleString()}
          icon={Database}
          tone="bg-health-mint/20 text-health-mint"
          hint="Governed sessions available in this workspace."
        />
        <StatCard
          label="Rows Indexed"
          value={formatCompactNumber(stats.rowCount)}
          icon={BarChart3}
          tone="bg-blue-500/20 text-blue-400"
          hint="Combined row volume across active governed datasets."
        />
        <StatCard
          label="Average Quality"
          value={`${stats.averageQuality}%`}
          icon={CheckCircle2}
          tone="bg-emerald-500/20 text-emerald-400"
          hint="Mean score across active datasets with backend profiling."
        />
        <StatCard
          label="AI Insights"
          value={stats.insightCount.toLocaleString()}
          icon={Sparkles}
          tone="bg-amber-500/20 text-amber-400"
          hint={`${stats.flaggedInsights} flagged, ${stats.attentionCount} datasets needing attention.`}
        />
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Governed Backend Status</CardTitle>
            <p className="text-xs text-muted-foreground">
              Role-aware health, queue, audit, and governance signals from the live FastAPI control plane.
            </p>
          </CardHeader>
          <CardContent className="space-y-3">
            {systemStatus ? (
              <>
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline" className="border-health-mint/30 text-health-mint">
                    {systemStatus.status}
                  </Badge>
                  <Badge variant="outline">{systemStatus.role}</Badge>
                  <Badge variant="outline">Actor {systemStatus.actor}</Badge>
                  <Badge variant="outline">Updated {formatTimestamp(systemStatus.timestamp)}</Badge>
                </div>
                <div className="space-y-2">
                  {systemStatus.alerts.map((alert, index) => (
                    <div key={`${alert.level}-${index}`} className={cn('rounded-2xl border px-4 py-3 text-sm', statusAlertTone(alert.level))}>
                      {alert.message}
                    </div>
                  ))}
                </div>
              </>
            ) : (
              <div className="rounded-2xl border border-border bg-background p-4 text-sm text-muted-foreground">
                {systemStatusError ?? 'Loading governed backend status...'}
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Control Plane Counters</CardTitle>
            <p className="text-xs text-muted-foreground">Live counts visible to the current backend role scope.</p>
          </CardHeader>
          <CardContent>
            {systemStatus ? (
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="rounded-2xl border border-border bg-background p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Visible Sessions</p>
                  <p className="mt-2 text-2xl font-bold text-foreground">{systemStatus.counts.visible_sessions}</p>
                </div>
                <div className="rounded-2xl border border-border bg-background p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Trusted Docs</p>
                  <p className="mt-2 text-2xl font-bold text-foreground">{systemStatus.counts.visible_documents}</p>
                </div>
                <div className="rounded-2xl border border-border bg-background p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Jobs</p>
                  <p className="mt-2 text-2xl font-bold text-foreground">
                    {systemStatus.counts.active_jobs + systemStatus.counts.queued_jobs}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {systemStatus.counts.active_jobs} active · {systemStatus.counts.queued_jobs} queued
                  </p>
                </div>
                <div className="rounded-2xl border border-border bg-background p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Review Queue</p>
                  <p className="mt-2 text-2xl font-bold text-foreground">
                    {systemStatus.counts.pending_sensitive_exports + systemStatus.counts.pending_workflow_reviews}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {systemStatus.counts.pending_sensitive_exports} export · {systemStatus.counts.pending_workflow_reviews} workflow
                  </p>
                </div>
                <div className="rounded-2xl border border-border bg-background p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Models</p>
                  <p className="mt-2 text-2xl font-bold text-foreground">{systemStatus.counts.active_models}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{systemStatus.counts.stale_models} stale</p>
                </div>
                <div className="rounded-2xl border border-border bg-background p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">Recent Audit Events</p>
                  <p className="mt-2 text-2xl font-bold text-foreground">{systemStatus.counts.recent_audit_events_24h}</p>
                  <p className="mt-1 text-xs text-muted-foreground">Last 24 hours</p>
                </div>
              </div>
            ) : (
              <div className="rounded-2xl border border-border bg-background p-4 text-sm text-muted-foreground">
                Waiting for live backend counters...
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <Card className="glass-card">
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">Governance Queue</CardTitle>
              <p className="text-xs text-muted-foreground">
                Actionable approvals, stale assets, failed jobs, and schedules surfaced from the governed backend.
              </p>
            </div>
            <Badge variant="outline">{reviewQueue.length} visible</Badge>
          </CardHeader>
          <CardContent className="space-y-3">
            {reviewQueue.length > 0 ? (
              reviewQueue.slice(0, 6).map((item) => (
                <div key={item.item_id} className={cn('rounded-2xl border p-4', reviewItemTone(item.severity))}>
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge variant="outline">{reviewCategoryLabel(item.category)}</Badge>
                        <Badge variant="outline">{item.status.replace(/_/g, ' ')}</Badge>
                        <Badge variant="outline">{item.dataset_label}</Badge>
                      </div>
                      <p className="mt-3 text-sm font-semibold text-foreground">{item.title}</p>
                      <p className="mt-2 text-sm text-muted-foreground">{item.summary}</p>
                      {item.action_hint ? <p className="mt-2 text-xs text-muted-foreground">{item.action_hint}</p> : null}
                      <p className="mt-3 text-xs text-muted-foreground">Updated {formatTimestamp(item.updated_at)}</p>
                    </div>
                    <Button variant="outline" size="sm" className="gap-2" onClick={() => handleOpenGovernedSession(item.dataset_id)}>
                      Open AI Analyst
                      <ArrowUpRight className="h-3 w-3" />
                    </Button>
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-2xl border border-border bg-background p-4 text-sm text-muted-foreground">
                {reviewQueueError ?? 'No governed review items are currently waiting in your visible scope.'}
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Recent Backend Jobs</CardTitle>
            <p className="text-xs text-muted-foreground">Latest governed jobs visible to the current backend role scope.</p>
          </CardHeader>
          <CardContent className="space-y-3">
            {recentJobs.length > 0 ? (
              recentJobs.map((job) => (
                <div key={job.job_id} className={cn('rounded-2xl border p-4', jobStatusTone(job.status))}>
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="outline">{job.type.replace(/_/g, ' ')}</Badge>
                    <Badge variant="outline">{job.status}</Badge>
                    <Badge variant="outline">{job.progress}%</Badge>
                  </div>
                  <p className="mt-3 break-all text-sm font-medium text-foreground">{job.dataset_id}</p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Updated {formatTimestamp(job.updated_at ?? job.created_at ?? new Date().toISOString())}
                  </p>
                  {job.error?.message ? <p className="mt-2 text-xs text-muted-foreground">{job.error.message}</p> : null}
                </div>
              ))
            ) : (
              <div className="rounded-2xl border border-border bg-background p-4 text-sm text-muted-foreground">
                {recentJobsError ?? 'No recent governed jobs are visible yet.'}
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.7fr_1fr]">
        <Card className="glass-card">
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">Dataset Quality Trend</CardTitle>
              <p className="text-xs text-muted-foreground">Recent governed sessions by quality score and surfaced issue count.</p>
            </div>
            <Button variant="outline" size="sm" className="gap-2" onClick={() => onViewChange('datasets')}>
              View Datasets
              <ArrowUpRight className="h-3 w-3" />
            </Button>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={qualityTrend}>
                  <defs>
                    <linearGradient id="qualityFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 15% 18%)" />
                  <XAxis dataKey="name" stroke="hsl(220 10% 60%)" fontSize={12} tickLine={false} />
                  <YAxis stroke="hsl(220 10% 60%)" fontSize={12} tickLine={false} domain={[0, 100]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(220 18% 8%)',
                      border: '1px solid hsl(220 15% 18%)',
                      borderRadius: '12px',
                    }}
                  />
                  <Area type="monotone" dataKey="quality" stroke="#10b981" fill="url(#qualityFill)" strokeWidth={2.5} />
                  <Bar dataKey="issues" fill="#f59e0b" radius={[6, 6, 0, 0]} maxBarSize={18} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Source Mix</CardTitle>
            <p className="text-xs text-muted-foreground">How the current workspace is composed.</p>
          </CardHeader>
          <CardContent>
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie data={sourceMix} dataKey="value" nameKey="name" innerRadius={52} outerRadius={78} paddingAngle={4}>
                    {sourceMix.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(220 18% 8%)',
                      border: '1px solid hsl(220 15% 18%)',
                      borderRadius: '12px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className="mt-4 grid gap-2">
              {sourceMix.map((entry) => (
                <div key={entry.name} className="flex items-center justify-between rounded-xl border border-border bg-background px-3 py-2">
                  <div className="flex items-center gap-2">
                    <div className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: entry.color }} />
                    <span className="text-sm capitalize text-foreground">{entry.name}</span>
                  </div>
                  <span className="text-sm text-muted-foreground">{entry.value}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <Card className="glass-card">
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">Dataset Scale</CardTitle>
              <p className="text-xs text-muted-foreground">Recent sessions compared by row and column footprint.</p>
            </div>
            <Badge variant="outline">{sortedDatasets.length} governed sessions</Badge>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={scaleTrend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 15% 18%)" vertical={false} />
                  <XAxis dataKey="name" stroke="hsl(220 10% 60%)" fontSize={12} tickLine={false} />
                  <YAxis stroke="hsl(220 10% 60%)" fontSize={12} tickLine={false} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'hsl(220 18% 8%)',
                      border: '1px solid hsl(220 15% 18%)',
                      borderRadius: '12px',
                    }}
                  />
                  <Bar dataKey="rows" fill="#3b82f6" radius={[6, 6, 0, 0]} maxBarSize={30} />
                  <Bar dataKey="columns" fill="#10b981" radius={[6, 6, 0, 0]} maxBarSize={30} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader>
            <CardTitle className="text-lg font-semibold">Governance Watchlist</CardTitle>
            <p className="text-xs text-muted-foreground">Quality and control-plane issues surfaced by the backend.</p>
          </CardHeader>
          <CardContent className="space-y-3">
            {datasetWatchlist.length === 0 ? (
              <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-4 text-sm text-muted-foreground">
                No active watchlist items. Current datasets are not surfacing major quality issues.
              </div>
            ) : (
              datasetWatchlist.map((item) => (
                <div key={`${item.datasetId}-${item.issue}`} className="rounded-2xl border border-border bg-background p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="outline" className={item.qualityScore >= 80 ? 'border-emerald-500/30 text-emerald-400' : 'border-amber-500/30 text-amber-400'}>
                      Quality {item.qualityScore}%
                    </Badge>
                    <span className="text-sm font-medium text-foreground">{item.datasetName}</span>
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground">{item.issue}</p>
                </div>
              ))
            )}

            <div className="rounded-2xl border border-health-mint/30 bg-health-mint/5 p-4">
              <div className="flex items-start gap-3">
                <Shield className="mt-0.5 h-5 w-5 text-health-mint" />
                <div>
                  <p className="text-sm font-medium text-foreground">Trust layer status</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Answers now flow through governed sessions, inspectable insights, and approved ask-data execution.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
        <Card className="glass-card">
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">Recent Datasets</CardTitle>
              <p className="text-xs text-muted-foreground">Latest governed sessions with quality and issue signals.</p>
            </div>
            <Button variant="outline" size="sm" className="gap-2" onClick={() => onViewChange('datasets')}>
              Open Datasets
              <ArrowUpRight className="h-3 w-3" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-3">
            {sortedDatasets.slice(0, 5).map((dataset) => (
              <div key={dataset.id} className="rounded-2xl border border-border bg-background p-4">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <p className="truncate text-sm font-medium text-foreground">{dataset.name}</p>
                      <Badge variant="outline" className={datasetStatusTone(dataset)}>
                        {dataset.status}
                      </Badge>
                      <Badge variant="outline">{dataset.type.toUpperCase()}</Badge>
                    </div>
                    <p className="mt-1 text-sm text-muted-foreground">{dataset.description}</p>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Badge variant="outline">{dataset.rowCount.toLocaleString()} rows</Badge>
                      <Badge variant="outline">{dataset.columnCount} columns</Badge>
                      <Badge variant="outline">Quality {dataset.qualityScore}%</Badge>
                      <Badge variant="outline" className="capitalize">
                        {dataset.source}
                      </Badge>
                    </div>
                  </div>
                  <div className="text-xs text-muted-foreground">Updated {formatTimestamp(dataset.lastUpdated)}</div>
                </div>

                {dataset.issues.length > 0 ? (
                  <div className="mt-3 rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-muted-foreground">
                    {dataset.issues[0]}
                  </div>
                ) : null}
              </div>
            ))}
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardHeader className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">Recent Insights</CardTitle>
              <p className="text-xs text-muted-foreground">Latest governed narratives produced in this workspace.</p>
            </div>
            <Button variant="outline" size="sm" className="gap-2" onClick={() => onViewChange('insights')}>
              Open Insights
              <ArrowUpRight className="h-3 w-3" />
            </Button>
          </CardHeader>
          <CardContent className="space-y-3">
            {sortedInsights.length === 0 ? (
              <div className="rounded-2xl border border-border bg-background p-5 text-sm text-muted-foreground">
                No governed insights yet. Run analysis on the latest dataset to generate explainable findings.
              </div>
            ) : (
              sortedInsights.slice(0, 4).map((insight) => (
                <div key={insight.id} className="rounded-2xl border border-border bg-background p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="outline" className={insightTone(insight)}>
                      {insightTypeLabel(insight)}
                    </Badge>
                    <Badge variant="outline">{insight.confidence}% confidence</Badge>
                    {insight.flagged ? (
                      <Badge variant="outline" className="border-amber-500/30 text-amber-400">
                        <Flag className="mr-1 h-3 w-3" />
                        Flagged
                      </Badge>
                    ) : null}
                    {insight.verified ? (
                      <Badge variant="outline" className="border-emerald-500/30 text-emerald-400">
                        <CheckCircle2 className="mr-1 h-3 w-3" />
                        Verified
                      </Badge>
                    ) : null}
                  </div>
                  <p className="mt-3 text-sm font-medium text-foreground">{insight.title}</p>
                  <p className="mt-1 line-clamp-3 text-sm text-muted-foreground">{insight.content}</p>
                  <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="outline">{insight.citations.length} citations</Badge>
                      {insight.inspection?.factsUsed?.length ? (
                        <Badge variant="outline">{insight.inspection.factsUsed.length} facts used</Badge>
                      ) : null}
                    </div>
                    <span className="text-xs text-muted-foreground">{formatTimestamp(insight.timestamp)}</span>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </section>

      <section className="rounded-2xl border border-border bg-card p-4 sm:p-5">
        <div className="flex items-start gap-3">
          <AlertCircle className="mt-0.5 h-5 w-5 text-health-mint" />
          <div className="space-y-2">
            <h3 className="font-medium text-foreground">Current Phase</h3>
            <p className="text-sm text-muted-foreground">
              The dashboard route is now governed workspace-aware. It surfaces live dataset and insight state instead
              of mock health metrics, while deeper chart reasoning and audit details remain in the dedicated AI
              analytics surface.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

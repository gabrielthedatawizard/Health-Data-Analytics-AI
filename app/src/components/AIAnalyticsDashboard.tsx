import { useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  BarChart3,
  BadgeCheck,
  Bot,
  Database,
  FileDown,
  FileText,
  FileSpreadsheet,
  Loader2,
  Plus,
  RefreshCw,
  Shield,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Upload,
  X,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import {
  API_TARGET_LABEL,
  type AnomalyAnalysisResponse,
  BACKEND_ROLE_STORAGE_KEY,
  BACKEND_USER_STORAGE_KEY,
  type AskResponsePayload,
  type DocumentAskResponse,
  type DocumentSummary,
  type CohortAnalysisResponse,
  type ModelEvaluationResponse,
  type ModelRegistryEntry,
  type ForecastDriftResponse,
  type ForecastRunRecord,
  type SavedInvestigationRecord,
  type SavedPlaybookRecord,
  type WorkflowActionRecord,
  type AuthContextResponse,
  type BackendUserRole,
  type AuditEvent,
  type FeedbackRecord,
  type JobStatus,
  type SessionMeta,
  askDocuments,
  askDataset,
  buildCohortAnalysis,
  createSession,
  datasetCsvUrl,
  draftWorkflowAction,
  executeWorkflowAction,
  factsJsonUrl,
  generateDashboardSpec,
  generateReport,
  getAudit,
  getAnomalyAnalysis,
  getAuthContext,
  getCohortAnalysis,
  getDashboardSpec,
  listFeedback,
  getFacts,
  getForecastDrift,
  getForecastRuns,
  getModelEvaluation,
  getModelRegistry,
  getJobStatus,
  getProfile,
  getSavedInvestigations,
  getSavedPlaybooks,
  getSession,
  getWorkflowActions,
  inferBackendUserRole,
  listDocuments,
  requestSensitiveExportApproval,
  reportHtmlUrl,
  reportPdfUrl,
  saveInvestigation,
  savePlaybook,
  reviewWorkflowAction,
  promoteModelRun,
  reviewSensitiveExportApproval,
  setSensitiveExportEnabled,
  submitFeedback,
  trainForecastRun,
  uploadDocument,
  uploadDataset,
} from '@/lib/backend-api';

const DATASET_STORAGE_KEY = 'healthai_backend_dataset_id';
const USER_STORAGE_KEY = BACKEND_USER_STORAGE_KEY;
const ROLE_STORAGE_KEY = BACKEND_ROLE_STORAGE_KEY;
const DATASET_ID_PATTERN = /^[a-f0-9-]{36}$/i;
const ROLE_FALLBACK_PERMISSIONS: Record<BackendUserRole, string[]> = {
  viewer: ['sessions:read_own', 'sessions:export_masked_own', 'docs:read_own'],
  analyst: [
    'sessions:create',
    'sessions:read_own',
    'sessions:write_own',
    'sessions:compute_own',
    'sessions:export_own',
    'sensitive_export:request_own',
    'docs:create',
    'docs:read_own',
  ],
  reviewer: [
    'sessions:create',
    'sessions:read_own',
    'sessions:write_own',
    'sessions:compute_own',
    'sessions:export_own',
    'sessions:read_all',
    'sessions:export_all',
    'sensitive_export:request_own',
    'sensitive_export:review',
    'ml:promote',
    'docs:create',
    'docs:read_own',
    'docs:read_all',
  ],
  admin: [
    'sessions:create',
    'sessions:read_all',
    'sessions:write_all',
    'sessions:compute_all',
    'sessions:export_all',
    'sensitive_export:review',
    'ml:promote',
    'docs:create',
    'docs:read_all',
    'admin:all',
  ],
};

function isDatasetId(value: string): boolean {
  return DATASET_ID_PATTERN.test(value.trim());
}

function getStoredDatasetId(): string {
  if (typeof window === 'undefined') {
    return '';
  }
  const storedValue = localStorage.getItem(DATASET_STORAGE_KEY)?.trim() ?? '';
  return isDatasetId(storedValue) ? storedValue : '';
}

function fallbackPermissionsForRole(role: BackendUserRole): string[] {
  return ROLE_FALLBACK_PERMISSIONS[role] ?? ROLE_FALLBACK_PERMISSIONS.analyst;
}

function asNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function asString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

type DashboardChartSpec = {
  key: string;
  title: string;
  type: string;
  x: string | null;
  y: string | null;
  groupBy: string | null;
  aggregation: string | null;
  factIds: string[];
  layout: Record<string, unknown> | null;
};

type ExplainPreset = 'summary' | 'significance' | 'contributors';
type CohortDraftCriterion = {
  id: string;
  field: string;
  operator: string;
  value: string;
};
type CohortRequestCriterion = {
  field: string;
  operator: string;
  value?: string | string[];
};
type CohortFieldOption = {
  name: string;
  inferredType: string;
  sampleValues: string[];
};

type ChartExplanationEntry = {
  question: string;
  preset: ExplainPreset;
  result: AskResponsePayload;
  generatedAt: string;
};

const EXPLAIN_PRESETS: Array<{
  id: ExplainPreset;
  label: string;
  instruction: string;
}> = [
  {
    id: 'summary',
    label: 'Explain',
    instruction: 'Explain the main pattern, likely drivers, and any caveats in plain English.',
  },
  {
    id: 'significance',
    label: 'Check Signal',
    instruction: 'Assess whether the visible change looks materially meaningful and call out what should be validated next.',
  },
  {
    id: 'contributors',
    label: 'Top Drivers',
    instruction: 'Identify which segment, category, or time slice appears to contribute most and what follow-up cut to inspect next.',
  },
];
const COHORT_OPERATOR_OPTIONS: Record<string, Array<{ value: string; label: string }>> = {
  string: [
    { value: 'eq', label: 'equals' },
    { value: 'neq', label: 'does not equal' },
    { value: 'contains', label: 'contains' },
    { value: 'starts_with', label: 'starts with' },
    { value: 'in', label: 'is one of' },
    { value: 'not_in', label: 'is not one of' },
    { value: 'is_null', label: 'is blank' },
    { value: 'not_null', label: 'is present' },
  ],
  number: [
    { value: 'eq', label: 'equals' },
    { value: 'neq', label: 'does not equal' },
    { value: 'gt', label: 'greater than' },
    { value: 'gte', label: 'at least' },
    { value: 'lt', label: 'less than' },
    { value: 'lte', label: 'at most' },
    { value: 'between', label: 'between' },
    { value: 'in', label: 'is one of' },
    { value: 'not_in', label: 'is not one of' },
    { value: 'is_null', label: 'is blank' },
    { value: 'not_null', label: 'is present' },
  ],
  datetime: [
    { value: 'eq', label: 'on' },
    { value: 'neq', label: 'not on' },
    { value: 'gt', label: 'after' },
    { value: 'gte', label: 'on or after' },
    { value: 'lt', label: 'before' },
    { value: 'lte', label: 'on or before' },
    { value: 'between', label: 'between' },
    { value: 'in', label: 'is one of' },
    { value: 'not_in', label: 'is not one of' },
    { value: 'is_null', label: 'is blank' },
    { value: 'not_null', label: 'is present' },
  ],
};
const WORKFLOW_ACTION_OPTIONS = [
  { value: 'draft_email', label: 'Draft Email' },
  { value: 'create_ticket', label: 'Create Ticket' },
  { value: 'action_plan', label: 'Action Plan' },
  { value: 'schedule_report', label: 'Schedule Report' },
] as const;

function cohortOperatorsForType(inferredType: string) {
  return COHORT_OPERATOR_OPTIONS[inferredType] ?? COHORT_OPERATOR_OPTIONS.string;
}

function cohortOperatorNeedsValue(operator: string) {
  return operator !== 'is_null' && operator !== 'not_null';
}

function createDraftCriterion(fieldOptions: CohortFieldOption[]): CohortDraftCriterion {
  const fallbackField = fieldOptions[0];
  const inferredType = fallbackField?.inferredType ?? 'string';
  return {
    id: `criterion-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
    field: fallbackField?.name ?? '',
    operator: cohortOperatorsForType(inferredType)[0]?.value ?? 'eq',
    value: '',
  };
}

function formatCohortCriterionValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((item) => String(item)).join(', ');
  }
  if (value === null || value === undefined) {
    return '';
  }
  return String(value);
}

function normalizeCohortFieldOptions(profile: Record<string, unknown> | null): CohortFieldOption[] {
  const columns = Array.isArray(profile?.columns) ? profile.columns : [];
  return columns
    .map((entry) => {
      const column = asRecord(entry);
      const name = asString(column?.name);
      if (!name || column?.is_pii_candidate || column?.is_id_like) {
        return null;
      }
      const inferredType = asString(column?.inferred_type) ?? 'string';
      const sampleValues = Array.isArray(column?.sample_values)
        ? column.sample_values.map((item) => String(item)).filter(Boolean).slice(0, 4)
        : [];
      return {
        name,
        inferredType,
        sampleValues,
      };
    })
    .filter((item): item is CohortFieldOption => Boolean(item));
}

function summarizeFactReference(fact: unknown): string | null {
  const item = asRecord(fact);
  if (!item) return null;

  const value = asRecord(item.value);
  const metric =
    asString(value?.metric) ??
    asString(value?.name) ??
    asString(item.metric) ??
    asString(item.type) ??
    'governed fact';
  const summary =
    asString(value?.narrative) ??
    asString(value?.summary) ??
    asString(value?.label) ??
    asString(item.summary);

  return summary ? `${metric}: ${summary}` : metric;
}

function formatDocumentFreshnessLabel(value: string | null | undefined): string {
  if (!value) return 'Current';
  return value
    .split('_')
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(' ');
}

function feedbackKey(surface: string, targetId: string): string {
  return `${surface}:${targetId}`;
}

function normalizeDashboardCharts(spec: Record<string, unknown> | null): DashboardChartSpec[] {
  const items = Array.isArray(spec?.charts) ? spec.charts : [];
  return items
    .map((item, index) => {
      const chart = asRecord(item);
      if (!chart) return null;
      return {
        key: `${index}-${asString(chart.title) ?? 'chart'}`,
        title: asString(chart.title) ?? `Chart ${index + 1}`,
        type: asString(chart.type) ?? 'table',
        x: asString(chart.x),
        y: asString(chart.y),
        groupBy: asString(chart.group_by),
        aggregation: asString(chart.aggregation),
        factIds: Array.isArray(chart.fact_ids)
          ? chart.fact_ids.map((value) => asString(value)).filter((value): value is string => Boolean(value))
          : [],
        layout: asRecord(chart.layout),
      };
    })
    .filter((chart): chart is DashboardChartSpec => Boolean(chart));
}

function buildChartExplanationQuestion(
  chart: DashboardChartSpec,
  preset: ExplainPreset,
  linkedFacts: string[],
  dashboardTitle?: string | null
): string {
  const presetConfig = EXPLAIN_PRESETS.find((candidate) => candidate.id === preset) ?? EXPLAIN_PRESETS[0];
  const parts = [
    dashboardTitle ? `Dashboard title: ${dashboardTitle}.` : null,
    `Explain the governed dashboard chart titled "${chart.title}".`,
    `Chart type: ${chart.type}.`,
    chart.x ? `X axis: ${chart.x}.` : null,
    chart.y ? `Y axis: ${chart.y}.` : null,
    chart.groupBy ? `Grouped by: ${chart.groupBy}.` : null,
    chart.aggregation ? `Aggregation: ${chart.aggregation}.` : null,
    linkedFacts.length ? `Linked governed facts: ${linkedFacts.join(' | ')}.` : null,
    presetConfig.instruction,
    'Use only approved dataset evidence, explain uncertainty clearly, and surface any metric or coverage caveats.',
  ];

  return parts.filter(Boolean).join(' ');
}

function buildDashboardExplanationQuestion(
  dashboardTitle: string | null,
  charts: DashboardChartSpec[],
  filters: Array<{ field: string; type: string }>
): string {
  const chartSummary = charts
    .slice(0, 4)
    .map((chart) => {
      const parts = [
        `${chart.title} (${chart.type})`,
        chart.x ? `x=${chart.x}` : null,
        chart.y ? `y=${chart.y}` : null,
        chart.groupBy ? `group_by=${chart.groupBy}` : null,
      ];
      return parts.filter(Boolean).join(', ');
    })
    .join(' | ');
  const filterSummary = filters.map((filter) => `${filter.field}:${filter.type}`).join(', ');

  return [
    dashboardTitle ? `Summarize the governed dashboard "${dashboardTitle}".` : 'Summarize this governed dashboard.',
    chartSummary ? `Chart layout: ${chartSummary}.` : null,
    filterSummary ? `Available filters: ${filterSummary}.` : null,
    'Highlight the most important patterns, caveats, and next investigation steps for an analyst or quality committee.',
    'Use only governed facts and semantic query logic.',
  ]
    .filter(Boolean)
    .join(' ');
}

function ResultRowsTable({
  rows,
  emptyMessage,
}: {
  rows: Array<Record<string, unknown>>;
  emptyMessage: string;
}) {
  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">{emptyMessage}</p>;
  }

  return (
    <div className="overflow-auto rounded-xl border border-border">
      <table className="min-w-full divide-y divide-border text-sm">
        <thead className="bg-muted/50">
          <tr>
            {Object.keys(rows[0] ?? {}).map((column) => (
              <th key={column} className="whitespace-nowrap px-3 py-2 text-left font-medium text-foreground">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 20).map((row, index) => (
            <tr key={index} className="border-t border-border">
              {Object.entries(row).map(([column, value]) => (
                <td key={`${index}-${column}`} className="whitespace-nowrap px-3 py-2 text-muted-foreground">
                  {String(value)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AskResponseInspector({
  result,
  title,
  emptyRowsMessage,
}: {
  result: AskResponsePayload;
  title: string;
  emptyRowsMessage: string;
}) {
  const queryRows = result.result_rows ?? [];

  return (
    <div className="space-y-4">
      <div className="rounded-xl border border-border bg-background p-4">
        <p className="text-sm text-foreground">{result.answer}</p>
        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant="outline">Confidence: {result.confidence}</Badge>
          <Badge variant="outline">Fact coverage: {result.fact_coverage}</Badge>
          <Badge variant="outline">Data coverage: {result.data_coverage}</Badge>
          <Badge variant="outline">Facts used: {result.facts_used.length}</Badge>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        <div>
          <p className="mb-2 text-sm font-medium text-foreground">Governance</p>
          <pre className="overflow-auto rounded-xl border border-border bg-background p-3 text-xs text-muted-foreground">
            {JSON.stringify(result.governance, null, 2)}
          </pre>
        </div>
        <div>
          <p className="mb-2 text-sm font-medium text-foreground">Query Plan</p>
          <pre className="overflow-auto rounded-xl border border-border bg-background p-3 text-xs text-muted-foreground">
            {JSON.stringify(result.query_plan ?? {}, null, 2)}
          </pre>
        </div>
        <div>
          <p className="mb-2 text-sm font-medium text-foreground">Chart Payload</p>
          <pre className="overflow-auto rounded-xl border border-border bg-background p-3 text-xs text-muted-foreground">
            {JSON.stringify(result.chart ?? {}, null, 2)}
          </pre>
        </div>
      </div>

      <div>
        <p className="mb-2 text-sm font-medium text-foreground">{title}</p>
        <ResultRowsTable rows={queryRows} emptyMessage={emptyRowsMessage} />
      </div>
    </div>
  );
}

export function AIAnalyticsDashboard() {
  const [userId, setUserId] = useState(() => localStorage.getItem(USER_STORAGE_KEY) || 'react_user');
  const [userRole, setUserRole] = useState<BackendUserRole>(() => {
    const storedRole = localStorage.getItem(ROLE_STORAGE_KEY);
    if (storedRole === 'viewer' || storedRole === 'analyst' || storedRole === 'reviewer' || storedRole === 'admin') {
      return storedRole;
    }
    return inferBackendUserRole(localStorage.getItem(USER_STORAGE_KEY) || 'react_user');
  });
  const [datasetId, setDatasetId] = useState(() => getStoredDatasetId());
  const [loadDatasetId, setLoadDatasetId] = useState(() => getStoredDatasetId());
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedDocumentFile, setSelectedDocumentFile] = useState<File | null>(null);
  const [documentTitle, setDocumentTitle] = useState('');
  const [documentSourceName, setDocumentSourceName] = useState('');
  const [documentVersionLabel, setDocumentVersionLabel] = useState('');
  const [documentEffectiveDate, setDocumentEffectiveDate] = useState('');
  const [documentSupersedesId, setDocumentSupersedesId] = useState('');
  const [authContext, setAuthContext] = useState<AuthContextResponse | null>(null);
  const [usingFallbackAuthContext, setUsingFallbackAuthContext] = useState(false);
  const [sessionMeta, setSessionMeta] = useState<SessionMeta | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [profile, setProfile] = useState<Record<string, unknown> | null>(null);
  const [factsBundle, setFactsBundle] = useState<Record<string, unknown> | null>(null);
  const [anomalyAnalysis, setAnomalyAnalysis] = useState<AnomalyAnalysisResponse | null>(null);
  const [cohortAnalysis, setCohortAnalysis] = useState<CohortAnalysisResponse | null>(null);
  const [forecastRuns, setForecastRuns] = useState<ForecastRunRecord[]>([]);
  const [forecastDrift, setForecastDrift] = useState<ForecastDriftResponse | null>(null);
  const [modelEvaluation, setModelEvaluation] = useState<ModelEvaluationResponse | null>(null);
  const [modelRegistryEntries, setModelRegistryEntries] = useState<ModelRegistryEntry[]>([]);
  const [savedInvestigations, setSavedInvestigations] = useState<SavedInvestigationRecord[]>([]);
  const [savedPlaybooks, setSavedPlaybooks] = useState<SavedPlaybookRecord[]>([]);
  const [workflowActions, setWorkflowActions] = useState<WorkflowActionRecord[]>([]);
  const [dashboardSpec, setDashboardSpec] = useState<Record<string, unknown> | null>(null);
  const [askQuestion, setAskQuestion] = useState('Show the main metric trend over time');
  const [investigationTitle, setInvestigationTitle] = useState('');
  const [investigationNote, setInvestigationNote] = useState('');
  const [playbookName, setPlaybookName] = useState('');
  const [playbookDescription, setPlaybookDescription] = useState('');
  const [cohortName, setCohortName] = useState('Priority follow-up cohort');
  const [cohortDescription, setCohortDescription] = useState('');
  const [cohortCriteria, setCohortCriteria] = useState<CohortDraftCriterion[]>([{ id: 'criterion-initial', field: '', operator: 'eq', value: '' }]);
  const [forecastName, setForecastName] = useState('');
  const [forecastTimeField, setForecastTimeField] = useState('');
  const [forecastMetricField, setForecastMetricField] = useState('');
  const [forecastAggregation, setForecastAggregation] = useState<'sum' | 'mean'>('sum');
  const [forecastHorizon, setForecastHorizon] = useState('3');
  const [modelPromotionNote, setModelPromotionNote] = useState('');
  const [workflowActionType, setWorkflowActionType] = useState<(typeof WORKFLOW_ACTION_OPTIONS)[number]['value']>('draft_email');
  const [workflowTitle, setWorkflowTitle] = useState('');
  const [workflowTarget, setWorkflowTarget] = useState('');
  const [workflowObjective, setWorkflowObjective] = useState('');
  const [workflowDecisionNote, setWorkflowDecisionNote] = useState('');
  const [askResult, setAskResult] = useState<AskResponsePayload | null>(null);
  const [documentQuestion, setDocumentQuestion] = useState('What do our trusted policy documents say about denominator exclusions?');
  const [documentAnswer, setDocumentAnswer] = useState<DocumentAskResponse | null>(null);
  const [feedbackRecords, setFeedbackRecords] = useState<FeedbackRecord[]>([]);
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([]);
  const [factsJob, setFactsJob] = useState<JobStatus | null>(null);
  const [reportJob, setReportJob] = useState<JobStatus | null>(null);
  const [dashboardExplanation, setDashboardExplanation] = useState<ChartExplanationEntry | null>(null);
  const [chartExplanations, setChartExplanations] = useState<Record<string, ChartExplanationEntry>>({});
  const [selectedChartKey, setSelectedChartKey] = useState<string | null>(null);
  const [exportApprovalInput, setExportApprovalInput] = useState('');
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyAction, setBusyAction] = useState<string | null>(null);

  useEffect(() => {
    localStorage.setItem(USER_STORAGE_KEY, userId);
  }, [userId]);

  useEffect(() => {
    localStorage.setItem(ROLE_STORAGE_KEY, userRole);
  }, [userRole]);

  useEffect(() => {
    if (datasetId) {
      localStorage.setItem(DATASET_STORAGE_KEY, datasetId);
      setLoadDatasetId(datasetId);
      return;
    }
    localStorage.removeItem(DATASET_STORAGE_KEY);
  }, [datasetId]);

  useEffect(() => {
    let active = true;

    async function resolveAuthContext() {
      try {
        const context = await getAuthContext(userId, userRole);
        if (!active) return;
        setAuthContext(context);
        setUsingFallbackAuthContext(false);
      } catch {
        if (!active) return;
        setAuthContext({
          actor: userId.trim() || 'anonymous',
          role: userRole,
          permissions: fallbackPermissionsForRole(userRole),
        });
        setUsingFallbackAuthContext(true);
      }
    }

    void resolveAuthContext();

    return () => {
      active = false;
    };
  }, [userId, userRole]);

  useEffect(() => {
    let active = true;

    async function hydrateDocuments() {
      try {
        const response = await listDocuments(userId);
        if (!active) return;
        setDocuments(response.documents ?? []);
      } catch {
        if (!active) return;
        setDocuments([]);
      }
    }

    void hydrateDocuments();

    return () => {
      active = false;
    };
  }, [userId, userRole]);

  useEffect(() => {
    if (!datasetId.trim()) {
      setSessionMeta(null);
      setProfile(null);
      setFactsBundle(null);
      setAnomalyAnalysis(null);
      setCohortAnalysis(null);
      setForecastRuns([]);
      setForecastDrift(null);
      setModelEvaluation(null);
      setModelRegistryEntries([]);
      setSavedInvestigations([]);
      setSavedPlaybooks([]);
      setWorkflowActions([]);
      setFeedbackRecords([]);
      setDashboardSpec(null);
      setAskResult(null);
      setDashboardExplanation(null);
      setChartExplanations({});
      setSelectedChartKey(null);
      setExportApprovalInput('');
      setAuditEvents([]);
      return;
    }

    let active = true;

    async function hydrateSession() {
      try {
        setForecastRuns([]);
        setForecastDrift(null);
        setModelEvaluation(null);
        setModelRegistryEntries([]);
        setSavedInvestigations([]);
        setSavedPlaybooks([]);
        setWorkflowActions([]);
        setFeedbackRecords([]);
        const meta = await getSession(datasetId, userId);
        if (!active) return;
        setSessionMeta(meta);

        if (meta.artifacts?.profile) {
          const profileResponse = await getProfile(datasetId, userId);
          if (!active) return;
          setProfile(profileResponse.profile);
        }

        if (meta.artifacts?.facts) {
          const factsResponse = await getFacts(datasetId, userId);
          if (!active) return;
          setFactsBundle(factsResponse.facts_bundle ?? null);
        }

        if (meta.artifacts?.anomaly_analysis) {
          const anomalyResponse = await getAnomalyAnalysis(datasetId, userId);
          if (!active) return;
          setAnomalyAnalysis(anomalyResponse);
        }

        if (meta.artifacts?.cohort_analysis) {
          const cohortResponse = await getCohortAnalysis(datasetId, userId);
          if (!active) return;
          setCohortAnalysis(cohortResponse);
        }

        if (meta.artifacts?.ml_runs) {
          const mlRunsResponse = await getForecastRuns(datasetId, userId);
          if (!active) return;
          setForecastRuns(mlRunsResponse.runs ?? []);
        }

        if (meta.artifacts?.ml_drift) {
          const driftResponse = await getForecastDrift(datasetId, userId);
          if (!active) return;
          setForecastDrift(driftResponse);
        }

        if (meta.artifacts?.ml_registry) {
          const registryResponse = await getModelRegistry(datasetId, userId);
          if (!active) return;
          setModelRegistryEntries(registryResponse.entries ?? []);
        }

        if (meta.artifacts?.ml_evaluation) {
          const evaluationResponse = await getModelEvaluation(datasetId, userId);
          if (!active) return;
          setModelEvaluation(evaluationResponse);
        }

        if (meta.artifacts?.investigations) {
          const investigationsResponse = await getSavedInvestigations(datasetId, userId);
          if (!active) return;
          setSavedInvestigations(investigationsResponse.investigations ?? []);
        }

        if (meta.artifacts?.playbooks) {
          const playbooksResponse = await getSavedPlaybooks(datasetId, userId);
          if (!active) return;
          setSavedPlaybooks(playbooksResponse.playbooks ?? []);
        }

        if (meta.artifacts?.workflow_actions) {
          const workflowResponse = await getWorkflowActions(datasetId, userId);
          if (!active) return;
          setWorkflowActions(workflowResponse.actions ?? []);
        }

        if (meta.artifacts?.dashboard_spec) {
          const specResponse = await getDashboardSpec(datasetId, userId);
          if (!active) return;
          setDashboardSpec(specResponse.dashboard_spec);
        }

        const auditResponse = await getAudit(datasetId, userId);
        if (!active) return;
        setAuditEvents(auditResponse.events ?? []);

        if (meta.artifacts?.feedback) {
          const feedbackResponse = await listFeedback(datasetId, userId);
          if (!active) return;
          setFeedbackRecords(feedbackResponse.feedback ?? []);
        }
      } catch (sessionError) {
        if (!active) return;
        setError(sessionError instanceof Error ? sessionError.message : 'Failed to load session.');
      }
    }

    void hydrateSession();

    return () => {
      active = false;
    };
  }, [datasetId, userId]);

  useEffect(() => {
    if (!factsJob?.job_id || !datasetId) {
      return;
    }

    const interval = window.setInterval(async () => {
      try {
        const status = await getJobStatus(factsJob.job_id, userId);
        setFactsJob(status);
        if (status.status === 'succeeded') {
          const factsResponse = await getFacts(datasetId, userId);
          setFactsBundle(factsResponse.facts_bundle ?? null);
          setBusyAction(null);
          setNotice('Facts bundle is ready.');
          const meta = await getSession(datasetId, userId);
          setSessionMeta(meta);
          const auditResponse = await getAudit(datasetId, userId);
          setAuditEvents(auditResponse.events ?? []);
        }
        if (status.status === 'failed' || status.status === 'succeeded') {
          window.clearInterval(interval);
        }
      } catch (pollError) {
        window.clearInterval(interval);
        setBusyAction(null);
        setError(pollError instanceof Error ? pollError.message : 'Failed to poll facts job.');
      }
    }, 3000);

    return () => window.clearInterval(interval);
  }, [datasetId, factsJob?.job_id, userId]);

  useEffect(() => {
    if (!reportJob?.job_id || !datasetId) {
      return;
    }

    const interval = window.setInterval(async () => {
      try {
        const status = await getJobStatus(reportJob.job_id, userId);
        setReportJob(status);
        if (status.status === 'succeeded') {
          setBusyAction(null);
          setNotice('Report PDF is ready.');
          const meta = await getSession(datasetId, userId);
          setSessionMeta(meta);
          const auditResponse = await getAudit(datasetId, userId);
          setAuditEvents(auditResponse.events ?? []);
        }
        if (status.status === 'failed' || status.status === 'succeeded') {
          window.clearInterval(interval);
        }
      } catch (pollError) {
        window.clearInterval(interval);
        setBusyAction(null);
        setError(pollError instanceof Error ? pollError.message : 'Failed to poll report job.');
      }
    }, 3000);

    return () => window.clearInterval(interval);
  }, [datasetId, reportJob?.job_id, userId]);

  const qualityScore = useMemo(() => {
    const profileScore = asNumber(profile?.quality_score);
    if (profileScore !== null) {
      return profileScore;
    }
    return asNumber((factsBundle?.quality as { score?: unknown } | undefined)?.score);
  }, [factsBundle, profile]);

  const dataCoverage = useMemo(() => {
    const coverage = factsBundle?.data_coverage;
    return coverage && typeof coverage === 'object' ? (coverage as Record<string, unknown>) : null;
  }, [factsBundle]);

  const kpis = useMemo(() => {
    const value = factsBundle?.kpis;
    return Array.isArray(value) ? value.slice(0, 6) : [];
  }, [factsBundle]);

  const qualityIssues = useMemo(() => {
    const quality = factsBundle?.quality;
    if (!quality || typeof quality !== 'object') {
      return [];
    }
    const issues = (quality as { issues?: unknown }).issues;
    return Array.isArray(issues) ? issues.slice(0, 5) : [];
  }, [factsBundle]);

  const anomalyFindings = useMemo(() => anomalyAnalysis?.analysis.anomalies ?? [], [anomalyAnalysis]);
  const anomalySuggestedQuestions = useMemo(
    () => anomalyAnalysis?.analysis.suggested_questions ?? [],
    [anomalyAnalysis]
  );
  const forecastTimeOptions = useMemo(() => {
    const columns = Array.isArray(profile?.columns) ? profile.columns : [];
    return columns
      .map(asRecord)
      .filter((column): column is Record<string, unknown> => Boolean(column))
      .filter((column) => asString(column.inferred_type) === 'datetime' && !Boolean(column.is_pii_candidate) && !Boolean(column.is_id_like))
      .map((column) => asString(column.name))
      .filter((value): value is string => Boolean(value));
  }, [profile]);
  const forecastMetricOptions = useMemo(() => {
    const columns = Array.isArray(profile?.columns) ? profile.columns : [];
    return columns
      .map(asRecord)
      .filter((column): column is Record<string, unknown> => Boolean(column))
      .filter((column) => asString(column.inferred_type) === 'number' && !Boolean(column.is_pii_candidate) && !Boolean(column.is_id_like))
      .map((column) => asString(column.name))
      .filter((value): value is string => Boolean(value));
  }, [profile]);
  const cohortFieldOptions = useMemo(() => normalizeCohortFieldOptions(profile), [profile]);
  const cohortFieldLookup = useMemo(
    () =>
      cohortFieldOptions.reduce<Record<string, CohortFieldOption>>((accumulator, option) => {
        accumulator[option.name] = option;
        return accumulator;
      }, {}),
    [cohortFieldOptions]
  );

  const dashboardCharts = useMemo(() => normalizeDashboardCharts(dashboardSpec), [dashboardSpec]);
  const latestForecastRun = forecastRuns[0] ?? null;
  const activeRegistryEntry = modelRegistryEntries.find((entry) => entry.status === 'active') ?? null;

  const dashboardFilters = useMemo(() => {
    const items = Array.isArray(dashboardSpec?.filters) ? dashboardSpec.filters : [];
    return items
      .map((item) => {
        const filter = asRecord(item);
        const field = asString(filter?.field);
        const type = asString(filter?.type);
        if (!field || !type) return null;
        return { field, type };
      })
      .filter((item): item is { field: string; type: string } => Boolean(item));
  }, [dashboardSpec]);

  const factReferenceLookup = useMemo(() => {
    const facts = Array.isArray(factsBundle?.insight_facts) ? factsBundle.insight_facts : [];
    return facts.reduce<Record<string, string>>((accumulator, fact) => {
      const item = asRecord(fact);
      const id = asString(item?.id);
      const summary = summarizeFactReference(fact);
      if (id && summary) {
        accumulator[id] = summary;
      }
      return accumulator;
    }, {});
  }, [factsBundle]);

  const activeChartExplanation = selectedChartKey ? chartExplanations[selectedChartKey] ?? null : null;
  const feedbackLookup = useMemo(() => {
    return feedbackRecords.reduce<Record<string, FeedbackRecord>>((accumulator, record) => {
      const key = feedbackKey(record.surface, record.target_id);
      if (!accumulator[key]) {
        accumulator[key] = record;
      }
      return accumulator;
    }, {});
  }, [feedbackRecords]);
  const piiCandidates = useMemo(() => {
    const raw = profile?.pii_candidates;
    return Array.isArray(raw) ? raw.map((value) => String(value)).slice(0, 6) : [];
  }, [profile]);
  const sensitiveExportApproval = sessionMeta?.sensitive_export_approval;
  const sensitiveExportStatus = asString(sensitiveExportApproval?.status) ?? 'not_requested';
  const maskedExportsActive = piiCandidates.length > 0 && !sessionMeta?.allow_sensitive_export;
  const effectivePermissions = useMemo(() => {
    const resolvedPermissions = authContext?.permissions ?? [];
    if (resolvedPermissions.length > 0) {
      return resolvedPermissions;
    }
    return fallbackPermissionsForRole(authContext?.role ?? userRole);
  }, [authContext, userRole]);
  const permissionSet = useMemo(() => new Set(effectivePermissions), [effectivePermissions]);
  const effectiveRole = authContext?.role ?? userRole;
  const canCreateSession = permissionSet.has('sessions:create') || permissionSet.has('admin:all');
  const canCompute = permissionSet.has('sessions:compute_own') || permissionSet.has('sessions:compute_all') || permissionSet.has('admin:all');
  const canWriteSession = permissionSet.has('sessions:write_own') || permissionSet.has('sessions:write_all') || permissionSet.has('admin:all');
  const canReviewSensitiveExport = permissionSet.has('sensitive_export:review') || permissionSet.has('admin:all');
  const canRequestSensitiveExport = permissionSet.has('sensitive_export:request_own') || permissionSet.has('admin:all');
  const canReadDocuments = permissionSet.has('docs:read_own') || permissionSet.has('docs:read_all') || permissionSet.has('admin:all');
  const canUploadDocuments = permissionSet.has('docs:create') || permissionSet.has('admin:all');
  const canPromoteModel = permissionSet.has('ml:promote') || permissionSet.has('admin:all');
  const canDraftWorkflow = permissionSet.has('workflow:create_own') || permissionSet.has('workflow:create_all') || permissionSet.has('admin:all');
  const canReviewWorkflow = permissionSet.has('workflow:review') || permissionSet.has('admin:all');
  const canExecuteWorkflow = permissionSet.has('workflow:execute_own') || permissionSet.has('workflow:execute_all') || permissionSet.has('admin:all');

  useEffect(() => {
    if (cohortFieldOptions.length === 0) {
      return;
    }

    setCohortCriteria((current) =>
      current.map((criterion) => {
        if (criterion.field && cohortFieldLookup[criterion.field]) {
          const allowedOperators = cohortOperatorsForType(cohortFieldLookup[criterion.field].inferredType).map((item) => item.value);
          if (allowedOperators.includes(criterion.operator)) {
            return criterion;
          }
          return { ...criterion, operator: allowedOperators[0] ?? 'eq', value: '' };
        }

        const fallback = cohortFieldOptions[0];
        const fallbackOperator = cohortOperatorsForType(fallback.inferredType)[0]?.value ?? 'eq';
        return {
          ...criterion,
          field: fallback.name,
          operator: fallbackOperator,
        };
      })
    );
  }, [cohortFieldLookup, cohortFieldOptions]);

  useEffect(() => {
    if (forecastTimeOptions.length === 0) {
      if (forecastTimeField) {
        setForecastTimeField('');
      }
      return;
    }
    if (!forecastTimeOptions.includes(forecastTimeField)) {
      setForecastTimeField(forecastTimeOptions[0]);
    }
  }, [forecastTimeField, forecastTimeOptions]);

  useEffect(() => {
    if (forecastMetricOptions.length === 0) {
      if (forecastMetricField) {
        setForecastMetricField('');
      }
      return;
    }
    if (!forecastMetricOptions.includes(forecastMetricField)) {
      setForecastMetricField(forecastMetricOptions[0]);
    }
  }, [forecastMetricField, forecastMetricOptions]);

  useEffect(() => {
    if (dashboardCharts.length === 0) {
      setSelectedChartKey(null);
      return;
    }

    setSelectedChartKey((current) =>
      current && dashboardCharts.some((chart) => chart.key === current) ? current : dashboardCharts[0].key
    );
  }, [dashboardCharts]);

  async function refreshDocuments() {
    const response = await listDocuments(userId);
    setDocuments(response.documents ?? []);
  }

  async function refreshSavedInvestigations(targetDatasetId: string) {
    const response = await getSavedInvestigations(targetDatasetId, userId);
    setSavedInvestigations(response.investigations ?? []);
  }

  async function refreshSavedPlaybooks(targetDatasetId: string) {
    const response = await getSavedPlaybooks(targetDatasetId, userId);
    setSavedPlaybooks(response.playbooks ?? []);
  }

  async function refreshForecastRuns(targetDatasetId: string) {
    const response = await getForecastRuns(targetDatasetId, userId);
    setForecastRuns(response.runs ?? []);
  }

  async function refreshModelRegistry(targetDatasetId: string) {
    const response = await getModelRegistry(targetDatasetId, userId);
    setModelRegistryEntries(response.entries ?? []);
  }

  async function refreshWorkflowActions(targetDatasetId: string) {
    const response = await getWorkflowActions(targetDatasetId, userId);
    setWorkflowActions(response.actions ?? []);
  }

  async function refreshFeedback(targetDatasetId: string) {
    const response = await listFeedback(targetDatasetId, userId);
    setFeedbackRecords(response.feedback ?? []);
  }

  async function handleCreateSession() {
    setBusyAction('create_session');
    setError(null);
    setNotice(null);
    try {
      const created = await createSession(userId);
      setSessionMeta(null);
      setProfile(null);
      setFactsBundle(null);
      setAnomalyAnalysis(null);
      setCohortAnalysis(null);
      setForecastRuns([]);
      setForecastDrift(null);
      setModelEvaluation(null);
      setModelRegistryEntries([]);
      setSavedInvestigations([]);
      setSavedPlaybooks([]);
      setWorkflowActions([]);
      setFeedbackRecords([]);
      setDashboardSpec(null);
      setAskResult(null);
      setDashboardExplanation(null);
      setChartExplanations({});
      setSelectedChartKey(null);
      setAuditEvents([]);
      setDatasetId(created.dataset_id);
      setSessionMeta(await getSession(created.dataset_id, userId));
      setNotice(`Created session ${created.dataset_id}.`);
    } catch (sessionError) {
      setError(sessionError instanceof Error ? sessionError.message : 'Failed to create session.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleUploadDocument() {
    if (!selectedDocumentFile) {
      setError('Choose a TXT, MD, HTML, JSON, or text-based PDF document first.');
      return;
    }

    setBusyAction('document_upload');
    setError(null);
    setNotice(null);
    try {
      await uploadDocument(userId, selectedDocumentFile, {
        title: documentTitle,
        source_name: documentSourceName,
        version_label: documentVersionLabel,
        effective_date: documentEffectiveDate,
        supersedes_document_id: documentSupersedesId,
      });
      await refreshDocuments();
      setSelectedDocumentFile(null);
      setDocumentTitle('');
      setDocumentSourceName('');
      setDocumentVersionLabel('');
      setDocumentEffectiveDate('');
      setDocumentSupersedesId('');
      setNotice(`${selectedDocumentFile.name} uploaded to the trusted document library.`);
    } catch (documentError) {
      setError(documentError instanceof Error ? documentError.message : 'Document upload failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleAskDocuments() {
    if (!documentQuestion.trim()) {
      setError('Enter a document question first.');
      return;
    }

    setBusyAction('document_ask');
    setError(null);
    setNotice(null);
    try {
      const response = await askDocuments(userId, documentQuestion.trim());
      setDocumentAnswer(response);
      setNotice(response.grounded ? 'Document answer generated with citations.' : 'No grounded document answer was found.');
    } catch (documentError) {
      setError(documentError instanceof Error ? documentError.message : 'Document Q&A failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleSubmitFeedback(
    surface: 'ask_data' | 'document_qa' | 'dashboard_summary' | 'chart_explanation',
    targetId: string,
    rating: 'positive' | 'negative',
    extra?: { question?: string; title?: string }
  ) {
    if (!datasetId) {
      setError('Create or load a governed session before recording feedback.');
      return;
    }

    setBusyAction(`feedback:${surface}:${targetId}:${rating}`);
    setError(null);
    setNotice(null);
    try {
      await submitFeedback(datasetId, userId, {
        surface,
        target_id: targetId,
        rating,
        question: extra?.question,
        title: extra?.title,
      });
      await refreshFeedback(datasetId);
      const meta = await getSession(datasetId, userId);
      setSessionMeta(meta);
      const auditResponse = await getAudit(datasetId, userId);
      setAuditEvents(auditResponse.events ?? []);
      setNotice(`Recorded ${rating} feedback for ${surface.replace('_', ' ')}.`);
    } catch (feedbackError) {
      setError(feedbackError instanceof Error ? feedbackError.message : 'Feedback submission failed.');
    } finally {
      setBusyAction(null);
    }
  }

  function renderFeedbackActions(
    surface: 'ask_data' | 'document_qa' | 'dashboard_summary' | 'chart_explanation',
    targetId: string,
    extra?: { question?: string; title?: string }
  ) {
    const selectedRating = feedbackLookup[feedbackKey(surface, targetId)]?.rating;
    return (
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-muted-foreground">Was this helpful?</span>
        <Button
          type="button"
          size="sm"
          variant={selectedRating === 'positive' ? 'default' : 'outline'}
          onClick={() => void handleSubmitFeedback(surface, targetId, 'positive', extra)}
          disabled={busyAction === `feedback:${surface}:${targetId}:positive` || !canWriteSession}
        >
          {busyAction === `feedback:${surface}:${targetId}:positive` ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <ThumbsUp className="mr-2 h-4 w-4" />
          )}
          Helpful
        </Button>
        <Button
          type="button"
          size="sm"
          variant={selectedRating === 'negative' ? 'default' : 'outline'}
          onClick={() => void handleSubmitFeedback(surface, targetId, 'negative', extra)}
          disabled={busyAction === `feedback:${surface}:${targetId}:negative` || !canWriteSession}
        >
          {busyAction === `feedback:${surface}:${targetId}:negative` ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <ThumbsDown className="mr-2 h-4 w-4" />
          )}
          Needs Work
        </Button>
      </div>
    );
  }

  async function handleLoadSession() {
    const targetDatasetId = loadDatasetId.trim();
    if (!targetDatasetId) {
      setError('Enter a session ID to load.');
      return;
    }
    if (!isDatasetId(targetDatasetId)) {
      setError('Session IDs are generated UUIDs. Click New Session or paste a valid session ID, not a short value like 1.');
      return;
    }
    setBusyAction('load_session');
    setError(null);
    setNotice(null);
    try {
      await getSession(targetDatasetId, userId);
      setSessionMeta(null);
      setProfile(null);
      setFactsBundle(null);
      setAnomalyAnalysis(null);
      setCohortAnalysis(null);
      setForecastRuns([]);
      setForecastDrift(null);
      setModelEvaluation(null);
      setModelRegistryEntries([]);
      setSavedInvestigations([]);
      setSavedPlaybooks([]);
      setWorkflowActions([]);
      setFeedbackRecords([]);
      setDashboardSpec(null);
      setAskResult(null);
      setDashboardExplanation(null);
      setChartExplanations({});
      setSelectedChartKey(null);
      setAuditEvents([]);
      setDatasetId(targetDatasetId);
      setNotice(`Loaded session ${targetDatasetId}.`);
    } catch (sessionError) {
      setError(sessionError instanceof Error ? sessionError.message : 'Failed to load session.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleUpload() {
    if (!selectedFile) {
      setError('Choose a CSV or XLSX file first.');
      return;
    }

    setBusyAction('upload');
    setError(null);
    setNotice(null);

    try {
      let targetDatasetId = datasetId;
      if (!targetDatasetId) {
        const created = await createSession(userId);
        targetDatasetId = created.dataset_id;
        setDatasetId(created.dataset_id);
      }

      await uploadDataset(targetDatasetId, selectedFile, userId);
      setProfile(null);
      setFactsBundle(null);
      setAnomalyAnalysis(null);
      setCohortAnalysis(null);
      setForecastRuns([]);
      setForecastDrift(null);
      setModelEvaluation(null);
      setModelRegistryEntries([]);
      setSavedInvestigations([]);
      setSavedPlaybooks([]);
      setWorkflowActions([]);
      setDashboardSpec(null);
      setAskResult(null);
      setDashboardExplanation(null);
      setChartExplanations({});
      setSelectedChartKey(null);
      setExportApprovalInput('');
      setFactsJob(null);
      setReportJob(null);
      setSelectedFile(null);
      setSessionMeta(await getSession(targetDatasetId, userId));
      await refreshForecastRuns(targetDatasetId);
      await refreshModelRegistry(targetDatasetId);
      setAuditEvents((await getAudit(targetDatasetId, userId)).events ?? []);
      setNotice(`${selectedFile.name} uploaded successfully.`);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Upload failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleProfile() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    setBusyAction('profile');
    setError(null);
    setNotice(null);
    try {
      const response = await getProfile(datasetId, userId);
      setProfile(response.profile);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice('Profile generated successfully.');
    } catch (profileError) {
      setError(profileError instanceof Error ? profileError.message : 'Profiling failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleFacts() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    setBusyAction('facts');
    setError(null);
    setNotice(null);
    try {
      const response = await getFacts(datasetId, userId);
      if (response.job_id) {
        setFactsJob({
          job_id: response.job_id,
          type: 'facts',
          dataset_id: datasetId,
          status: response.status ?? 'queued',
          progress: 0,
        });
        setNotice(`Facts job queued: ${response.job_id}`);
        return;
      }
      setFactsBundle(response.facts_bundle ?? null);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice('Facts bundle generated successfully.');
    } catch (factsError) {
      setError(factsError instanceof Error ? factsError.message : 'Facts generation failed.');
    } finally {
      if (!factsJob?.job_id) {
        setBusyAction(null);
      }
    }
  }

  async function handleDashboardSpec() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    setBusyAction('dashboard');
    setError(null);
    setNotice(null);
    try {
      const response = await generateDashboardSpec(datasetId, userId);
      setDashboardSpec(response.dashboard_spec);
      setDashboardExplanation(null);
      setChartExplanations({});
      setSelectedChartKey(normalizeDashboardCharts(response.dashboard_spec)[0]?.key ?? null);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice('Dashboard spec generated successfully.');
    } catch (dashboardError) {
      setError(dashboardError instanceof Error ? dashboardError.message : 'Dashboard generation failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleDetectAnomalies() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    setBusyAction('anomalies');
    setError(null);
    setNotice(null);
    try {
      const response = await getAnomalyAnalysis(datasetId, userId);
      setAnomalyAnalysis(response);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice(
        response.analysis.anomaly_count > 0
          ? `Detected ${response.analysis.anomaly_count} governed anomaly signal(s).`
          : 'No high-signal anomalies were detected.'
      );
    } catch (anomalyError) {
      setError(anomalyError instanceof Error ? anomalyError.message : 'Anomaly detection failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleTrainForecast() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    if (!profile) {
      setBusyAction('forecast_profile');
      setError(null);
      setNotice(null);
      try {
        const response = await getProfile(datasetId, userId);
        setProfile(response.profile);
        setSessionMeta(await getSession(datasetId, userId));
        setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
        setNotice('Profile loaded. Choose governed time and metric fields, then train the forecast again.');
      } catch (profileError) {
        setError(profileError instanceof Error ? profileError.message : 'Failed to load profile for forecast training.');
      } finally {
        setBusyAction(null);
      }
      return;
    }
    if (!forecastTimeField || !forecastMetricField) {
      setError('Choose governed time and metric fields for the forecast.');
      return;
    }

    const parsedHorizon = Number.parseInt(forecastHorizon, 10);
    if (!Number.isFinite(parsedHorizon) || parsedHorizon < 1 || parsedHorizon > 12) {
      setError('Forecast horizon must be between 1 and 12 periods.');
      return;
    }

    setBusyAction('forecast');
    setError(null);
    setNotice(null);
    try {
      const run = await trainForecastRun(datasetId, userId, {
        name: forecastName.trim() || undefined,
        time_field: forecastTimeField,
        metric_field: forecastMetricField,
        horizon: parsedHorizon,
        aggregation: forecastAggregation,
      });
      await refreshForecastRuns(datasetId);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setForecastDrift(null);
      setModelEvaluation(null);
      setForecastName('');
      setNotice(`Forecast run completed with champion model ${run.payload.champion_model}.`);
    } catch (forecastError) {
      setError(forecastError instanceof Error ? forecastError.message : 'Forecast training failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleScanForecastDrift(runId?: string) {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    if (forecastRuns.length === 0) {
      setError('Train or load a forecast run before scanning drift.');
      return;
    }

    setBusyAction('forecast_drift');
    setError(null);
    setNotice(null);
    try {
      const response = await getForecastDrift(datasetId, userId, runId ?? forecastRuns[0]?.run_id);
      setForecastDrift(response);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice(`Forecast drift scanned for ${response.drift.metric_field}.`);
    } catch (driftError) {
      setError(driftError instanceof Error ? driftError.message : 'Forecast drift scan failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handlePromoteModel(runId: string) {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }

    setBusyAction(`promote_model:${runId}`);
    setError(null);
    setNotice(null);
    try {
      const promoted = await promoteModelRun(datasetId, runId, userId, {
        note: modelPromotionNote.trim() || undefined,
      });
      await refreshModelRegistry(datasetId);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setModelEvaluation(null);
      setModelPromotionNote('');
      setNotice(`Promoted model ${promoted.name} to the governed registry.`);
    } catch (promoteError) {
      setError(promoteError instanceof Error ? promoteError.message : 'Model promotion failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleEvaluateModels(challengerRunId?: string) {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    if (!activeRegistryEntry) {
      setError('Promote an active governed model before running champion-versus-challenger evaluation.');
      return;
    }
    if (forecastRuns.length < 2) {
      setError('Train at least two forecast runs before running model evaluation.');
      return;
    }

    setBusyAction('model_evaluation');
    setError(null);
    setNotice(null);
    try {
      const response = await getModelEvaluation(datasetId, userId, challengerRunId);
      setModelEvaluation(response);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice(`Model evaluation completed. Recommended winner: ${response.evaluation.winner}.`);
    } catch (evaluationError) {
      setError(evaluationError instanceof Error ? evaluationError.message : 'Model evaluation failed.');
    } finally {
      setBusyAction(null);
    }
  }

  function handleAddCohortCriterion() {
    setCohortCriteria((current) => [...current, createDraftCriterion(cohortFieldOptions)]);
  }

  function handleRemoveCohortCriterion(criterionId: string) {
    setCohortCriteria((current) => {
      if (current.length === 1) {
        return [{ id: 'criterion-initial', field: cohortFieldOptions[0]?.name ?? '', operator: cohortOperatorsForType(cohortFieldOptions[0]?.inferredType ?? 'string')[0]?.value ?? 'eq', value: '' }];
      }
      return current.filter((criterion) => criterion.id !== criterionId);
    });
  }

  function handleUpdateCohortCriterion(
    criterionId: string,
    patch: Partial<CohortDraftCriterion> & { field?: string; operator?: string; value?: string }
  ) {
    setCohortCriteria((current) =>
      current.map((criterion) => {
        if (criterion.id !== criterionId) {
          return criterion;
        }

        const nextField = patch.field ?? criterion.field;
        const nextFieldType = cohortFieldLookup[nextField]?.inferredType ?? 'string';
        const allowedOperators = cohortOperatorsForType(nextFieldType).map((item) => item.value);
        const requestedOperator = patch.operator ?? criterion.operator;
        const nextOperator = allowedOperators.includes(requestedOperator) ? requestedOperator : allowedOperators[0] ?? 'eq';
        const nextValue =
          patch.value !== undefined
            ? patch.value
            : nextOperator !== criterion.operator && !cohortOperatorNeedsValue(nextOperator)
              ? ''
              : criterion.value;

        return {
          ...criterion,
          ...patch,
          field: nextField,
          operator: nextOperator,
          value: cohortOperatorNeedsValue(nextOperator) ? nextValue : '',
        };
      })
    );
  }

  async function handleBuildCohort() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    if (!profile) {
      setBusyAction('cohort_profile');
      setError(null);
      setNotice(null);
      try {
        const response = await getProfile(datasetId, userId);
        setProfile(response.profile);
        setSessionMeta(await getSession(datasetId, userId));
        setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
        setNotice('Profile loaded. Review the governed fields and run the cohort builder again.');
      } catch (profileError) {
        setError(profileError instanceof Error ? profileError.message : 'Failed to load profile for cohort builder.');
      } finally {
        setBusyAction(null);
      }
      return;
    }

    const criteriaPayload: CohortRequestCriterion[] = cohortCriteria
      .map((criterion) => {
        const field = criterion.field.trim();
        const operator = criterion.operator.trim();
        const value = criterion.value.trim();
        if (!field || !operator) {
          return null;
        }
        if (!cohortOperatorNeedsValue(operator)) {
          return { field, operator };
        }
        return {
          field,
          operator,
          value: operator === 'in' || operator === 'not_in' || operator === 'between'
            ? value.split(',').map((item) => item.trim()).filter(Boolean)
            : value,
        };
      })
      .filter((item): item is CohortRequestCriterion => item !== null);

    if (criteriaPayload.length === 0) {
      setError('Add at least one governed cohort criterion.');
      return;
    }

    const missingValueCriterion = cohortCriteria.find(
      (criterion) => criterion.field.trim() && criterion.operator.trim() && cohortOperatorNeedsValue(criterion.operator) && !criterion.value.trim()
    );
    if (missingValueCriterion) {
      setError(`Enter a value for cohort field ${missingValueCriterion.field}.`);
      return;
    }

    setBusyAction('cohort');
    setError(null);
    setNotice(null);
    try {
      const response = await buildCohortAnalysis(datasetId, userId, {
        name: cohortName.trim() || undefined,
        description: cohortDescription.trim() || undefined,
        criteria: criteriaPayload,
        limit: 25,
      });
      setCohortAnalysis(response);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice(
        response.cohort.row_count > 0
          ? `Cohort generated with ${response.cohort.row_count} matching row(s).`
          : 'Cohort generated, but no rows matched the current governed criteria.'
      );
    } catch (cohortError) {
      setError(cohortError instanceof Error ? cohortError.message : 'Cohort generation failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleSaveInvestigation() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    if (!askResult || !askQuestion.trim()) {
      setError('Run a governed ask-data investigation before saving it.');
      return;
    }

    setBusyAction('save_investigation');
    setError(null);
    setNotice(null);
    try {
      const saved = await saveInvestigation(datasetId, userId, {
        title: investigationTitle.trim() || undefined,
        question: askQuestion.trim(),
        context_type: 'ask',
        note: investigationNote.trim() || undefined,
        result: askResult as unknown as Record<string, unknown>,
      });
      await refreshSavedInvestigations(datasetId);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setInvestigationTitle('');
      setInvestigationNote('');
      setNotice(`Saved investigation: ${saved.title}.`);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Saving the investigation failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleSavePlaybook() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    if (!askQuestion.trim()) {
      setError('Enter a governed question before saving a playbook.');
      return;
    }
    if (!playbookName.trim()) {
      setError('Enter a playbook name first.');
      return;
    }

    setBusyAction('save_playbook');
    setError(null);
    setNotice(null);
    try {
      const saved = await savePlaybook(datasetId, userId, {
        name: playbookName.trim(),
        question_template: askQuestion.trim(),
        description: playbookDescription.trim() || undefined,
        context_type: 'ask',
      });
      await refreshSavedPlaybooks(datasetId);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setPlaybookName('');
      setPlaybookDescription('');
      setNotice(`Saved playbook: ${saved.name}.`);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Saving the playbook failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleDraftWorkflow() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }

    setBusyAction('workflow_draft');
    setError(null);
    setNotice(null);
    try {
      const drafted = await draftWorkflowAction(datasetId, userId, {
        action_type: workflowActionType,
        title: workflowTitle.trim() || undefined,
        target: workflowTarget.trim() || undefined,
        objective: workflowObjective.trim() || undefined,
      });
      await refreshWorkflowActions(datasetId);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setWorkflowTitle('');
      setWorkflowTarget('');
      setWorkflowObjective('');
      setNotice(`Workflow draft ready for approval: ${drafted.title}.`);
    } catch (workflowError) {
      setError(workflowError instanceof Error ? workflowError.message : 'Workflow draft failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleReviewWorkflow(actionId: string, approved: boolean) {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }

    setBusyAction(approved ? `workflow_approve:${actionId}` : `workflow_reject:${actionId}`);
    setError(null);
    setNotice(null);
    try {
      const action = await reviewWorkflowAction(datasetId, actionId, userId, {
        approved,
        note: workflowDecisionNote.trim() || undefined,
      });
      await refreshWorkflowActions(datasetId);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setWorkflowDecisionNote('');
      setNotice(
        approved
          ? `Workflow action approved: ${action.title}.`
          : `Workflow action rejected: ${action.title}.`
      );
    } catch (workflowError) {
      setError(workflowError instanceof Error ? workflowError.message : 'Workflow review failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleExecuteWorkflow(actionId: string) {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }

    setBusyAction(`workflow_execute:${actionId}`);
    setError(null);
    setNotice(null);
    try {
      const action = await executeWorkflowAction(datasetId, actionId, userId, {
        note: workflowDecisionNote.trim() || undefined,
      });
      await refreshWorkflowActions(datasetId);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setWorkflowDecisionNote('');
      setNotice(`Workflow action executed: ${action.title}.`);
    } catch (workflowError) {
      setError(workflowError instanceof Error ? workflowError.message : 'Workflow execution failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleInvestigateQuestion(question: string, successNotice = 'Investigation generated successfully.') {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    if (!question.trim()) {
      setError('Investigation question is empty.');
      return;
    }
    setBusyAction('investigate');
    setError(null);
    setNotice(null);
    setAskQuestion(question);
    try {
      const response = await askDataset(datasetId, userId, question.trim());
      setAskResult(response);
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice(successNotice);
    } catch (askError) {
      setError(askError instanceof Error ? askError.message : 'Investigation request failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleAsk() {
    if (!askQuestion.trim()) {
      setError('Enter a question first.');
      return;
    }
    await handleInvestigateQuestion(askQuestion.trim(), 'Ask-data answer generated successfully.');
  }

  async function handleRequestSensitiveExportApproval() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    if (exportApprovalInput.trim().length < 8) {
      setError('Enter a short justification before requesting sensitive export access.');
      return;
    }

    setBusyAction('request_sensitive_export');
    setError(null);
    setNotice(null);
    try {
      await requestSensitiveExportApproval(datasetId, userId, exportApprovalInput.trim());
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice('Sensitive export approval requested.');
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'Could not request export approval.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleReviewSensitiveExportApproval(approved: boolean) {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }

    setBusyAction(approved ? 'approve_sensitive_export' : 'reject_sensitive_export');
    setError(null);
    setNotice(null);
    try {
      await reviewSensitiveExportApproval(datasetId, userId, {
        approved,
        note: exportApprovalInput.trim() || undefined,
      });
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice(approved ? 'Sensitive export approved and enabled.' : 'Sensitive export request rejected.');
      setExportApprovalInput('');
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : 'Could not review export approval.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleDisableSensitiveExport() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }

    setBusyAction('disable_sensitive_export');
    setError(null);
    setNotice(null);
    try {
      await setSensitiveExportEnabled(datasetId, userId, false);
      setSessionMeta(await getSession(datasetId, userId));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice('Sensitive export disabled. Future CSV and JSON downloads will be masked again.');
    } catch (disableError) {
      setError(disableError instanceof Error ? disableError.message : 'Could not disable sensitive export.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleExplainDashboard() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    if (dashboardCharts.length === 0) {
      setError('Generate a dashboard spec first.');
      return;
    }

    const question = buildDashboardExplanationQuestion(
      asString(dashboardSpec?.title),
      dashboardCharts,
      dashboardFilters
    );

    setBusyAction('dashboard_explanation');
    setError(null);
    setNotice(null);
    try {
      const response = await askDataset(datasetId, userId, question);
      setDashboardExplanation({
        generatedAt: new Date().toISOString(),
        preset: 'summary',
        question,
        result: response,
      });
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice('Dashboard explanation generated successfully.');
    } catch (explainError) {
      setError(explainError instanceof Error ? explainError.message : 'Dashboard explanation failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleExplainChart(chart: DashboardChartSpec, preset: ExplainPreset) {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }

    const linkedFacts = chart.factIds
      .map((factId) => factReferenceLookup[factId] ?? factId)
      .filter((value): value is string => Boolean(value));
    const question = buildChartExplanationQuestion(chart, preset, linkedFacts, asString(dashboardSpec?.title));
    const busyKey = `chart_explanation:${chart.key}:${preset}`;

    setBusyAction(busyKey);
    setSelectedChartKey(chart.key);
    setError(null);
    setNotice(null);
    try {
      const response = await askDataset(datasetId, userId, question);
      setChartExplanations((previous) => ({
        ...previous,
        [chart.key]: {
          generatedAt: new Date().toISOString(),
          preset,
          question,
          result: response,
        },
      }));
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice(`${chart.title} explanation generated successfully.`);
    } catch (explainError) {
      setError(explainError instanceof Error ? explainError.message : 'Chart explanation failed.');
    } finally {
      setBusyAction(null);
    }
  }

  async function handleReport() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    setBusyAction('report');
    setError(null);
    setNotice(null);
    try {
      const response = await generateReport(datasetId, userId);
      setReportJob({
        job_id: response.job_id,
        type: 'report',
        dataset_id: response.dataset_id,
        status: response.status,
        progress: 0,
      });
      setNotice(`Report job queued: ${response.job_id}`);
      setBusyAction(null);
    } catch (reportError) {
      setError(reportError instanceof Error ? reportError.message : 'Report generation failed.');
      setBusyAction(null);
    }
  }

  const isBusy = (action: string) => busyAction === action;
  const reportReady = Boolean(sessionMeta?.artifacts?.report_pdf);
  const hasFacts = Boolean(factsBundle);
  const hasDashboardSpec = Boolean(dashboardSpec);

  return (
    <div className="space-y-4 sm:space-y-6">
      <section className="rounded-2xl border border-border bg-card p-4 sm:p-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <div className="rounded-xl bg-health-mint/20 p-2 text-health-mint">
                <Bot className="h-5 w-5" />
              </div>
              <h2 className="text-lg font-semibold text-foreground sm:text-xl">Governed AI Analytics Engine</h2>
            </div>
            <p className="mt-2 text-sm text-muted-foreground">
              This surface uses the FastAPI control plane for sessions, profiling, facts, semantic ask-data,
              reports, jobs, exports, and audit events.
            </p>
            <p className="mt-1 text-xs text-muted-foreground">API target: {API_TARGET_LABEL}</p>
          </div>

          <div className="grid gap-2 sm:grid-cols-2">
            <div className="rounded-xl border border-border bg-background px-3 py-2">
              <p className="text-xs text-muted-foreground">Session ID</p>
              <p className="break-all text-sm text-foreground">{datasetId || 'Not selected'}</p>
            </div>
            <div className="rounded-xl border border-border bg-background px-3 py-2">
              <p className="text-xs text-muted-foreground">Status</p>
              <p className="text-sm text-foreground">{sessionMeta?.status ?? 'Idle'}</p>
            </div>
          </div>
        </div>

        <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_220px_1fr_auto]">
          <Input value={userId} onChange={(event) => setUserId(event.target.value)} placeholder="User ID" />
          <select
            value={userRole}
            onChange={(event) => setUserRole(event.target.value as BackendUserRole)}
            className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm"
          >
            <option value="viewer">viewer</option>
            <option value="analyst">analyst</option>
            <option value="reviewer">reviewer</option>
            <option value="admin">admin</option>
          </select>
          <Input
            value={loadDatasetId}
            onChange={(event) => setLoadDatasetId(event.target.value)}
            placeholder="Existing session ID (UUID)"
          />
          <div className="flex gap-2">
            <Button type="button" variant="outline" onClick={() => void handleLoadSession()} disabled={isBusy('load_session')}>
              {isBusy('load_session') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              Load
            </Button>
            <Button type="button" onClick={() => void handleCreateSession()} disabled={isBusy('create_session') || !canCreateSession}>
              {isBusy('create_session') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              New Session
            </Button>
          </div>
        </div>

        <div className="mt-3 rounded-xl border border-border bg-background px-4 py-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline">Actor: {authContext?.actor ?? userId ?? 'anonymous'}</Badge>
            <Badge variant="outline">Role: {effectiveRole}</Badge>
            <Badge variant="outline">Permissions: {effectivePermissions.length}</Badge>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            {usingFallbackAuthContext
              ? 'Using local role fallback permissions because the auth context check could not be loaded. New Session and other governed actions stay available while connectivity recovers.'
              : 'This dev auth layer is permission-aware but still header-driven. Production rollout still needs real identity and policy enforcement.'}
          </p>
          <p className="mt-2 text-xs text-muted-foreground">
            Use New Session to generate a fresh session ID automatically. Do not type small values like 1 into the load field.
          </p>
          {effectivePermissions.length ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {effectivePermissions.slice(0, 8).map((permission) => (
                <Badge key={permission} variant="secondary">
                  {permission}
                </Badge>
              ))}
            </div>
          ) : null}
        </div>

        <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_auto]">
          <Input
            type="file"
            accept=".csv,.xlsx"
            onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
          />
          <Button
            type="button"
            onClick={() => void handleUpload()}
            disabled={isBusy('upload') || !selectedFile || (!datasetId && !canCreateSession)}
          >
            {isBusy('upload') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Upload className="mr-2 h-4 w-4" />}
            Upload Dataset
          </Button>
        </div>

        {selectedFile ? (
          <div className="mt-3 rounded-xl border border-border bg-background px-3 py-2 text-sm text-foreground">
            Selected file: {selectedFile.name}
          </div>
        ) : null}

        {notice ? (
          <div className="mt-3 rounded-xl border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-400">
            {notice}
          </div>
        ) : null}
        {error ? (
          <div className="mt-3 rounded-xl border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-400">
            {error}
          </div>
        ) : null}
      </section>

      <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
        <Card className="border-border">
          <CardContent className="p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Rows / Columns</p>
            <p className="mt-2 text-lg font-semibold text-foreground">
              {profile?.shape && typeof profile.shape === 'object'
                ? `${String((profile.shape as { rows?: unknown }).rows ?? 'n/a')} / ${String((profile.shape as { cols?: unknown }).cols ?? 'n/a')}`
                : 'Pending'}
            </p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardContent className="p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Quality Score</p>
            <p className="mt-2 text-lg font-semibold text-foreground">{qualityScore ?? 'Pending'}</p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardContent className="p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Coverage Mode</p>
            <p className="mt-2 text-lg font-semibold text-foreground">
              {typeof dataCoverage?.mode === 'string' ? dataCoverage.mode : 'Pending'}
            </p>
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardContent className="p-4">
            <p className="text-xs uppercase tracking-wide text-muted-foreground">Current File</p>
            <p className="mt-2 truncate text-sm font-medium text-foreground">{sessionMeta?.file?.name ?? 'No file uploaded'}</p>
          </CardContent>
        </Card>
      </section>

      <section className="rounded-2xl border border-border bg-card p-4 sm:p-6">
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => void handleProfile()}
            disabled={isBusy('profile') || !datasetId || !canCompute}
          >
            {isBusy('profile') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <BadgeCheck className="mr-2 h-4 w-4" />}
            Run Profiling
          </Button>
          <Button type="button" variant="outline" onClick={() => void handleFacts()} disabled={isBusy('facts') || !datasetId || !canCompute}>
            {isBusy('facts') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
            Generate Facts
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() => void handleDetectAnomalies()}
            disabled={isBusy('anomalies') || !datasetId || !canCompute}
          >
            {isBusy('anomalies') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <AlertCircle className="mr-2 h-4 w-4" />}
            Detect Anomalies
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() => void handleDashboardSpec()}
            disabled={isBusy('dashboard') || !datasetId || !canCompute}
          >
            {isBusy('dashboard') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Database className="mr-2 h-4 w-4" />}
            Generate Dashboard
          </Button>
          <Button type="button" variant="outline" onClick={() => void handleReport()} disabled={isBusy('report') || !datasetId || !canCompute}>
            {isBusy('report') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <FileDown className="mr-2 h-4 w-4" />}
            Generate Report
          </Button>
        </div>

        {factsJob ? (
          <div className="mt-4 rounded-xl border border-border bg-background p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-medium text-foreground">Facts Job</p>
                <p className="text-xs text-muted-foreground">
                  {factsJob.job_id} · {factsJob.status}
                </p>
              </div>
              <Button type="button" size="sm" variant="ghost" onClick={() => void handleFacts()}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
            </div>
            <Progress className="mt-3 h-2" value={factsJob.progress ?? 0} />
          </div>
        ) : null}

        {reportJob ? (
          <div className="mt-4 rounded-xl border border-border bg-background p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-medium text-foreground">Report Job</p>
                <p className="text-xs text-muted-foreground">
                  {reportJob.job_id} · {reportJob.status}
                </p>
              </div>
              <Badge variant="outline">{reportJob.progress ?? 0}%</Badge>
            </div>
            <Progress className="mt-3 h-2" value={reportJob.progress ?? 0} />
          </div>
        ) : null}
      </section>

      <section className="rounded-2xl border border-border bg-card p-4 sm:p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex items-center gap-2">
            <div className="rounded-xl bg-health-mint/20 p-2 text-health-mint">
              <BadgeCheck className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-lg font-medium text-foreground">Governed Cohort Builder</h3>
              <p className="text-sm text-muted-foreground">
                Build editable cohort rules from approved fields only. PII and ID-like columns stay blocked from criteria and preview rows.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button type="button" variant="outline" onClick={handleAddCohortCriterion} disabled={!cohortFieldOptions.length || !canCompute}>
              <Plus className="mr-2 h-4 w-4" />
              Add Criterion
            </Button>
            <Button type="button" onClick={() => void handleBuildCohort()} disabled={isBusy('cohort') || isBusy('cohort_profile') || !datasetId || !canCompute}>
              {isBusy('cohort') || isBusy('cohort_profile') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <BadgeCheck className="mr-2 h-4 w-4" />}
              Build Cohort
            </Button>
          </div>
        </div>

        {!datasetId ? (
          <div className="mt-4 rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
            Create or load a session before building governed cohort criteria.
          </div>
        ) : !profile ? (
          <div className="mt-4 rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
            Run profiling once to load governed cohort fields. The builder only exposes approved, non-PHI columns.
          </div>
        ) : cohortFieldOptions.length === 0 ? (
          <div className="mt-4 rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
            No governed cohort fields are currently available after applying PII and identifier controls.
          </div>
        ) : (
          <div className="mt-4 space-y-4">
            <div className="grid gap-3 lg:grid-cols-[1fr_1.2fr]">
              <Input value={cohortName} onChange={(event) => setCohortName(event.target.value)} placeholder="Cohort name" />
              <Textarea
                value={cohortDescription}
                onChange={(event) => setCohortDescription(event.target.value)}
                placeholder="Optional description of the operational or quality use case."
                className="min-h-[72px]"
              />
            </div>

            <div className="grid gap-3">
              {cohortCriteria.map((criterion) => {
                const fieldMeta = cohortFieldLookup[criterion.field];
                const operatorOptions = cohortOperatorsForType(fieldMeta?.inferredType ?? 'string');
                return (
                  <div key={criterion.id} className="rounded-xl border border-border bg-background p-4">
                    <div className="grid gap-3 lg:grid-cols-[1.2fr_0.9fr_1fr_auto]">
                      <select
                        value={criterion.field}
                        onChange={(event) => handleUpdateCohortCriterion(criterion.id, { field: event.target.value })}
                        className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm"
                      >
                        {cohortFieldOptions.map((option) => (
                          <option key={option.name} value={option.name}>
                            {option.name} ({option.inferredType})
                          </option>
                        ))}
                      </select>

                      <select
                        value={criterion.operator}
                        onChange={(event) => handleUpdateCohortCriterion(criterion.id, { operator: event.target.value })}
                        className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm"
                      >
                        {operatorOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>

                      <Input
                        value={criterion.value}
                        onChange={(event) => handleUpdateCohortCriterion(criterion.id, { value: event.target.value })}
                        placeholder={
                          criterion.operator === 'between'
                            ? 'two values separated by comma'
                            : criterion.operator === 'in' || criterion.operator === 'not_in'
                              ? 'comma-separated values'
                              : 'value'
                        }
                        disabled={!cohortOperatorNeedsValue(criterion.operator)}
                      />

                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => handleRemoveCohortCriterion(criterion.id)}
                        aria-label="Remove cohort criterion"
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    </div>

                    {fieldMeta?.sampleValues.length ? (
                      <p className="mt-3 text-xs text-muted-foreground">
                        Sample values: {fieldMeta.sampleValues.join(', ')}
                      </p>
                    ) : null}
                  </div>
                );
              })}
            </div>

            <div className="flex flex-wrap gap-2">
              {cohortCriteria
                .filter((criterion) => criterion.field)
                .map((criterion) => {
                  const fieldMeta = cohortFieldLookup[criterion.field];
                  const operatorLabel =
                    cohortOperatorsForType(fieldMeta?.inferredType ?? 'string').find((item) => item.value === criterion.operator)?.label ??
                    criterion.operator;
                  const valueLabel = cohortOperatorNeedsValue(criterion.operator) && criterion.value.trim() ? ` ${criterion.value.trim()}` : '';
                  return (
                    <Badge key={`${criterion.id}-badge`} variant="outline">
                      {criterion.field} {operatorLabel}
                      {valueLabel}
                    </Badge>
                  );
                })}
            </div>
          </div>
        )}

        {cohortAnalysis ? (
          <div className="mt-4 space-y-4">
            <div className="rounded-xl border border-border bg-background p-4">
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">Rows: {cohortAnalysis.cohort.row_count}</Badge>
                <Badge variant="outline">Population: {cohortAnalysis.cohort.population_row_count}</Badge>
                <Badge variant="outline">Criteria: {cohortAnalysis.cohort.criteria_count}</Badge>
              </div>
              <p className="mt-3 text-sm text-foreground">{cohortAnalysis.cohort.summary}</p>
              {cohortAnalysis.cohort.excluded_columns.length > 0 ? (
                <p className="mt-2 text-xs text-muted-foreground">
                  Excluded by governance: {cohortAnalysis.cohort.excluded_columns.join(', ')}
                </p>
              ) : null}
            </div>

            {cohortAnalysis.cohort.criteria.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {cohortAnalysis.cohort.criteria.map((criterion, index) => (
                  <Badge key={`${criterion.field}-${index}`} variant="outline">
                    {criterion.field} {criterion.operator_label ?? criterion.operator}
                    {criterion.value !== undefined && criterion.value !== null && formatCohortCriterionValue(criterion.value)
                      ? ` ${formatCohortCriterionValue(criterion.value)}`
                      : ''}
                  </Badge>
                ))}
              </div>
            ) : null}

            {cohortAnalysis.cohort.suggested_questions.length > 0 ? (
              <div className="rounded-xl border border-border bg-background p-4">
                <p className="text-sm font-medium text-foreground">Suggested follow-up questions</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {cohortAnalysis.cohort.suggested_questions.map((question) => (
                    <Button
                      key={question}
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => void handleInvestigateQuestion(question, 'Cohort follow-up generated successfully.')}
                      disabled={busyAction === 'investigate' || !canCompute}
                    >
                      {question}
                    </Button>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="rounded-xl border border-border bg-background p-4">
              <p className="text-sm font-medium text-foreground">Preview Rows</p>
              {cohortAnalysis.cohort.preview_rows.length === 0 ? (
                <p className="mt-3 text-sm text-muted-foreground">No preview rows matched the current governed cohort rules.</p>
              ) : (
                <div className="mt-3 overflow-auto">
                  <table className="min-w-full divide-y divide-border text-sm">
                    <thead>
                      <tr>
                        {cohortAnalysis.cohort.preview_columns.map((column) => (
                          <th key={column} className="px-3 py-2 text-left font-medium text-muted-foreground">
                            {column}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border">
                      {cohortAnalysis.cohort.preview_rows.map((row, rowIndex) => (
                        <tr key={`cohort-row-${rowIndex}`}>
                          {cohortAnalysis.cohort.preview_columns.map((column) => (
                            <td key={`${rowIndex}-${column}`} className="px-3 py-2 text-foreground">
                              {String(row[column] ?? '—')}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        ) : null}
      </section>

      <section className="rounded-2xl border border-border bg-card p-4 sm:p-6">
        <div className="flex items-center gap-2">
          <div className="rounded-xl bg-health-mint/20 p-2 text-health-mint">
            <FileText className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-lg font-medium text-foreground">Trusted Document Q&amp;A</h3>
            <p className="text-sm text-muted-foreground">
              Upload trusted SOPs, policy notes, and measure definitions, then answer only from cited document snippets.
            </p>
          </div>
        </div>

        <div className="mt-4 grid gap-3 lg:grid-cols-2">
          <Input
            type="file"
            accept=".txt,.md,.html,.htm,.json,.pdf"
            onChange={(event) => setSelectedDocumentFile(event.target.files?.[0] ?? null)}
          />
          <Input
            value={documentTitle}
            onChange={(event) => setDocumentTitle(event.target.value)}
            placeholder="Optional title override"
          />
          <Input
            value={documentSourceName}
            onChange={(event) => setDocumentSourceName(event.target.value)}
            placeholder="Optional source or policy group"
          />
          <Input
            value={documentVersionLabel}
            onChange={(event) => setDocumentVersionLabel(event.target.value)}
            placeholder="Version label, e.g. v2026.03"
          />
          <Input
            type="date"
            value={documentEffectiveDate}
            onChange={(event) => setDocumentEffectiveDate(event.target.value)}
          />
          <Input
            value={documentSupersedesId}
            onChange={(event) => setDocumentSupersedesId(event.target.value)}
            placeholder="Optional superseded document ID"
          />
          <Button
            type="button"
            variant="outline"
            onClick={() => void handleUploadDocument()}
            disabled={busyAction === 'document_upload' || !selectedDocumentFile || !canUploadDocuments}
          >
            {busyAction === 'document_upload' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Upload className="mr-2 h-4 w-4" />}
            Add Trusted Doc
          </Button>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <Badge variant="outline">Documents: {documents.length}</Badge>
          <Badge variant="outline">Readable: {canReadDocuments ? 'yes' : 'no'}</Badge>
          <Badge variant="outline">Upload enabled: {canUploadDocuments ? 'yes' : 'no'}</Badge>
        </div>

        {documents.length > 0 ? (
          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            {documents.slice(0, 6).map((document) => (
              <div key={document.document_id} className="rounded-xl border border-border bg-background px-4 py-3">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline">{document.file_type.toUpperCase()}</Badge>
                  <Badge variant="outline">{document.chunk_count} chunks</Badge>
                  <Badge variant="outline">{formatDocumentFreshnessLabel(document.freshness)}</Badge>
                  {document.version_label ? <Badge variant="outline">{document.version_label}</Badge> : null}
                </div>
                <p className="mt-2 text-sm font-medium text-foreground">{document.title}</p>
                <p className="mt-1 text-xs text-muted-foreground">{document.source_name}</p>
                <p className="mt-1 text-xs text-muted-foreground break-all">
                  ID: {document.document_id}
                  {document.effective_date ? ` · effective ${document.effective_date}` : ''}
                </p>
                {document.freshness_note ? (
                  <p className="mt-1 text-xs text-amber-400">{document.freshness_note}</p>
                ) : null}
                {document.snippet_preview ? (
                  <p className="mt-2 text-xs text-muted-foreground">{document.snippet_preview}</p>
                ) : null}
              </div>
            ))}
          </div>
        ) : (
          <div className="mt-4 rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
            No trusted documents uploaded yet. This slice supports TXT, MD, HTML, JSON, and text-based PDF sources.
          </div>
        )}

        <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_auto]">
          <Textarea
            value={documentQuestion}
            onChange={(event) => setDocumentQuestion(event.target.value)}
            placeholder="Ask a source-backed document question..."
            className="min-h-[110px]"
          />
          <Button type="button" onClick={() => void handleAskDocuments()} disabled={busyAction === 'document_ask' || !canReadDocuments}>
            {busyAction === 'document_ask' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Bot className="mr-2 h-4 w-4" />}
            Ask Docs
          </Button>
        </div>

        {documentAnswer ? (
          <div className="mt-4 space-y-4">
            <div className="rounded-xl border border-border bg-background p-4">
              <p className="text-sm text-foreground">{documentAnswer.answer}</p>
              {documentAnswer.freshness_summary ? (
                <p className="mt-2 text-xs text-amber-400">{documentAnswer.freshness_summary}</p>
              ) : null}
              <div className="mt-3 flex flex-wrap gap-2">
                <Badge variant="outline">Grounded: {documentAnswer.grounded ? 'yes' : 'no'}</Badge>
                <Badge variant="outline">Confidence: {documentAnswer.confidence}</Badge>
                <Badge variant="outline">Citations: {documentAnswer.citations.length}</Badge>
              </div>
              <div className="mt-4">
                {renderFeedbackActions('document_qa', 'document_qa', {
                  question: documentQuestion,
                  title: 'Trusted document answer',
                })}
              </div>
            </div>
            {documentAnswer.citations.length > 0 ? (
              <div className="grid gap-3 lg:grid-cols-2">
                {documentAnswer.citations.map((citation) => (
                  <div
                    key={citation.citation_key}
                    className="rounded-xl border border-border bg-background px-4 py-3"
                  >
                    <p className="text-sm font-medium text-foreground">{citation.title}</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {citation.source_name} · chunk {citation.chunk_index + 1}
                      {citation.effective_date ? ` · effective ${citation.effective_date}` : ''}
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <Badge variant="outline">{formatDocumentFreshnessLabel(citation.freshness)}</Badge>
                      {citation.version_label ? <Badge variant="outline">{citation.version_label}</Badge> : null}
                    </div>
                    <p className="mt-2 text-sm text-muted-foreground">{citation.snippet}</p>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="rounded-2xl border border-border bg-card p-4 sm:p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex items-center gap-2">
            <div className="rounded-xl bg-health-mint/20 p-2 text-health-mint">
              <Shield className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-lg font-medium text-foreground">Governed Workflow Assistant</h3>
              <p className="text-sm text-muted-foreground">
                Draft operational follow-ups, require reviewer approval, and record execution without bypassing governance controls.
              </p>
            </div>
          </div>
          <Button
            type="button"
            onClick={() => void handleDraftWorkflow()}
            disabled={busyAction === 'workflow_draft' || !datasetId || !canDraftWorkflow}
          >
            {busyAction === 'workflow_draft' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Shield className="mr-2 h-4 w-4" />}
            Draft Action
          </Button>
        </div>

        <div className="mt-4 grid gap-3 lg:grid-cols-[220px_1fr_1fr]">
          <select
            value={workflowActionType}
            onChange={(event) => setWorkflowActionType(event.target.value as (typeof WORKFLOW_ACTION_OPTIONS)[number]['value'])}
            className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm"
          >
            {WORKFLOW_ACTION_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <Input
            value={workflowTitle}
            onChange={(event) => setWorkflowTitle(event.target.value)}
            placeholder="Optional title override"
          />
          <Input
            value={workflowTarget}
            onChange={(event) => setWorkflowTarget(event.target.value)}
            placeholder="Target team, owner, or audience"
          />
        </div>

        <div className="mt-3 grid gap-3 lg:grid-cols-[1.4fr_1fr]">
          <Textarea
            value={workflowObjective}
            onChange={(event) => setWorkflowObjective(event.target.value)}
            placeholder="Describe the operational objective for this workflow draft."
            className="min-h-[110px]"
          />
          <Textarea
            value={workflowDecisionNote}
            onChange={(event) => setWorkflowDecisionNote(event.target.value)}
            placeholder="Optional approval or execution note."
            className="min-h-[110px]"
          />
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          <Badge variant="outline">Draft enabled: {canDraftWorkflow ? 'yes' : 'no'}</Badge>
          <Badge variant="outline">Review enabled: {canReviewWorkflow ? 'yes' : 'no'}</Badge>
          <Badge variant="outline">Execute enabled: {canExecuteWorkflow ? 'yes' : 'no'}</Badge>
          <Badge variant="outline">Saved actions: {workflowActions.length}</Badge>
        </div>

        {workflowActions.length === 0 ? (
          <div className="mt-4 rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
            No governed workflow drafts yet. Start with a draft email, investigation ticket, action plan, or report schedule request.
          </div>
        ) : (
          <div className="mt-4 grid gap-4 xl:grid-cols-2">
            {workflowActions.map((action) => {
              const actionLabel =
                WORKFLOW_ACTION_OPTIONS.find((option) => option.value === action.action_type)?.label ?? action.action_type;
              const canApproveThis = canReviewWorkflow && action.status === 'pending_approval';
              const canExecuteThis = canExecuteWorkflow && action.status === 'approved';
              const busyApprove = busyAction === `workflow_approve:${action.action_id}`;
              const busyReject = busyAction === `workflow_reject:${action.action_id}`;
              const busyExecute = busyAction === `workflow_execute:${action.action_id}`;

              return (
                <div key={action.action_id} className="rounded-xl border border-border bg-background p-4">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <p className="text-sm font-medium text-foreground">{action.title}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{action.summary}</p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="outline">{actionLabel}</Badge>
                      <Badge variant="outline">{action.status}</Badge>
                    </div>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-2">
                    {action.target ? <Badge variant="outline">Target: {action.target}</Badge> : null}
                    {action.reviewed_by ? <Badge variant="outline">Reviewed by: {action.reviewed_by}</Badge> : null}
                    {action.executed_by ? <Badge variant="outline">Executed by: {action.executed_by}</Badge> : null}
                  </div>

                  {action.evidence.length > 0 ? (
                    <div className="mt-3 space-y-2">
                      {action.evidence.map((item, index) => (
                        <div key={`${action.action_id}-evidence-${index}`} className="rounded-lg border border-border/70 bg-card px-3 py-2 text-xs text-muted-foreground">
                          {item}
                        </div>
                      ))}
                    </div>
                  ) : null}

                  <div className="mt-3 rounded-xl border border-border/70 bg-card p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Draft payload</p>
                    <pre className="mt-2 overflow-auto text-xs text-muted-foreground">
                      {JSON.stringify(action.payload, null, 2)}
                    </pre>
                  </div>

                  {action.review_note ? (
                    <p className="mt-3 text-xs text-muted-foreground">Review note: {action.review_note}</p>
                  ) : null}
                  {action.execution_note ? (
                    <p className="mt-2 text-xs text-muted-foreground">Execution note: {action.execution_note}</p>
                  ) : null}

                  <div className="mt-4 flex flex-wrap gap-2">
                    {canApproveThis ? (
                      <>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => void handleReviewWorkflow(action.action_id, true)}
                          disabled={busyApprove}
                        >
                          {busyApprove ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                          Approve
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => void handleReviewWorkflow(action.action_id, false)}
                          disabled={busyReject}
                        >
                          {busyReject ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                          Reject
                        </Button>
                      </>
                    ) : null}

                    {canExecuteThis ? (
                      <Button
                        type="button"
                        size="sm"
                        onClick={() => void handleExecuteWorkflow(action.action_id)}
                        disabled={busyExecute}
                      >
                        {busyExecute ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                        Execute
                      </Button>
                    ) : null}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base">Facts Overview</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {!hasFacts ? <p className="text-sm text-muted-foreground">Generate facts to populate this panel.</p> : null}

            {hasFacts ? (
              <>
                <div className="flex flex-wrap gap-2">
                  <Badge variant="outline">
                    Mode: {typeof dataCoverage?.mode === 'string' ? dataCoverage.mode : 'n/a'}
                  </Badge>
                  <Badge variant="outline">
                    Rows used: {String(dataCoverage?.rows_used ?? 'n/a')}
                  </Badge>
                  <Badge variant="outline">
                    Rows total: {String(dataCoverage?.rows_total ?? 'n/a')}
                  </Badge>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  {kpis.map((kpi, index) => {
                    const item = kpi as { name?: unknown; value?: unknown; unit?: unknown };
                    return (
                      <div key={`${String(item.name ?? 'kpi')}-${index}`} className="rounded-xl border border-border bg-background p-3">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">{String(item.name ?? 'KPI')}</p>
                        <p className="mt-1 text-base font-semibold text-foreground">{String(item.value ?? 'n/a')}</p>
                        <p className="text-xs text-muted-foreground">{String(item.unit ?? '')}</p>
                      </div>
                    );
                  })}
                </div>

                <div>
                  <p className="text-sm font-medium text-foreground">Quality Issues</p>
                  <div className="mt-2 space-y-2">
                    {qualityIssues.length === 0 ? (
                      <p className="text-sm text-muted-foreground">No major quality issues were surfaced in the facts bundle.</p>
                    ) : (
                      qualityIssues.map((issue, index) => (
                        <div key={index} className="rounded-xl border border-border bg-background px-3 py-2 text-sm text-foreground">
                          {JSON.stringify(issue)}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </>
            ) : null}
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
              <div>
                <CardTitle className="text-base">Dashboard Spec</CardTitle>
                <p className="mt-1 text-sm text-muted-foreground">
                  Inspect governed chart blueprints and run explanation flows against approved data.
                </p>
              </div>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void handleExplainDashboard()}
                disabled={!hasDashboardSpec || busyAction === 'dashboard_explanation' || !canCompute}
              >
                {busyAction === 'dashboard_explanation' ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <BarChart3 className="mr-2 h-4 w-4" />
                )}
                Explain Dashboard
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {!hasDashboardSpec ? (
              <p className="text-sm text-muted-foreground">Generate a dashboard spec to inspect governed chart layout.</p>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  <Badge variant="outline">Title: {String(dashboardSpec?.title ?? 'Untitled')}</Badge>
                  <Badge variant="outline">
                    Charts:{' '}
                    {Array.isArray(dashboardSpec?.charts) ? dashboardSpec.charts.length : Array.isArray(dashboardSpec?.components) ? dashboardSpec.components.length : 0}
                  </Badge>
                  <Badge variant="outline">
                    Filters: {Array.isArray(dashboardSpec?.filters) ? dashboardSpec.filters.length : 0}
                  </Badge>
                </div>

                {dashboardCharts.length > 0 ? (
                  <div className="grid gap-3 lg:grid-cols-2">
                    {dashboardCharts.map((chart) => {
                      const isSelected = selectedChartKey === chart.key;
                      const explanation = chartExplanations[chart.key];

                      return (
                        <div
                          key={chart.key}
                          className={`rounded-xl border p-4 transition-colors ${
                            isSelected ? 'border-health-mint/40 bg-health-mint/5' : 'border-border bg-background'
                          }`}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="text-sm font-medium text-foreground">{chart.title}</p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                {chart.type} chart
                                {chart.layout?.row !== undefined && chart.layout?.col !== undefined
                                  ? ` · layout ${String(chart.layout.row)},${String(chart.layout.col)}`
                                  : ''}
                              </p>
                            </div>
                            <Badge variant="outline">{chart.type}</Badge>
                          </div>

                          <div className="mt-3 flex flex-wrap gap-2">
                            {chart.x ? <Badge variant="outline">X: {chart.x}</Badge> : null}
                            {chart.y ? <Badge variant="outline">Y: {chart.y}</Badge> : null}
                            {chart.groupBy ? <Badge variant="outline">Group: {chart.groupBy}</Badge> : null}
                            {chart.aggregation ? <Badge variant="outline">Agg: {chart.aggregation}</Badge> : null}
                            <Badge variant="outline">Facts: {chart.factIds.length}</Badge>
                          </div>

                          {chart.factIds.length > 0 ? (
                            <div className="mt-3 space-y-2">
                              {chart.factIds.slice(0, 2).map((factId) => (
                                <div key={factId} className="rounded-lg border border-border/70 bg-card px-3 py-2 text-xs text-muted-foreground">
                                  {factReferenceLookup[factId] ?? factId}
                                </div>
                              ))}
                            </div>
                          ) : null}

                          <div className="mt-4 flex flex-wrap gap-2">
                            {EXPLAIN_PRESETS.map((preset) => {
                              const buttonBusyKey = `chart_explanation:${chart.key}:${preset.id}`;
                              return (
                                <Button
                                  key={preset.id}
                                  type="button"
                                  size="sm"
                                  variant={explanation?.preset === preset.id ? 'default' : 'outline'}
                                  onClick={() => void handleExplainChart(chart, preset.id)}
                                  disabled={busyAction === buttonBusyKey || !canCompute}
                                >
                                  {busyAction === buttonBusyKey ? (
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                  ) : null}
                                  {preset.label}
                                </Button>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : null}

                {dashboardExplanation ? (
                  <div className="rounded-xl border border-border bg-background p-4">
                    <div className="mb-3">
                      <p className="text-sm font-medium text-foreground">Dashboard Narrative</p>
                      <p className="mt-1 text-xs text-muted-foreground">{dashboardExplanation.question}</p>
                    </div>
                    <AskResponseInspector
                      result={dashboardExplanation.result}
                      title="Dashboard result rows"
                      emptyRowsMessage="No tabular rows were returned for the dashboard summary."
                    />
                    <div className="mt-4">
                      {renderFeedbackActions('dashboard_summary', 'dashboard_summary', {
                        question: dashboardExplanation.question,
                        title: 'Dashboard narrative',
                      })}
                    </div>
                  </div>
                ) : null}

                {selectedChartKey && activeChartExplanation ? (
                  <div className="rounded-xl border border-border bg-background p-4">
                    <div className="mb-3">
                      <p className="text-sm font-medium text-foreground">Selected Chart Explanation</p>
                      <p className="mt-1 text-xs text-muted-foreground">{activeChartExplanation.question}</p>
                    </div>
                    <AskResponseInspector
                      result={activeChartExplanation.result}
                      title="Chart result rows"
                      emptyRowsMessage="No tabular rows were returned for this chart explanation."
                    />
                    <div className="mt-4">
                      {renderFeedbackActions('chart_explanation', selectedChartKey, {
                        question: activeChartExplanation.question,
                        title: 'Chart explanation',
                      })}
                    </div>
                  </div>
                ) : null}

                <pre className="overflow-auto rounded-xl border border-border bg-background p-3 text-xs text-muted-foreground">
                  {JSON.stringify(dashboardSpec, null, 2)}
                </pre>
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      <section className="rounded-2xl border border-border bg-card p-4 sm:p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex items-center gap-2">
            <div className="rounded-xl bg-health-mint/20 p-2 text-health-mint">
              <AlertCircle className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-lg font-medium text-foreground">Anomaly and Root-Cause Scan</h3>
              <p className="text-sm text-muted-foreground">
                Detect governed quality, segment, and time-based anomalies, then investigate them with auditable follow-up questions.
              </p>
            </div>
          </div>
          <Button
            type="button"
            variant="outline"
            onClick={() => void handleDetectAnomalies()}
            disabled={isBusy('anomalies') || !datasetId || !canCompute}
          >
            {isBusy('anomalies') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
            Refresh Anomaly Scan
          </Button>
        </div>

        {!anomalyAnalysis ? (
          <div className="mt-4 rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
            Run anomaly detection to surface governed quality flags, segment outliers, and time-based spikes or dips.
          </div>
        ) : (
          <div className="mt-4 space-y-4">
            <div className="rounded-xl border border-border bg-background p-4">
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">Anomalies: {anomalyAnalysis.analysis.anomaly_count}</Badge>
                <Badge variant="outline">Generated: {new Date(anomalyAnalysis.analysis.generated_at).toLocaleString()}</Badge>
              </div>
              <p className="mt-3 text-sm text-foreground">{anomalyAnalysis.analysis.summary}</p>
            </div>

            {anomalyFindings.length > 0 ? (
              <div className="grid gap-3 lg:grid-cols-2">
                {anomalyFindings.map((finding) => (
                  <div key={finding.anomaly_id} className="rounded-xl border border-border bg-background p-4">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <p className="text-sm font-medium text-foreground">{finding.title}</p>
                        <p className="mt-1 text-xs text-muted-foreground">{finding.summary}</p>
                      </div>
                      <div className="flex flex-wrap gap-2">
                        <Badge variant="outline">{finding.kind}</Badge>
                        <Badge variant="outline">{finding.severity}</Badge>
                      </div>
                    </div>

                    <div className="mt-3 flex flex-wrap gap-2">
                      {finding.metric ? <Badge variant="outline">Metric: {finding.metric}</Badge> : null}
                      {finding.dimension ? <Badge variant="outline">Dimension: {finding.dimension}</Badge> : null}
                      {finding.segment ? <Badge variant="outline">Segment: {finding.segment}</Badge> : null}
                      {finding.period ? <Badge variant="outline">Period: {finding.period}</Badge> : null}
                    </div>

                    {finding.evidence.length > 0 ? (
                      <div className="mt-3 space-y-2">
                        {finding.evidence.slice(0, 3).map((evidence, index) => (
                          <div key={`${finding.anomaly_id}-evidence-${index}`} className="rounded-lg border border-border/70 bg-card px-3 py-2 text-xs text-muted-foreground">
                            {evidence}
                          </div>
                        ))}
                      </div>
                    ) : null}

                    {finding.root_cause_hints.length > 0 ? (
                      <div className="mt-3 rounded-lg border border-health-mint/20 bg-health-mint/10 px-3 py-2">
                        <p className="text-xs font-medium uppercase tracking-wide text-health-mint">Root-cause hints</p>
                        <div className="mt-2 space-y-1">
                          {finding.root_cause_hints.slice(0, 3).map((hint, index) => (
                            <p key={`${finding.anomaly_id}-hint-${index}`} className="text-xs text-foreground">
                              {hint}
                            </p>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    {finding.recommended_question ? (
                      <div className="mt-4 flex flex-wrap gap-2">
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => void handleInvestigateQuestion(finding.recommended_question!, `${finding.title} investigation generated successfully.`)}
                          disabled={busyAction === 'investigate' || !canCompute}
                        >
                          {busyAction === 'investigate' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Bot className="mr-2 h-4 w-4" />}
                          Investigate
                        </Button>
                      </div>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
                No high-signal anomaly findings were returned for this dataset.
              </div>
            )}

            {anomalySuggestedQuestions.length > 0 ? (
              <div className="rounded-xl border border-border bg-background p-4">
                <p className="text-sm font-medium text-foreground">Suggested follow-up questions</p>
                <div className="mt-3 flex flex-wrap gap-2">
                  {anomalySuggestedQuestions.map((question) => (
                    <Button
                      key={question}
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={() => void handleInvestigateQuestion(question, 'Anomaly follow-up generated successfully.')}
                      disabled={busyAction === 'investigate' || !canCompute}
                    >
                      {question}
                    </Button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-border bg-card p-4 sm:p-6">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex items-center gap-2">
            <div className="rounded-xl bg-health-mint/20 p-2 text-health-mint">
              <BarChart3 className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-lg font-medium text-foreground">Governed Forecast Lab</h3>
              <p className="text-sm text-muted-foreground">
                Train audited forecast runs on governed time-series fields, compare candidate models, and monitor drift after data refreshes.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={() => void handleEvaluateModels()}
              disabled={busyAction === 'model_evaluation' || !activeRegistryEntry || forecastRuns.length < 2 || !datasetId || !canCompute}
            >
              {busyAction === 'model_evaluation' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <BadgeCheck className="mr-2 h-4 w-4" />}
              Evaluate Models
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={() => void handleScanForecastDrift()}
              disabled={busyAction === 'forecast_drift' || forecastRuns.length === 0 || !datasetId || !canCompute}
            >
              {busyAction === 'forecast_drift' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
              Scan Drift
            </Button>
          </div>
        </div>

        <div className="mt-4 grid gap-3 xl:grid-cols-[1.2fr_1fr_1fr_160px_150px_auto]">
          <Input
            value={forecastName}
            onChange={(event) => setForecastName(event.target.value)}
            placeholder="Optional run name"
          />
          <select
            value={forecastTimeField}
            onChange={(event) => setForecastTimeField(event.target.value)}
            className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm"
            disabled={forecastTimeOptions.length === 0}
          >
            {forecastTimeOptions.length === 0 ? <option value="">Load profile first</option> : null}
            {forecastTimeOptions.map((field) => (
              <option key={field} value={field}>
                {field}
              </option>
            ))}
          </select>
          <select
            value={forecastMetricField}
            onChange={(event) => setForecastMetricField(event.target.value)}
            className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm"
            disabled={forecastMetricOptions.length === 0}
          >
            {forecastMetricOptions.length === 0 ? <option value="">Load profile first</option> : null}
            {forecastMetricOptions.map((field) => (
              <option key={field} value={field}>
                {field}
              </option>
            ))}
          </select>
          <select
            value={forecastAggregation}
            onChange={(event) => setForecastAggregation(event.target.value as 'sum' | 'mean')}
            className="h-10 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-sm"
          >
            <option value="sum">Monthly sum</option>
            <option value="mean">Monthly mean</option>
          </select>
          <Input
            value={forecastHorizon}
            onChange={(event) => setForecastHorizon(event.target.value)}
            inputMode="numeric"
            placeholder="Horizon"
          />
          <Button type="button" onClick={() => void handleTrainForecast()} disabled={busyAction === 'forecast' || busyAction === 'forecast_profile' || !datasetId || !canCompute}>
            {busyAction === 'forecast' || busyAction === 'forecast_profile' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
            Train Forecast
          </Button>
        </div>

        {forecastTimeOptions.length === 0 || forecastMetricOptions.length === 0 ? (
          <div className="mt-4 rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
            Load a governed profile with at least one datetime field and one numeric metric to enable forecast training.
          </div>
        ) : null}

        {canPromoteModel ? (
          <div className="mt-4 rounded-xl border border-border bg-background p-4">
            <p className="text-sm font-medium text-foreground">Registry promotion note</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Reviewer promotion is required before a forecast run becomes the active governed model for this session.
            </p>
            <Textarea
              value={modelPromotionNote}
              onChange={(event) => setModelPromotionNote(event.target.value)}
              placeholder="Optional reviewer note about why this run is being promoted."
              className="mt-3 min-h-[90px]"
            />
          </div>
        ) : null}

        {latestForecastRun ? (
          <div className="mt-4 space-y-4">
            <div className="rounded-xl border border-border bg-background p-4">
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline">Champion: {latestForecastRun.payload.champion_model}</Badge>
                <Badge variant="outline">Metric: {latestForecastRun.payload.metric_field}</Badge>
                <Badge variant="outline">Periods used: {latestForecastRun.payload.periods_used}</Badge>
                <Badge variant="outline">Horizon: {latestForecastRun.payload.horizon}</Badge>
                {latestForecastRun.payload.training_data_hash && sessionMeta?.file_hash && latestForecastRun.payload.training_data_hash !== sessionMeta.file_hash ? (
                  <Badge variant="outline">Stale data</Badge>
                ) : null}
              </div>
              <p className="mt-3 text-sm text-foreground">{latestForecastRun.payload.summary}</p>
              {latestForecastRun.payload.warnings.length > 0 ? (
                <div className="mt-3 space-y-2">
                  {latestForecastRun.payload.warnings.map((warning) => (
                    <div key={warning} className="rounded-lg border border-health-mint/20 bg-health-mint/10 px-3 py-2 text-xs text-foreground">
                      {warning}
                    </div>
                  ))}
                </div>
              ) : null}
            </div>

            <div className="rounded-xl border border-border bg-background p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-foreground">Governed model registry</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Active production models are reviewer-promoted and remain visible across data refreshes.
                  </p>
                </div>
                {activeRegistryEntry ? <Badge variant="outline">Active run: {activeRegistryEntry.run_id.slice(0, 8)}</Badge> : <Badge variant="outline">No active model</Badge>}
              </div>
              {activeRegistryEntry ? (
                <div className="mt-3 space-y-3">
                  <div className="flex flex-wrap gap-2">
                    <Badge variant="outline">{activeRegistryEntry.name}</Badge>
                    <Badge variant="outline">{activeRegistryEntry.champion_model}</Badge>
                    <Badge variant="outline">{activeRegistryEntry.metric_field}</Badge>
                    {activeRegistryEntry.source_data_hash && sessionMeta?.file_hash && activeRegistryEntry.source_data_hash !== sessionMeta.file_hash ? (
                      <Badge variant="outline">Needs retrain</Badge>
                    ) : null}
                  </div>
                  {activeRegistryEntry.note ? (
                    <div className="rounded-lg border border-border/70 bg-card px-3 py-2 text-xs text-foreground">
                      {activeRegistryEntry.note}
                    </div>
                  ) : null}
                  <p className="text-xs text-muted-foreground">
                    Promoted by {activeRegistryEntry.promoted_by} on {new Date(activeRegistryEntry.promoted_at).toLocaleString()}.
                  </p>
                </div>
              ) : (
                <p className="mt-3 text-sm text-muted-foreground">Promote a reviewed forecast run to create the active governed model entry.</p>
              )}
            </div>

            {modelEvaluation ? (
              <div className="rounded-xl border border-border bg-background p-4">
                <div className="flex flex-wrap gap-2">
                  <Badge variant="outline">Winner: {modelEvaluation.evaluation.winner}</Badge>
                  <Badge variant="outline">Active MAE: {modelEvaluation.evaluation.active_run.mae.toFixed(2)}</Badge>
                  <Badge variant="outline">Challenger MAE: {modelEvaluation.evaluation.challenger_run.mae.toFixed(2)}</Badge>
                </div>
                <p className="mt-3 text-sm text-foreground">{modelEvaluation.evaluation.recommendation}</p>
                <div className="mt-4 grid gap-4 xl:grid-cols-2">
                  <div className="rounded-lg border border-border/70 bg-card p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Rationale</p>
                    <div className="mt-2 space-y-2">
                      {modelEvaluation.evaluation.rationale.map((item) => (
                        <div key={item} className="rounded-md border border-border/60 px-3 py-2 text-xs text-foreground">
                          {item}
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-lg border border-border/70 bg-card p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Suggested actions</p>
                    <div className="mt-2 space-y-2">
                      {modelEvaluation.evaluation.suggested_actions.map((item) => (
                        <div key={item} className="rounded-md border border-border/60 px-3 py-2 text-xs text-foreground">
                          {item}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : null}

            {forecastDrift ? (
              <div className="rounded-xl border border-border bg-background p-4">
                <div className="flex flex-wrap gap-2">
                  <Badge variant="outline">Drift score: {forecastDrift.drift.drift_score.toFixed(2)}</Badge>
                  <Badge variant="outline">Window: {forecastDrift.drift.periods_analyzed} periods</Badge>
                  <Badge variant="outline">Stale model: {forecastDrift.drift.stale_model ? 'yes' : 'no'}</Badge>
                </div>
                <p className="mt-3 text-sm text-foreground">{forecastDrift.drift.summary}</p>
                <div className="mt-4 grid gap-4 xl:grid-cols-2">
                  <div className="rounded-lg border border-border/70 bg-card p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Signals</p>
                    {forecastDrift.drift.signals.length === 0 ? (
                      <p className="mt-2 text-xs text-muted-foreground">No material drift signals were detected.</p>
                    ) : (
                      <div className="mt-2 space-y-2">
                        {forecastDrift.drift.signals.map((signal) => (
                          <div key={`${signal.code}-${signal.message}`} className="rounded-md border border-border/60 px-3 py-2 text-xs text-foreground">
                            <span className="font-medium">{signal.code}</span>: {signal.message}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="rounded-lg border border-border/70 bg-card p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Recommended actions</p>
                    <div className="mt-2 space-y-2">
                      {forecastDrift.drift.recommended_actions.map((action) => (
                        <div key={action} className="rounded-md border border-border/60 px-3 py-2 text-xs text-foreground">
                          {action}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : null}

            <div className="grid gap-4 xl:grid-cols-2">
              <div className="rounded-xl border border-border bg-background p-4">
                <p className="text-sm font-medium text-foreground">Candidate model comparison</p>
                <div className="mt-3 overflow-x-auto">
                  <table className="min-w-full text-left text-xs">
                    <thead className="text-muted-foreground">
                      <tr>
                        <th className="pb-2 pr-3 font-medium">Model</th>
                        <th className="pb-2 pr-3 font-medium">MAE</th>
                        <th className="pb-2 pr-3 font-medium">RMSE</th>
                        <th className="pb-2 pr-3 font-medium">MAPE</th>
                      </tr>
                    </thead>
                    <tbody className="text-foreground">
                      {latestForecastRun.payload.candidate_models.map((candidate) => (
                        <tr key={candidate.model_name} className="border-t border-border/60">
                          <td className="py-2 pr-3">
                            <div className="flex items-center gap-2">
                              <span>{candidate.model_name}</span>
                              {candidate.model_name === latestForecastRun.payload.champion_model ? <Badge variant="outline">Champion</Badge> : null}
                            </div>
                          </td>
                          <td className="py-2 pr-3">{candidate.mae.toFixed(2)}</td>
                          <td className="py-2 pr-3">{candidate.rmse.toFixed(2)}</td>
                          <td className="py-2 pr-3">{candidate.mape !== null && candidate.mape !== undefined ? `${candidate.mape.toFixed(2)}%` : 'n/a'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-xl border border-border bg-background p-4">
                <p className="text-sm font-medium text-foreground">Forecast output</p>
                <div className="mt-3 overflow-x-auto">
                  <table className="min-w-full text-left text-xs">
                    <thead className="text-muted-foreground">
                      <tr>
                        <th className="pb-2 pr-3 font-medium">Period</th>
                        <th className="pb-2 pr-3 font-medium">Projected value</th>
                      </tr>
                    </thead>
                    <tbody className="text-foreground">
                      {latestForecastRun.payload.forecast.map((point) => (
                        <tr key={point.period} className="border-t border-border/60">
                          <td className="py-2 pr-3">{point.period}</td>
                          <td className="py-2 pr-3">{point.value.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="grid gap-4 xl:grid-cols-2">
              <div className="rounded-xl border border-border bg-background p-4">
                <p className="text-sm font-medium text-foreground">Recent historical periods</p>
                <div className="mt-3 overflow-x-auto">
                  <table className="min-w-full text-left text-xs">
                    <thead className="text-muted-foreground">
                      <tr>
                        <th className="pb-2 pr-3 font-medium">Period</th>
                        <th className="pb-2 pr-3 font-medium">Actual value</th>
                      </tr>
                    </thead>
                    <tbody className="text-foreground">
                      {latestForecastRun.payload.historical.slice(-8).map((point) => (
                        <tr key={point.period} className="border-t border-border/60">
                          <td className="py-2 pr-3">{point.period}</td>
                          <td className="py-2 pr-3">{point.value.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="rounded-xl border border-border bg-background p-4">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-medium text-foreground">Saved model runs</p>
                  <Badge variant="outline">{forecastRuns.length}</Badge>
                </div>
                {forecastRuns.length === 0 ? (
                  <p className="mt-3 text-sm text-muted-foreground">No forecast runs have been saved for this session yet.</p>
                ) : (
                  <div className="mt-3 space-y-3">
                    {forecastRuns.slice(0, 5).map((run) => (
                      <div key={run.run_id} className="rounded-lg border border-border/70 bg-card p-3">
                        <div className="flex flex-wrap items-start justify-between gap-2">
                          <div>
                            <p className="text-sm font-medium text-foreground">{run.payload.name}</p>
                            <p className="mt-1 text-xs text-muted-foreground">{run.payload.summary}</p>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            {run.payload.training_data_hash && sessionMeta?.file_hash && run.payload.training_data_hash !== sessionMeta.file_hash ? (
                              <Badge variant="outline">Stale</Badge>
                            ) : null}
                            <Badge variant="outline">{run.payload.champion_model}</Badge>
                          </div>
                        </div>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <Badge variant="outline">{run.payload.metric_field}</Badge>
                          <Badge variant="outline">{run.payload.aggregation}</Badge>
                          <Badge variant="outline">{new Date(run.created_at).toLocaleString()}</Badge>
                          {canPromoteModel ? (
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            onClick={() => void handlePromoteModel(run.run_id)}
                              disabled={
                                busyAction === `promote_model:${run.run_id}` ||
                                (activeRegistryEntry?.run_id === run.run_id) ||
                                (Boolean(run.payload.training_data_hash) && Boolean(sessionMeta?.file_hash) && run.payload.training_data_hash !== sessionMeta?.file_hash)
                              }
                            >
                              {busyAction === `promote_model:${run.run_id}` ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <BadgeCheck className="mr-2 h-4 w-4" />}
                              {activeRegistryEntry?.run_id === run.run_id ? 'Active' : 'Promote'}
                            </Button>
                          ) : null}
                          {activeRegistryEntry?.run_id !== run.run_id ? (
                            <Button
                              type="button"
                              size="sm"
                              variant="outline"
                              onClick={() => void handleEvaluateModels(run.run_id)}
                              disabled={busyAction === 'model_evaluation' || !activeRegistryEntry || !canCompute}
                            >
                              Evaluate
                            </Button>
                          ) : null}
                          <Button
                            type="button"
                            size="sm"
                            variant="outline"
                            onClick={() => void handleScanForecastDrift(run.run_id)}
                            disabled={busyAction === 'forecast_drift' || !canCompute}
                          >
                            Scan Drift
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="mt-4 rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
            No governed forecast runs have been trained for this session yet.
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-border bg-card p-4 sm:p-6">
        <div className="flex items-center gap-2">
          <div className="rounded-xl bg-health-mint/20 p-2 text-health-mint">
            <Sparkles className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-lg font-medium text-foreground">Ask Your Data</h3>
            <p className="text-sm text-muted-foreground">Runs through the governed semantic query-plan path.</p>
          </div>
        </div>

        <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_auto]">
          <Textarea
            value={askQuestion}
            onChange={(event) => setAskQuestion(event.target.value)}
            placeholder="Ask a governed analytics question..."
            className="min-h-[110px]"
          />
          <Button type="button" onClick={() => void handleAsk()} disabled={isBusy('ask') || !datasetId || !canCompute}>
            {isBusy('ask') ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Bot className="mr-2 h-4 w-4" />}
            Ask Data
          </Button>
        </div>

        {askResult ? (
          <div className="mt-4 space-y-4">
            <AskResponseInspector
              result={askResult}
              title="Result Rows"
              emptyRowsMessage="No tabular rows were returned for this question."
            />
            {renderFeedbackActions('ask_data', 'current_ask', {
              question: askQuestion,
              title: 'Ask Your Data response',
            })}

            <div className="grid gap-4 xl:grid-cols-2">
              <div className="rounded-xl border border-border bg-background p-4">
                <p className="text-sm font-medium text-foreground">Save Investigation</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Capture this governed result with its question and inspector payload so it can be reopened later.
                </p>
                <div className="mt-3 space-y-3">
                  <Input
                    value={investigationTitle}
                    onChange={(event) => setInvestigationTitle(event.target.value)}
                    placeholder="Optional investigation title"
                  />
                  <Textarea
                    value={investigationNote}
                    onChange={(event) => setInvestigationNote(event.target.value)}
                    placeholder="Optional note about why this investigation matters."
                    className="min-h-[90px]"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => void handleSaveInvestigation()}
                    disabled={busyAction === 'save_investigation' || !canWriteSession}
                  >
                    {busyAction === 'save_investigation' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <FileText className="mr-2 h-4 w-4" />}
                    Save Investigation
                  </Button>
                </div>
              </div>

              <div className="rounded-xl border border-border bg-background p-4">
                <p className="text-sm font-medium text-foreground">Save Playbook</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Turn the current governed question into a reusable playbook template for future runs.
                </p>
                <div className="mt-3 space-y-3">
                  <Input
                    value={playbookName}
                    onChange={(event) => setPlaybookName(event.target.value)}
                    placeholder="Playbook name"
                  />
                  <Textarea
                    value={playbookDescription}
                    onChange={(event) => setPlaybookDescription(event.target.value)}
                    placeholder="Optional description of when to use this playbook."
                    className="min-h-[90px]"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => void handleSavePlaybook()}
                    disabled={busyAction === 'save_playbook' || !canWriteSession}
                  >
                    {busyAction === 'save_playbook' ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <BadgeCheck className="mr-2 h-4 w-4" />}
                    Save Playbook
                  </Button>
                </div>
              </div>
            </div>
          </div>
        ) : null}

        {(savedInvestigations.length > 0 || savedPlaybooks.length > 0) ? (
          <div className="mt-4 grid gap-4 xl:grid-cols-2">
            <div className="rounded-xl border border-border bg-background p-4">
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm font-medium text-foreground">Saved Investigations</p>
                <Badge variant="outline">{savedInvestigations.length}</Badge>
              </div>
              {savedInvestigations.length === 0 ? (
                <p className="mt-3 text-sm text-muted-foreground">No investigations saved for this session yet.</p>
              ) : (
                <div className="mt-3 space-y-3">
                  {savedInvestigations.slice(0, 6).map((item) => (
                    <div key={item.investigation_id} className="rounded-lg border border-border/70 bg-card p-3">
                      <p className="text-sm font-medium text-foreground">{item.title}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{item.question}</p>
                      {item.note ? <p className="mt-2 text-xs text-muted-foreground">{item.note}</p> : null}
                      <div className="mt-3 flex flex-wrap gap-2">
                        <Badge variant="outline">{item.context_type}</Badge>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            setAskQuestion(item.question);
                            setAskResult(item.result as unknown as AskResponsePayload);
                            setNotice(`Loaded saved investigation: ${item.title}.`);
                          }}
                        >
                          Load
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="rounded-xl border border-border bg-background p-4">
              <div className="flex items-center justify-between gap-2">
                <p className="text-sm font-medium text-foreground">Saved Playbooks</p>
                <Badge variant="outline">{savedPlaybooks.length}</Badge>
              </div>
              {savedPlaybooks.length === 0 ? (
                <p className="mt-3 text-sm text-muted-foreground">No playbooks saved for this session yet.</p>
              ) : (
                <div className="mt-3 space-y-3">
                  {savedPlaybooks.slice(0, 6).map((item) => (
                    <div key={item.playbook_id} className="rounded-lg border border-border/70 bg-card p-3">
                      <p className="text-sm font-medium text-foreground">{item.name}</p>
                      <p className="mt-1 text-xs text-muted-foreground">{item.question_template}</p>
                      {item.description ? <p className="mt-2 text-xs text-muted-foreground">{item.description}</p> : null}
                      <div className="mt-3 flex flex-wrap gap-2">
                        <Badge variant="outline">{item.context_type}</Badge>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => {
                            setAskQuestion(item.question_template);
                            setNotice(`Loaded playbook: ${item.name}.`);
                          }}
                        >
                          Load
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ) : null}
      </section>

      <section className="grid gap-4 xl:grid-cols-2">
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base">Exports</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {!datasetId ? <p className="text-sm text-muted-foreground">Create or load a session to enable exports.</p> : null}
            {datasetId ? (
              <div className="space-y-4">
                {profile ? (
                  <div
                    className={`rounded-xl border px-4 py-3 ${
                      piiCandidates.length > 0
                        ? maskedExportsActive
                          ? 'border-amber-500/30 bg-amber-500/10'
                          : 'border-red-500/30 bg-red-500/10'
                        : 'border-emerald-500/30 bg-emerald-500/10'
                    }`}
                  >
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="outline">
                        Approval: {sensitiveExportStatus}
                      </Badge>
                      <Badge variant="outline">
                        Mode: {maskedExportsActive ? 'masked' : sessionMeta?.allow_sensitive_export ? 'sensitive' : 'standard'}
                      </Badge>
                      <Badge variant="outline">
                        PII fields: {piiCandidates.length}
                      </Badge>
                    </div>
                    <p className="mt-3 text-sm text-foreground">
                      {piiCandidates.length > 0
                        ? maskedExportsActive
                          ? 'CSV and JSON exports are currently masked because potential PII was detected.'
                          : 'Sensitive export is enabled. CSV and JSON downloads may include raw PII field names and values.'
                        : 'No likely PII columns were detected in the current profile. Standard exports are available.'}
                    </p>
                    {piiCandidates.length > 0 ? (
                      <p className="mt-2 text-xs text-muted-foreground">
                        Candidate fields: {piiCandidates.join(', ')}
                      </p>
                    ) : null}
                    {sensitiveExportApproval?.justification ? (
                      <p className="mt-2 text-xs text-muted-foreground">
                        Request rationale: {sensitiveExportApproval.justification}
                      </p>
                    ) : null}
                    {sensitiveExportApproval?.review_note ? (
                      <p className="mt-2 text-xs text-muted-foreground">
                        Review note: {sensitiveExportApproval.review_note}
                      </p>
                    ) : null}
                  </div>
                ) : (
                  <div className="rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
                    Run profiling to evaluate PII exposure before requesting sensitive export approval.
                  </div>
                )}

                {piiCandidates.length > 0 ? (
                  <div className="rounded-xl border border-border bg-background p-4 space-y-3">
                    <p className="text-sm font-medium text-foreground">Sensitive Export Approval</p>
                    <Textarea
                      value={exportApprovalInput}
                      onChange={(event) => setExportApprovalInput(event.target.value)}
                      placeholder={
                        sensitiveExportStatus === 'pending'
                          ? 'Optional reviewer note before approving or rejecting...'
                          : 'Describe why unmasked export is necessary and how it will be handled safely.'
                      }
                      className="min-h-[96px]"
                    />

                    <div className="flex flex-wrap gap-2">
                      {canRequestSensitiveExport && sensitiveExportStatus !== 'pending' && !sessionMeta?.allow_sensitive_export ? (
                        <Button
                          type="button"
                          variant="outline"
                          onClick={() => void handleRequestSensitiveExportApproval()}
                          disabled={busyAction === 'request_sensitive_export'}
                        >
                          {busyAction === 'request_sensitive_export' ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <AlertCircle className="mr-2 h-4 w-4" />
                          )}
                          Request Approval
                        </Button>
                      ) : null}

                      {canReviewSensitiveExport && sensitiveExportStatus === 'pending' ? (
                        <>
                          <Button
                            type="button"
                            onClick={() => void handleReviewSensitiveExportApproval(true)}
                            disabled={busyAction === 'approve_sensitive_export'}
                          >
                            {busyAction === 'approve_sensitive_export' ? (
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            ) : (
                              <BadgeCheck className="mr-2 h-4 w-4" />
                            )}
                            Approve Request
                          </Button>
                          <Button
                            type="button"
                            variant="outline"
                            onClick={() => void handleReviewSensitiveExportApproval(false)}
                            disabled={busyAction === 'reject_sensitive_export'}
                          >
                            {busyAction === 'reject_sensitive_export' ? (
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            ) : (
                              <AlertCircle className="mr-2 h-4 w-4" />
                            )}
                            Reject Request
                          </Button>
                        </>
                      ) : null}

                      {sessionMeta?.allow_sensitive_export ? (
                        <Button
                          type="button"
                          variant="outline"
                          onClick={() => void handleDisableSensitiveExport()}
                          disabled={busyAction === 'disable_sensitive_export'}
                        >
                          {busyAction === 'disable_sensitive_export' ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          ) : (
                            <Shield className="mr-2 h-4 w-4" />
                          )}
                          Disable Sensitive Export
                        </Button>
                      ) : null}
                    </div>
                  </div>
                ) : null}

                <div className="flex flex-wrap gap-2">
                  <a href={factsJsonUrl(datasetId)} target="_blank" rel="noreferrer">
                    <Button type="button" variant="outline">
                      <Database className="mr-2 h-4 w-4" />
                      Facts JSON
                    </Button>
                  </a>
                  <a href={datasetCsvUrl(datasetId)} target="_blank" rel="noreferrer">
                    <Button type="button" variant="outline">
                      <FileSpreadsheet className="mr-2 h-4 w-4" />
                      CSV Export
                    </Button>
                  </a>
                  {reportReady ? (
                    <>
                      <a href={reportPdfUrl(datasetId)} target="_blank" rel="noreferrer">
                        <Button type="button" variant="outline">
                          <FileDown className="mr-2 h-4 w-4" />
                          Report PDF
                        </Button>
                      </a>
                      <a href={reportHtmlUrl(datasetId)} target="_blank" rel="noreferrer">
                        <Button type="button" variant="outline">
                          <FileDown className="mr-2 h-4 w-4" />
                          Report HTML
                        </Button>
                      </a>
                    </>
                  ) : (
                    <p className="text-sm text-muted-foreground">Report PDF will appear here after the report job succeeds.</p>
                  )}
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-base">Audit Trail</CardTitle>
          </CardHeader>
          <CardContent>
            {auditEvents.length === 0 ? (
              <p className="text-sm text-muted-foreground">No audit events yet for this session.</p>
            ) : (
              <div className="space-y-2">
                {auditEvents.slice(0, 8).map((event) => (
                  <div key={`${event.timestamp}-${event.action}`} className="rounded-xl border border-border bg-background px-3 py-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="outline">{event.action}</Badge>
                      <span className="text-xs text-muted-foreground">{event.actor}</span>
                      <span className="text-xs text-muted-foreground">{event.timestamp}</span>
                    </div>
                    {Object.keys(event.details ?? {}).length > 0 ? (
                      <p className="mt-2 text-xs text-muted-foreground">{JSON.stringify(event.details)}</p>
                    ) : null}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      <section className="rounded-2xl border border-border bg-card p-4 sm:p-6">
        <div className="flex items-start gap-3">
          <AlertCircle className="mt-0.5 h-5 w-5 text-health-mint" />
          <div>
            <h3 className="font-medium text-foreground">Current Phase</h3>
            <p className="mt-1 text-sm text-muted-foreground">
              This governed React surface now supports inspectable ask-data, dashboard summaries, chart-level
              explanations, and source-backed trusted document Q&amp;A. The rest of the product still needs migration
              away from the legacy browser-local analytics context.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

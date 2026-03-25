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
  RefreshCw,
  Shield,
  Sparkles,
  Upload,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Textarea } from '@/components/ui/textarea';
import {
  API_TARGET_LABEL,
  BACKEND_ROLE_STORAGE_KEY,
  BACKEND_USER_STORAGE_KEY,
  type AskResponsePayload,
  type DocumentAskResponse,
  type DocumentSummary,
  type AuthContextResponse,
  type BackendUserRole,
  type AuditEvent,
  type JobStatus,
  type SessionMeta,
  askDocuments,
  askDataset,
  createSession,
  datasetCsvUrl,
  factsJsonUrl,
  generateDashboardSpec,
  generateReport,
  getAudit,
  getAuthContext,
  getDashboardSpec,
  getFacts,
  getJobStatus,
  getProfile,
  getSession,
  inferBackendUserRole,
  listDocuments,
  requestSensitiveExportApproval,
  reportHtmlUrl,
  reportPdfUrl,
  reviewSensitiveExportApproval,
  setSensitiveExportEnabled,
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
  const [authContext, setAuthContext] = useState<AuthContextResponse | null>(null);
  const [usingFallbackAuthContext, setUsingFallbackAuthContext] = useState(false);
  const [sessionMeta, setSessionMeta] = useState<SessionMeta | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [profile, setProfile] = useState<Record<string, unknown> | null>(null);
  const [factsBundle, setFactsBundle] = useState<Record<string, unknown> | null>(null);
  const [dashboardSpec, setDashboardSpec] = useState<Record<string, unknown> | null>(null);
  const [askQuestion, setAskQuestion] = useState('Show the main metric trend over time');
  const [askResult, setAskResult] = useState<AskResponsePayload | null>(null);
  const [documentQuestion, setDocumentQuestion] = useState('What do our trusted policy documents say about denominator exclusions?');
  const [documentAnswer, setDocumentAnswer] = useState<DocumentAskResponse | null>(null);
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

        if (meta.artifacts?.dashboard_spec) {
          const specResponse = await getDashboardSpec(datasetId, userId);
          if (!active) return;
          setDashboardSpec(specResponse.dashboard_spec);
        }

        const auditResponse = await getAudit(datasetId, userId);
        if (!active) return;
        setAuditEvents(auditResponse.events ?? []);
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

  const dashboardCharts = useMemo(() => normalizeDashboardCharts(dashboardSpec), [dashboardSpec]);

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
  const canReviewSensitiveExport = permissionSet.has('sensitive_export:review') || permissionSet.has('admin:all');
  const canRequestSensitiveExport = permissionSet.has('sensitive_export:request_own') || permissionSet.has('admin:all');
  const canReadDocuments = permissionSet.has('docs:read_own') || permissionSet.has('docs:read_all') || permissionSet.has('admin:all');
  const canUploadDocuments = permissionSet.has('docs:create') || permissionSet.has('admin:all');

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

  async function handleCreateSession() {
    setBusyAction('create_session');
    setError(null);
    setNotice(null);
    try {
      const created = await createSession(userId);
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
      setError('Choose a TXT, MD, HTML, or JSON document first.');
      return;
    }

    setBusyAction('document_upload');
    setError(null);
    setNotice(null);
    try {
      await uploadDocument(userId, selectedDocumentFile);
      await refreshDocuments();
      setSelectedDocumentFile(null);
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

  async function handleAsk() {
    if (!datasetId) {
      setError('Create or load a session first.');
      return;
    }
    if (!askQuestion.trim()) {
      setError('Enter a question first.');
      return;
    }
    setBusyAction('ask');
    setError(null);
    setNotice(null);
    try {
      const response = await askDataset(datasetId, userId, askQuestion.trim());
      setAskResult(response);
      setAuditEvents((await getAudit(datasetId, userId)).events ?? []);
      setNotice('Ask-data answer generated successfully.');
    } catch (askError) {
      setError(askError instanceof Error ? askError.message : 'Ask-data request failed.');
    } finally {
      setBusyAction(null);
    }
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

        <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_auto]">
          <Input
            type="file"
            accept=".txt,.md,.html,.htm,.json"
            onChange={(event) => setSelectedDocumentFile(event.target.files?.[0] ?? null)}
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
                </div>
                <p className="mt-2 text-sm font-medium text-foreground">{document.title}</p>
                <p className="mt-1 text-xs text-muted-foreground">{document.source_name}</p>
                {document.snippet_preview ? (
                  <p className="mt-2 text-xs text-muted-foreground">{document.snippet_preview}</p>
                ) : null}
              </div>
            ))}
          </div>
        ) : (
          <div className="mt-4 rounded-xl border border-border bg-background px-4 py-3 text-sm text-muted-foreground">
            No trusted documents uploaded yet. This first slice supports TXT, MD, HTML, and JSON sources.
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
              <div className="mt-3 flex flex-wrap gap-2">
                <Badge variant="outline">Grounded: {documentAnswer.grounded ? 'yes' : 'no'}</Badge>
                <Badge variant="outline">Confidence: {documentAnswer.confidence}</Badge>
                <Badge variant="outline">Citations: {documentAnswer.citations.length}</Badge>
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
                    </p>
                    <p className="mt-2 text-sm text-muted-foreground">{citation.snippet}</p>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
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

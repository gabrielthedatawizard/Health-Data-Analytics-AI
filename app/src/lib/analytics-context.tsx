import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import {
  askDataset as askBackendDataset,
  BACKEND_USER_STORAGE_KEY,
  createSession as createBackendSession,
  deleteSession as deleteBackendSession,
  getFacts,
  getJobStatus,
  getPreview,
  getProfile,
  listSessions,
  updateSession as updateBackendSession,
  uploadDataset as uploadBackendDataset,
  type AskResponsePayload,
  type PreviewResponse,
  type SessionSummary,
} from '@/lib/backend-api';
import {
  createEntityId,
  createSampleDataset,
  datasetRowsToCsv,
  formatBytes,
  type AIEnvironmentState,
  type DatasetColumnProfile,
  type DatasetFileType,
  type DatasetRecord,
  type DatasetRow,
  type InsightRecord,
  type SampleDatasetKind,
  type UploadProgressHandler,
} from '@/lib/ai-engine';

const STORAGE_KEY = 'healthai_workspace_state_v2';

interface PersistedWorkspaceState {
  insights: InsightRecord[];
  aiEnvironment: AIEnvironmentState;
}

export interface AnalyticsExportSnapshot {
  exportedAt: string;
  datasets: DatasetRecord[];
  insights: InsightRecord[];
  aiEnvironment: AIEnvironmentState;
}

interface AnalyticsContextValue {
  datasets: DatasetRecord[];
  insights: InsightRecord[];
  aiEnvironment: AIEnvironmentState;
  uploadFile: (file: File, onProgress?: UploadProgressHandler) => Promise<DatasetRecord>;
  addSampleDataset: (kind?: SampleDatasetKind) => Promise<DatasetRecord>;
  removeDataset: (datasetId: string) => Promise<void>;
  renameDataset: (datasetId: string, nextName: string) => Promise<void>;
  analyzeDataset: (datasetId: string, sourceQuestion?: string) => Promise<InsightRecord[]>;
  askAi: (question: string) => Promise<InsightRecord>;
  loadDatasetPreview: (datasetId: string) => Promise<DatasetRecord | null>;
  toggleInsightExpanded: (insightId: string) => void;
  setInsightFeedback: (insightId: string, feedback: 'positive' | 'negative') => void;
  flagInsight: (insightId: string) => void;
  exportSnapshot: () => AnalyticsExportSnapshot;
  clearAllData: () => Promise<void>;
}

interface InitialWorkspaceState {
  datasets: DatasetRecord[];
  insights: InsightRecord[];
  aiEnvironment: AIEnvironmentState;
}

const DEFAULT_AI_ENVIRONMENT: AIEnvironmentState = {
  status: 'ready',
  mode: 'governed-backend',
  model: 'HealthAI Governed Analytics',
  version: '1.1.0',
  lastRunAt: null,
};

const AnalyticsContext = createContext<AnalyticsContextValue | null>(null);

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function loadPersistedState(): PersistedWorkspaceState | null {
  if (typeof window === 'undefined') return null;
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as Partial<PersistedWorkspaceState>;
    if (!Array.isArray(parsed.insights)) {
      return null;
    }
    return {
      insights: parsed.insights as InsightRecord[],
      aiEnvironment: parsed.aiEnvironment ?? DEFAULT_AI_ENVIRONMENT,
    };
  } catch {
    return null;
  }
}

function buildInitialWorkspaceState(): InitialWorkspaceState {
  const persisted = loadPersistedState();
  return {
    datasets: [],
    insights: persisted?.insights ?? [],
    aiEnvironment: persisted?.aiEnvironment ?? DEFAULT_AI_ENVIRONMENT,
  };
}

function getOrCreateBackendUserId(): string {
  if (typeof window === 'undefined') {
    return 'healthai_react_user';
  }
  const existing = localStorage.getItem(BACKEND_USER_STORAGE_KEY)?.trim();
  if (existing) {
    return existing;
  }
  const generated =
    window.crypto?.randomUUID?.() ?? `healthai_user_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  localStorage.setItem(BACKEND_USER_STORAGE_KEY, generated);
  return generated;
}

function stripFileExtension(name: string): string {
  return name.replace(/\.[^.]+$/, '');
}

function inferDatasetType(fileType?: string | null, fileName?: string | null): DatasetFileType {
  const normalized = (fileType || fileName?.split('.').pop() || 'csv').toLowerCase();
  if (normalized === 'xlsx' || normalized === 'xls' || normalized === 'excel') {
    return 'excel';
  }
  if (normalized === 'json') {
    return 'json';
  }
  return 'csv';
}

function inferDatasetSource(name: string): DatasetRecord['source'] {
  const normalized = name.toLowerCase();
  if (normalized.includes('dhis2') || normalized.includes('database')) {
    return 'integration';
  }
  if (normalized.includes('demo') || normalized.includes('sample')) {
    return 'sample';
  }
  return 'upload';
}

function mapSessionStatus(status: string): DatasetRecord['status'] {
  const normalized = status.toLowerCase();
  if (normalized.includes('error') || normalized.includes('failed')) {
    return 'error';
  }
  if (normalized === 'created' || normalized === 'uploaded') {
    return 'processing';
  }
  return 'active';
}

function mergeTags(existing: string[], incoming: string[]): string[] {
  return Array.from(new Set([...existing, ...incoming].filter(Boolean))).slice(0, 6);
}

function normalizePreviewRows(rows: Array<Record<string, unknown>>): DatasetRow[] {
  return rows.map((row) => {
    const normalized = Object.fromEntries(
      Object.entries(row).map(([key, value]) => {
        if (value === null || value === undefined) return [key, null];
        if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
          return [key, value];
        }
        return [key, String(value)];
      })
    );
    return normalized as DatasetRow;
  });
}

function toColumnProfile(column: Record<string, any>): DatasetColumnProfile {
  const rawType = String(column.inferred_type ?? column.dtype ?? 'unknown').toLowerCase();
  const inferredType =
    rawType === 'datetime'
      ? 'date'
      : rawType === 'number' || rawType === 'string' || rawType === 'boolean'
        ? rawType
        : 'unknown';

  return {
    name: String(column.name ?? 'column'),
    inferredType,
    nullPercentage: Number(column.missing_percent ?? 0),
    uniqueCount: Number(column.unique_count ?? 0),
    sampleValues: Array.isArray(column.sample_values)
      ? column.sample_values.map((value: unknown) => String(value))
      : [],
    min: typeof column.min === 'number' ? column.min : undefined,
    max: typeof column.max === 'number' ? column.max : undefined,
    mean: typeof column.mean === 'number' ? column.mean : undefined,
  };
}

function buildProfileIssues(profile: Record<string, any>): string[] {
  const issues: string[] = [];
  const duplicatePercent = Number(profile.duplicate_percent ?? 0);
  if (duplicatePercent > 2) {
    issues.push(`Duplicate row percentage is ${duplicatePercent.toFixed(1)}%.`);
  }

  const columns = Array.isArray(profile.columns) ? profile.columns : [];
  columns.forEach((column) => {
    const missingPercent = Number(column?.missing_percent ?? 0);
    if (missingPercent >= 20) {
      issues.push(`${String(column?.name ?? 'Column')} has ${missingPercent.toFixed(1)}% missing values.`);
    }
  });

  const piiCandidates = Array.isArray(profile.pii_candidates) ? profile.pii_candidates : [];
  if (piiCandidates.length > 0) {
    issues.push(`Potential PII detected in ${piiCandidates.slice(0, 3).join(', ')}.`);
  }

  return Array.from(new Set(issues)).slice(0, 4);
}

function buildProfileTags(profile: Record<string, any>): string[] {
  const tags: string[] = [];
  const templateName = String(profile.health_template?.name ?? '').trim();
  if (templateName) {
    tags.push(templateName.toUpperCase());
  }

  const healthSignals = (profile.health_signals ?? {}) as Record<string, unknown>;
  if (Array.isArray(healthSignals.time_columns) && healthSignals.time_columns.length > 0) {
    tags.push('Time Series');
  }
  if (Array.isArray(healthSignals.geography_columns) && healthSignals.geography_columns.length > 0) {
    tags.push('Geography');
  }
  if (Array.isArray(healthSignals.service_columns) && healthSignals.service_columns.length > 0) {
    tags.push('Health Ops');
  }

  return tags;
}

function buildDatasetMetrics(profile: Record<string, any>): Record<string, number> {
  return {
    quality_score: Number(profile.quality_score ?? 0),
    completeness_percent: Number(profile.completeness_percent ?? 0),
    duplicate_rows: Number(profile.duplicate_rows ?? 0),
    duplicate_percent: Number(profile.duplicate_percent ?? 0),
    numeric_columns: Array.isArray(profile.numeric_cols) ? profile.numeric_cols.length : 0,
    categorical_columns: Array.isArray(profile.categorical_cols) ? profile.categorical_cols.length : 0,
    datetime_columns: Array.isArray(profile.datetime_cols) ? profile.datetime_cols.length : 0,
  };
}

function defaultColumnsFromPreview(preview?: PreviewResponse): DatasetColumnProfile[] {
  if (!preview) return [];
  return preview.columns.map((name) => ({
    name,
    inferredType: 'unknown',
    nullPercentage: 0,
    uniqueCount: 0,
    sampleValues: [],
  }));
}

function mapSessionSummaryToDataset(summary: SessionSummary, existing?: DatasetRecord): DatasetRecord {
  const displayName = summary.display_name || summary.file_name || existing?.name || summary.dataset_id;
  const nextSizeBytes = summary.size_bytes || existing?.sizeBytes || 0;
  const summaryTags = [
    summary.file_type ? summary.file_type.toUpperCase() : '',
    'Governed',
    summary.pii_masking_enabled ? 'PII Masked' : '',
  ].filter(Boolean);

  return {
    id: summary.dataset_id,
    name: displayName,
    description:
      summary.description ||
      existing?.description ||
      (summary.row_count > 0
        ? `${summary.row_count.toLocaleString()} rows available through the governed analytics backend.`
        : 'Dataset session ready for upload and governed analytics.'),
    type: inferDatasetType(summary.file_type, summary.file_name),
    source: existing?.source ?? inferDatasetSource(displayName),
    status: mapSessionStatus(summary.status),
    rowCount: summary.row_count || existing?.rowCount || 0,
    columnCount: summary.column_count || existing?.columnCount || 0,
    qualityScore: Math.round(summary.quality_score || existing?.qualityScore || 0),
    sizeBytes: nextSizeBytes,
    sizeLabel: formatBytes(nextSizeBytes),
    issues: summary.quality_issues.length > 0 ? summary.quality_issues : existing?.issues ?? [],
    tags: mergeTags(existing?.tags ?? [], summaryTags),
    createdBy: summary.created_by || existing?.createdBy || 'You',
    uploadedAt: summary.created_at || existing?.uploadedAt || new Date().toISOString(),
    lastUpdated: summary.updated_at || existing?.lastUpdated || summary.created_at || new Date().toISOString(),
    columns: existing?.columns ?? [],
    sampleRows: existing?.sampleRows ?? [],
    metrics: existing?.metrics ?? {},
  };
}

function hydrateDatasetRecord(
  dataset: DatasetRecord,
  profile: Record<string, any>,
  preview?: PreviewResponse
): DatasetRecord {
  const shape = (profile.shape ?? {}) as Record<string, unknown>;
  const hasProfileData = Object.keys(profile).length > 0;
  const columns = Array.isArray(profile.columns) && profile.columns.length > 0
    ? profile.columns.map((column: Record<string, any>) => toColumnProfile(column))
    : dataset.columns.length > 0
      ? dataset.columns
      : defaultColumnsFromPreview(preview);
  const rowCount = Number(shape.rows ?? preview?.row_count ?? dataset.rowCount);
  const columnCount = Number(shape.cols ?? columns.length ?? dataset.columnCount);
  const issues = buildProfileIssues(profile);

  return {
    ...dataset,
    status: 'active',
    rowCount,
    columnCount,
    qualityScore: Math.round(Number(profile.quality_score ?? dataset.qualityScore)),
    sizeLabel: formatBytes(dataset.sizeBytes),
    issues: issues.length > 0 ? issues : dataset.issues,
    tags: hasProfileData ? mergeTags(dataset.tags, buildProfileTags(profile)) : dataset.tags,
    columns: columns.length > 0 ? columns : dataset.columns,
    sampleRows: preview ? normalizePreviewRows(preview.rows) : dataset.sampleRows,
    metrics: hasProfileData ? buildDatasetMetrics(profile) : dataset.metrics,
    lastUpdated: new Date().toISOString(),
  };
}

function upsertDataset(datasets: DatasetRecord[], nextDataset: DatasetRecord): DatasetRecord[] {
  const existingIndex = datasets.findIndex((dataset) => dataset.id === nextDataset.id);
  if (existingIndex === -1) {
    return [nextDataset, ...datasets];
  }

  const updated = [...datasets];
  updated[existingIndex] = nextDataset;
  return updated;
}

function reconcileDatasets(datasets: DatasetRecord[], sessions: SessionSummary[]): DatasetRecord[] {
  const existingById = new Map(datasets.map((dataset) => [dataset.id, dataset]));
  return sessions.map((session) => mapSessionSummaryToDataset(session, existingById.get(session.dataset_id)));
}

function filterInsightsForDatasets(insights: InsightRecord[], datasetIds: Set<string>): InsightRecord[] {
  return insights.filter((insight) => !insight.datasetId || datasetIds.has(insight.datasetId));
}

function confidenceFromLabel(label?: string): number {
  const normalized = String(label ?? '').toLowerCase();
  if (normalized === 'high') return 94;
  if (normalized === 'medium') return 82;
  if (normalized === 'low') return 68;
  return 78;
}

function buildUploadInsight(dataset: DatasetRecord): InsightRecord {
  return {
    id: createEntityId('insight'),
    datasetId: dataset.id,
    type: 'trend',
    title: `${dataset.name} profile ready`,
    content: `${dataset.rowCount.toLocaleString()} rows and ${dataset.columnCount.toLocaleString()} columns were profiled through the governed backend. Current quality score is ${dataset.qualityScore}%.`,
    confidence: 96,
    citations: [dataset.name],
    timestamp: new Date().toISOString(),
    verified: true,
    expanded: true,
  };
}

function formatQualityIssue(issue: Record<string, any>): string {
  const code = String(issue.code ?? 'ISSUE');
  const severity = String(issue.severity ?? 'medium');
  const message = String(issue.message ?? 'Quality issue detected.');
  const columns = Array.isArray(issue.columns) && issue.columns.length > 0 ? ` Columns: ${issue.columns.slice(0, 4).join(', ')}.` : '';
  return `${code} (${severity}): ${message}${columns}`;
}

function buildAnalysisInsights(
  dataset: DatasetRecord,
  factsBundle: Record<string, any>,
  sourceQuestion?: string
): InsightRecord[] {
  const now = new Date().toISOString();
  const coverageMode = String(factsBundle.data_coverage?.mode ?? 'full');
  const insights: InsightRecord[] = [
    {
      id: createEntityId('insight'),
      datasetId: dataset.id,
      type: 'trend',
      title: `${dataset.name} analysis ready`,
      content: `${dataset.rowCount.toLocaleString()} rows and ${dataset.columnCount.toLocaleString()} columns were analyzed. Quality score is ${dataset.qualityScore}%, and the current evidence coverage mode is ${coverageMode}.`,
      confidence: 95,
      citations: [dataset.name],
      timestamp: now,
      verified: true,
      expanded: true,
      sourceQuestion,
    },
  ];

  const qualityIssues = Array.isArray(factsBundle.quality?.issues) ? factsBundle.quality.issues : [];
  if (qualityIssues.length > 0) {
    insights.push({
      id: createEntityId('insight'),
      datasetId: dataset.id,
      type: 'anomaly',
      title: 'Data quality and governance flags',
      content: qualityIssues.slice(0, 2).map((issue: Record<string, any>) => formatQualityIssue(issue)).join(' '),
      confidence: 91,
      citations: [dataset.name],
      timestamp: now,
      verified: true,
      expanded: false,
      sourceQuestion,
    });
  }

  const insightFacts = Array.isArray(factsBundle.insight_facts) ? factsBundle.insight_facts : [];
  const trendFact = insightFacts.find((fact: Record<string, any>) => fact?.type === 'trend') as Record<string, any> | undefined;
  if (trendFact) {
    const value = (trendFact.value ?? {}) as Record<string, any>;
    const latest = Number(value.latest_period_value ?? 0);
    const previous = Number(value.previous_period_value ?? 0);
    const pctChange = value.pct_change;
    const metric = String(value.metric ?? 'metric');
    const content =
      pctChange !== null && pctChange !== undefined
        ? `${metric} moved from ${previous.toFixed(2)} to ${latest.toFixed(2)} for the latest period, a ${Number(pctChange).toFixed(2)}% change.`
        : `${metric} has comparable recent periods available for trend inspection.`;

    insights.push({
      id: createEntityId('insight'),
      datasetId: dataset.id,
      type: 'trend',
      title: `${metric} period change`,
      content,
      confidence: 89,
      citations: [dataset.name, String(trendFact.id ?? 'fact')],
      timestamp: now,
      verified: true,
      expanded: false,
      sourceQuestion,
    });
  }

  const comparisonFact = insightFacts.find((fact: Record<string, any>) => {
    const metric = String(fact?.value?.metric ?? '');
    return fact?.type === 'comparison' && metric.startsWith('top_');
  }) as Record<string, any> | undefined;

  if (comparisonFact) {
    const value = (comparisonFact.value ?? {}) as Record<string, any>;
    insights.push({
      id: createEntityId('insight'),
      datasetId: dataset.id,
      type: 'recommendation',
      title: 'Top contributing segment',
      content: `${String(value.segment_field ?? 'Segment')} leader is ${String(value.segment ?? 'N/A')} with ${String(value.metric ?? 'metric')} at ${Number(value.value ?? 0).toFixed(2)}.`,
      confidence: 86,
      citations: [dataset.name, String(comparisonFact.id ?? 'fact')],
      timestamp: now,
      verified: true,
      expanded: false,
      sourceQuestion,
    });
  }

  insights.push({
    id: createEntityId('insight'),
    datasetId: dataset.id,
    type: 'recommendation',
    title: 'Recommended next step',
    content:
      dataset.qualityScore >= 85
        ? coverageMode === 'sample'
          ? 'The dataset is strong enough for dashboards, but run full coverage facts before final reporting or export.'
          : 'The dataset is ready for dashboard generation, narrative reporting, and follow-up analysis.'
        : 'Address the flagged data quality issues before using this dataset for executive reporting.',
    confidence: 88,
    citations: [dataset.name],
    timestamp: now,
    verified: true,
    expanded: false,
    sourceQuestion,
  });

  return insights;
}

function inferInsightTypeFromQuestion(question: string, response: AskResponsePayload): InsightRecord['type'] {
  const normalized = question.toLowerCase();
  if (normalized.includes('recommend') || normalized.includes('action') || normalized.includes('next step')) {
    return 'recommendation';
  }
  if (normalized.includes('predict') || normalized.includes('forecast')) {
    return 'prediction';
  }
  if (normalized.includes('anomaly') || normalized.includes('issue') || normalized.includes('why')) {
    return 'anomaly';
  }
  if ((response.result_rows ?? []).length > 0 || response.chart) {
    return 'trend';
  }
  return 'recommendation';
}

function buildAskInsight(dataset: DatasetRecord, question: string, response: AskResponsePayload): InsightRecord {
  const coverageNote =
    response.data_coverage === 'sample' ? ' Evidence was generated from sampled coverage, so validate before final reporting.' : '';
  return {
    id: createEntityId('insight'),
    datasetId: dataset.id,
    type: inferInsightTypeFromQuestion(question, response),
    title: question.trim().length > 72 ? `${question.trim().slice(0, 72)}...` : question.trim(),
    content: `${response.answer}${coverageNote}`,
    confidence: confidenceFromLabel(response.confidence),
    citations: Array.from(new Set([dataset.name, ...(response.facts_used ?? [])])).filter(Boolean),
    timestamp: new Date().toISOString(),
    verified: true,
    expanded: true,
    sourceQuestion: question.trim(),
  };
}

function buildNoDatasetInsight(question: string): InsightRecord {
  return {
    id: createEntityId('insight'),
    type: 'recommendation',
    title: 'Upload data to start analysis',
    content: 'No governed dataset session is available yet. Upload a CSV or Excel file, then ask your question again.',
    confidence: 99,
    citations: [],
    timestamp: new Date().toISOString(),
    verified: true,
    expanded: true,
    sourceQuestion: question.trim(),
  };
}

function pickDatasetForQuestion(question: string, datasets: DatasetRecord[]): DatasetRecord | null {
  if (datasets.length === 0) return null;
  const normalizedQuestion = question.toLowerCase();
  const namedMatch = datasets.find((dataset) =>
    normalizedQuestion.includes(dataset.name.toLowerCase().replace(/\.[^.]+$/, ''))
  );
  if (namedMatch) {
    return namedMatch;
  }

  const activeDatasets = datasets.filter((dataset) => dataset.status === 'active');
  const candidates = activeDatasets.length > 0 ? activeDatasets : datasets;
  return [...candidates].sort((left, right) => right.uploadedAt.localeCompare(left.uploadedAt))[0] ?? null;
}

async function resolveFactsBundle(datasetId: string, userId: string): Promise<Record<string, any>> {
  const response = await getFacts(datasetId, userId);
  if (response.facts_bundle) {
    return response.facts_bundle as Record<string, any>;
  }
  if (!response.job_id) {
    throw new Error('Facts generation did not return a facts bundle.');
  }

  for (let attempt = 0; attempt < 60; attempt += 1) {
    await wait(1500);
    const job = await getJobStatus(response.job_id, userId);
    if (job.status === 'failed') {
      throw new Error(job.error?.message || 'Facts generation job failed.');
    }
    if (job.status === 'succeeded') {
      const completed = await getFacts(datasetId, userId);
      if (completed.facts_bundle) {
        return completed.facts_bundle as Record<string, any>;
      }
    }
  }

  throw new Error('Facts generation is taking longer than expected.');
}

function createFallbackDataset(file: File, datasetId: string, userId: string, description?: string): DatasetRecord {
  const now = new Date().toISOString();
  return {
    id: datasetId,
    name: stripFileExtension(file.name) || file.name,
    description: description || 'Dataset uploaded to the governed analytics backend.',
    type: inferDatasetType(file.name.split('.').pop(), file.name),
    source: 'upload',
    status: 'processing',
    rowCount: 0,
    columnCount: 0,
    qualityScore: 0,
    sizeBytes: file.size,
    sizeLabel: formatBytes(file.size),
    issues: [],
    tags: ['Governed'],
    createdBy: userId,
    uploadedAt: now,
    lastUpdated: now,
    columns: [],
    sampleRows: [],
    metrics: {},
  };
}

export function AnalyticsProvider({ children }: { children: ReactNode }) {
  const initialState = useMemo(() => buildInitialWorkspaceState(), []);
  const [datasets, setDatasets] = useState<DatasetRecord[]>(initialState.datasets);
  const [insights, setInsights] = useState<InsightRecord[]>(initialState.insights);
  const [aiEnvironment, setAiEnvironment] = useState<AIEnvironmentState>(initialState.aiEnvironment);
  const [backendUserId] = useState(() => getOrCreateBackendUserId());

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const snapshot: PersistedWorkspaceState = {
      insights,
      aiEnvironment,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
  }, [aiEnvironment, insights]);

  const markEnvironmentBusy = useCallback(() => {
    setAiEnvironment((previous) => ({
      ...previous,
      status: 'busy',
      lastError: undefined,
    }));
  }, []);

  const markEnvironmentReady = useCallback(() => {
    setAiEnvironment((previous) => ({
      ...previous,
      status: 'ready',
      lastRunAt: new Date().toISOString(),
      lastError: undefined,
    }));
  }, []);

  const markEnvironmentError = useCallback((errorMessage: string) => {
    setAiEnvironment((previous) => ({
      ...previous,
      status: 'error',
      lastRunAt: new Date().toISOString(),
      lastError: errorMessage,
    }));
  }, []);

  const refreshDatasets = useCallback(async () => {
    const response = await listSessions(backendUserId);
    const datasetIds = new Set(response.sessions.map((session) => session.dataset_id));
    setDatasets((previous) => reconcileDatasets(previous, response.sessions));
    setInsights((previous) => filterInsightsForDatasets(previous, datasetIds));
    return response.sessions;
  }, [backendUserId]);

  useEffect(() => {
    let active = true;
    markEnvironmentBusy();
    refreshDatasets()
      .then(() => {
        if (active) {
          markEnvironmentReady();
        }
      })
      .catch((error) => {
        if (!active) return;
        const message = error instanceof Error ? error.message : 'Could not load governed datasets.';
        markEnvironmentError(message);
      });

    return () => {
      active = false;
    };
  }, [markEnvironmentBusy, markEnvironmentError, markEnvironmentReady, refreshDatasets]);

  const uploadDatasetToBackend = useCallback(
    async (
      file: File,
      options?: {
        description?: string;
        sourceOverride?: DatasetRecord['source'];
        initialTags?: string[];
        onProgress?: UploadProgressHandler;
      }
    ): Promise<DatasetRecord> => {
      markEnvironmentBusy();
      let datasetId: string | null = null;
      let uploadCommitted = false;

      try {
        options?.onProgress?.({
          progress: 10,
          status: 'uploading',
          message: 'Creating governed session',
        });
        const created = await createBackendSession(backendUserId, {
          display_name: stripFileExtension(file.name),
          description: options?.description,
        });
        datasetId = created.dataset_id;

        options?.onProgress?.({
          progress: 40,
          status: 'uploading',
          message: 'Uploading dataset to backend',
        });
        await uploadBackendDataset(datasetId, file, backendUserId);
        uploadCommitted = true;

        options?.onProgress?.({
          progress: 75,
          status: 'processing',
          message: 'Profiling dataset',
        });

        const [profileResponse, previewResponse] = await Promise.all([
          getProfile(datasetId, backendUserId),
          getPreview(datasetId, backendUserId, 50),
        ]);

        let baseDataset = createFallbackDataset(file, datasetId, backendUserId, options?.description);
        try {
          const sessions = await listSessions(backendUserId);
          const summary = sessions.sessions.find((candidate) => candidate.dataset_id === datasetId);
          if (summary) {
            baseDataset = mapSessionSummaryToDataset(summary, baseDataset);
          }
        } catch {
          // Keep the upload resilient even if the refresh call fails.
        }

        let hydratedDataset = hydrateDatasetRecord(
          baseDataset,
          profileResponse.profile as Record<string, any>,
          previewResponse
        );
        if (options?.sourceOverride) {
          hydratedDataset = { ...hydratedDataset, source: options.sourceOverride };
        }
        if (options?.initialTags?.length) {
          hydratedDataset = {
            ...hydratedDataset,
            tags: mergeTags(hydratedDataset.tags, options.initialTags),
          };
        }

        setDatasets((previous) => upsertDataset(previous, hydratedDataset));
        setInsights((previous) => [buildUploadInsight(hydratedDataset), ...previous]);
        void refreshDatasets();

        options?.onProgress?.({
          progress: 100,
          status: 'processing',
          message: 'Dataset ready',
        });
        markEnvironmentReady();
        return hydratedDataset;
      } catch (error) {
        if (datasetId && !uploadCommitted) {
          try {
            await deleteBackendSession(datasetId, backendUserId);
          } catch {
            // Ignore cleanup failures during optimistic session creation.
          }
        }
        if (uploadCommitted) {
          void refreshDatasets();
        }
        const message = error instanceof Error ? error.message : 'Upload failed unexpectedly.';
        markEnvironmentError(message);
        throw error instanceof Error ? error : new Error(message);
      }
    },
    [backendUserId, markEnvironmentBusy, markEnvironmentError, markEnvironmentReady, refreshDatasets]
  );

  const uploadFile = useCallback(
    async (file: File, onProgress?: UploadProgressHandler): Promise<DatasetRecord> => {
      return uploadDatasetToBackend(file, { onProgress });
    },
    [uploadDatasetToBackend]
  );

  const addSampleDataset = useCallback(
    async (kind: SampleDatasetKind = 'demo'): Promise<DatasetRecord> => {
      const sampleDataset = createSampleDataset(kind);
      const csv = datasetRowsToCsv(sampleDataset);
      const file = new File([csv], sampleDataset.name, { type: 'text/csv;charset=utf-8' });
      return uploadDatasetToBackend(file, {
        description: sampleDataset.description,
        sourceOverride: sampleDataset.source,
        initialTags: sampleDataset.tags,
      });
    },
    [uploadDatasetToBackend]
  );

  const removeDataset = useCallback(
    async (datasetId: string) => {
      markEnvironmentBusy();
      try {
        await deleteBackendSession(datasetId, backendUserId);
        setDatasets((previous) => previous.filter((dataset) => dataset.id !== datasetId));
        setInsights((previous) => previous.filter((insight) => insight.datasetId !== datasetId));
        markEnvironmentReady();
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Could not delete dataset.';
        markEnvironmentError(message);
        throw error instanceof Error ? error : new Error(message);
      }
    },
    [backendUserId, markEnvironmentBusy, markEnvironmentError, markEnvironmentReady]
  );

  const renameDataset = useCallback(
    async (datasetId: string, nextName: string) => {
      const normalizedName = nextName.trim();
      if (!normalizedName) return;

      const previousName = datasets.find((dataset) => dataset.id === datasetId)?.name;
      if (!previousName) return;

      markEnvironmentBusy();
      try {
        await updateBackendSession(datasetId, backendUserId, { display_name: normalizedName });
        setDatasets((previous) =>
          previous.map((dataset) =>
            dataset.id === datasetId
              ? {
                  ...dataset,
                  name: normalizedName,
                  lastUpdated: new Date().toISOString(),
                }
              : dataset
          )
        );
        setInsights((previous) =>
          previous.map((insight) => {
            if (insight.datasetId !== datasetId) return insight;
            return {
              ...insight,
              title: insight.title.replace(previousName, normalizedName),
              citations: insight.citations.map((citation) =>
                citation === previousName ? normalizedName : citation
              ),
            };
          })
        );
        markEnvironmentReady();
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Could not rename dataset.';
        markEnvironmentError(message);
        throw error instanceof Error ? error : new Error(message);
      }
    },
    [backendUserId, datasets, markEnvironmentBusy, markEnvironmentError, markEnvironmentReady]
  );

  const loadDatasetPreview = useCallback(
    async (datasetId: string): Promise<DatasetRecord | null> => {
      const existing = datasets.find((dataset) => dataset.id === datasetId);
      if (!existing) {
        return null;
      }
      if (existing.sampleRows.length > 0 && existing.columns.length > 0) {
        return existing;
      }

      const [previewResponse, profileResponse] = await Promise.all([
        getPreview(datasetId, backendUserId, 50),
        existing.columns.length === 0 ? getProfile(datasetId, backendUserId) : Promise.resolve(null),
      ]);

      const hydrated = hydrateDatasetRecord(
        existing,
        (profileResponse?.profile as Record<string, any> | undefined) ?? {},
        previewResponse
      );
      setDatasets((previous) => upsertDataset(previous, hydrated));
      return hydrated;
    },
    [backendUserId, datasets]
  );

  const analyzeDataset = useCallback(
    async (datasetId: string, sourceQuestion?: string): Promise<InsightRecord[]> => {
      const existing = datasets.find((dataset) => dataset.id === datasetId);
      if (!existing) {
        throw new Error('Dataset not found.');
      }

      markEnvironmentBusy();
      try {
        const [profileResponse, previewResponse, factsBundle] = await Promise.all([
          getProfile(datasetId, backendUserId),
          getPreview(datasetId, backendUserId, 50),
          resolveFactsBundle(datasetId, backendUserId),
        ]);

        const hydrated = hydrateDatasetRecord(
          existing,
          profileResponse.profile as Record<string, any>,
          previewResponse
        );
        const generated = buildAnalysisInsights(hydrated, factsBundle, sourceQuestion);

        setDatasets((previous) => upsertDataset(previous, hydrated));
        setInsights((previous) => [...generated, ...previous]);
        void refreshDatasets();
        markEnvironmentReady();
        return generated;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Analysis failed unexpectedly.';
        markEnvironmentError(message);
        throw error instanceof Error ? error : new Error(message);
      }
    },
    [backendUserId, datasets, markEnvironmentBusy, markEnvironmentError, markEnvironmentReady, refreshDatasets]
  );

  const askAi = useCallback(
    async (question: string): Promise<InsightRecord> => {
      const trimmedQuestion = question.trim();
      if (!trimmedQuestion) {
        throw new Error('Question cannot be empty.');
      }

      const dataset = pickDatasetForQuestion(trimmedQuestion, datasets);
      if (!dataset) {
        const insight = buildNoDatasetInsight(trimmedQuestion);
        setInsights((previous) => [insight, ...previous]);
        return insight;
      }

      markEnvironmentBusy();
      try {
        const response = await askBackendDataset(dataset.id, backendUserId, trimmedQuestion);
        const insight = buildAskInsight(dataset, trimmedQuestion, response);
        setInsights((previous) => [insight, ...previous]);
        markEnvironmentReady();
        return insight;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Could not answer the question.';
        markEnvironmentError(message);
        throw error instanceof Error ? error : new Error(message);
      }
    },
    [backendUserId, datasets, markEnvironmentBusy, markEnvironmentError, markEnvironmentReady]
  );

  const toggleInsightExpanded = useCallback((insightId: string) => {
    setInsights((previous) =>
      previous.map((insight) =>
        insight.id === insightId ? { ...insight, expanded: !insight.expanded } : insight
      )
    );
  }, []);

  const setInsightFeedback = useCallback((insightId: string, feedback: 'positive' | 'negative') => {
    setInsights((previous) =>
      previous.map((insight) =>
        insight.id === insightId ? { ...insight, userFeedback: feedback } : insight
      )
    );
  }, []);

  const flagInsight = useCallback((insightId: string) => {
    setInsights((previous) =>
      previous.map((insight) =>
        insight.id === insightId ? { ...insight, flagged: !insight.flagged } : insight
      )
    );
  }, []);

  const exportSnapshot = useCallback((): AnalyticsExportSnapshot => {
    return {
      exportedAt: new Date().toISOString(),
      datasets,
      insights,
      aiEnvironment,
    };
  }, [aiEnvironment, datasets, insights]);

  const clearAllData = useCallback(async () => {
    markEnvironmentBusy();
    try {
      const results = await Promise.allSettled(
        datasets.map((dataset) => deleteBackendSession(dataset.id, backendUserId))
      );
      const failedDeletes = results.filter((result) => result.status === 'rejected');
      if (failedDeletes.length > 0) {
        void refreshDatasets();
        throw new Error(`Could not delete ${failedDeletes.length} dataset session(s).`);
      }
      setDatasets([]);
      setInsights([]);
      setAiEnvironment({
        ...DEFAULT_AI_ENVIRONMENT,
        lastRunAt: new Date().toISOString(),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not clear workspace data.';
      markEnvironmentError(message);
      throw error instanceof Error ? error : new Error(message);
    }
  }, [backendUserId, datasets, markEnvironmentBusy, markEnvironmentError, refreshDatasets]);

  const value = useMemo<AnalyticsContextValue>(
    () => ({
      datasets,
      insights,
      aiEnvironment,
      uploadFile,
      addSampleDataset,
      removeDataset,
      renameDataset,
      analyzeDataset,
      askAi,
      loadDatasetPreview,
      toggleInsightExpanded,
      setInsightFeedback,
      flagInsight,
      exportSnapshot,
      clearAllData,
    }),
    [
      addSampleDataset,
      aiEnvironment,
      analyzeDataset,
      askAi,
      clearAllData,
      datasets,
      exportSnapshot,
      flagInsight,
      insights,
      loadDatasetPreview,
      removeDataset,
      renameDataset,
      setInsightFeedback,
      toggleInsightExpanded,
      uploadFile,
    ]
  );

  return <AnalyticsContext.Provider value={value}>{children}</AnalyticsContext.Provider>;
}

export function useAnalytics() {
  const context = useContext(AnalyticsContext);
  if (!context) {
    throw new Error('useAnalytics must be used within AnalyticsProvider');
  }
  return context;
}

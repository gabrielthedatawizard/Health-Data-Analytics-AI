export interface SensitiveExportApprovalState {
  status: string;
  requested_by?: string | null;
  requested_at?: string | null;
  justification?: string | null;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
  review_note?: string | null;
}

export interface SessionMeta {
  dataset_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  created_by: string;
  user_id?: string | null;
  pii_masking_enabled: boolean;
  allow_sensitive_export: boolean;
  sensitive_export_approval: SensitiveExportApprovalState;
  file?: {
    name?: string;
    rows?: number;
    size_bytes?: number;
    content_type?: string;
  } | null;
  file_hash?: string | null;
  artifacts: Record<string, string>;
}

export interface SessionSummary {
  dataset_id: string;
  display_name: string;
  description: string;
  status: string;
  created_at: string;
  updated_at: string;
  created_by: string;
  pii_masking_enabled: boolean;
  allow_sensitive_export: boolean;
  sensitive_export_approval?: SensitiveExportApprovalState;
  file_name?: string | null;
  file_type?: string | null;
  size_bytes: number;
  row_count: number;
  column_count: number;
  quality_score: number;
  quality_issues: string[];
  artifacts: Record<string, string>;
}

export interface ProfileResponse {
  dataset_id: string;
  profile: Record<string, unknown>;
}

export interface FactsResponse {
  dataset_id: string;
  facts_bundle?: Record<string, unknown>;
  cached?: boolean;
  job_id?: string;
  status?: string;
  queued?: boolean;
}

export interface DashboardSpecResponse {
  dataset_id: string;
  dashboard_spec: Record<string, unknown>;
}

export interface AnomalyFinding {
  anomaly_id: string;
  kind: string;
  severity: string;
  title: string;
  summary: string;
  metric?: string | null;
  dimension?: string | null;
  segment?: string | null;
  period?: string | null;
  evidence: string[];
  root_cause_hints: string[];
  recommended_question?: string | null;
}

export interface AnomalyAnalysisPayload {
  generated_at: string;
  anomaly_count: number;
  summary: string;
  anomalies: AnomalyFinding[];
  suggested_questions: string[];
}

export interface AnomalyAnalysisResponse {
  dataset_id: string;
  analysis: AnomalyAnalysisPayload;
}

export interface CohortCriterionPayload {
  field: string;
  operator: string;
  value?: unknown;
  inferred_type?: string;
  operator_label?: string;
}

export interface CohortAnalysisPayload {
  generated_at: string;
  name: string;
  description?: string | null;
  row_count: number;
  population_row_count: number;
  criteria_count: number;
  criteria: CohortCriterionPayload[];
  preview_columns: string[];
  preview_rows: Array<Record<string, unknown>>;
  excluded_columns: string[];
  summary: string;
  suggested_questions: string[];
}

export interface CohortAnalysisResponse {
  dataset_id: string;
  cohort: CohortAnalysisPayload;
}

export interface WorkflowActionRecord {
  action_id: string;
  action_type: string;
  status: string;
  title: string;
  target?: string | null;
  objective?: string | null;
  summary: string;
  payload: Record<string, unknown>;
  evidence: string[];
  generated_at: string;
  updated_at: string;
  created_by: string;
  review_note?: string | null;
  reviewed_by?: string | null;
  reviewed_at?: string | null;
  executed_by?: string | null;
  executed_at?: string | null;
  execution_note?: string | null;
  requires_approval: boolean;
}

export interface WorkflowActionsResponse {
  dataset_id: string;
  actions: WorkflowActionRecord[];
}

export interface SavedInvestigationRecord {
  investigation_id: string;
  title: string;
  question: string;
  context_type: string;
  note?: string | null;
  result: Record<string, unknown>;
  created_at: string;
  created_by: string;
}

export interface SavedInvestigationsResponse {
  dataset_id: string;
  investigations: SavedInvestigationRecord[];
}

export interface SavedPlaybookRecord {
  playbook_id: string;
  name: string;
  question_template: string;
  description?: string | null;
  context_type: string;
  created_at: string;
  created_by: string;
}

export interface SavedPlaybooksResponse {
  dataset_id: string;
  playbooks: SavedPlaybookRecord[];
}

export interface ForecastSeriesPoint {
  period: string;
  value: number;
}

export interface ForecastCandidateMetrics {
  model_name: string;
  mae: number;
  rmse: number;
  mape?: number | null;
  holdout_points: number;
}

export interface ForecastRunPayload {
  generated_at: string;
  name: string;
  time_field: string;
  metric_field: string;
  aggregation: string;
  periods_used: number;
  holdout_points: number;
  horizon: number;
  champion_model: string;
  training_data_hash?: string | null;
  baseline_mean?: number | null;
  baseline_std?: number | null;
  latest_actual?: number | null;
  summary: string;
  warnings: string[];
  candidate_models: ForecastCandidateMetrics[];
  historical: ForecastSeriesPoint[];
  forecast: ForecastSeriesPoint[];
}

export interface ForecastRunRecord {
  run_id: string;
  model_kind: string;
  status: string;
  created_at: string;
  created_by: string;
  payload: ForecastRunPayload;
}

export interface ForecastRunsResponse {
  dataset_id: string;
  runs: ForecastRunRecord[];
}

export interface ForecastDriftSignal {
  code: string;
  severity: string;
  message: string;
}

export interface ForecastDriftPayload {
  generated_at: string;
  run_id: string;
  run_name: string;
  champion_model: string;
  time_field: string;
  metric_field: string;
  aggregation: string;
  training_data_hash?: string | null;
  current_data_hash?: string | null;
  stale_model: boolean;
  drift_score: number;
  periods_analyzed: number;
  baseline_mean: number;
  recent_mean: number;
  baseline_std: number;
  recent_std: number;
  summary: string;
  signals: ForecastDriftSignal[];
  recommended_actions: string[];
}

export interface ForecastDriftResponse {
  dataset_id: string;
  drift: ForecastDriftPayload;
}

export interface ModelRegistryEntry {
  registry_id: string;
  run_id: string;
  model_kind: string;
  status: string;
  promoted_at: string;
  promoted_by: string;
  note?: string | null;
  name: string;
  champion_model: string;
  metric_field: string;
  time_field: string;
  aggregation: string;
  source_data_hash?: string | null;
}

export interface ModelRegistryResponse {
  dataset_id: string;
  entries: ModelRegistryEntry[];
}

export interface ModelEvaluationRunSummary {
  run_id: string;
  name: string;
  champion_model: string;
  metric_field: string;
  mae: number;
  rmse: number;
  mape?: number | null;
  drift_score: number;
  stale_model: boolean;
  source: string;
}

export interface ModelEvaluationPayload {
  generated_at: string;
  active_run: ModelEvaluationRunSummary;
  challenger_run: ModelEvaluationRunSummary;
  recommendation: string;
  winner: string;
  rationale: string[];
  suggested_actions: string[];
}

export interface ModelEvaluationResponse {
  dataset_id: string;
  evaluation: ModelEvaluationPayload;
}

export interface AskResponsePayload {
  dataset_id: string;
  answer: string;
  facts_used: string[];
  confidence: string;
  fact_coverage: number;
  data_coverage: string;
  query_plan?: Record<string, unknown> | null;
  result_rows: Array<Record<string, unknown>>;
  chart?: Record<string, unknown> | null;
  governance: Record<string, unknown>;
}

export interface ReportJobResponse {
  dataset_id: string;
  job_id: string;
  status: string;
}

export interface JobStatus {
  job_id: string;
  type: string;
  dataset_id: string;
  status: string;
  progress: number;
  result?: Record<string, unknown> | null;
  artifacts?: Record<string, string | null> | null;
  error?: {
    code?: string;
    message?: string;
    trace?: string;
  } | null;
}

export interface AuditEvent {
  timestamp: string;
  dataset_id: string;
  action: string;
  actor: string;
  details: Record<string, unknown>;
}

export interface AuditResponse {
  dataset_id: string;
  events: AuditEvent[];
}

export interface PreviewResponse {
  dataset_id: string;
  rows: Array<Record<string, unknown>>;
  columns: string[];
  row_count: number;
}

export interface SensitiveExportStatusResponse {
  dataset_id: string;
  allow_sensitive_export: boolean;
  sensitive_export_approval: SensitiveExportApprovalState;
}

export interface DocumentSummary {
  document_id: string;
  title: string;
  source_name: string;
  status: string;
  created_at: string;
  updated_at: string;
  created_by: string;
  file_name: string;
  file_type: string;
  chunk_count: number;
  char_count: number;
  version_label?: string | null;
  effective_date?: string | null;
  supersedes_document_id?: string | null;
  superseded_by_document_id?: string | null;
  is_current?: boolean;
  freshness?: string;
  freshness_note?: string;
  snippet_preview: string;
}

export interface DocumentCitation {
  citation_key: string;
  document_id: string;
  title: string;
  source_name: string;
  snippet: string;
  chunk_index: number;
  version_label?: string | null;
  effective_date?: string | null;
  freshness?: string;
}

export interface DocumentAskResponse {
  answer: string;
  grounded: boolean;
  confidence: string;
  citations: DocumentCitation[];
  freshness_summary: string;
}

export type BackendUserRole = 'viewer' | 'analyst' | 'reviewer' | 'admin';

export interface AuthContextResponse {
  actor: string;
  role: BackendUserRole;
  permissions: string[];
}

export interface SystemStatusCounts {
  visible_sessions: number;
  visible_documents: number;
  active_jobs: number;
  queued_jobs: number;
  failed_jobs: number;
  active_models: number;
  stale_models: number;
  pending_sensitive_exports: number;
  pending_workflow_reviews: number;
  superseded_documents: number;
  recent_audit_events_24h: number;
}

export interface SystemStatusAlert {
  level: string;
  message: string;
}

export interface SystemStatusResponse {
  status: string;
  timestamp: string;
  actor: string;
  role: BackendUserRole;
  counts: SystemStatusCounts;
  alerts: SystemStatusAlert[];
}

export const ENV_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? '';

export function computeDefaultApiBase(): string {
  if (typeof window === 'undefined') {
    return '';
  }
  const { hostname } = window.location;
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  return '/api';
}

export const API_BASE = ENV_API_BASE || computeDefaultApiBase();
export const API_TARGET_LABEL = API_BASE || 'same-origin';
export const BACKEND_USER_STORAGE_KEY = 'healthai_backend_user_id';
export const BACKEND_ROLE_STORAGE_KEY = 'healthai_backend_user_role';

const VALID_BACKEND_ROLES = new Set<BackendUserRole>(['viewer', 'analyst', 'reviewer', 'admin']);

function normalizeBackendUserRole(role: string | null | undefined): BackendUserRole | undefined {
  if (!role) {
    return undefined;
  }
  const normalized = role.trim().toLowerCase();
  return VALID_BACKEND_ROLES.has(normalized as BackendUserRole) ? (normalized as BackendUserRole) : undefined;
}

function storedBackendUserId(): string {
  if (typeof window === 'undefined') {
    return '';
  }
  return localStorage.getItem(BACKEND_USER_STORAGE_KEY)?.trim() ?? '';
}

function storedBackendUserRole(): BackendUserRole | undefined {
  if (typeof window === 'undefined') {
    return undefined;
  }
  return normalizeBackendUserRole(localStorage.getItem(BACKEND_ROLE_STORAGE_KEY));
}

export function inferBackendUserRole(userId: string): BackendUserRole {
  const identity = userId.trim().toLowerCase();
  if (!identity || identity === 'anonymous') {
    return 'viewer';
  }
  if (identity === 'admin' || identity.startsWith('admin') || identity.endsWith('_admin')) {
    return 'admin';
  }
  if (['manager', 'reviewer', 'approver', 'auditor'].some((token) => identity.includes(token))) {
    return 'reviewer';
  }
  if (['viewer', 'read_only', 'readonly'].some((token) => identity.includes(token))) {
    return 'viewer';
  }
  return 'analyst';
}

function authenticatedUrl(path: string, options?: { userId?: string; userRole?: BackendUserRole }): string {
  const url = new URL(apiUrl(path), typeof window !== 'undefined' ? window.location.origin : 'http://localhost');
  const actor = options?.userId?.trim() || storedBackendUserId();
  const role = options?.userRole || storedBackendUserRole();
  if (actor) {
    url.searchParams.set('actor', actor);
  }
  if (role) {
    url.searchParams.set('role', role);
  }
  return API_BASE ? url.toString() : `${url.pathname}${url.search}`;
}

function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function parseResponse<T>(response: Response): Promise<T> {
  const rawBody = await response.text();
  const trimmed = rawBody.trim();
  const contentType = (response.headers.get('content-type') ?? '').toLowerCase();
  let payload: unknown = null;

  if (trimmed) {
    const looksLikeJson =
      contentType.includes('application/json') || trimmed.startsWith('{') || trimmed.startsWith('[');
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
    throw new Error(`${detail} [${response.status} ${response.url}]`);
  }

  return (payload ?? {}) as T;
}

export async function apiRequest<T>(
  path: string,
  init?: RequestInit,
  options?: {
    timeoutMs?: number;
    userId?: string;
    userRole?: BackendUserRole;
  }
): Promise<T> {
  const url = apiUrl(path);
  const controller = new AbortController();
  const timeoutMs = options?.timeoutMs ?? 60000;
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const headers = new Headers(init?.headers ?? {});
  const resolvedUserId = options?.userId?.trim() || storedBackendUserId();
  const resolvedUserRole = options?.userRole || storedBackendUserRole();
  if (resolvedUserId) {
    headers.set('X-API-Key', resolvedUserId);
  }
  if (resolvedUserRole) {
    headers.set('X-User-Role', resolvedUserRole);
  }

  try {
    const response = await fetch(url, {
      ...init,
      headers,
      signal: controller.signal,
    });
    return await parseResponse<T>(response);
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(`Request timed out for ${url}.`);
    }
    if (error instanceof Error) {
      throw new Error(`Could not reach API at ${url}: ${error.message}`);
    }
    throw new Error(`Could not reach API at ${url}.`);
  } finally {
    window.clearTimeout(timeout);
  }
}

export function getAuthContext(userId: string, userRole?: BackendUserRole) {
  return apiRequest<AuthContextResponse>('/auth/me', undefined, { userId, userRole });
}

export function getSystemStatus(userId: string, userRole?: BackendUserRole) {
  return apiRequest<SystemStatusResponse>('/system/status', undefined, { userId, userRole });
}

export function listDocuments(userId: string) {
  return apiRequest<{ documents: DocumentSummary[] }>('/documents', undefined, { userId });
}

export async function uploadDocument(
  userId: string,
  file: File,
  payload?: { title?: string; source_name?: string; version_label?: string; effective_date?: string; supersedes_document_id?: string }
) {
  const form = new FormData();
  form.append('file', file);
  if (payload?.title?.trim()) {
    form.append('title', payload.title.trim());
  }
  if (payload?.source_name?.trim()) {
    form.append('source_name', payload.source_name.trim());
  }
  if (payload?.version_label?.trim()) {
    form.append('version_label', payload.version_label.trim());
  }
  if (payload?.effective_date?.trim()) {
    form.append('effective_date', payload.effective_date.trim());
  }
  if (payload?.supersedes_document_id?.trim()) {
    form.append('supersedes_document_id', payload.supersedes_document_id.trim());
  }
  return apiRequest<DocumentSummary>(
    '/documents',
    {
      method: 'POST',
      body: form,
    },
    { userId, timeoutMs: 120000 }
  );
}

export function askDocuments(userId: string, question: string, documentIds?: string[], limit = 4) {
  return apiRequest<DocumentAskResponse>(
    '/documents/ask',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        document_ids: documentIds ?? [],
        limit,
      }),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function createSession(
  userId: string,
  payload?: {
    display_name?: string;
    description?: string;
  }
) {
  return apiRequest<{ dataset_id: string; created_at: string }>(
    '/sessions',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        created_by: userId || 'react_user',
        ...(payload ?? {}),
      }),
    },
    { userId }
  );
}

export function getSession(datasetId: string, userId: string) {
  return apiRequest<SessionMeta>(`/sessions/${datasetId}`, undefined, { userId });
}

export function listSessions(userId: string) {
  return apiRequest<{ sessions: SessionSummary[] }>('/sessions', undefined, { userId });
}

export function updateSession(datasetId: string, userId: string, payload: { display_name?: string; description?: string }) {
  return apiRequest<{ dataset_id: string; session: SessionSummary }>(
    `/sessions/${datasetId}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    { userId }
  );
}

export function deleteSession(datasetId: string, userId: string) {
  return apiRequest<{ dataset_id: string; deleted: boolean }>(
    `/sessions/${datasetId}`,
    {
      method: 'DELETE',
    },
    { userId }
  );
}

export async function uploadDataset(datasetId: string, file: File, userId: string) {
  const form = new FormData();
  form.append('file', file);
  form.append('uploaded_by', userId || 'react_user');
  return apiRequest<Record<string, unknown>>(
    `/sessions/${datasetId}/upload`,
    {
      method: 'POST',
      body: form,
    },
    { userId, timeoutMs: 120000 }
  );
}

export function getProfile(datasetId: string, userId: string) {
  return apiRequest<ProfileResponse>(`/sessions/${datasetId}/profile`, undefined, { userId });
}

export function getPreview(datasetId: string, userId: string, limit = 100) {
  return apiRequest<PreviewResponse>(`/sessions/${datasetId}/preview?limit=${encodeURIComponent(String(limit))}`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function getFacts(datasetId: string, userId: string, mode = 'auto') {
  return apiRequest<FactsResponse>(`/sessions/${datasetId}/facts?mode=${encodeURIComponent(mode)}`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function getAnomalyAnalysis(datasetId: string, userId: string, limit = 6) {
  return apiRequest<AnomalyAnalysisResponse>(
    `/sessions/${datasetId}/anomalies?limit=${encodeURIComponent(String(limit))}`,
    undefined,
    {
      userId,
      timeoutMs: 120000,
    }
  );
}

export function getCohortAnalysis(datasetId: string, userId: string) {
  return apiRequest<CohortAnalysisResponse>(`/sessions/${datasetId}/cohorts`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function buildCohortAnalysis(
  datasetId: string,
  userId: string,
  payload: {
    name?: string;
    description?: string;
    criteria: Array<{
      field: string;
      operator: string;
      value?: unknown;
    }>;
    limit?: number;
  }
) {
  return apiRequest<CohortAnalysisResponse>(
    `/sessions/${datasetId}/cohorts`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function getWorkflowActions(datasetId: string, userId: string) {
  return apiRequest<WorkflowActionsResponse>(`/sessions/${datasetId}/workflow-actions`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function getSavedInvestigations(datasetId: string, userId: string) {
  return apiRequest<SavedInvestigationsResponse>(`/sessions/${datasetId}/investigations`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function saveInvestigation(
  datasetId: string,
  userId: string,
  payload: {
    title?: string;
    question: string;
    context_type?: string;
    note?: string;
    result: Record<string, unknown>;
  }
) {
  return apiRequest<SavedInvestigationRecord>(
    `/sessions/${datasetId}/investigations`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function getSavedPlaybooks(datasetId: string, userId: string) {
  return apiRequest<SavedPlaybooksResponse>(`/sessions/${datasetId}/playbooks`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function getForecastRuns(datasetId: string, userId: string) {
  return apiRequest<ForecastRunsResponse>(`/sessions/${datasetId}/ml-runs`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function getForecastDrift(datasetId: string, userId: string, runId?: string, window = 6) {
  const params = new URLSearchParams();
  params.set('window', String(window));
  if (runId) {
    params.set('run_id', runId);
  }
  return apiRequest<ForecastDriftResponse>(`/sessions/${datasetId}/ml-runs/drift?${params.toString()}`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function getModelRegistry(datasetId: string, userId: string) {
  return apiRequest<ModelRegistryResponse>(`/sessions/${datasetId}/ml-registry`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function getModelEvaluation(datasetId: string, userId: string, challengerRunId?: string) {
  const params = new URLSearchParams();
  if (challengerRunId) {
    params.set('challenger_run_id', challengerRunId);
  }
  const suffix = params.toString() ? `?${params.toString()}` : '';
  return apiRequest<ModelEvaluationResponse>(`/sessions/${datasetId}/ml-evaluation${suffix}`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function promoteModelRun(
  datasetId: string,
  runId: string,
  userId: string,
  payload?: {
    note?: string;
  }
) {
  return apiRequest<ModelRegistryEntry>(
    `/sessions/${datasetId}/ml-registry/promote/${runId}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload ?? {}),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function trainForecastRun(
  datasetId: string,
  userId: string,
  payload: {
    name?: string;
    time_field?: string;
    metric_field?: string;
    horizon?: number;
    aggregation?: 'sum' | 'mean';
  }
) {
  return apiRequest<ForecastRunRecord>(
    `/sessions/${datasetId}/ml-runs/forecast`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function savePlaybook(
  datasetId: string,
  userId: string,
  payload: {
    name: string;
    question_template: string;
    description?: string;
    context_type?: string;
  }
) {
  return apiRequest<SavedPlaybookRecord>(
    `/sessions/${datasetId}/playbooks`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function draftWorkflowAction(
  datasetId: string,
  userId: string,
  payload: {
    action_type: string;
    title?: string;
    target?: string;
    objective?: string;
  }
) {
  return apiRequest<WorkflowActionRecord>(
    `/sessions/${datasetId}/workflow-actions/draft`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function reviewWorkflowAction(
  datasetId: string,
  actionId: string,
  userId: string,
  payload: {
    approved: boolean;
    note?: string;
  }
) {
  return apiRequest<WorkflowActionRecord>(
    `/sessions/${datasetId}/workflow-actions/${actionId}/decision`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function executeWorkflowAction(
  datasetId: string,
  actionId: string,
  userId: string,
  payload?: {
    note?: string;
  }
) {
  return apiRequest<WorkflowActionRecord>(
    `/sessions/${datasetId}/workflow-actions/${actionId}/execute`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload ?? {}),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function generateDashboardSpec(datasetId: string, userId: string, template = 'health_core') {
  return apiRequest<DashboardSpecResponse>(
    `/sessions/${datasetId}/dashboard-spec`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ template }),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function getDashboardSpec(datasetId: string, userId: string) {
  return apiRequest<DashboardSpecResponse>(`/sessions/${datasetId}/dashboard-spec`, undefined, {
    userId,
    timeoutMs: 120000,
  });
}

export function askDataset(datasetId: string, userId: string, question: string) {
  return apiRequest<AskResponsePayload>(
    `/sessions/${datasetId}/ask`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, mode: 'safe' }),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function generateReport(datasetId: string, userId: string) {
  return apiRequest<ReportJobResponse>(
    `/sessions/${datasetId}/report`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    },
    { userId, timeoutMs: 120000 }
  );
}

export function getJobStatus(jobId: string, userId: string) {
  return apiRequest<JobStatus>(`/jobs/${jobId}`, undefined, { userId });
}

export function getAudit(datasetId: string, userId: string) {
  return apiRequest<AuditResponse>(`/sessions/${datasetId}/audit`, undefined, { userId });
}

export function getSensitiveExportStatus(datasetId: string, userId: string) {
  return apiRequest<SensitiveExportStatusResponse>(`/sessions/${datasetId}/sensitive-export`, undefined, { userId });
}

export function requestSensitiveExportApproval(datasetId: string, userId: string, justification: string) {
  return apiRequest<SensitiveExportStatusResponse>(
    `/sessions/${datasetId}/sensitive-export/request`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ justification }),
    },
    { userId }
  );
}

export function reviewSensitiveExportApproval(
  datasetId: string,
  userId: string,
  payload: {
    approved: boolean;
    note?: string;
  }
) {
  return apiRequest<SensitiveExportStatusResponse>(
    `/sessions/${datasetId}/sensitive-export/decision`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    },
    { userId }
  );
}

export function setSensitiveExportEnabled(datasetId: string, userId: string, enabled: boolean) {
  return apiRequest<SensitiveExportStatusResponse>(
    `/sessions/${datasetId}/sensitive-export`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled }),
    },
    { userId }
  );
}

export function reportPdfUrl(datasetId: string): string {
  return authenticatedUrl(`/sessions/${datasetId}/export/pdf`);
}

export function reportHtmlUrl(datasetId: string): string {
  return authenticatedUrl(`/sessions/${datasetId}/report/html`);
}

export function factsJsonUrl(datasetId: string): string {
  return authenticatedUrl(`/sessions/${datasetId}/export/json`);
}

export function datasetCsvUrl(datasetId: string): string {
  return authenticatedUrl(`/sessions/${datasetId}/export/csv`);
}

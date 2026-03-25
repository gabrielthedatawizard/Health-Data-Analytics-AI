export interface SessionMeta {
  dataset_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  created_by: string;
  user_id?: string | null;
  pii_masking_enabled: boolean;
  allow_sensitive_export: boolean;
  file?: {
    name?: string;
    rows?: number;
    size_bytes?: number;
    content_type?: string;
  } | null;
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

export const ENV_API_BASE = import.meta.env.VITE_API_BASE_URL?.trim() ?? '';

export function computeDefaultApiBase(): string {
  if (typeof window === 'undefined') {
    return '';
  }
  const { hostname } = window.location;
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  return '';
}

export const API_BASE = ENV_API_BASE || computeDefaultApiBase();
export const API_TARGET_LABEL = API_BASE || 'same-origin';

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
  }
): Promise<T> {
  const url = apiUrl(path);
  const controller = new AbortController();
  const timeoutMs = options?.timeoutMs ?? 60000;
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const headers = new Headers(init?.headers ?? {});
  if (options?.userId?.trim()) {
    headers.set('X-API-Key', options.userId.trim());
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

export function createSession(userId: string) {
  return apiRequest<{ dataset_id: string; created_at: string }>(
    '/sessions',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ created_by: userId || 'react_user' }),
    },
    { userId }
  );
}

export function getSession(datasetId: string, userId: string) {
  return apiRequest<SessionMeta>(`/sessions/${datasetId}`, undefined, { userId });
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

export function getFacts(datasetId: string, userId: string, mode = 'auto') {
  return apiRequest<FactsResponse>(`/sessions/${datasetId}/facts?mode=${encodeURIComponent(mode)}`, undefined, {
    userId,
    timeoutMs: 120000,
  });
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

export function reportPdfUrl(datasetId: string): string {
  return apiUrl(`/sessions/${datasetId}/export/pdf`);
}

export function reportHtmlUrl(datasetId: string): string {
  return apiUrl(`/sessions/${datasetId}/report/html`);
}

export function factsJsonUrl(datasetId: string): string {
  return apiUrl(`/sessions/${datasetId}/export/json`);
}

export function datasetCsvUrl(datasetId: string): string {
  return apiUrl(`/sessions/${datasetId}/export/csv`);
}

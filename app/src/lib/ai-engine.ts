export type DatasetFileType = 'csv' | 'excel' | 'json';
export type DatasetStatus = 'active' | 'processing' | 'error';
export type DatasetSource = 'upload' | 'sample' | 'integration';
export type ColumnInferredType = 'number' | 'string' | 'boolean' | 'date' | 'unknown';
export type InsightType = 'trend' | 'anomaly' | 'prediction' | 'recommendation';
export type DatasetCellValue = string | number | boolean | null;
export type DatasetRow = Record<string, DatasetCellValue>;

export interface DatasetColumnProfile {
  name: string;
  inferredType: ColumnInferredType;
  nullPercentage: number;
  uniqueCount: number;
  sampleValues: string[];
  min?: number;
  max?: number;
  mean?: number;
}

export interface DatasetRecord {
  id: string;
  name: string;
  description: string;
  type: DatasetFileType;
  source: DatasetSource;
  status: DatasetStatus;
  rowCount: number;
  columnCount: number;
  qualityScore: number;
  sizeBytes: number;
  sizeLabel: string;
  issues: string[];
  tags: string[];
  createdBy: string;
  uploadedAt: string;
  lastUpdated: string;
  columns: DatasetColumnProfile[];
  sampleRows: DatasetRow[];
  metrics: Record<string, number>;
}

export interface InsightRecord {
  id: string;
  datasetId?: string;
  type: InsightType;
  title: string;
  content: string;
  confidence: number;
  citations: string[];
  timestamp: string;
  verified: boolean;
  userFeedback?: 'positive' | 'negative';
  expanded: boolean;
  flagged?: boolean;
  sourceQuestion?: string;
}

export interface AIEnvironmentState {
  status: 'ready' | 'busy' | 'error';
  mode: 'local-guarded';
  model: string;
  version: string;
  lastRunAt: string | null;
  lastError?: string;
}

export interface UploadProgressEvent {
  progress: number;
  status: 'uploading' | 'processing';
  message?: string;
}

export type UploadProgressHandler = (event: UploadProgressEvent) => void;

export type SampleDatasetKind = 'demo' | 'database' | 'dhis2';

interface BuildDatasetOptions {
  id?: string;
  name: string;
  type: DatasetFileType;
  source: DatasetSource;
  rows: DatasetRow[];
  sizeBytes: number;
  createdBy?: string;
  description?: string;
  tags?: string[];
  status?: DatasetStatus;
}

const MAX_ANALYSIS_ROWS = 10_000;
const MAX_PREVIEW_ROWS = 250;
const NUMBER_REGEX = /^[-+]?\d+(\.\d+)?$/;

export function createEntityId(prefix: string): string {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** index;
  const fixed = value >= 10 || index === 0 ? 0 : 1;
  return `${value.toFixed(fixed)} ${units[index]}`;
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, decimals = 1): number {
  const multiplier = 10 ** decimals;
  return Math.round(value * multiplier) / multiplier;
}

function normalizeHeader(raw: string, index: number, seen: Set<string>): string {
  const base = raw.trim().replace(/\s+/g, '_') || `column_${index + 1}`;
  let candidate = base;
  let counter = 2;
  while (seen.has(candidate)) {
    candidate = `${base}_${counter}`;
    counter += 1;
  }
  seen.add(candidate);
  return candidate;
}

function normalizeCell(value: unknown): DatasetCellValue {
  if (value === null || value === undefined) return null;
  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : null;
  }
  if (typeof value === 'boolean') return value;
  if (typeof value === 'string') {
    const trimmed = value.trim();
    if (!trimmed) return null;
    if (/^(true|false)$/i.test(trimmed)) return trimmed.toLowerCase() === 'true';
    if (NUMBER_REGEX.test(trimmed)) {
      const parsed = Number(trimmed);
      if (Number.isFinite(parsed)) return parsed;
    }
    return trimmed;
  }
  if (value instanceof Date) {
    return value.toISOString();
  }
  try {
    const serialized = JSON.stringify(value);
    return serialized === undefined ? null : serialized;
  } catch {
    return null;
  }
}

function inferColumnType(values: DatasetCellValue[]): ColumnInferredType {
  if (values.length === 0) return 'unknown';
  if (values.every((value) => typeof value === 'number')) return 'number';
  if (values.every((value) => typeof value === 'boolean')) return 'boolean';
  const stringValues = values.filter((value): value is string => typeof value === 'string');
  if (stringValues.length === values.length) {
    const dateLike = stringValues.filter((value) => !Number.isNaN(Date.parse(value))).length;
    if (dateLike / stringValues.length >= 0.8) return 'date';
    return 'string';
  }
  return 'string';
}

function parseDelimitedRows(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let currentField = '';
  let currentRow: string[] = [];
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    const nextCharacter = text[index + 1];

    if (character === '"') {
      if (inQuotes && nextCharacter === '"') {
        currentField += '"';
        index += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (!inQuotes && character === delimiter) {
      currentRow.push(currentField);
      currentField = '';
      continue;
    }

    if (!inQuotes && (character === '\n' || character === '\r')) {
      if (character === '\r' && nextCharacter === '\n') {
        index += 1;
      }
      currentRow.push(currentField);
      rows.push(currentRow);
      currentRow = [];
      currentField = '';
      continue;
    }

    currentField += character;
  }

  currentRow.push(currentField);
  if (currentRow.some((field) => field.trim().length > 0) || rows.length === 0) {
    rows.push(currentRow);
  }

  return rows;
}

function detectDelimiter(text: string): string {
  const firstNonEmptyLine = text
    .split(/\r?\n/)
    .find((line) => line.trim().length > 0) ?? '';
  const delimiters = [',', ';', '\t', '|'];
  let selected = ',';
  let highestCount = -1;

  delimiters.forEach((delimiter) => {
    const count = firstNonEmptyLine.split(delimiter).length - 1;
    if (count > highestCount) {
      highestCount = count;
      selected = delimiter;
    }
  });

  return selected;
}

function parseCsv(text: string): DatasetRow[] {
  const cleaned = text.replace(/^\uFEFF/, '');
  const delimiter = detectDelimiter(cleaned);
  const parsedRows = parseDelimitedRows(cleaned, delimiter).filter((row) =>
    row.some((cell) => cell.trim().length > 0)
  );

  if (parsedRows.length === 0) {
    throw new Error('The CSV file appears to be empty.');
  }

  const headerRow = parsedRows[0];
  const seenHeaders = new Set<string>();
  const headers = headerRow.map((header, index) => normalizeHeader(header, index, seenHeaders));
  const bodyRows = parsedRows.slice(1).slice(0, MAX_ANALYSIS_ROWS);

  if (bodyRows.length === 0) {
    throw new Error('The CSV file does not contain any data rows.');
  }

  return bodyRows.map((rowCells) => {
    const row: DatasetRow = {};
    headers.forEach((header, index) => {
      row[header] = normalizeCell(rowCells[index] ?? null);
    });
    return row;
  });
}

function extractRecordArray(value: unknown): Record<string, unknown>[] | null {
  if (Array.isArray(value)) {
    if (value.every((item) => item && typeof item === 'object' && !Array.isArray(item))) {
      return value as Record<string, unknown>[];
    }
    for (const item of value) {
      const nested = extractRecordArray(item);
      if (nested) return nested;
    }
    return null;
  }

  if (value && typeof value === 'object') {
    const objectValue = value as Record<string, unknown>;
    for (const nestedValue of Object.values(objectValue)) {
      const nested = extractRecordArray(nestedValue);
      if (nested) return nested;
    }
    if (Object.keys(objectValue).length > 0) {
      return [objectValue];
    }
  }

  return null;
}

function normalizeJsonRows(rawRows: Record<string, unknown>[]): DatasetRow[] {
  const headers = new Set<string>();
  rawRows.forEach((row) => {
    Object.keys(row).forEach((key) => headers.add(key.trim() || 'value'));
  });
  const orderedHeaders = Array.from(headers);
  if (orderedHeaders.length === 0) {
    throw new Error('JSON data does not contain object keys that can be analyzed.');
  }

  return rawRows.slice(0, MAX_ANALYSIS_ROWS).map((row) => {
    const normalized: DatasetRow = {};
    orderedHeaders.forEach((header) => {
      normalized[header] = normalizeCell(row[header]);
    });
    return normalized;
  });
}

function parseJson(text: string): DatasetRow[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error('The JSON file is not valid JSON.');
  }

  const records = extractRecordArray(parsed);
  if (!records || records.length === 0) {
    throw new Error('Could not find an array of records in this JSON file.');
  }

  return normalizeJsonRows(records);
}

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Failed reading ${file.name}.`));
    reader.onload = () => resolve(String(reader.result ?? ''));
    reader.readAsText(file);
  });
}

function detectDatasetType(file: File): DatasetFileType | null {
  const extension = file.name.split('.').pop()?.toLowerCase();
  if (extension === 'csv') return 'csv';
  if (extension === 'json') return 'json';
  if (extension === 'xlsx' || extension === 'xls') return 'excel';
  if (file.type.includes('csv')) return 'csv';
  if (file.type.includes('json')) return 'json';
  return null;
}

function buildColumnProfiles(rows: DatasetRow[], headers: string[]): DatasetColumnProfile[] {
  return headers.map((header) => {
    const values = rows.map((row) => row[header] ?? null);
    const nonNullValues = values.filter((value): value is Exclude<DatasetCellValue, null> => value !== null);
    const inferredType = inferColumnType(nonNullValues);
    const uniqueCount = new Set(nonNullValues.map((value) => String(value))).size;
    const nullPercentage = rows.length === 0 ? 0 : round(((rows.length - nonNullValues.length) / rows.length) * 100);
    const sampleValues = nonNullValues.slice(0, 3).map((value) => String(value));

    if (inferredType === 'number') {
      const numericValues = nonNullValues.filter((value): value is number => typeof value === 'number');
      const min = Math.min(...numericValues);
      const max = Math.max(...numericValues);
      const mean = round(numericValues.reduce((sum, value) => sum + value, 0) / numericValues.length, 2);
      return {
        name: header,
        inferredType,
        nullPercentage,
        uniqueCount,
        sampleValues,
        min,
        max,
        mean,
      };
    }

    return {
      name: header,
      inferredType,
      nullPercentage,
      uniqueCount,
      sampleValues,
    };
  });
}

function buildIssues(
  rows: DatasetRow[],
  columns: DatasetColumnProfile[],
  missingPercent: number,
  duplicateRows: number
): string[] {
  const issues: string[] = [];
  if (missingPercent > 5) {
    issues.push(`${round(missingPercent)}% of cells are missing values.`);
  }
  columns
    .filter((column) => column.nullPercentage >= 20)
    .slice(0, 3)
    .forEach((column) => {
      issues.push(`${column.name} has ${round(column.nullPercentage)}% missing values.`);
    });
  if (duplicateRows > 0) {
    issues.push(`${duplicateRows.toLocaleString()} duplicate rows detected.`);
  }
  if (rows.length < 20) {
    issues.push('Small sample size: fewer than 20 rows.');
  }
  if (columns.every((column) => column.inferredType !== 'number')) {
    issues.push('No numeric columns found for statistical analysis.');
  }
  return issues.slice(0, 4);
}

function buildTags(name: string, columns: DatasetColumnProfile[], type: DatasetFileType): string[] {
  const cleanedName = name
    .replace(/\.[^.]+$/, '')
    .split(/[_\-\s]+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 2)
    .slice(0, 2);
  const columnTags = columns.slice(0, 2).map((column) => column.name.replace(/_/g, ' '));
  return Array.from(new Set([type.toUpperCase(), ...cleanedName, ...columnTags]));
}

export function createDatasetFromRows(options: BuildDatasetOptions): DatasetRecord {
  const now = new Date().toISOString();
  const rows = options.rows.slice(0, MAX_ANALYSIS_ROWS);
  if (rows.length === 0) {
    throw new Error('No rows were found to analyze.');
  }
  const headers = Object.keys(rows[0]);
  if (headers.length === 0) {
    throw new Error('No columns were found to analyze.');
  }

  const columns = buildColumnProfiles(rows, headers);
  const totalCells = rows.length * headers.length;
  const missingCells = rows.reduce((sum, row) => {
    return (
      sum +
      headers.reduce((columnSum, header) => {
        return columnSum + (row[header] === null ? 1 : 0);
      }, 0)
    );
  }, 0);
  const missingPercent = totalCells === 0 ? 0 : (missingCells / totalCells) * 100;
  const rowFingerprints = new Set<string>();
  let duplicateRows = 0;
  rows.forEach((row) => {
    const fingerprint = JSON.stringify(row);
    if (rowFingerprints.has(fingerprint)) {
      duplicateRows += 1;
    } else {
      rowFingerprints.add(fingerprint);
    }
  });
  const duplicatePercent = rows.length === 0 ? 0 : (duplicateRows / rows.length) * 100;
  const qualityScore = clamp(
    Math.round(100 - missingPercent * 0.65 - duplicatePercent * 0.35),
    0,
    100
  );
  const issues = buildIssues(rows, columns, missingPercent, duplicateRows);
  const numericColumns = columns.filter((column) => column.inferredType === 'number');
  const dateColumns = columns.filter((column) => column.inferredType === 'date');

  const dataset: DatasetRecord = {
    id: options.id ?? createEntityId('dataset'),
    name: options.name,
    description:
      options.description ??
      `${rows.length.toLocaleString()} rows across ${headers.length} columns. Quality score ${qualityScore}%.`,
    type: options.type,
    source: options.source,
    status: options.status ?? 'active',
    rowCount: rows.length,
    columnCount: headers.length,
    qualityScore,
    sizeBytes: options.sizeBytes,
    sizeLabel: formatBytes(options.sizeBytes),
    issues,
    tags: options.tags ?? buildTags(options.name, columns, options.type),
    createdBy: options.createdBy ?? 'You',
    uploadedAt: now,
    lastUpdated: now,
    columns,
    sampleRows: rows.slice(0, MAX_PREVIEW_ROWS),
    metrics: {
      completeness: round(100 - missingPercent),
      duplicateRows,
      numericColumns: numericColumns.length,
      dateColumns: dateColumns.length,
    },
  };

  return dataset;
}

export async function parseUploadedFile(
  file: File,
  onProgress?: UploadProgressHandler
): Promise<DatasetRecord> {
  const detectedType = detectDatasetType(file);
  if (!detectedType) {
    throw new Error('Unsupported file format. Please upload CSV, JSON, or Excel.');
  }

  onProgress?.({ progress: 10, status: 'uploading', message: `Reading ${file.name}` });
  await wait(120);

  if (detectedType === 'excel') {
    throw new Error(
      'Excel parsing is not available in this browser-only mode yet. Please upload CSV or JSON.'
    );
  }

  const text = await readFileAsText(file);
  onProgress?.({ progress: 58, status: 'processing', message: 'Profiling dataset columns' });
  await wait(150);

  const rows = detectedType === 'csv' ? parseCsv(text) : parseJson(text);
  const dataset = createDatasetFromRows({
    name: file.name,
    type: detectedType,
    source: 'upload',
    rows,
    sizeBytes: file.size,
  });

  onProgress?.({ progress: 100, status: 'processing', message: 'Generating AI insights' });
  await wait(150);

  return dataset;
}

export function buildFailedDataset(file: File, errorMessage: string): DatasetRecord {
  const now = new Date().toISOString();
  const detectedType = detectDatasetType(file) ?? 'csv';
  return {
    id: createEntityId('dataset'),
    name: file.name,
    description: `Upload failed: ${errorMessage}`,
    type: detectedType,
    source: 'upload',
    status: 'error',
    rowCount: 0,
    columnCount: 0,
    qualityScore: 0,
    sizeBytes: file.size,
    sizeLabel: formatBytes(file.size),
    issues: [errorMessage],
    tags: [detectedType.toUpperCase(), 'Upload Error'],
    createdBy: 'You',
    uploadedAt: now,
    lastUpdated: now,
    columns: [],
    sampleRows: [],
    metrics: {},
  };
}

function pickPrimaryNumericColumn(dataset: DatasetRecord): DatasetColumnProfile | undefined {
  const numericColumns = dataset.columns.filter(
    (column) => column.inferredType === 'number' && column.mean !== undefined
  );
  if (numericColumns.length === 0) return undefined;
  return numericColumns.sort((left, right) => {
    const leftRange = (left.max ?? 0) - (left.min ?? 0);
    const rightRange = (right.max ?? 0) - (right.min ?? 0);
    return rightRange - leftRange;
  })[0];
}

function confidenceFromQuality(qualityScore: number, minConfidence = 65): number {
  return clamp(Math.round(qualityScore * 0.6 + 35), minConfidence, 99);
}

interface InsightGenerationOptions {
  expandedFirst?: boolean;
  sourceQuestion?: string;
}

export function generateInsightsForDataset(
  dataset: DatasetRecord,
  options: InsightGenerationOptions = {}
): InsightRecord[] {
  const now = new Date().toISOString();
  const citations = [dataset.name];
  const insights: InsightRecord[] = [];

  insights.push({
    id: createEntityId('insight'),
    datasetId: dataset.id,
    type: 'trend',
    title: `${dataset.name} profile ready`,
    content: `${dataset.rowCount.toLocaleString()} rows and ${dataset.columnCount} columns analyzed. Data quality is ${dataset.qualityScore}%.`,
    confidence: confidenceFromQuality(dataset.qualityScore),
    citations,
    timestamp: now,
    verified: true,
    expanded: options.expandedFirst ?? true,
    sourceQuestion: options.sourceQuestion,
  });

  if (dataset.issues.length > 0) {
    insights.push({
      id: createEntityId('insight'),
      datasetId: dataset.id,
      type: 'anomaly',
      title: 'Data quality checks detected issues',
      content: dataset.issues.join(' '),
      confidence: clamp(95 - dataset.issues.length * 3, 75, 98),
      citations,
      timestamp: now,
      verified: true,
      expanded: false,
      sourceQuestion: options.sourceQuestion,
    });
  }

  const numericColumn = pickPrimaryNumericColumn(dataset);
  if (numericColumn && numericColumn.mean !== undefined) {
    insights.push({
      id: createEntityId('insight'),
      datasetId: dataset.id,
      type: 'prediction',
      title: `${numericColumn.name} baseline estimate`,
      content: `Observed mean is ${round(numericColumn.mean, 2)} with a range of ${round(
        numericColumn.min ?? 0,
        2
      )} to ${round(numericColumn.max ?? 0, 2)}. This baseline can support short-term forecasting.`,
      confidence: confidenceFromQuality(dataset.qualityScore, 60),
      citations,
      timestamp: now,
      verified: true,
      expanded: false,
      sourceQuestion: options.sourceQuestion,
    });
  }

  insights.push({
    id: createEntityId('insight'),
    datasetId: dataset.id,
    type: 'recommendation',
    title: 'Recommended next step',
    content:
      dataset.qualityScore >= 85
        ? 'Data quality is strong enough for dashboard generation. Proceed to KPI and trend visualization.'
        : 'Clean missing values and duplicates before publishing dashboards to avoid biased insights.',
    confidence: confidenceFromQuality(dataset.qualityScore, 70),
    citations,
    timestamp: now,
    verified: true,
    expanded: false,
    sourceQuestion: options.sourceQuestion,
  });

  return insights;
}

function pickDatasetForQuestion(question: string, datasets: DatasetRecord[]): DatasetRecord | null {
  if (datasets.length === 0) return null;
  const lowerQuestion = question.toLowerCase();
  const byName = datasets.find((dataset) =>
    lowerQuestion.includes(dataset.name.toLowerCase().replace(/\.[^.]+$/, ''))
  );
  if (byName) return byName;
  const activeDatasets = datasets.filter((dataset) => dataset.status === 'active');
  if (activeDatasets.length > 0) {
    return [...activeDatasets].sort((left, right) =>
      right.uploadedAt.localeCompare(left.uploadedAt)
    )[0];
  }
  return [...datasets].sort((left, right) => right.uploadedAt.localeCompare(left.uploadedAt))[0];
}

function createQuestionInsight(
  dataset: DatasetRecord | null,
  question: string
): Omit<InsightRecord, 'expanded'> {
  const normalizedQuestion = question.trim();
  const now = new Date().toISOString();

  if (!dataset) {
    return {
      id: createEntityId('insight'),
      type: 'recommendation',
      title: 'Upload data to start AI analysis',
      content: 'No dataset is available yet. Upload a CSV or JSON file, then ask your question again.',
      confidence: 99,
      citations: [],
      timestamp: now,
      verified: true,
      sourceQuestion: normalizedQuestion,
    };
  }

  const lowerQuestion = normalizedQuestion.toLowerCase();
  const citations = [dataset.name];
  let type: InsightType = 'trend';
  let title = `Answer from ${dataset.name}`;
  let content = `${dataset.rowCount.toLocaleString()} rows and ${dataset.columnCount} columns are currently available.`;
  let confidence = confidenceFromQuality(dataset.qualityScore, 70);

  const numericColumn = pickPrimaryNumericColumn(dataset);
  if (lowerQuestion.includes('quality') || lowerQuestion.includes('clean')) {
    type = 'anomaly';
    title = 'Data quality status';
    content = `Quality score is ${dataset.qualityScore}%. ${dataset.issues.length > 0 ? dataset.issues.join(' ') : 'No major quality issues were detected.'}`;
    confidence = 97;
  } else if (
    lowerQuestion.includes('missing') ||
    lowerQuestion.includes('issue') ||
    lowerQuestion.includes('duplicate')
  ) {
    type = 'anomaly';
    title = 'Detected data issues';
    content =
      dataset.issues.length > 0
        ? dataset.issues.join(' ')
        : 'No critical missing-value or duplicate-row issue was detected in the current sample.';
    confidence = 95;
  } else if (
    lowerQuestion.includes('row') ||
    lowerQuestion.includes('record') ||
    lowerQuestion.includes('column')
  ) {
    type = 'trend';
    title = 'Dataset size summary';
    content = `${dataset.name} contains ${dataset.rowCount.toLocaleString()} rows and ${dataset.columnCount.toLocaleString()} columns.`;
    confidence = 98;
  } else if (
    numericColumn &&
    (lowerQuestion.includes('average') ||
      lowerQuestion.includes('mean') ||
      lowerQuestion.includes('max') ||
      lowerQuestion.includes('min'))
  ) {
    type = 'trend';
    title = `${numericColumn.name} statistical summary`;
    content = `Mean ${round(numericColumn.mean ?? 0, 2)}, minimum ${round(
      numericColumn.min ?? 0,
      2
    )}, and maximum ${round(numericColumn.max ?? 0, 2)}.`;
    confidence = confidenceFromQuality(dataset.qualityScore, 75);
  } else if (
    lowerQuestion.includes('predict') ||
    lowerQuestion.includes('forecast') ||
    lowerQuestion.includes('next')
  ) {
    type = 'prediction';
    title = 'Short-term forecast guidance';
    content =
      numericColumn && numericColumn.mean !== undefined
        ? `Using ${numericColumn.name} as baseline (${round(
            numericColumn.mean,
            2
          )}), use moving averages over the latest periods for a near-term forecast.`
        : 'No numeric feature is available for a statistical forecast. Add at least one numeric metric column.';
    confidence = confidenceFromQuality(dataset.qualityScore, 62);
  } else if (
    lowerQuestion.includes('recommend') ||
    lowerQuestion.includes('action') ||
    lowerQuestion.includes('improve')
  ) {
    type = 'recommendation';
    title = 'Recommended action';
    content =
      dataset.qualityScore >= 85
        ? 'Proceed to dashboard generation and set alerts for metric drift.'
        : 'Start with missing-value remediation and duplicate cleanup, then regenerate insights.';
    confidence = confidenceFromQuality(dataset.qualityScore, 72);
  } else {
    const previewColumns = dataset.columns.slice(0, 3).map((column) => column.name);
    content = `Top columns include ${previewColumns.join(', ')}. Quality is ${dataset.qualityScore}%, and ${
      dataset.issues.length
    } issue(s) are currently flagged.`;
  }

  return {
    id: createEntityId('insight'),
    datasetId: dataset.id,
    type,
    title,
    content,
    confidence,
    citations,
    timestamp: now,
    verified: true,
    sourceQuestion: normalizedQuestion,
  };
}

export function answerQuestion(question: string, datasets: DatasetRecord[]): InsightRecord {
  const targetDataset = pickDatasetForQuestion(question, datasets);
  const insight = createQuestionInsight(targetDataset, question);
  return {
    ...insight,
    expanded: true,
  };
}

function buildDemoRows(): DatasetRow[] {
  const districts = ['Chamwino', 'Bahi', 'Dodoma Municipal', 'Chemba', 'Kondoa', 'Mpwapwa'];
  const months = ['2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12'];
  const rows: DatasetRow[] = [];
  months.forEach((month, monthIndex) => {
    districts.forEach((district, districtIndex) => {
      const ancCoverage = 58 + monthIndex * 2 + districtIndex;
      rows.push({
        month,
        district,
        anc_coverage: ancCoverage,
        facility_delivery: ancCoverage + 8,
        maternal_mortality: Math.max(52, 110 - monthIndex * 5 - districtIndex * 2),
        missing_records: districtIndex % 3 === 0 ? 2 + monthIndex : 1 + (monthIndex % 2),
      });
    });
  });
  return rows;
}

function buildDatabaseRows(): DatasetRow[] {
  return Array.from({ length: 32 }, (_, index) => {
    const day = (index % 30) + 1;
    return {
      facility_id: `FAC-${String((index % 8) + 1).padStart(3, '0')}`,
      event_date: `2026-01-${String(day).padStart(2, '0')}`,
      total_visits: 120 + ((index * 7) % 40),
      completed_forms: 110 + ((index * 5) % 35),
      data_source: 'postgresql',
    };
  });
}

function buildDhisRows(): DatasetRow[] {
  const districts = ['Dodoma', 'Iringa', 'Morogoro', 'Tabora'];
  return Array.from({ length: 24 }, (_, index) => {
    const district = districts[index % districts.length];
    return {
      period: `2026Q${(index % 4) + 1}`,
      district,
      indicator: 'ANC4_COVERAGE',
      value: 62 + ((index * 3) % 19),
      source: 'DHIS2',
      verified: index % 5 !== 0,
    };
  });
}

export function createSampleDataset(kind: SampleDatasetKind = 'demo'): DatasetRecord {
  const rows = kind === 'database' ? buildDatabaseRows() : kind === 'dhis2' ? buildDhisRows() : buildDemoRows();
  const name =
    kind === 'database'
      ? 'Database_Import_Sample.csv'
      : kind === 'dhis2'
        ? 'DHIS2_Import_Sample.csv'
        : 'HealthAI_Demo_Dataset.csv';
  const description =
    kind === 'database'
      ? 'Simulated relational database extract for facility operations.'
      : kind === 'dhis2'
        ? 'Simulated DHIS2 indicator export.'
        : 'Demo maternal health indicators for six districts.';

  const roughSize = new Blob([JSON.stringify(rows)]).size;
  return createDatasetFromRows({
    name,
    type: 'csv',
    source: kind === 'demo' ? 'sample' : 'integration',
    rows,
    sizeBytes: roughSize,
    createdBy: 'HealthAI Demo',
    description,
    tags: kind === 'demo' ? ['DEMO', 'Maternal Health', 'Coverage'] : undefined,
  });
}

function escapeCsvValue(value: DatasetCellValue): string {
  if (value === null) return '';
  const stringValue = String(value);
  if (/[",\n\r]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }
  return stringValue;
}

export function datasetRowsToCsv(dataset: DatasetRecord): string {
  const headers = dataset.columns.map((column) => column.name);
  if (headers.length === 0) {
    return '';
  }
  const lines = [headers.join(',')];
  dataset.sampleRows.forEach((row) => {
    const line = headers.map((header) => escapeCsvValue(row[header] ?? null)).join(',');
    lines.push(line);
  });
  return lines.join('\n');
}

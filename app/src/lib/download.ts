import { datasetRowsToCsv, type DatasetRecord } from '@/lib/ai-engine';

function triggerDownload(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export function downloadTextFile(
  filename: string,
  content: string,
  mimeType = 'text/plain;charset=utf-8'
): void {
  const blob = new Blob([content], { type: mimeType });
  triggerDownload(filename, blob);
}

export function downloadJsonFile(filename: string, payload: unknown): void {
  const content = JSON.stringify(payload, null, 2);
  downloadTextFile(filename, content, 'application/json;charset=utf-8');
}

export function downloadDatasetCsv(dataset: DatasetRecord): void {
  const csv = datasetRowsToCsv(dataset);
  downloadTextFile(`${dataset.name.replace(/\.[^.]+$/, '') || 'dataset'}.csv`, csv, 'text/csv;charset=utf-8');
}

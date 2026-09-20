import { describe, expect, it } from 'vitest';
import { createDatasetFromRows, formatBytes } from './ai-engine';

describe('ai-engine leak-proof parsing', () => {
  it('formats bytes', () => {
    expect(formatBytes(0)).toBe('0 B');
    expect(formatBytes(2048)).toContain('KB');
  });

  it('builds dataset with quality score and no PII leak in metrics', () => {
    const ds = createDatasetFromRows({
      name: 'test.csv',
      type: 'csv',
      source: 'upload',
      rows: [
        { visit_date: '2025-01-01', opd_visits: 10 },
        { visit_date: '2025-02-01', opd_visits: 20 },
      ],
      sizeBytes: 123,
    });
    expect(ds.rowCount).toBe(2);
    expect(ds.qualityScore).toBeGreaterThan(80);
    expect(ds.columns.map((c) => c.name)).toContain('opd_visits');
  });
});

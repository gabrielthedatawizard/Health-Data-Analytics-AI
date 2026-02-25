import { useCallback, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  Database,
  FileSpreadsheet,
  FileText,
  Loader2,
  Sparkles,
  Upload,
  X,
} from 'lucide-react';
import type { ViewType } from '@/App';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useI18n } from '@/lib/i18n';
import { cn } from '@/lib/utils';
import { useAnalytics } from '@/lib/analytics-context';
import type { DatasetRecord, SampleDatasetKind } from '@/lib/ai-engine';

interface DataUploadProps {
  onViewChange: (view: ViewType) => void;
}

interface ActiveUploadState {
  name: string;
  progress: number;
  status: 'uploading' | 'processing';
  message?: string;
  error?: string;
}

function pause(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function typeLabel(type: DatasetRecord['type']): string {
  if (type === 'csv') return 'CSV';
  if (type === 'json') return 'JSON';
  return 'Excel';
}

function datasetTypeIcon(type: DatasetRecord['type']) {
  if (type === 'csv') return FileSpreadsheet;
  if (type === 'json') return Database;
  return FileText;
}

export function DataUpload({ onViewChange }: DataUploadProps) {
  const { t, formatTime } = useI18n();
  const { datasets, uploadFile, removeDataset, analyzeDataset, addSampleDataset, aiEnvironment } =
    useAnalytics();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadingFile, setUploadingFile] = useState<ActiveUploadState | null>(null);
  const [analyzingId, setAnalyzingId] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);

  const sortedDatasets = useMemo(() => {
    return [...datasets].sort((left, right) => right.uploadedAt.localeCompare(left.uploadedAt));
  }, [datasets]);

  const processFiles = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;
      for (const file of files) {
        setUploadingFile({
          name: file.name,
          progress: 0,
          status: 'uploading',
          message: t.uploading,
        });
        try {
          await uploadFile(file, (progressEvent) => {
            setUploadingFile({
              name: file.name,
              progress: progressEvent.progress,
              status: progressEvent.status,
              message: progressEvent.message,
            });
          });
          setActionMessage(`${file.name} uploaded successfully.`);
          await pause(300);
        } catch (error) {
          const message = error instanceof Error ? error.message : t.error;
          setUploadingFile((previous) =>
            previous
              ? {
                  ...previous,
                  error: message,
                }
              : {
                  name: file.name,
                  progress: 0,
                  status: 'uploading',
                  error: message,
                }
          );
          setActionMessage(`Upload failed for ${file.name}.`);
          await pause(1100);
        }
      }
      setUploadingFile(null);
    },
    [t.error, t.uploading, uploadFile]
  );

  const handleFileSelect = useCallback(
    async (event: ChangeEvent<HTMLInputElement>) => {
      const selectedFiles = event.target.files ? Array.from(event.target.files) : [];
      await processFiles(selectedFiles);
      event.target.value = '';
    },
    [processFiles]
  );

  const handleDrop = useCallback(
    async (event: DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(false);
      const droppedFiles = Array.from(event.dataTransfer.files);
      await processFiles(droppedFiles);
    },
    [processFiles]
  );

  const handleAnalyze = useCallback(
    async (dataset: DatasetRecord) => {
      if (dataset.status !== 'active') return;
      setAnalyzingId(dataset.id);
      try {
        await analyzeDataset(dataset.id);
        onViewChange('insights');
      } finally {
        setAnalyzingId(null);
      }
    },
    [analyzeDataset, onViewChange]
  );

  const handleSourceImport = useCallback(
    (kind: SampleDatasetKind, label: string) => {
      addSampleDataset(kind);
      setActionMessage(`${label} imported successfully.`);
      onViewChange('datasets');
    },
    [addSampleDataset, onViewChange]
  );

  return (
    <div className="space-y-6">
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.json,.xlsx,.xls"
        multiple
        className="hidden"
        onChange={handleFileSelect}
      />

      <div>
        <h1 className="text-2xl font-bold text-foreground">{t.dataUpload}</h1>
        <p className="text-muted-foreground mt-1">{t.dragDropFiles}</p>
      </div>

      {actionMessage && (
        <div className="flex items-center justify-between rounded-xl border border-border bg-muted/40 px-4 py-3">
          <span className="text-sm text-foreground">{actionMessage}</span>
          <Button variant="ghost" size="sm" onClick={() => setActionMessage(null)}>
            {t.dismiss}
          </Button>
        </div>
      )}

      <div
        onDragOver={(event) => {
          event.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={(event) => {
          event.preventDefault();
          setIsDragging(false);
        }}
        onDrop={handleDrop}
        className={cn(
          'relative rounded-2xl border-2 border-dashed p-12 text-center transition-all duration-300',
          isDragging
            ? 'border-health-mint bg-health-mint/10'
            : 'border-border bg-card/50 hover:border-muted-foreground/50'
        )}
      >
        <div className="absolute inset-0 gradient-glow-mint opacity-0 hover:opacity-30 transition-opacity" />
        <div className="relative">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl gradient-mint flex items-center justify-center">
            <Upload className="w-8 h-8 text-background" />
          </div>
          <h3 className="text-lg font-semibold text-foreground mb-2">{t.dragDropFiles}</h3>
          <p className="text-sm text-muted-foreground mb-4">{t.orClickToBrowse}</p>

          <div className="flex items-center justify-center gap-2 mb-6">
            <Badge variant="outline" className="gap-1.5">
              <FileSpreadsheet className="w-3 h-3" />
              CSV
            </Badge>
            <Badge variant="outline" className="gap-1.5">
              <FileText className="w-3 h-3" />
              Excel
            </Badge>
            <Badge variant="outline" className="gap-1.5">
              <Database className="w-3 h-3" />
              JSON
            </Badge>
          </div>

          <Button
            onClick={() => fileInputRef.current?.click()}
            className="gap-2 gradient-mint text-background hover:opacity-90"
          >
            <Upload className="w-4 h-4" />
            {t.upload}
          </Button>
        </div>
      </div>

      {uploadingFile && (
        <Card
          className={cn(
            'glass-card',
            uploadingFile.error ? 'border-red-500/30' : 'border-health-mint/30'
          )}
        >
          <CardContent className="p-5">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-lg bg-health-mint/20 flex items-center justify-center">
                {uploadingFile.error ? (
                  <AlertCircle className="w-5 h-5 text-red-400" />
                ) : (
                  <Loader2 className="w-5 h-5 text-health-mint animate-spin" />
                )}
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2 gap-4">
                  <div>
                    <p className="font-medium text-foreground">{uploadingFile.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {uploadingFile.error ? uploadingFile.error : uploadingFile.message}
                    </p>
                  </div>
                  {!uploadingFile.error && (
                    <span className="text-sm text-muted-foreground">{uploadingFile.progress}%</span>
                  )}
                </div>
                {!uploadingFile.error && <Progress value={uploadingFile.progress} className="h-2" />}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <button
          type="button"
          className="text-left"
          onClick={() => handleSourceImport('database', 'Database sample')}
        >
          <Card className="glass-card card-hover cursor-pointer">
            <CardContent className="p-5">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <Database className="w-5 h-5 text-blue-400" />
                </div>
                <div>
                  <h4 className="font-medium text-foreground">{t.connect}</h4>
                  <p className="text-xs text-muted-foreground mt-1">PostgreSQL, MySQL, SQL Server</p>
                </div>
                <ChevronRight className="w-4 h-4 text-muted-foreground ml-auto" />
              </div>
            </CardContent>
          </Card>
        </button>

        <button type="button" className="text-left" onClick={() => handleSourceImport('dhis2', 'DHIS2 sample')}>
          <Card className="glass-card card-hover cursor-pointer">
            <CardContent className="p-5">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center">
                  <Database className="w-5 h-5 text-purple-400" />
                </div>
                <div>
                  <h4 className="font-medium text-foreground">DHIS2</h4>
                  <p className="text-xs text-muted-foreground mt-1">{t.import} DHIS2</p>
                </div>
                <ChevronRight className="w-4 h-4 text-muted-foreground ml-auto" />
              </div>
            </CardContent>
          </Card>
        </button>

        <button type="button" className="text-left" onClick={() => handleSourceImport('demo', 'Demo dataset')}>
          <Card className="glass-card card-hover cursor-pointer">
            <CardContent className="p-5">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-emerald-400" />
                </div>
                <div>
                  <h4 className="font-medium text-foreground">{t.sample}</h4>
                  <p className="text-xs text-muted-foreground mt-1">{t.exploreDemo}</p>
                </div>
                <ChevronRight className="w-4 h-4 text-muted-foreground ml-auto" />
              </div>
            </CardContent>
          </Card>
        </button>
      </div>

      <div>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-foreground">{t.recentUploads}</h3>
          <Badge variant="outline" className="gap-1">
            <Sparkles className="w-3 h-3" />
            AI engine: {aiEnvironment.status}
          </Badge>
        </div>

        <div className="space-y-3">
          {sortedDatasets.map((dataset) => {
            const Icon = datasetTypeIcon(dataset.type);
            const isAnalyzing = analyzingId === dataset.id;
            const hasIssues = dataset.issues.length > 0;
            return (
              <Card key={dataset.id} className="glass-card card-hover">
                <CardContent className="p-5">
                  <div className="flex items-start gap-4">
                    <div
                      className={cn(
                        'w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0',
                        dataset.type === 'csv'
                          ? 'bg-emerald-500/20'
                          : dataset.type === 'json'
                            ? 'bg-purple-500/20'
                            : 'bg-blue-500/20'
                      )}
                    >
                      <Icon
                        className={cn(
                          'w-6 h-6',
                          dataset.type === 'csv'
                            ? 'text-emerald-400'
                            : dataset.type === 'json'
                              ? 'text-purple-400'
                              : 'text-blue-400'
                        )}
                      />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <h4 className="font-medium text-foreground truncate">{dataset.name}</h4>
                          <div className="flex items-center gap-3 mt-1 flex-wrap">
                            <span className="text-xs text-muted-foreground">{dataset.sizeLabel}</span>
                            <span className="text-xs text-muted-foreground">•</span>
                            <span className="text-xs text-muted-foreground">
                              {dataset.rowCount.toLocaleString()} {t.rows}
                            </span>
                            <span className="text-xs text-muted-foreground">•</span>
                            <span className="text-xs text-muted-foreground">
                              {dataset.columnCount.toLocaleString()} {t.columns}
                            </span>
                            <span className="text-xs text-muted-foreground">•</span>
                            <span className="text-xs text-muted-foreground">{formatTime(dataset.uploadedAt)}</span>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">
                            {typeLabel(dataset.type)}
                          </Badge>
                          <Badge
                            variant="outline"
                            className={cn(
                              'gap-1',
                              dataset.status === 'error'
                                ? 'border-red-500/30 text-red-400'
                                : dataset.qualityScore >= 85
                                  ? 'border-emerald-500/30 text-emerald-400'
                                  : dataset.qualityScore >= 70
                                    ? 'border-amber-500/30 text-amber-400'
                                    : 'border-red-500/30 text-red-400'
                            )}
                          >
                            {dataset.status === 'error' ? (
                              <>
                                <AlertCircle className="w-3 h-3" />
                                {t.failed}
                              </>
                            ) : (
                              <>
                                <CheckCircle2 className="w-3 h-3" />
                                {t.qualityScore}: {dataset.qualityScore}%
                              </>
                            )}
                          </Badge>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => removeDataset(dataset.id)}
                          >
                            <X className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>

                      {hasIssues ? (
                        <div className="flex items-center gap-2 mt-3 flex-wrap">
                          <AlertCircle className="w-4 h-4 text-amber-500" />
                          <span className="text-xs text-amber-400">{dataset.issues.join(' ')}</span>
                          <button
                            type="button"
                            className="text-xs text-health-mint hover:underline"
                            onClick={() => {
                              window.alert(dataset.issues.join('\n'));
                            }}
                          >
                            {t.review}
                          </button>
                        </div>
                      ) : (
                        <div className="flex items-center gap-2 mt-3">
                          <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                          <span className="text-xs text-emerald-400">{t.noIssues}</span>
                        </div>
                      )}
                    </div>

                    <Button
                      variant="outline"
                      size="sm"
                      disabled={dataset.status !== 'active' || isAnalyzing}
                      className="gap-1 border-health-mint/30 text-health-mint hover:bg-health-mint/10"
                      onClick={() => {
                        void handleAnalyze(dataset);
                      }}
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2 className="w-3 h-3 animate-spin" />
                          {t.analyzing}
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-3 h-3" />
                          {t.analyze}
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
}

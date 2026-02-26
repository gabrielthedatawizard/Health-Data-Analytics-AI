import { useMemo, useState } from 'react';
import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Clock,
  Database,
  Download,
  Edit,
  Eye,
  FileSpreadsheet,
  FileText,
  Filter,
  MoreHorizontal,
  Plus,
  Search,
  Sparkles,
  Trash2,
} from 'lucide-react';
import type { ViewType } from '@/App';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Input } from '@/components/ui/input';
import { useAnalytics } from '@/lib/analytics-context';
import type { DatasetRecord } from '@/lib/ai-engine';
import { downloadDatasetCsv, downloadJsonFile } from '@/lib/download';
import { cn } from '@/lib/utils';

interface DatasetsViewProps {
  onViewChange: (view: ViewType) => void;
}

type StatusFilter = 'all' | 'active' | 'processing' | 'error';

const statusLabels: Record<StatusFilter, string> = {
  all: 'All',
  active: 'Active',
  processing: 'Processing',
  error: 'Error',
};

function statusCycle(next: StatusFilter): StatusFilter {
  if (next === 'all') return 'active';
  if (next === 'active') return 'processing';
  if (next === 'processing') return 'error';
  return 'all';
}

function datasetIcon(dataset: DatasetRecord) {
  if (dataset.type === 'csv') return FileSpreadsheet;
  if (dataset.type === 'json') return Database;
  return FileText;
}

function datasetColorClasses(type: DatasetRecord['type']): string {
  if (type === 'csv') return 'bg-emerald-500/20 text-emerald-400';
  if (type === 'json') return 'bg-purple-500/20 text-purple-400';
  return 'bg-blue-500/20 text-blue-400';
}

export function DatasetsView({ onViewChange }: DatasetsViewProps) {
  const { datasets, removeDataset, renameDataset, analyzeDataset } = useAnalytics();
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [previewDataset, setPreviewDataset] = useState<DatasetRecord | null>(null);
  const [analyzingId, setAnalyzingId] = useState<string | null>(null);

  const filteredDatasets = useMemo(() => {
    const normalizedQuery = searchQuery.trim().toLowerCase();
    return [...datasets]
      .sort((left, right) => right.uploadedAt.localeCompare(left.uploadedAt))
      .filter((dataset) => {
        if (statusFilter !== 'all' && dataset.status !== statusFilter) {
          return false;
        }
        if (!normalizedQuery) return true;
        return (
          dataset.name.toLowerCase().includes(normalizedQuery) ||
          dataset.description.toLowerCase().includes(normalizedQuery) ||
          dataset.tags.some((tag) => tag.toLowerCase().includes(normalizedQuery))
        );
      });
  }, [datasets, searchQuery, statusFilter]);

  const stats = useMemo(() => {
    const totalRows = datasets.reduce((sum, dataset) => sum + dataset.rowCount, 0);
    const activeDatasets = datasets.filter((dataset) => dataset.status === 'active');
    const avgQuality =
      activeDatasets.length === 0
        ? 0
        : Math.round(
            activeDatasets.reduce((sum, dataset) => sum + dataset.qualityScore, 0) /
              activeDatasets.length
          );
    return {
      totalDatasets: datasets.length,
      totalRows,
      avgQuality,
      processingCount: datasets.filter((dataset) => dataset.status === 'processing').length,
    };
  }, [datasets]);

  const handleAnalyze = async (datasetId: string) => {
    setAnalyzingId(datasetId);
    try {
      await analyzeDataset(datasetId);
      onViewChange('insights');
    } finally {
      setAnalyzingId(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Datasets</h1>
          <p className="text-muted-foreground mt-1">Manage and explore your health data</p>
        </div>

        <Button className="gap-2 gradient-mint text-background hover:opacity-90" onClick={() => onViewChange('upload')}>
          <Plus className="w-4 h-4" />
          Add Dataset
        </Button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        {[
          { label: 'Total Datasets', value: stats.totalDatasets.toLocaleString(), icon: Database, color: 'bg-health-mint/20 text-health-mint' },
          { label: 'Total Records', value: stats.totalRows.toLocaleString(), icon: BarChart3, color: 'bg-blue-500/20 text-blue-400' },
          { label: 'Avg Quality', value: `${stats.avgQuality}%`, icon: CheckCircle2, color: 'bg-emerald-500/20 text-emerald-400' },
          { label: 'Processing', value: stats.processingCount.toLocaleString(), icon: Clock, color: 'bg-amber-500/20 text-amber-400' },
        ].map((stat) => (
          <Card key={stat.label} className="glass-card">
            <CardContent className="p-4 flex items-center gap-4">
              <div className={cn('w-10 h-10 rounded-lg flex items-center justify-center', stat.color)}>
                <stat.icon className="w-5 h-5" />
              </div>
              <div>
                <p className="text-2xl font-bold text-foreground">{stat.value}</p>
                <p className="text-xs text-muted-foreground">{stat.label}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search datasets by name, description, or tags..."
            className="pl-10"
          />
        </div>
        <div className="flex gap-2">
          <Button variant="outline" className="gap-2" onClick={() => setStatusFilter((previous) => statusCycle(previous))}>
            <Filter className="w-4 h-4" />
            Filter: {statusLabels[statusFilter]}
          </Button>
          <Button
            variant="outline"
            className="gap-2"
            onClick={() => {
              const snapshot = datasets.map((dataset) => ({
                id: dataset.id,
                name: dataset.name,
                status: dataset.status,
                qualityScore: dataset.qualityScore,
                rowCount: dataset.rowCount,
                columnCount: dataset.columnCount,
                uploadedAt: dataset.uploadedAt,
              }));
              downloadJsonFile('datasets-export.json', snapshot);
            }}
          >
            <Download className="w-4 h-4" />
            Export
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {filteredDatasets.map((dataset) => {
          const Icon = datasetIcon(dataset);
          const analyzing = analyzingId === dataset.id;
          return (
            <Card key={dataset.id} className="glass-card card-hover">
              <CardContent className="p-5">
                <div className="flex items-start gap-4">
                  <div className={cn('w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0', datasetColorClasses(dataset.type))}>
                    <Icon className="w-6 h-6" />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-start justify-between gap-2">
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="font-semibold text-foreground">{dataset.name}</h4>
                          {dataset.status === 'processing' && (
                            <Badge variant="outline" className="text-xs gap-1 border-amber-500/30 text-amber-400">
                              <Clock className="w-3 h-3 animate-spin" />
                              Processing
                            </Badge>
                          )}
                          {dataset.status === 'error' && (
                            <Badge variant="outline" className="text-xs gap-1 border-red-500/30 text-red-400">
                              <AlertCircle className="w-3 h-3" />
                              Error
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{dataset.description}</p>
                      </div>

                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <MoreHorizontal className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem className="gap-2" onClick={() => setPreviewDataset(dataset)}>
                            <Eye className="w-4 h-4" />
                            Preview
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="gap-2"
                            disabled={dataset.status !== 'active' || analyzing}
                            onClick={() => {
                              void handleAnalyze(dataset.id);
                            }}
                          >
                            <Sparkles className="w-4 h-4" />
                            Analyze
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="gap-2"
                            onClick={() => {
                              const nextName = window.prompt('Rename dataset', dataset.name);
                              if (nextName && nextName.trim()) {
                                renameDataset(dataset.id, nextName.trim());
                              }
                            }}
                          >
                            <Edit className="w-4 h-4" />
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem className="gap-2" onClick={() => downloadDatasetCsv(dataset)}>
                            <Download className="w-4 h-4" />
                            Download
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="gap-2 text-destructive"
                            onClick={() => {
                              if (window.confirm(`Delete ${dataset.name}?`)) {
                                removeDataset(dataset.id);
                              }
                            }}
                          >
                            <Trash2 className="w-4 h-4" />
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    <div className="flex flex-wrap gap-1.5 mt-3">
                      {dataset.tags.slice(0, 4).map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>

                    <div className="flex items-center gap-4 mt-4 pt-4 border-t border-border">
                      <div className="flex items-center gap-1.5">
                        <BarChart3 className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm text-foreground">{dataset.rowCount.toLocaleString()}</span>
                        <span className="text-xs text-muted-foreground">rows</span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <Database className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm text-foreground">{dataset.columnCount}</span>
                        <span className="text-xs text-muted-foreground">columns</span>
                      </div>
                      <div className="ml-auto flex items-center gap-1.5">
                        {dataset.qualityScore >= 85 ? (
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        ) : (
                          <AlertCircle className="w-4 h-4 text-amber-400" />
                        )}
                        <span
                          className={cn(
                            'text-sm font-medium',
                            dataset.qualityScore >= 85
                              ? 'text-emerald-400'
                              : dataset.qualityScore >= 70
                                ? 'text-amber-400'
                                : 'text-red-400'
                          )}
                        >
                          {dataset.qualityScore}%
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between mt-3 text-xs text-muted-foreground">
                      <span>By {dataset.createdBy}</span>
                      <span>Updated {new Date(dataset.lastUpdated).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {filteredDatasets.length === 0 && (
        <Card className="glass-card">
          <CardContent className="p-12 text-center">
            <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-muted flex items-center justify-center">
              <Database className="w-8 h-8 text-muted-foreground" />
            </div>
            <h3 className="text-lg font-semibold text-foreground mb-2">No datasets found</h3>
            <p className="text-sm text-muted-foreground mb-4">Try adjusting your search or upload a new dataset.</p>
            <Button className="gap-2 gradient-mint text-background hover:opacity-90" onClick={() => onViewChange('upload')}>
              <Plus className="w-4 h-4" />
              Upload Dataset
            </Button>
          </CardContent>
        </Card>
      )}

      <Dialog open={Boolean(previewDataset)} onOpenChange={(open) => (!open ? setPreviewDataset(null) : undefined)}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>{previewDataset?.name}</DialogTitle>
            <DialogDescription>
              Showing up to {previewDataset?.sampleRows.length ?? 0} preview rows and{' '}
              {previewDataset?.columnCount ?? 0} columns.
            </DialogDescription>
          </DialogHeader>

          {previewDataset && previewDataset.sampleRows.length > 0 ? (
            <div className="max-h-[420px] overflow-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-card">
                  <tr className="border-b border-border">
                    {previewDataset.columns.map((column) => (
                      <th key={column.name} className="px-3 py-2 text-left font-medium text-muted-foreground">
                        {column.name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewDataset.sampleRows.slice(0, 50).map((row, index) => (
                    <tr key={`${previewDataset.id}-${index}`} className="border-b border-border/50">
                      {previewDataset.columns.map((column) => (
                        <td key={`${index}-${column.name}`} className="px-3 py-2 text-foreground">
                          {row[column.name] === null ? (
                            <span className="text-muted-foreground">-</span>
                          ) : (
                            String(row[column.name])
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="rounded-lg border border-border p-6 text-sm text-muted-foreground">
              No preview rows available for this dataset.
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

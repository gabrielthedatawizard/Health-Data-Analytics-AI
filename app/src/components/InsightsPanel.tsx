import { useMemo, useState, type ElementType } from 'react';
import {
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Copy,
  Database,
  Flag,
  Share2,
  Shield,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  TrendingUp,
  AlertCircle,
} from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Textarea } from '@/components/ui/textarea';
import { useAnalytics } from '@/lib/analytics-context';
import { downloadJsonFile } from '@/lib/download';
import { useI18n } from '@/lib/i18n';
import { cn } from '@/lib/utils';
import type { InsightRecord } from '@/lib/ai-engine';

const quickPrompts = [
  'What is the current data quality score?',
  'Which metrics have missing values?',
  'Give me a recommendation before dashboard publishing.',
];

const typeConfig: Record<
  InsightRecord['type'],
  {
    icon: ElementType;
    label: string;
    iconClasses: string;
    badgeClasses: string;
  }
> = {
  trend: {
    icon: TrendingUp,
    label: 'Trend',
    iconClasses: 'bg-emerald-500/20 text-emerald-400',
    badgeClasses: 'border-emerald-500/30 text-emerald-400',
  },
  anomaly: {
    icon: AlertCircle,
    label: 'Anomaly',
    iconClasses: 'bg-amber-500/20 text-amber-400',
    badgeClasses: 'border-amber-500/30 text-amber-400',
  },
  prediction: {
    icon: Brain,
    label: 'Prediction',
    iconClasses: 'bg-blue-500/20 text-blue-400',
    badgeClasses: 'border-blue-500/30 text-blue-400',
  },
  recommendation: {
    icon: Sparkles,
    label: 'Recommendation',
    iconClasses: 'bg-health-mint/20 text-health-mint',
    badgeClasses: 'border-health-mint/30 text-health-mint',
  },
};

type InspectionView = 'sources' | 'logic' | 'preview' | 'governance';

const inspectionViewConfig: Record<
  InspectionView,
  {
    label: string;
    icon: ElementType;
  }
> = {
  sources: {
    label: 'Show Sources',
    icon: BarChart3,
  },
  logic: {
    label: 'Show Logic',
    icon: Brain,
  },
  preview: {
    label: 'Show Preview',
    icon: Database,
  },
  governance: {
    label: 'Show Governance',
    icon: Shield,
  },
};

function hasTraceContent(insight: InsightRecord): boolean {
  const inspection = insight.inspection;
  if (!inspection) return false;
  return Boolean(
    (inspection.factsUsed?.length ?? 0) > 0 ||
      inspection.queryPlan ||
      inspection.chart ||
      (inspection.chartCandidates?.length ?? 0) > 0 ||
      (inspection.resultRows?.length ?? 0) > 0 ||
      (inspection.kpis?.length ?? 0) > 0 ||
      (inspection.qualityIssues?.length ?? 0) > 0 ||
      (inspection.notes?.length ?? 0) > 0 ||
      (inspection.governance && Object.keys(inspection.governance).length > 0)
  );
}

function availableInspectionViews(insight: InsightRecord): InspectionView[] {
  const inspection = insight.inspection;
  if (!inspection) return [];

  const views: InspectionView[] = [];
  if (
    inspection.coverageMode ||
    inspection.coverageNote ||
    (inspection.factsUsed?.length ?? 0) > 0 ||
    (inspection.kpis?.length ?? 0) > 0 ||
    (inspection.qualityIssues?.length ?? 0) > 0 ||
    (inspection.notes?.length ?? 0) > 0
  ) {
    views.push('sources');
  }
  if (inspection.queryPlan || inspection.chart || (inspection.chartCandidates?.length ?? 0) > 0) {
    views.push('logic');
  }
  if ((inspection.resultRows?.length ?? 0) > 0) {
    views.push('preview');
  }
  if (inspection.governance && Object.keys(inspection.governance).length > 0) {
    views.push('governance');
  }
  return views;
}

function defaultInspectionView(insight: InsightRecord): InspectionView {
  const available = availableInspectionViews(insight);
  return available[0] ?? 'sources';
}

function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2) ?? '';
}

function formatInspectionCell(value: string | number | boolean | null | undefined): string {
  if (value === null || value === undefined) return '-';
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
  }
  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }
  return String(value);
}

export function InsightsPanel() {
  const { t, formatTime } = useI18n();
  const {
    insights,
    aiEnvironment,
    askAi,
    toggleInsightExpanded,
    setInsightFeedback,
    flagInsight,
  } = useAnalytics();
  const [question, setQuestion] = useState('');
  const [isAsking, setIsAsking] = useState(false);
  const [copiedInsightId, setCopiedInsightId] = useState<string | null>(null);
  const [inspectionViews, setInspectionViews] = useState<Record<string, InspectionView>>({});

  const sortedInsights = useMemo(() => {
    return [...insights].sort((left, right) => right.timestamp.localeCompare(left.timestamp));
  }, [insights]);

  const handleAsk = async () => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion) return;
    setIsAsking(true);
    try {
      await askAi(trimmedQuestion);
      setQuestion('');
    } finally {
      setIsAsking(false);
    }
  };

  const copyInsight = async (insight: InsightRecord) => {
    const payload = `${insight.title}\n\n${insight.content}\n\nCitations: ${insight.citations.join(', ')}`;
    try {
      await navigator.clipboard.writeText(payload);
      setCopiedInsightId(insight.id);
      window.setTimeout(() => setCopiedInsightId(null), 1200);
    } catch {
      window.alert('Could not copy to clipboard in this browser.');
    }
  };

  const shareInsight = async (insight: InsightRecord) => {
    const sharePayload = {
      title: insight.title,
      text: insight.content,
    };

    if (navigator.share) {
      try {
        await navigator.share(sharePayload);
        return;
      } catch {
        return;
      }
    }

    await copyInsight(insight);
  };

  const setInspectionView = (insight: InsightRecord, nextView: InspectionView) => {
    setInspectionViews((previous) => ({
      ...previous,
      [insight.id]: nextView,
    }));
  };

  return (
    <div className="space-y-4 sm:space-y-6">
      <div>
        <h1 className="text-xl font-bold text-foreground sm:text-2xl">AI Insights</h1>
        <p className="mt-1 text-sm text-muted-foreground sm:text-base">
          AI-generated insights with verified facts and citations
        </p>
      </div>

      <Card className="glass-card border-health-mint/20">
        <CardContent className="p-5">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start">
            <div className="w-10 h-10 rounded-xl gradient-mint flex items-center justify-center flex-shrink-0">
              <Sparkles className="w-5 h-5 text-background" />
            </div>
            <div className="flex-1">
              <h3 className="font-semibold text-foreground mb-2">{t.askYourData}</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Ask questions about your uploaded data. Responses are based on computed metrics and stored profiles.
              </p>
              <div className="flex gap-2">
                <Textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  placeholder={t.askQuestionPlaceholder}
                  className="flex-1 min-h-[80px] bg-muted border-0 resize-none"
                />
              </div>
              <div className="flex flex-wrap items-center justify-between mt-3 gap-2">
                <div className="flex items-center gap-2 flex-wrap">
                  {quickPrompts.map((prompt) => (
                    <Badge
                      key={prompt}
                      variant="outline"
                      className="text-xs gap-1 border-health-mint/30 text-health-mint cursor-pointer hover:bg-health-mint/10"
                      onClick={() => setQuestion(prompt)}
                    >
                      {prompt}
                    </Badge>
                  ))}
                </div>
                <Button
                  onClick={() => {
                    void handleAsk();
                  }}
                  disabled={!question.trim() || isAsking}
                  className="w-full gap-2 gradient-mint text-background hover:opacity-90 sm:w-auto"
                >
                  {isAsking ? (
                    <>
                      <div className="w-4 h-4 border-2 border-background/30 border-t-background rounded-full animate-spin" />
                      {t.analyzing}
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      Ask AI
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="flex flex-col md:flex-row md:items-center gap-4 p-4 rounded-xl bg-muted/50 border border-border">
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-health-mint" />
          <span className="text-sm font-medium text-foreground">{t.zeroHallucination}</span>
        </div>
        <div className="h-4 w-px bg-border hidden md:block" />
        <div className="flex items-center gap-4 text-sm text-muted-foreground flex-wrap">
          <span className="flex items-center gap-1.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            {t.allNumbersComputed}
          </span>
          <span className="flex items-center gap-1.5">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            {t.citationsProvided}
          </span>
          <Badge variant="outline" className="text-xs">
            AI env: {aiEnvironment.status}
          </Badge>
          <Badge variant="outline" className="text-xs">
            {aiEnvironment.mode} / {aiEnvironment.version}
          </Badge>
        </div>
      </div>

      <div className="space-y-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
          <h3 className="text-lg font-semibold text-foreground">Generated Insights</h3>
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="gap-1">
              <BarChart3 className="w-3 h-3" />
              {sortedInsights.length} insights
            </Badge>
            <Button
              variant="outline"
              size="sm"
              className="gap-1"
              onClick={() => downloadJsonFile('insights-export.json', sortedInsights)}
            >
              <Share2 className="w-3 h-3" />
              Export
            </Button>
          </div>
        </div>

        {sortedInsights.length === 0 && (
          <Card className="glass-card">
            <CardContent className="p-8 text-center text-muted-foreground">
              Upload a dataset and run analysis to generate your first insight.
            </CardContent>
          </Card>
        )}

        {sortedInsights.map((insight) => {
          const config = typeConfig[insight.type];
          const Icon = config.icon;
          const inspection = insight.inspection;
          const traceAvailable = hasTraceContent(insight);
          const inspectionOptions = availableInspectionViews(insight);
          const activeInspectionView = inspectionViews[insight.id] ?? defaultInspectionView(insight);
          const previewColumns =
            inspection?.resultColumns && inspection.resultColumns.length > 0
              ? inspection.resultColumns
              : inspection?.resultRows && inspection.resultRows.length > 0
                ? Object.keys(inspection.resultRows[0])
                : [];
          return (
            <Card
              key={insight.id}
              className={cn(
                'glass-card overflow-hidden transition-all duration-300',
                insight.expanded && 'border-health-mint/30',
                insight.flagged && 'border-amber-500/40'
              )}
            >
              <CardContent className="p-0">
                <div
                  className="p-5 cursor-pointer"
                  onClick={() => toggleInsightExpanded(insight.id)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      toggleInsightExpanded(insight.id);
                    }
                  }}
                >
                  <div className="flex items-start gap-4">
                    <div className={cn('w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0', config.iconClasses)}>
                      <Icon className="w-5 h-5" />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <Badge variant="outline" className={cn('text-xs', config.badgeClasses)}>
                          {config.label}
                        </Badge>
                        <Badge
                          variant="outline"
                          className={cn(
                            'text-xs',
                            insight.confidence >= 90
                              ? 'border-emerald-500/30 text-emerald-400'
                              : insight.confidence >= 70
                                ? 'border-amber-500/30 text-amber-400'
                                : 'border-red-500/30 text-red-400'
                          )}
                        >
                          {insight.confidence}% confidence
                        </Badge>
                        {insight.verified && (
                          <Badge variant="outline" className="text-xs gap-1 border-emerald-500/30 text-emerald-400">
                            <CheckCircle2 className="w-3 h-3" />
                            Verified
                          </Badge>
                        )}
                        {insight.flagged && (
                          <Badge variant="outline" className="text-xs gap-1 border-amber-500/30 text-amber-400">
                            <Flag className="w-3 h-3" />
                            Flagged
                          </Badge>
                        )}
                      </div>

                      <h4 className="font-semibold text-foreground mt-2">{insight.title}</h4>

                      {!insight.expanded && (
                        <p className="text-sm text-muted-foreground mt-1 line-clamp-2">{insight.content}</p>
                      )}

                      <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                        <span>{formatTime(insight.timestamp)}</span>
                        <span>•</span>
                        <span>Based on {insight.citations.length} source(s)</span>
                      </div>
                    </div>

                    <Button variant="ghost" size="icon" className="hidden flex-shrink-0 sm:inline-flex">
                      {insight.expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                    </Button>
                  </div>
                </div>

                {insight.expanded && (
                  <div className="px-5 pb-5 border-t border-border">
                    <div className="pt-4">
                      <p className="text-sm text-foreground leading-relaxed">{insight.content}</p>

                      {insight.sourceQuestion && (
                        <p className="text-xs text-muted-foreground mt-3">Question: {insight.sourceQuestion}</p>
                      )}

                      <div className="mt-4">
                        <p className="text-xs font-medium text-muted-foreground uppercase mb-2">Sources</p>
                        <div className="flex flex-wrap gap-2">
                          {insight.citations.map((citation) => (
                            <Badge key={`${insight.id}-${citation}`} variant="secondary" className="text-xs gap-1">
                              <BarChart3 className="w-3 h-3" />
                              {citation}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {traceAvailable && inspection && (
                        <div className="mt-4 rounded-xl border border-border bg-muted/30">
                          <div className="flex flex-col gap-3 border-b border-border px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                            <div>
                              <p className="text-xs font-medium text-muted-foreground uppercase">Inspection</p>
                              <p className="text-xs text-muted-foreground">
                                Inspect governed sources, logic, previews, and policy checks.
                              </p>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              {inspectionOptions.map((view) => {
                                const ViewIcon = inspectionViewConfig[view].icon;
                                return (
                                  <Button
                                    key={`${insight.id}-${view}`}
                                    variant={activeInspectionView === view ? 'default' : 'outline'}
                                    size="sm"
                                    className={cn(
                                      'h-8 gap-1',
                                      activeInspectionView === view &&
                                        'bg-health-mint text-background hover:bg-health-mint/90'
                                    )}
                                    onClick={() => setInspectionView(insight, view)}
                                  >
                                    <ViewIcon className="w-3 h-3" />
                                    {inspectionViewConfig[view].label}
                                  </Button>
                                );
                              })}
                            </div>
                          </div>

                          <div className="px-4 py-4">
                            {activeInspectionView === 'sources' && (
                              <div className="space-y-4">
                                <div className="flex flex-wrap gap-2">
                                  {inspection.coverageMode && (
                                    <Badge variant="outline" className="text-xs">
                                      Coverage: {inspection.coverageMode}
                                    </Badge>
                                  )}
                                  {typeof inspection.factCoverage === 'number' && (
                                    <Badge variant="outline" className="text-xs">
                                      Fact coverage: {Math.round(inspection.factCoverage * 100)}%
                                    </Badge>
                                  )}
                                  <Badge variant="outline" className="text-xs">
                                    Trace origin: {inspection.origin}
                                  </Badge>
                                </div>

                                {inspection.coverageNote && (
                                  <div className="rounded-lg border border-border bg-card/40 px-3 py-2 text-xs text-muted-foreground">
                                    {inspection.coverageNote}
                                  </div>
                                )}

                                {(inspection.factsUsed?.length ?? 0) > 0 && (
                                  <div>
                                    <p className="mb-2 text-xs font-medium text-muted-foreground uppercase">
                                      Fact References
                                    </p>
                                    <div className="flex flex-wrap gap-2">
                                      {inspection.factsUsed?.map((factId) => (
                                        <Badge key={`${insight.id}-${factId}`} variant="secondary" className="text-xs">
                                          {factId}
                                        </Badge>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                {(inspection.kpis?.length ?? 0) > 0 && (
                                  <div>
                                    <p className="mb-2 text-xs font-medium text-muted-foreground uppercase">
                                      KPI Snapshot
                                    </p>
                                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                                      {inspection.kpis?.map((kpi) => (
                                        <div
                                          key={`${insight.id}-${kpi.id ?? kpi.name}`}
                                          className="rounded-lg border border-border bg-card/40 px-3 py-2"
                                        >
                                          <p className="text-xs text-muted-foreground">{kpi.name}</p>
                                          <p className="text-sm font-medium text-foreground">{kpi.value}</p>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                {(inspection.qualityIssues?.length ?? 0) > 0 && (
                                  <div>
                                    <p className="mb-2 text-xs font-medium text-muted-foreground uppercase">
                                      Quality Issues
                                    </p>
                                    <div className="space-y-2">
                                      {inspection.qualityIssues?.map((issue) => (
                                        <div
                                          key={`${insight.id}-${issue}`}
                                          className="rounded-lg border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs text-amber-300"
                                        >
                                          {issue}
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                {(inspection.notes?.length ?? 0) > 0 && (
                                  <div>
                                    <p className="mb-2 text-xs font-medium text-muted-foreground uppercase">
                                      Notes
                                    </p>
                                    <div className="space-y-2">
                                      {inspection.notes?.map((note) => (
                                        <div
                                          key={`${insight.id}-${note}`}
                                          className="rounded-lg border border-border bg-card/40 px-3 py-2 text-xs text-muted-foreground"
                                        >
                                          {note}
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}

                            {activeInspectionView === 'logic' && (
                              <div className="space-y-4">
                                {inspection.queryPlan && (
                                  <div>
                                    <p className="mb-2 text-xs font-medium text-muted-foreground uppercase">
                                      Query Plan
                                    </p>
                                    <pre className="max-h-64 overflow-auto rounded-lg border border-border bg-card/50 p-3 text-xs text-foreground">
                                      {prettyJson(inspection.queryPlan)}
                                    </pre>
                                  </div>
                                )}

                                {inspection.chart && (
                                  <div>
                                    <p className="mb-2 text-xs font-medium text-muted-foreground uppercase">
                                      Chart Payload
                                    </p>
                                    <pre className="max-h-64 overflow-auto rounded-lg border border-border bg-card/50 p-3 text-xs text-foreground">
                                      {prettyJson(inspection.chart)}
                                    </pre>
                                  </div>
                                )}

                                {(inspection.chartCandidates?.length ?? 0) > 0 && (
                                  <div>
                                    <p className="mb-2 text-xs font-medium text-muted-foreground uppercase">
                                      Chart Candidates
                                    </p>
                                    <pre className="max-h-64 overflow-auto rounded-lg border border-border bg-card/50 p-3 text-xs text-foreground">
                                      {prettyJson(inspection.chartCandidates)}
                                    </pre>
                                  </div>
                                )}

                                {!inspection.queryPlan &&
                                  !inspection.chart &&
                                  (inspection.chartCandidates?.length ?? 0) === 0 && (
                                    <div className="rounded-lg border border-border bg-card/40 px-3 py-3 text-xs text-muted-foreground">
                                      No logic payload was captured for this insight.
                                    </div>
                                  )}
                              </div>
                            )}

                            {activeInspectionView === 'preview' && (
                              <div className="space-y-4">
                                {(inspection.resultRows?.length ?? 0) > 0 ? (
                                  <div className="overflow-auto rounded-lg border border-border">
                                    <table className="w-full text-sm">
                                      <thead className="sticky top-0 bg-card">
                                        <tr className="border-b border-border">
                                          {previewColumns.map((column) => (
                                            <th
                                              key={`${insight.id}-${column}`}
                                              className="px-3 py-2 text-left text-xs font-medium text-muted-foreground"
                                            >
                                              {column}
                                            </th>
                                          ))}
                                        </tr>
                                      </thead>
                                      <tbody>
                                        {inspection.resultRows?.slice(0, 8).map((row, rowIndex) => (
                                          <tr
                                            key={`${insight.id}-${rowIndex}`}
                                            className="border-b border-border/50"
                                          >
                                            {previewColumns.map((column) => (
                                              <td
                                                key={`${insight.id}-${rowIndex}-${column}`}
                                                className="px-3 py-2 text-xs text-foreground"
                                              >
                                                {formatInspectionCell(row[column])}
                                              </td>
                                            ))}
                                          </tr>
                                        ))}
                                      </tbody>
                                    </table>
                                  </div>
                                ) : (
                                  <div className="rounded-lg border border-border bg-card/40 px-3 py-3 text-xs text-muted-foreground">
                                    No tabular preview was returned for this insight.
                                  </div>
                                )}
                              </div>
                            )}

                            {activeInspectionView === 'governance' && (
                              <div className="space-y-4">
                                {inspection.governance && Object.keys(inspection.governance).length > 0 ? (
                                  <pre className="max-h-64 overflow-auto rounded-lg border border-border bg-card/50 p-3 text-xs text-foreground">
                                    {prettyJson(inspection.governance)}
                                  </pre>
                                ) : (
                                  <div className="rounded-lg border border-border bg-card/40 px-3 py-3 text-xs text-muted-foreground">
                                    No governance payload was captured for this insight.
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mt-4 pt-4 border-t border-border gap-3">
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground">{t.wasThisHelpful}</span>
                          <Button
                            variant="ghost"
                            size="sm"
                            className={cn(
                              'h-8 gap-1',
                              insight.userFeedback === 'positive' && 'text-emerald-400 bg-emerald-500/10'
                            )}
                            onClick={() => setInsightFeedback(insight.id, 'positive')}
                          >
                            <ThumbsUp className="w-4 h-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className={cn(
                              'h-8 gap-1',
                              insight.userFeedback === 'negative' && 'text-red-400 bg-red-500/10'
                            )}
                            onClick={() => setInsightFeedback(insight.id, 'negative')}
                          >
                            <ThumbsDown className="w-4 h-4" />
                          </Button>
                        </div>

                        <div className="flex flex-wrap items-center gap-2">
                          <Button variant="ghost" size="sm" className="h-8 gap-1" onClick={() => void copyInsight(insight)}>
                            <Copy className="w-4 h-4" />
                            {copiedInsightId === insight.id ? 'Copied' : 'Copy'}
                          </Button>
                          <Button variant="ghost" size="sm" className="h-8 gap-1" onClick={() => void shareInsight(insight)}>
                            <Share2 className="w-4 h-4" />
                            Share
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className={cn(
                              'h-8 gap-1',
                              insight.flagged
                                ? 'text-amber-400 hover:text-amber-300 hover:bg-amber-500/10'
                                : 'text-amber-400/80 hover:text-amber-400 hover:bg-amber-500/10'
                            )}
                            onClick={() => flagInsight(insight.id)}
                          >
                            <Flag className="w-4 h-4" />
                            {insight.flagged ? 'Unflag' : 'Flag'}
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Brain className="w-5 h-5 text-health-mint" />
            How AI Insights Work
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[
              { step: '1', title: 'File Parsing', desc: 'Uploaded rows are parsed and normalized' },
              { step: '2', title: 'Profile Build', desc: 'Column quality and metrics are computed' },
              { step: '3', title: 'Guarded AI', desc: 'Answers are generated from computed metrics' },
              { step: '4', title: 'Human Review', desc: 'Feedback and flags improve trust over time' },
            ].map((item) => (
              <div key={item.step} className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-health-mint/20 flex items-center justify-center flex-shrink-0">
                  <span className="text-sm font-bold text-health-mint">{item.step}</span>
                </div>
                <div>
                  <p className="font-medium text-foreground text-sm">{item.title}</p>
                  <p className="text-xs text-muted-foreground">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

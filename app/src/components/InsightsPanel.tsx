import { useMemo, useState, type ElementType } from 'react';
import {
  BarChart3,
  Brain,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Copy,
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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">AI Insights</h1>
        <p className="text-muted-foreground mt-1">
          AI-generated insights with verified facts and citations
        </p>
      </div>

      <Card className="glass-card border-health-mint/20">
        <CardContent className="p-5">
          <div className="flex items-start gap-4">
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
                  className="gap-2 gradient-mint text-background hover:opacity-90"
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
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-lg font-semibold text-foreground">Generated Insights</h3>
          <div className="flex items-center gap-2">
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

                    <Button variant="ghost" size="icon" className="flex-shrink-0">
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

                        <div className="flex items-center gap-2">
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
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
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

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
  answerQuestion,
  buildFailedDataset,
  createSampleDataset,
  generateInsightsForDataset,
  parseUploadedFile,
  type AIEnvironmentState,
  type DatasetRecord,
  type InsightRecord,
  type SampleDatasetKind,
  type UploadProgressHandler,
} from '@/lib/ai-engine';

const STORAGE_KEY = 'healthai_workspace_state_v1';

interface PersistedWorkspaceState {
  datasets: DatasetRecord[];
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
  addSampleDataset: (kind?: SampleDatasetKind) => DatasetRecord;
  removeDataset: (datasetId: string) => void;
  renameDataset: (datasetId: string, nextName: string) => void;
  analyzeDataset: (datasetId: string, sourceQuestion?: string) => Promise<InsightRecord[]>;
  askAi: (question: string) => Promise<InsightRecord>;
  toggleInsightExpanded: (insightId: string) => void;
  setInsightFeedback: (insightId: string, feedback: 'positive' | 'negative') => void;
  flagInsight: (insightId: string) => void;
  exportSnapshot: () => AnalyticsExportSnapshot;
  clearAllData: () => void;
}

interface InitialWorkspaceState {
  datasets: DatasetRecord[];
  insights: InsightRecord[];
  aiEnvironment: AIEnvironmentState;
}

const DEFAULT_AI_ENVIRONMENT: AIEnvironmentState = {
  status: 'ready',
  mode: 'local-guarded',
  model: 'HealthAI Local Intelligence',
  version: '1.0.0',
  lastRunAt: null,
};

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
    if (!Array.isArray(parsed.datasets) || !Array.isArray(parsed.insights)) {
      return null;
    }
    return {
      datasets: parsed.datasets as DatasetRecord[],
      insights: parsed.insights as InsightRecord[],
      aiEnvironment: parsed.aiEnvironment ?? DEFAULT_AI_ENVIRONMENT,
    };
  } catch {
    return null;
  }
}

function buildInitialWorkspaceState(): InitialWorkspaceState {
  const persisted = loadPersistedState();
  if (persisted && persisted.datasets.length > 0) {
    return persisted;
  }

  const seedDataset = createSampleDataset('demo');
  return {
    datasets: [seedDataset],
    insights: generateInsightsForDataset(seedDataset, { expandedFirst: true }),
    aiEnvironment: DEFAULT_AI_ENVIRONMENT,
  };
}

const AnalyticsContext = createContext<AnalyticsContextValue | null>(null);

export function AnalyticsProvider({ children }: { children: ReactNode }) {
  const initialState = useMemo(() => buildInitialWorkspaceState(), []);
  const [datasets, setDatasets] = useState<DatasetRecord[]>(initialState.datasets);
  const [insights, setInsights] = useState<InsightRecord[]>(initialState.insights);
  const [aiEnvironment, setAiEnvironment] = useState<AIEnvironmentState>(initialState.aiEnvironment);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const snapshot: PersistedWorkspaceState = {
      datasets,
      insights,
      aiEnvironment,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
  }, [datasets, insights, aiEnvironment]);

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

  const uploadFile = useCallback(
    async (file: File, onProgress?: UploadProgressHandler): Promise<DatasetRecord> => {
      markEnvironmentBusy();
      try {
        const dataset = await parseUploadedFile(file, onProgress);
        const generatedInsights = generateInsightsForDataset(dataset, {
          expandedFirst: true,
        });
        setDatasets((previous) => [dataset, ...previous]);
        setInsights((previous) => [...generatedInsights, ...previous]);
        markEnvironmentReady();
        return dataset;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Upload failed unexpectedly.';
        const failedDataset = buildFailedDataset(file, message);
        setDatasets((previous) => [failedDataset, ...previous]);
        markEnvironmentError(message);
        throw error instanceof Error ? error : new Error(message);
      }
    },
    [markEnvironmentBusy, markEnvironmentError, markEnvironmentReady]
  );

  const addSampleDataset = useCallback(
    (kind: SampleDatasetKind = 'demo'): DatasetRecord => {
      markEnvironmentBusy();
      const dataset = createSampleDataset(kind);
      const generatedInsights = generateInsightsForDataset(dataset, {
        expandedFirst: true,
      });
      setDatasets((previous) => [dataset, ...previous]);
      setInsights((previous) => [...generatedInsights, ...previous]);
      markEnvironmentReady();
      return dataset;
    },
    [markEnvironmentBusy, markEnvironmentReady]
  );

  const removeDataset = useCallback((datasetId: string) => {
    setDatasets((previous) => previous.filter((dataset) => dataset.id !== datasetId));
    setInsights((previous) => previous.filter((insight) => insight.datasetId !== datasetId));
  }, []);

  const renameDataset = useCallback(
    (datasetId: string, nextName: string) => {
      const normalizedName = nextName.trim();
      if (!normalizedName) return;
      const previousName = datasets.find((dataset) => dataset.id === datasetId)?.name;
      if (!previousName) return;

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
    },
    [datasets]
  );

  const analyzeDataset = useCallback(
    async (datasetId: string, sourceQuestion?: string): Promise<InsightRecord[]> => {
      const dataset = datasets.find((candidate) => candidate.id === datasetId);
      if (!dataset) {
        throw new Error('Dataset not found.');
      }

      markEnvironmentBusy();
      await wait(350);
      const generated = generateInsightsForDataset(dataset, {
        expandedFirst: true,
        sourceQuestion,
      });
      setInsights((previous) => [...generated, ...previous]);
      setDatasets((previous) =>
        previous.map((candidate) =>
          candidate.id === datasetId
            ? {
                ...candidate,
                lastUpdated: new Date().toISOString(),
                status: 'active',
              }
            : candidate
        )
      );
      markEnvironmentReady();
      return generated;
    },
    [datasets, markEnvironmentBusy, markEnvironmentReady]
  );

  const askAi = useCallback(
    async (question: string): Promise<InsightRecord> => {
      markEnvironmentBusy();
      await wait(450);
      const response = answerQuestion(question, datasets);
      setInsights((previous) => [response, ...previous]);
      markEnvironmentReady();
      return response;
    },
    [datasets, markEnvironmentBusy, markEnvironmentReady]
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

  const clearAllData = useCallback(() => {
    setDatasets([]);
    setInsights([]);
    setAiEnvironment({
      ...DEFAULT_AI_ENVIRONMENT,
      lastRunAt: new Date().toISOString(),
    });
  }, []);

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

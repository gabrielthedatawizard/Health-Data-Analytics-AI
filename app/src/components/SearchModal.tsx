import { useCallback, useEffect, useMemo, useState } from 'react';
import { 
  Search, 
  X, 
  FileSpreadsheet, 
  Lightbulb, 
  BarChart3, 
  Database,
  ArrowRight,
  Clock,
  TrendingUp,
  Sparkles
} from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { useAnalytics } from '@/lib/analytics-context';
import type { DatasetRecord, InsightRecord } from '@/lib/ai-engine';
import { cn } from '@/lib/utils';

const searchableData = {
  kpis: [
    { id: '1', name: 'ANC Coverage', value: '68.4%', trend: '+12.3%', category: 'Maternal Health' },
    { id: '2', name: 'Facility Delivery', value: '74.2%', trend: '+5.1%', category: 'Maternal Health' },
    { id: '3', name: 'Maternal Mortality', value: '89', trend: '-15%', category: 'Outcomes' },
    { id: '4', name: 'Total ANC Visits', value: '12,473', trend: '+8.7%', category: 'Service Delivery' },
  ],
  districts: [
    { id: '1', name: 'Chamwino', coverage: 82, status: 'on-target' },
    { id: '2', name: 'Bahi', coverage: 78, status: 'on-target' },
    { id: '3', name: 'Dodoma Municipal', coverage: 75, status: 'on-target' },
    { id: '4', name: 'Chemba', coverage: 71, status: 'at-risk' },
    { id: '5', name: 'Kondoa', coverage: 64, status: 'below-target' },
    { id: '6', name: 'Mpwapwa', coverage: 69, status: 'at-risk' },
  ],
  facilities: [
    { id: '1', name: 'Chamwino District Hospital', type: 'Hospital', district: 'Chamwino', visits: 2847 },
    { id: '2', name: 'Bahi Health Center', type: 'Health Center', district: 'Bahi', visits: 1923 },
    { id: '3', name: 'Dodoma Regional Hospital', type: 'Hospital', district: 'Dodoma Municipal', visits: 4521 },
    { id: '4', name: 'Chemba Dispensary', type: 'Dispensary', district: 'Chemba', visits: 987 },
    { id: '5', name: 'Kondoa Health Center', type: 'Health Center', district: 'Kondoa', visits: 1456 },
  ],
};

// Recent searches (persisted in localStorage)
const getRecentSearches = () => {
  if (typeof window === 'undefined') return [];
  const saved = localStorage.getItem('healthai_recent_searches');
  if (!saved) return [];
  try {
    const parsed = JSON.parse(saved) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((item): item is string => typeof item === 'string');
  } catch {
    return [];
  }
};

const saveRecentSearch = (query: string) => {
  if (typeof window === 'undefined' || !query.trim()) return;
  const recent = getRecentSearches();
  const updated = [query, ...recent.filter((s: string) => s !== query)].slice(0, 5);
  localStorage.setItem('healthai_recent_searches', JSON.stringify(updated));
};

type KpiResultItem = (typeof searchableData.kpis)[number];
type DistrictResultItem = (typeof searchableData.districts)[number];
type FacilityResultItem = (typeof searchableData.facilities)[number];

type SearchResult =
  | { type: 'dataset'; item: DatasetRecord; relevance: number }
  | { type: 'insight'; item: InsightRecord; relevance: number }
  | { type: 'kpi'; item: KpiResultItem; relevance: number }
  | { type: 'district'; item: DistrictResultItem; relevance: number }
  | { type: 'facility'; item: FacilityResultItem; relevance: number };

interface SearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  onResultClick?: (result: SearchResult) => void;
}

export function SearchModal({ isOpen, onClose, onResultClick }: SearchModalProps) {
  const { datasets, insights } = useAnalytics();
  const [query, setQuery] = useState('');
  const [recentSearches, setRecentSearches] = useState<string[]>(() => getRecentSearches());
  const [selectedIndex, setSelectedIndex] = useState(0);

  // Calculate relevance score
  const calculateRelevance = (query: string, fields: string[]): number => {
    let score = 0;
    fields.forEach(field => {
      const normalizedField = field.toLowerCase();
      if (normalizedField === query) score += 10; // Exact match
      else if (normalizedField.startsWith(query)) score += 5; // Starts with
      else if (normalizedField.includes(query)) score += 3; // Contains
    });
    return score;
  };

  const handleResultClick = useCallback(
    (result: SearchResult) => {
      saveRecentSearch(query || getResultTitle(result));
      setRecentSearches(getRecentSearches());
      setQuery('');
      setSelectedIndex(0);
      onResultClick?.(result);
      onClose();
    },
    [onClose, onResultClick, query]
  );

  const handleClose = useCallback(() => {
    setQuery('');
    setSelectedIndex(0);
    onClose();
  }, [onClose]);

  // Perform search across all data
  const performSearch = useCallback((searchQuery: string): SearchResult[] => {
    if (!searchQuery.trim()) return [];

    const normalizedQuery = searchQuery.toLowerCase();
    const allResults: SearchResult[] = [];

    datasets.forEach((dataset) => {
      const relevance = calculateRelevance(normalizedQuery, [
        dataset.name,
        dataset.description,
        ...dataset.tags,
      ]);
      if (relevance > 0) {
        allResults.push({ type: 'dataset', item: dataset, relevance });
      }
    });

    insights.forEach((insight) => {
      const relevance = calculateRelevance(normalizedQuery, [
        insight.title,
        insight.content,
        insight.type,
        ...insight.citations,
      ]);
      if (relevance > 0) {
        allResults.push({ type: 'insight', item: insight, relevance });
      }
    });

    searchableData.kpis.forEach((kpi) => {
      const relevance = calculateRelevance(normalizedQuery, [kpi.name, kpi.category, kpi.value]);
      if (relevance > 0) {
        allResults.push({ type: 'kpi', item: kpi, relevance });
      }
    });

    searchableData.districts.forEach((district) => {
      const relevance = calculateRelevance(normalizedQuery, [district.name, district.status]);
      if (relevance > 0) {
        allResults.push({ type: 'district', item: district, relevance });
      }
    });

    searchableData.facilities.forEach((facility) => {
      const relevance = calculateRelevance(normalizedQuery, [
        facility.name,
        facility.type,
        facility.district,
      ]);
      if (relevance > 0) {
        allResults.push({ type: 'facility', item: facility, relevance });
      }
    });

    return allResults.sort((left, right) => right.relevance - left.relevance);
  }, [datasets, insights]);

  const results = useMemo(() => performSearch(query), [performSearch, query]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case 'Escape':
          handleClose();
          break;
        case 'ArrowDown':
          e.preventDefault();
          setSelectedIndex(prev => Math.min(prev + 1, Math.max(results.length - 1, 0)));
          break;
        case 'ArrowUp':
          e.preventDefault();
          setSelectedIndex(prev => Math.max(prev - 1, 0));
          break;
        case 'Enter':
          e.preventDefault();
          if (results[selectedIndex]) {
            handleResultClick(results[selectedIndex]);
          }
          break;
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleClose, handleResultClick, isOpen, results, selectedIndex]);

  const handleRecentClick = (search: string) => {
    setQuery(search);
    setSelectedIndex(0);
  };

  const clearRecentSearches = () => {
    localStorage.removeItem('healthai_recent_searches');
    setRecentSearches([]);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-20">
      {/* Backdrop */}
        <div 
          className="absolute inset-0 bg-background/80 backdrop-blur-sm"
          onClick={handleClose}
        />
      
      {/* Modal */}
      <div className="relative w-full max-w-2xl mx-4 bg-card border border-border rounded-2xl shadow-2xl overflow-hidden">
        {/* Search Input */}
        <div className="flex items-center gap-3 p-4 border-b border-border">
          <Search className="w-5 h-5 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            placeholder="Search datasets, insights, KPIs, districts, facilities..."
            className="flex-1 border-0 bg-transparent text-lg focus-visible:ring-0 placeholder:text-muted-foreground"
            autoFocus
          />
          {query && (
            <button 
              onClick={() => {
                setQuery('');
                setSelectedIndex(0);
              }}
              className="p-1 rounded-full hover:bg-muted"
            >
              <X className="w-4 h-4 text-muted-foreground" />
            </button>
          )}
          <kbd className="hidden sm:inline-block px-2 py-1 text-xs bg-muted rounded">ESC</kbd>
        </div>

        {/* Results */}
        <div className="max-h-[60vh] overflow-auto">
          {query && results.length > 0 && (
            <div className="p-2">
              <p className="px-3 py-2 text-xs font-medium text-muted-foreground uppercase">
                {results.length} Results
              </p>
              {results.map((result, index) => (
                <button
                  key={`${result.type}-${result.item.id}`}
                  onClick={() => handleResultClick(result)}
                  className={cn(
                    "w-full flex items-start gap-3 px-3 py-3 rounded-xl text-left transition-colors",
                    selectedIndex === index 
                      ? "bg-primary/10 border border-primary/20" 
                      : "hover:bg-muted/50"
                  )}
                >
                  <ResultIcon type={result.type} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-foreground">
                        {getResultTitle(result)}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {result.type}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground truncate">
                      {getResultDescription(result)}
                    </p>
                    {result.type === 'insight' && (
                      <div className="flex items-center gap-2 mt-1">
                        <Badge 
                          variant="outline" 
                          className={cn(
                            "text-xs",
                            result.item.confidence >= 90 ? "border-emerald-500/30 text-emerald-400" :
                            result.item.confidence >= 70 ? "border-amber-500/30 text-amber-400" :
                            "border-red-500/30 text-red-400"
                          )}
                        >
                          {result.item.confidence}% confidence
                        </Badge>
                      </div>
                    )}
                  </div>
                  <ArrowRight className="w-4 h-4 text-muted-foreground flex-shrink-0" />
                </button>
              ))}
            </div>
          )}

          {query && results.length === 0 && (
            <div className="p-8 text-center">
              <div className="w-12 h-12 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
                <Search className="w-6 h-6 text-muted-foreground" />
              </div>
              <p className="text-foreground font-medium">No results found</p>
              <p className="text-sm text-muted-foreground mt-1">
                Try searching for datasets, insights, KPIs, or facilities
              </p>
            </div>
          )}

          {!query && recentSearches.length > 0 && (
            <div className="p-2">
              <div className="flex items-center justify-between px-3 py-2">
                <p className="text-xs font-medium text-muted-foreground uppercase">
                  Recent Searches
                </p>
                <button 
                  onClick={clearRecentSearches}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  Clear
                </button>
              </div>
              {recentSearches.map((search, index) => (
                <button
                  key={index}
                  onClick={() => handleRecentClick(search)}
                  className="w-full flex items-center gap-3 px-3 py-2 rounded-xl text-left hover:bg-muted/50 transition-colors"
                >
                  <Clock className="w-4 h-4 text-muted-foreground" />
                  <span className="text-sm text-foreground">{search}</span>
                </button>
              ))}
            </div>
          )}

          {!query && recentSearches.length === 0 && (
            <div className="p-4">
              <p className="px-3 py-2 text-xs font-medium text-muted-foreground uppercase">
                Try searching for
              </p>
              <div className="flex flex-wrap gap-2 p-3">
                {['ANC coverage', 'Chamwino district', 'Q3 dataset', 'facility performance', 'malaria surveillance'].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setQuery(suggestion)}
                    className="px-3 py-1.5 text-sm bg-muted rounded-full hover:bg-muted/80 transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-4 py-2 border-t border-border bg-muted/30">
          <div className="flex items-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-muted rounded">↑↓</kbd>
              Navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-1.5 py-0.5 bg-muted rounded">↵</kbd>
              Select
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Sparkles className="w-3 h-3 text-health-mint" />
            <span className="text-xs text-muted-foreground">AI-powered search</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper components
function ResultIcon({ type }: { type: SearchResult['type'] }) {
  const iconClass = "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0";
  
  switch (type) {
    case 'dataset':
      return (
        <div className={cn(iconClass, "bg-emerald-500/20")}>
          <FileSpreadsheet className="w-4 h-4 text-emerald-400" />
        </div>
      );
    case 'insight':
      return (
        <div className={cn(iconClass, "bg-purple-500/20")}>
          <Lightbulb className="w-4 h-4 text-purple-400" />
        </div>
      );
    case 'kpi':
      return (
        <div className={cn(iconClass, "bg-blue-500/20")}>
          <BarChart3 className="w-4 h-4 text-blue-400" />
        </div>
      );
    case 'district':
      return (
        <div className={cn(iconClass, "bg-amber-500/20")}>
          <Database className="w-4 h-4 text-amber-400" />
        </div>
      );
    case 'facility':
      return (
        <div className={cn(iconClass, "bg-health-mint/20")}>
          <TrendingUp className="w-4 h-4 text-health-mint" />
        </div>
      );
  }
}

function getResultTitle(result: SearchResult): string {
  switch (result.type) {
    case 'dataset':
      return result.item.name;
    case 'insight':
      return result.item.title;
    case 'kpi':
      return result.item.name;
    case 'district':
      return result.item.name;
    case 'facility':
      return result.item.name;
  }
}

function getResultDescription(result: SearchResult): string {
  switch (result.type) {
    case 'dataset':
      return `${result.item.description} • ${result.item.rowCount.toLocaleString()} rows`;
    case 'insight':
      return result.item.content;
    case 'kpi':
      return `${result.item.value} (${result.item.trend}) • ${result.item.category}`;
    case 'district':
      return `${result.item.coverage}% coverage • ${result.item.status}`;
    case 'facility':
      return `${result.item.type} • ${result.item.district} • ${result.item.visits.toLocaleString()} visits`;
  }
}

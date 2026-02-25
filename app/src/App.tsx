import { useState, useEffect, useCallback } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Dashboard } from './components/Dashboard';
import { DataUpload } from './components/DataUpload';
import { InsightsPanel } from './components/InsightsPanel';
import { DatasetsView } from './components/DatasetsView';
import { Settings } from './components/Settings';
import { SearchModal } from './components/SearchModal';
import { LandingPage } from './landing/LandingPage';
import { I18nProvider } from './lib/i18n';
import { AnalyticsProvider } from './lib/analytics-context';

export type ViewType = 'dashboard' | 'upload' | 'datasets' | 'insights' | 'settings';

function AppContent() {
  const [currentView, setCurrentView] = useState<ViewType>('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const [showLanding, setShowLanding] = useState(() => !localStorage.getItem('healthai_visited'));

  const handleEnterDashboard = () => {
    localStorage.setItem('healthai_visited', 'true');
    setShowLanding(false);
  };

  const handleLogout = () => {
    localStorage.removeItem('healthai_visited');
    setShowLanding(true);
  };

  // Handle keyboard shortcut (Cmd+K or Ctrl+K)
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      setIsSearchOpen(prev => !prev);
    }
  }, []);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const renderView = () => {
    switch (currentView) {
      case 'dashboard':
        return <Dashboard />;
      case 'upload':
        return <DataUpload onViewChange={setCurrentView} />;
      case 'datasets':
        return <DatasetsView onViewChange={setCurrentView} />;
      case 'insights':
        return <InsightsPanel />;
      case 'settings':
        return <Settings onLogout={handleLogout} />;
      default:
        return <Dashboard />;
    }
  };

  // Handle search result click - navigate to appropriate view
  const handleSearchResult = (result: { type: string; item: unknown }) => {
    switch (result.type) {
      case 'dataset':
        setCurrentView('datasets');
        break;
      case 'insight':
        setCurrentView('insights');
        break;
      case 'kpi':
      case 'district':
      case 'facility':
        setCurrentView('dashboard');
        break;
    }
  };

  // Show landing page for new visitors
  if (showLanding) {
    return <LandingPage onEnterDashboard={handleEnterDashboard} />;
  }

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      <Sidebar 
        collapsed={sidebarCollapsed} 
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        currentView={currentView}
        onViewChange={setCurrentView}
      />
      
      <div className="flex-1 flex flex-col min-w-0">
        <Header 
          currentView={currentView}
          onSearchClick={() => setIsSearchOpen(true)}
          onViewChange={setCurrentView}
          onLogout={handleLogout}
        />
        
        <main className="flex-1 overflow-auto custom-scrollbar p-6">
          {renderView()}
        </main>
      </div>

      {/* Search Modal */}
      <SearchModal 
        isOpen={isSearchOpen}
        onClose={() => setIsSearchOpen(false)}
        onResultClick={handleSearchResult}
      />
    </div>
  );
}

function App() {
  return (
    <I18nProvider>
      <AnalyticsProvider>
        <AppContent />
      </AnalyticsProvider>
    </I18nProvider>
  );
}

export default App;

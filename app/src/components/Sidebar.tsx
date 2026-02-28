import { 
  LayoutDashboard, 
  Upload, 
  Database, 
  Lightbulb, 
  Settings, 
  ChevronLeft, 
  ChevronRight,
  Activity,
  Shield,
  Sparkles,
  X
} from 'lucide-react';
import { cn } from '@/lib/utils';
import type { ViewType } from '@/App';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  currentView: ViewType;
  onViewChange: (view: ViewType) => void;
  mobileOpen: boolean;
  onMobileClose: () => void;
}

const menuItems = [
  { id: 'dashboard' as ViewType, label: 'Dashboard', icon: LayoutDashboard },
  { id: 'upload' as ViewType, label: 'Data Upload', icon: Upload },
  { id: 'datasets' as ViewType, label: 'Datasets', icon: Database },
  { id: 'insights' as ViewType, label: 'AI Insights', icon: Lightbulb },
  { id: 'ai_analytics' as ViewType, label: 'AI Analytics', icon: Sparkles },
];

const bottomItems = [
  { id: 'settings' as ViewType, label: 'Settings', icon: Settings },
];

export function Sidebar({ collapsed, onToggle, currentView, onViewChange, mobileOpen, onMobileClose }: SidebarProps) {
  const isCollapsed = collapsed && !mobileOpen;

  return (
    <>
      <button
        type="button"
        aria-label="Close navigation menu"
        onClick={onMobileClose}
        className={cn(
          "fixed inset-0 z-40 bg-background/70 backdrop-blur-sm transition-opacity lg:hidden",
          mobileOpen ? "opacity-100 pointer-events-auto" : "opacity-0 pointer-events-none"
        )}
      />

      <aside 
        className={cn(
          "fixed inset-y-0 left-0 z-50 flex flex-col border-r border-border bg-sidebar-background transition-all duration-300 ease-in-out lg:static lg:z-auto lg:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
          "w-[86vw] max-w-80 lg:max-w-none",
          isCollapsed ? "lg:w-20" : "lg:w-64"
        )}
      >
        {/* Logo */}
        <div className="flex items-center justify-between border-b border-sidebar-border p-4">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-10 h-10 rounded-xl gradient-mint flex items-center justify-center flex-shrink-0">
              <Activity className="w-5 h-5 text-background" />
            </div>
            {!isCollapsed && (
              <div className="min-w-0">
                <h1 className="truncate font-bold text-lg text-foreground">HealthAI</h1>
                <p className="text-xs text-muted-foreground">Analytics Platform</p>
              </div>
            )}
          </div>

          <button
            onClick={onMobileClose}
            className="p-1.5 rounded-lg hover:bg-sidebar-accent text-muted-foreground hover:text-foreground transition-colors lg:hidden"
          >
            <X className="w-4 h-4" />
          </button>

          <button 
            onClick={onToggle}
            className="hidden p-1.5 rounded-lg hover:bg-sidebar-accent text-muted-foreground hover:text-foreground transition-colors lg:flex"
          >
            {isCollapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>
        </div>

        {/* Main Navigation */}
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto custom-scrollbar">
          {!isCollapsed && (
            <p className="px-3 py-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
              Main Menu
            </p>
          )}
          {menuItems.map((item) => {
            const Icon = item.icon;
            const isActive = currentView === item.id;
            
            return (
              <button
                key={item.id}
                onClick={() => {
                  onViewChange(item.id);
                  onMobileClose();
                }}
                className={cn(
                  "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group",
                  isActive 
                    ? "bg-primary/10 text-primary border border-primary/20" 
                    : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-foreground"
                )}
              >
                <Icon className={cn(
                  "w-5 h-5 flex-shrink-0 transition-colors",
                  isActive ? "text-primary" : "text-muted-foreground group-hover:text-foreground"
                )} />
                {!isCollapsed && (
                  <span className="font-medium text-sm">{item.label}</span>
                )}
                {isActive && !isCollapsed && (
                  <div className="ml-auto w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                )}
              </button>
            );
          })}

          {!isCollapsed && (
            <div className="mt-8">
              <p className="px-3 py-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                System Status
              </p>
              <div className="mx-3 p-4 rounded-xl bg-sidebar-accent/50 border border-sidebar-border">
                <div className="flex items-center gap-2 mb-3">
                  <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-xs text-emerald-400 font-medium">System Online</span>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">API Status</span>
                    <span className="text-emerald-400">99.9%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Data Quality</span>
                    <span className="text-primary">87%</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Last Sync</span>
                    <span className="text-muted-foreground">2m ago</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </nav>

        {/* Bottom Navigation */}
        <div className="p-3 border-t border-sidebar-border space-y-1">
          {!isCollapsed && (
            <div className="mx-3 mb-3 p-3 rounded-xl bg-gradient-to-r from-health-mint/10 to-health-purple/10 border border-health-mint/20">
              <div className="flex items-center gap-2 mb-2">
                <Shield className="w-4 h-4 text-health-mint" />
                <span className="text-xs font-medium text-foreground">HIPAA Compliant</span>
              </div>
              <p className="text-xs text-muted-foreground">End-to-end encrypted</p>
            </div>
          )}
          
          {bottomItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                onClick={() => {
                  onViewChange(item.id);
                  onMobileClose();
                }}
                className={cn(
                  "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200",
                  currentView === item.id 
                    ? "bg-primary/10 text-primary" 
                    : "text-sidebar-foreground hover:bg-sidebar-accent hover:text-foreground"
                )}
              >
                <Icon className="w-5 h-5 flex-shrink-0" />
                {!isCollapsed && <span className="font-medium text-sm">{item.label}</span>}
              </button>
            );
          })}
        </div>
      </aside>
    </>
  );
}

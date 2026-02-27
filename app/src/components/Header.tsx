import { 
  Search, 
  Bell, 
  User, 
  LogOut,
  ChevronDown,
  Sparkles,
  Keyboard
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import type { ViewType } from '@/App';

interface HeaderProps {
  currentView: ViewType;
  onSearchClick: () => void;
  onViewChange: (view: ViewType) => void;
  onLogout: () => void;
}

const viewTitles: Record<ViewType, string> = {
  dashboard: 'Dashboard Overview',
  upload: 'Data Upload',
  datasets: 'Dataset Management',
  insights: 'AI Insights',
  ai_analytics: 'AI Analytics Engine',
  settings: 'Settings',
};

export function Header({ currentView, onSearchClick, onViewChange, onLogout }: HeaderProps) {
  return (
    <header className="h-16 border-b border-border bg-card/50 backdrop-blur-sm flex items-center justify-between px-6 sticky top-0 z-40">
      {/* Left side - Title & Search */}
      <div className="flex items-center gap-6">
        <div>
          <h2 className="text-xl font-semibold text-foreground">{viewTitles[currentView]}</h2>
          <p className="text-xs text-muted-foreground">
            {new Date().toLocaleDateString('en-US', { 
              weekday: 'long', 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}
          </p>
        </div>
        
        {/* Search Bar - Clickable */}
        <button 
          onClick={onSearchClick}
          className="relative w-80 flex items-center h-10 px-3 bg-muted rounded-lg border border-transparent hover:border-health-mint/30 transition-colors text-left group"
        >
          <Search className="w-4 h-4 text-muted-foreground mr-3" />
          <span className="flex-1 text-sm text-muted-foreground">
            Search datasets, insights, or ask a question...
          </span>
          <div className="flex items-center gap-1">
            <kbd className="px-1.5 py-0.5 text-xs bg-muted-foreground/20 rounded text-muted-foreground group-hover:bg-health-mint/20 group-hover:text-health-mint transition-colors">
              ⌘
            </kbd>
            <kbd className="px-1.5 py-0.5 text-xs bg-muted-foreground/20 rounded text-muted-foreground group-hover:bg-health-mint/20 group-hover:text-health-mint transition-colors">
              K
            </kbd>
          </div>
        </button>
      </div>

      {/* Right side - Actions */}
      <div className="flex items-center gap-3">
        {/* AI Assistant Button */}
        <Button 
          variant="outline" 
          className="gap-2 border-health-mint/30 text-health-mint hover:bg-health-mint/10"
          onClick={() => onViewChange('ai_analytics')}
        >
          <Sparkles className="w-4 h-4" />
          <span className="hidden sm:inline">Ask AI</span>
        </Button>

        {/* Notifications */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="relative">
              <Bell className="w-5 h-5" />
              <span className="absolute top-1 right-1 w-2 h-2 bg-health-mint rounded-full animate-pulse" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-80">
            <DropdownMenuLabel>Notifications</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <div className="max-h-64 overflow-auto">
              <DropdownMenuItem className="flex flex-col items-start gap-1 p-3 cursor-pointer">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-amber-500" />
                  <span className="font-medium text-sm">Data Quality Alert</span>
                </div>
                <p className="text-xs text-muted-foreground pl-4">
                  15% of ANC records missing gestational age in Q3 dataset
                </p>
                <span className="text-xs text-muted-foreground pl-4">2 minutes ago</span>
              </DropdownMenuItem>
              
              <DropdownMenuItem className="flex flex-col items-start gap-1 p-3 cursor-pointer">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                  <span className="font-medium text-sm">Dashboard Generated</span>
                </div>
                <p className="text-xs text-muted-foreground pl-4">
                  Maternal Health Dashboard is ready for review
                </p>
                <span className="text-xs text-muted-foreground pl-4">1 hour ago</span>
              </DropdownMenuItem>
              
              <DropdownMenuItem className="flex flex-col items-start gap-1 p-3 cursor-pointer">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                  <span className="font-medium text-sm">Forecast Update</span>
                </div>
                <p className="text-xs text-muted-foreground pl-4">
                  ANC coverage forecast updated with October data
                </p>
                <span className="text-xs text-muted-foreground pl-4">3 hours ago</span>
              </DropdownMenuItem>
            </div>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="justify-center text-primary" onClick={() => onViewChange('insights')}>
              View all notifications
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>

        {/* User Menu */}
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="gap-2 pl-2 pr-3">
              <div className="w-8 h-8 rounded-full gradient-mint flex items-center justify-center">
                <User className="w-4 h-4 text-background" />
              </div>
              <div className="hidden md:block text-left">
                <p className="text-sm font-medium">Dr. Sarah Kimaro</p>
                <p className="text-xs text-muted-foreground">District Health Manager</p>
              </div>
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuLabel>My Account</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={() => onViewChange('settings')}>
              <User className="w-4 h-4 mr-2" />
              Profile
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => onViewChange('insights')}>
              <Sparkles className="w-4 h-4 mr-2" />
              AI Preferences
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem onClick={onSearchClick}>
              <Keyboard className="w-4 h-4 mr-2" />
              Keyboard Shortcuts
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem className="text-destructive" onClick={onLogout}>
              <LogOut className="w-4 h-4 mr-2" />
              Log out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}

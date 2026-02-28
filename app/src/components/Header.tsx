import {
  Search,
  Bell,
  User,
  LogOut,
  ChevronDown,
  Sparkles,
  Keyboard,
  Menu,
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
  onMenuClick: () => void;
}

const viewTitles: Record<ViewType, string> = {
  dashboard: 'Dashboard Overview',
  upload: 'Data Upload',
  datasets: 'Dataset Management',
  insights: 'AI Insights',
  ai_analytics: 'AI Analytics Engine',
  settings: 'Settings',
};

export function Header({ currentView, onSearchClick, onViewChange, onLogout, onMenuClick }: HeaderProps) {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-card/60 backdrop-blur-sm">
      <div className="flex h-16 items-center justify-between gap-2 px-3 sm:px-4 lg:px-6">
        <div className="flex min-w-0 items-center gap-2 sm:gap-4 lg:gap-6">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={onMenuClick}>
            <Menu className="w-5 h-5" />
          </Button>

          <div className="min-w-0">
            <h2 className="truncate text-base font-semibold text-foreground sm:text-lg lg:text-xl">
              {viewTitles[currentView]}
            </h2>
            <p className="hidden text-xs text-muted-foreground sm:block">
              {new Date().toLocaleDateString('en-US', {
                weekday: 'long',
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}
            </p>
          </div>

          <button
            onClick={onSearchClick}
            className="group relative hidden h-10 w-56 items-center rounded-lg border border-transparent bg-muted px-3 text-left transition-colors hover:border-health-mint/30 lg:flex xl:w-80"
          >
            <Search className="mr-3 w-4 h-4 text-muted-foreground" />
            <span className="flex-1 truncate text-sm text-muted-foreground">Search datasets, insights...</span>
            <div className="flex items-center gap-1">
              <kbd className="rounded bg-muted-foreground/20 px-1.5 py-0.5 text-xs text-muted-foreground transition-colors group-hover:bg-health-mint/20 group-hover:text-health-mint">
                ⌘
              </kbd>
              <kbd className="rounded bg-muted-foreground/20 px-1.5 py-0.5 text-xs text-muted-foreground transition-colors group-hover:bg-health-mint/20 group-hover:text-health-mint">
                K
              </kbd>
            </div>
          </button>
        </div>

        <div className="flex items-center gap-1 sm:gap-2">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={onSearchClick}>
            <Search className="w-5 h-5" />
          </Button>

          <Button
            variant="outline"
            className="h-9 gap-2 border-health-mint/30 px-2 text-health-mint hover:bg-health-mint/10 sm:px-3"
            onClick={() => onViewChange('ai_analytics')}
          >
            <Sparkles className="w-4 h-4" />
            <span className="hidden xl:inline">Ask AI</span>
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="relative">
                <Bell className="w-5 h-5" />
                <span className="absolute top-1 right-1 w-2 h-2 rounded-full bg-health-mint animate-pulse" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-[calc(100vw-2rem)] max-w-80">
              <DropdownMenuLabel>Notifications</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <div className="max-h-64 overflow-auto">
                <DropdownMenuItem className="cursor-pointer flex flex-col items-start gap-1 p-3">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-amber-500" />
                    <span className="text-sm font-medium">Data Quality Alert</span>
                  </div>
                  <p className="pl-4 text-xs text-muted-foreground">
                    15% of ANC records missing gestational age in Q3 dataset
                  </p>
                  <span className="pl-4 text-xs text-muted-foreground">2 minutes ago</span>
                </DropdownMenuItem>

                <DropdownMenuItem className="cursor-pointer flex flex-col items-start gap-1 p-3">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span className="text-sm font-medium">Dashboard Generated</span>
                  </div>
                  <p className="pl-4 text-xs text-muted-foreground">
                    Maternal Health Dashboard is ready for review
                  </p>
                  <span className="pl-4 text-xs text-muted-foreground">1 hour ago</span>
                </DropdownMenuItem>

                <DropdownMenuItem className="cursor-pointer flex flex-col items-start gap-1 p-3">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500" />
                    <span className="text-sm font-medium">Forecast Update</span>
                  </div>
                  <p className="pl-4 text-xs text-muted-foreground">
                    ANC coverage forecast updated with October data
                  </p>
                  <span className="pl-4 text-xs text-muted-foreground">3 hours ago</span>
                </DropdownMenuItem>
              </div>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="justify-center text-primary" onClick={() => onViewChange('insights')}>
                View all notifications
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="h-9 gap-2 pl-1.5 pr-1.5 sm:pl-2 sm:pr-3">
                <div className="w-8 h-8 rounded-full gradient-mint flex items-center justify-center">
                  <User className="w-4 h-4 text-background" />
                </div>
                <div className="hidden text-left md:block">
                  <p className="text-sm font-medium">Dr. Sarah Kimaro</p>
                  <p className="text-xs text-muted-foreground">District Health Manager</p>
                </div>
                <ChevronDown className="hidden w-4 h-4 text-muted-foreground sm:block" />
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
      </div>
    </header>
  );
}

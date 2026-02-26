import { useEffect, useMemo, useState } from 'react';
import { 
  Moon, 
  Sun, 
  Monitor, 
  Bell, 
  Shield, 
  User, 
  Database,
  Check,
  ChevronRight,
  Key,
  FileText,
  LogOut
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import { useI18n, type Language } from '@/lib/i18n';
import { useAnalytics } from '@/lib/analytics-context';
import { downloadJsonFile } from '@/lib/download';

type Theme = 'system' | 'light' | 'dark';

interface SettingsProps {
  onLogout?: () => void;
}

interface ConnectedService {
  name: string;
  status: 'connected' | 'not_connected';
  icon: typeof Database;
}

interface PersistedSettings {
  theme: Theme;
  notifications: {
    email: boolean;
    push: boolean;
    insights: boolean;
    alerts: boolean;
    weekly: boolean;
  };
  privacy: {
    analytics: boolean;
    shareData: boolean;
  };
  services: ConnectedService[];
}

const SETTINGS_STORAGE_KEY = 'healthai_settings_v1';

const DEFAULT_SERVICES: ConnectedService[] = [
  { name: 'DHIS2', status: 'connected', icon: Database },
  { name: 'Google Drive', status: 'not_connected', icon: Database },
];

function isTheme(value: string | null): value is Theme {
  return value === 'system' || value === 'light' || value === 'dark';
}

function loadPersistedSettings(): Partial<PersistedSettings> {
  const rawSettings = localStorage.getItem(SETTINGS_STORAGE_KEY);
  if (!rawSettings) return {};
  try {
    return JSON.parse(rawSettings) as Partial<PersistedSettings>;
  } catch {
    return {};
  }
}

export function Settings({ onLogout }: SettingsProps) {
  const { t, language, setLanguage } = useI18n();
  const { exportSnapshot, clearAllData, aiEnvironment } = useAnalytics();
  const persistedSettings = useMemo(() => loadPersistedSettings(), []);

  const [theme, setTheme] = useState<Theme>(() => {
    const savedTheme = localStorage.getItem('healthai_theme');
    if (isTheme(savedTheme)) return savedTheme;
    const persistedTheme = persistedSettings.theme;
    return persistedTheme && isTheme(persistedTheme) ? persistedTheme : 'system';
  });
  const [notifications, setNotifications] = useState(() => persistedSettings.notifications ?? {
    email: true,
    push: true,
    insights: true,
    alerts: true,
    weekly: false,
  });
  const [privacy, setPrivacy] = useState(() => persistedSettings.privacy ?? {
    analytics: true,
    shareData: false,
  });
  const [services, setServices] = useState<ConnectedService[]>(() => {
    if (!Array.isArray(persistedSettings.services)) return DEFAULT_SERVICES;
    const normalized = persistedSettings.services.filter((service): service is ConnectedService => {
      return (
        typeof service?.name === 'string' &&
        (service.status === 'connected' || service.status === 'not_connected')
      );
    });
    if (normalized.length === 0) return DEFAULT_SERVICES;
    return normalized.map((service) => ({ ...service, icon: Database }));
  });

  // Apply theme
  useEffect(() => {
    const root = document.documentElement;
    const body = document.body;
    
    if (theme === 'system') {
      const systemTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      if (systemTheme === 'dark') {
        root.classList.add('dark');
        body.classList.remove('light');
      } else {
        root.classList.remove('dark');
        body.classList.add('light');
      }
    } else if (theme === 'dark') {
      root.classList.add('dark');
      body.classList.remove('light');
    } else {
      root.classList.remove('dark');
      body.classList.add('light');
    }
    
    localStorage.setItem('healthai_theme', theme);
  }, [theme]);

  useEffect(() => {
    const payload: PersistedSettings = {
      theme,
      notifications,
      privacy,
      services: services.map((service) => ({ ...service, icon: Database })),
    };
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(payload));
  }, [theme, notifications, privacy, services]);

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">{t.settings}</h1>
        <p className="text-muted-foreground mt-1">
          {language === 'sw' ? 'Dhibiti mapendeleo yako na mipangilio ya akaunti' : 'Manage your preferences and account settings'}
        </p>
      </div>

      {/* Appearance Section */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Monitor className="w-5 h-5 text-health-mint" />
            {t.appearance}
          </CardTitle>
          <CardDescription>{t.themeDescription}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Theme Selection */}
          <div>
            <Label className="text-sm font-medium mb-3 block">{t.theme}</Label>
            <div className="grid grid-cols-3 gap-3">
              {[
                { value: 'system', icon: Monitor, label: t.system },
                { value: 'light', icon: Sun, label: t.light },
                { value: 'dark', icon: Moon, label: t.dark },
              ].map((option) => (
                <button
                  key={option.value}
                  onClick={() => setTheme(option.value as Theme)}
                  className={cn(
                    "flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-all",
                    theme === option.value
                      ? "border-health-mint bg-health-mint/10"
                      : "border-border hover:border-muted-foreground/50"
                  )}
                >
                  <option.icon className={cn(
                    "w-6 h-6",
                    theme === option.value ? "text-health-mint" : "text-muted-foreground"
                  )} />
                  <span className={cn(
                    "text-sm font-medium",
                    theme === option.value ? "text-foreground" : "text-muted-foreground"
                  )}>
                    {option.label}
                  </span>
                  {theme === option.value && (
                    <Check className="w-4 h-4 text-health-mint" />
                  )}
                </button>
              ))}
            </div>
          </div>

          <Separator />

          {/* Language Selection */}
          <div>
            <Label className="text-sm font-medium mb-3 block">{t.language}</Label>
            <div className="grid grid-cols-2 gap-3">
              {[
                { value: 'en' as Language, label: t.english, flag: '🇺🇸' },
                { value: 'sw' as Language, label: t.swahili, flag: '🇹🇿' },
              ].map((option) => (
                <button
                  key={option.value}
                  onClick={() => setLanguage(option.value)}
                  className={cn(
                    "flex items-center gap-3 p-4 rounded-xl border-2 transition-all",
                    language === option.value
                      ? "border-health-mint bg-health-mint/10"
                      : "border-border hover:border-muted-foreground/50"
                  )}
                >
                  <span className="text-2xl">{option.flag}</span>
                  <span className={cn(
                    "flex-1 text-left font-medium",
                    language === option.value ? "text-foreground" : "text-muted-foreground"
                  )}>
                    {option.label}
                  </span>
                  {language === option.value && (
                    <Check className="w-4 h-4 text-health-mint" />
                  )}
                </button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Notifications Section */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="w-5 h-5 text-health-mint" />
            {t.notifications}
          </CardTitle>
          <CardDescription>{t.notificationsDescription}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {[
            { key: 'email', label: t.emailNotifications, icon: '✉️' },
            { key: 'push', label: t.pushNotifications, icon: '🔔' },
            { key: 'insights', label: t.aiInsightNotifications, icon: '🤖' },
            { key: 'alerts', label: t.dataQualityAlerts, icon: '⚠️' },
            { key: 'weekly', label: t.weeklySummary, icon: '📊' },
          ].map((item) => (
            <div key={item.key} className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span>{item.icon}</span>
                <Label htmlFor={item.key} className="text-sm cursor-pointer">
                  {item.label}
                </Label>
              </div>
              <Switch
                id={item.key}
                checked={notifications[item.key as keyof typeof notifications]}
                onCheckedChange={(checked) => 
                  setNotifications(prev => ({ ...prev, [item.key]: checked }))
                }
              />
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Privacy Section */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-health-mint" />
            {t.privacy}
          </CardTitle>
          <CardDescription>{t.privacyDescription}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {[
            { key: 'analytics', label: t.usageAnalytics, desc: language === 'sw' ? 'Tusaidie kuboresha kwa kushiriki takwimu za matumizi' : 'Help us improve by sharing anonymous usage data' },
            { key: 'shareData', label: t.shareAnonymousData, desc: language === 'sw' ? 'Changia utafiti wa afya kwa data isiyojulikana' : 'Contribute to health research with anonymized data' },
          ].map((item) => (
            <div key={item.key} className="flex items-start justify-between">
              <div className="space-y-1">
                <Label htmlFor={item.key} className="text-sm font-medium cursor-pointer">
                  {item.label}
                </Label>
                <p className="text-xs text-muted-foreground">{item.desc}</p>
              </div>
              <Switch
                id={item.key}
                checked={privacy[item.key as keyof typeof privacy]}
                onCheckedChange={(checked) => 
                  setPrivacy(prev => ({ ...prev, [item.key]: checked }))
                }
              />
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Account Section */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <User className="w-5 h-5 text-health-mint" />
            {t.account}
          </CardTitle>
          <CardDescription>{t.accountDescription}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Connected Services */}
          <div className="space-y-3">
            <Label className="text-sm font-medium">{t.connectedServices}</Label>
            {services.map((service) => (
              <button
                type="button"
                key={service.name}
                className="flex items-center justify-between p-3 rounded-lg bg-muted/50"
                onClick={() =>
                  setServices((prev) =>
                    prev.map((item) =>
                      item.name === service.name
                        ? {
                            ...item,
                            status: item.status === 'connected' ? 'not_connected' : 'connected',
                          }
                        : item
                    )
                  )
                }
              >
                <div className="flex items-center gap-3">
                  <service.icon className="w-5 h-5 text-muted-foreground" />
                  <span className="text-sm">{service.name}</span>
                </div>
                <Badge 
                  variant="outline" 
                  className={cn(
                    "text-xs",
                    service.status === 'connected' 
                      ? "border-emerald-500/30 text-emerald-400" 
                      : "border-muted text-muted-foreground"
                  )}
                >
                  {service.status === 'connected' ? t.connected : t.notConnected}
                </Badge>
              </button>
            ))}
          </div>

          <Separator />

          {/* API Keys */}
          <button
            type="button"
            className="w-full flex items-center justify-between p-3 rounded-lg hover:bg-muted/50 transition-colors"
            onClick={() => {
              const existing = localStorage.getItem('healthai_api_key') ?? '';
              const next = window.prompt('Set local AI API key (optional)', existing);
              if (next === null) return;
              if (next.trim()) {
                localStorage.setItem('healthai_api_key', next.trim());
              } else {
                localStorage.removeItem('healthai_api_key');
              }
            }}
          >
            <div className="flex items-center gap-3">
              <Key className="w-5 h-5 text-muted-foreground" />
              <span className="text-sm">{t.apiKeys} ({aiEnvironment.status})</span>
            </div>
            <ChevronRight className="w-4 h-4 text-muted-foreground" />
          </button>
        </CardContent>
      </Card>

      {/* Data Management Section */}
      <Card className="glass-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="w-5 h-5 text-health-mint" />
            {t.dataManagement}
          </CardTitle>
          <CardDescription>{t.dataManagementDescription}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Button
            variant="outline"
            className="w-full justify-start gap-3"
            onClick={() => {
              const snapshot = exportSnapshot();
              downloadJsonFile('healthai-workspace-export.json', snapshot);
            }}
          >
            <FileText className="w-4 h-4" />
            {t.exportData}
          </Button>
          <Button
            variant="outline"
            className="w-full justify-start gap-3 text-destructive hover:text-destructive"
            onClick={() => {
              const confirmed = window.confirm(
                'This will remove all uploaded datasets and generated insights from this browser. Continue?'
              );
              if (!confirmed) return;
              clearAllData();
              onLogout?.();
            }}
          >
            <Database className="w-4 h-4" />
            {t.deleteAccount}
          </Button>
        </CardContent>
      </Card>

      {/* Logout */}
      <Button 
        variant="outline" 
        className="w-full gap-2 text-destructive hover:text-destructive"
        onClick={onLogout}
      >
        <LogOut className="w-4 h-4" />
        {t.logout}
      </Button>
    </div>
  );
}

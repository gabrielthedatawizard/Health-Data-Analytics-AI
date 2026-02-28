import { 
  TrendingUp, 
  TrendingDown,
  Users, 
  Activity, 
  Baby, 
  Heart,
  Calendar,
  Filter,
  Download,
  MoreHorizontal,
  ArrowUpRight,
  AlertCircle
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import { cn } from '@/lib/utils';

// Mock data for charts
const trendData = [
  { month: 'Jan', anc: 62, delivery: 68, pnc: 55 },
  { month: 'Feb', anc: 58, delivery: 65, pnc: 52 },
  { month: 'Mar', anc: 65, delivery: 70, pnc: 58 },
  { month: 'Apr', anc: 63, delivery: 69, pnc: 56 },
  { month: 'May', anc: 67, delivery: 72, pnc: 61 },
  { month: 'Jun', anc: 70, delivery: 74, pnc: 64 },
  { month: 'Jul', anc: 68, delivery: 73, pnc: 62 },
  { month: 'Aug', anc: 72, delivery: 76, pnc: 67 },
  { month: 'Sep', anc: 75, delivery: 78, pnc: 70 },
  { month: 'Oct', anc: 73, delivery: 77, pnc: 68 },
  { month: 'Nov', anc: 76, delivery: 79, pnc: 71 },
  { month: 'Dec', anc: 78, delivery: 81, pnc: 73 },
];

const districtData = [
  { name: 'Chamwino', value: 82 },
  { name: 'Bahi', value: 78 },
  { name: 'Dodoma Muni', value: 75 },
  { name: 'Chemba', value: 71 },
  { name: 'Kondoa', value: 64 },
  { name: 'Mpwapwa', value: 69 },
];

const ageDistribution = [
  { name: '15-19', value: 15, color: '#34d399' },
  { name: '20-24', value: 32, color: '#10b981' },
  { name: '25-29', value: 28, color: '#059669' },
  { name: '30-34', value: 18, color: '#047857' },
  { name: '35+', value: 7, color: '#065f46' },
];

interface KPICardProps {
  title: string;
  value: string;
  change: number;
  changeLabel: string;
  icon: React.ElementType;
  trend: 'up' | 'down' | 'neutral';
  target?: string;
  confidence: 'high' | 'medium' | 'low';
}

function KPICard({ title, value, change, changeLabel, icon: Icon, trend, target, confidence }: KPICardProps) {
  return (
    <Card className="glass-card card-hover overflow-hidden">
      <CardContent className="p-4 sm:p-5">
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <div className={cn(
                "w-8 h-8 rounded-lg flex items-center justify-center",
                trend === 'up' ? "bg-emerald-500/20" : 
                trend === 'down' ? "bg-red-500/20" : "bg-blue-500/20"
              )}>
                <Icon className={cn(
                  "w-4 h-4",
                  trend === 'up' ? "text-emerald-400" : 
                  trend === 'down' ? "text-red-400" : "text-blue-400"
                )} />
              </div>
              <span className="text-xs text-muted-foreground uppercase tracking-wider">{title}</span>
            </div>
            
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold text-foreground sm:text-3xl">{value}</span>
              {target && (
                <span className="text-xs text-muted-foreground">/ {target}</span>
              )}
            </div>
            
            <div className="flex items-center gap-2 mt-2">
              <div className={cn(
                "flex items-center gap-1 text-sm font-medium",
                trend === 'up' ? "text-emerald-400" : 
                trend === 'down' ? "text-red-400" : "text-blue-400"
              )}>
                {trend === 'up' ? <TrendingUp className="w-4 h-4" /> : 
                 trend === 'down' ? <TrendingDown className="w-4 h-4" /> : null}
                <span>{change > 0 ? '+' : ''}{change}%</span>
              </div>
              <span className="text-xs text-muted-foreground">{changeLabel}</span>
            </div>
          </div>
          
          <div className="flex flex-col items-end gap-2">
            <Badge 
              variant="outline" 
              className={cn(
                "text-xs",
                confidence === 'high' ? "border-emerald-500/30 text-emerald-400" :
                confidence === 'medium' ? "border-amber-500/30 text-amber-400" :
                "border-red-500/30 text-red-400"
              )}
            >
              {confidence} confidence
            </Badge>
          </div>
        </div>
        
        {target && (
          <div className="mt-4">
            <div className="flex justify-between text-xs mb-1">
              <span className="text-muted-foreground">Progress to target</span>
              <span className="text-foreground font-medium">87%</span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
              <div className="h-full w-[87%] gradient-mint rounded-full" />
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function Dashboard() {
  return (
    <div className="space-y-6">
      {/* Welcome & Actions */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-foreground sm:text-2xl">
            Welcome back, <span className="text-gradient-mint">Dr. Kimaro</span>
          </h1>
          <p className="text-sm text-muted-foreground mt-1 sm:text-base">
            Here's what's happening in Dodoma Region health facilities
          </p>
        </div>
        
        <div className="flex flex-wrap items-center gap-2 sm:flex-nowrap">
          <Button variant="outline" size="sm" className="gap-2 flex-1 sm:flex-none">
            <Calendar className="w-4 h-4" />
            <span>Last 12 months</span>
            <Filter className="w-3 h-3" />
          </Button>
          <Button variant="outline" size="sm" className="gap-2 flex-1 sm:flex-none">
            <Download className="w-4 h-4" />
            Export
          </Button>
        </div>
      </div>

      {/* AI Insights Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-health-mint/30 bg-gradient-to-r from-health-mint/10 via-health-purple/5 to-transparent p-4 sm:p-5">
        <div className="absolute top-0 right-0 w-64 h-64 gradient-glow-mint opacity-50" />
        <div className="relative flex flex-col gap-4 sm:flex-row sm:items-start">
          <div className="w-10 h-10 rounded-xl bg-health-mint/20 flex items-center justify-center flex-shrink-0">
            <Activity className="w-5 h-5 text-health-mint" />
          </div>
          <div className="flex-1">
            <div className="mb-1 flex flex-wrap items-center gap-2">
              <h3 className="font-semibold text-foreground">AI-Generated Insight</h3>
              <Badge variant="outline" className="text-xs border-health-mint/30 text-health-mint">
                94% confidence
              </Badge>
            </div>
            <p className="text-sm leading-relaxed text-muted-foreground">
              ANC coverage has improved <span className="text-emerald-400 font-medium">+12.3%</span> this quarter, 
              with particularly strong gains in <span className="text-foreground">Chamwino (+18%)</span> and 
              <span className="text-foreground"> Bahi (+15%)</span> districts. However, 
              <span className="text-amber-400"> Kondoa remains below target at 52%</span> coverage.
            </p>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <Button size="sm" variant="outline" className="h-8 text-xs gap-1 border-health-mint/30 text-health-mint hover:bg-health-mint/10">
                View Details
                <ArrowUpRight className="w-3 h-3" />
              </Button>
              <Button size="sm" variant="ghost" className="h-8 text-xs gap-1 text-muted-foreground">
                <AlertCircle className="w-3 h-3" />
                Flag for Review
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title="ANC Coverage"
          value="68.4%"
          change={12.3}
          changeLabel="vs last quarter"
          icon={Users}
          trend="up"
          target="80%"
          confidence="high"
        />
        <KPICard
          title="Facility Delivery"
          value="74.2%"
          change={5.1}
          changeLabel="vs last quarter"
          icon={Baby}
          trend="up"
          target="85%"
          confidence="high"
        />
        <KPICard
          title="Maternal Mortality"
          value="89"
          change={-15}
          changeLabel="vs last year"
          icon={Heart}
          trend="up"
          confidence="medium"
        />
        <KPICard
          title="Total ANC Visits"
          value="12,473"
          change={8.7}
          changeLabel="vs last quarter"
          icon={Activity}
          trend="up"
          confidence="high"
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trend Chart */}
        <Card className="glass-card lg:col-span-2">
          <CardHeader className="flex flex-col gap-3 pb-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">ANC Coverage Trend</CardTitle>
              <p className="text-xs text-muted-foreground">Monthly coverage rate over time</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-full bg-emerald-500" />
                <span className="text-xs text-muted-foreground">ANC</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-full bg-blue-500" />
                <span className="text-xs text-muted-foreground">Delivery</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-full bg-purple-500" />
                <span className="text-xs text-muted-foreground">PNC</span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-60 sm:h-72">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={trendData}>
                  <defs>
                    <linearGradient id="ancGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="deliveryGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 15% 18%)" />
                  <XAxis 
                    dataKey="month" 
                    stroke="hsl(220 10% 60%)" 
                    fontSize={12}
                    tickLine={false}
                  />
                  <YAxis 
                    stroke="hsl(220 10% 60%)" 
                    fontSize={12}
                    tickLine={false}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(220 18% 8%)', 
                      border: '1px solid hsl(220 15% 18%)',
                      borderRadius: '8px'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="anc" 
                    stroke="#10b981" 
                    fillOpacity={1} 
                    fill="url(#ancGradient)" 
                    strokeWidth={2}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="delivery" 
                    stroke="#3b82f6" 
                    fillOpacity={1} 
                    fill="url(#deliveryGradient)" 
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="pnc" 
                    stroke="#8b5cf6" 
                    strokeWidth={2}
                    dot={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Age Distribution */}
        <Card className="glass-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-lg font-semibold">ANC by Age Group</CardTitle>
            <p className="text-xs text-muted-foreground">Distribution of visits</p>
          </CardHeader>
          <CardContent>
            <div className="h-52 sm:h-56">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={ageDistribution}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={4}
                    dataKey="value"
                  >
                    {ageDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(220 18% 8%)', 
                      border: '1px solid hsl(220 15% 18%)',
                      borderRadius: '8px'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-4 sm:grid-cols-3">
              {ageDistribution.slice(0, 3).map((item) => (
                <div key={item.name} className="text-center">
                  <div className="flex items-center justify-center gap-1">
                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                    <span className="text-xs text-muted-foreground">{item.name}</span>
                  </div>
                  <span className="text-sm font-medium">{item.value}%</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* District Comparison */}
        <Card className="glass-card">
          <CardHeader className="flex flex-col gap-3 pb-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">ANC Coverage by District</CardTitle>
              <p className="text-xs text-muted-foreground">Performance comparison</p>
            </div>
            <Button variant="ghost" size="icon">
              <MoreHorizontal className="w-4 h-4" />
            </Button>
          </CardHeader>
          <CardContent>
            <div className="h-60 sm:h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={districtData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(220 15% 18%)" horizontal={false} />
                  <XAxis 
                    type="number" 
                    stroke="hsl(220 10% 60%)" 
                    fontSize={12}
                    tickFormatter={(v) => `${v}%`}
                  />
                  <YAxis 
                    type="category" 
                    dataKey="name" 
                    stroke="hsl(220 10% 60%)" 
                    fontSize={12}
                    width={90}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(220 18% 8%)', 
                      border: '1px solid hsl(220 15% 18%)',
                      borderRadius: '8px'
                    }}
                    formatter={(v) => [`${v}%`, 'Coverage']}
                  />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {districtData.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.value >= 75 ? '#10b981' : entry.value >= 65 ? '#f59e0b' : '#ef4444'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Facility Performance Table */}
        <Card className="glass-card">
          <CardHeader className="flex flex-col gap-3 pb-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <CardTitle className="text-lg font-semibold">Facility Performance</CardTitle>
              <p className="text-xs text-muted-foreground">Top performing facilities</p>
            </div>
            <Button variant="outline" size="sm" className="text-xs">
              View All
            </Button>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px]">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-2 text-xs font-medium text-muted-foreground uppercase">Facility</th>
                    <th className="text-right py-3 px-2 text-xs font-medium text-muted-foreground uppercase">ANC Visits</th>
                    <th className="text-right py-3 px-2 text-xs font-medium text-muted-foreground uppercase">Coverage</th>
                    <th className="text-center py-3 px-2 text-xs font-medium text-muted-foreground uppercase">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { name: 'Chamwino District Hospital', visits: 2847, coverage: 82, status: 'on-target' },
                    { name: 'Bahi Health Center', visits: 1923, coverage: 78, status: 'on-target' },
                    { name: 'Dodoma Regional Hospital', visits: 4521, coverage: 75, status: 'on-target' },
                    { name: 'Chemba Dispensary', visits: 987, coverage: 71, status: 'at-risk' },
                    { name: 'Kondoa Health Center', visits: 1456, coverage: 64, status: 'below-target' },
                  ].map((facility, i) => (
                    <tr key={i} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                      <td className="py-3 px-2">
                        <div>
                          <p className="text-sm font-medium text-foreground">{facility.name}</p>
                          <p className="text-xs text-muted-foreground">District {i + 1}</p>
                        </div>
                      </td>
                      <td className="text-right py-3 px-2 text-sm">{facility.visits.toLocaleString()}</td>
                      <td className="text-right py-3 px-2">
                        <span className={cn(
                          "text-sm font-medium",
                          facility.coverage >= 75 ? "text-emerald-400" :
                          facility.coverage >= 65 ? "text-amber-400" : "text-red-400"
                        )}>
                          {facility.coverage}%
                        </span>
                      </td>
                      <td className="text-center py-3 px-2">
                        <Badge 
                          variant="outline" 
                          className={cn(
                            "text-xs",
                            facility.status === 'on-target' ? "border-emerald-500/30 text-emerald-400" :
                            facility.status === 'at-risk' ? "border-amber-500/30 text-amber-400" :
                            "border-red-500/30 text-red-400"
                          )}
                        >
                          {facility.status === 'on-target' ? 'On Target' :
                           facility.status === 'at-risk' ? 'At Risk' : 'Below Target'}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Data Quality Warning */}
      <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
        <AlertCircle className="w-5 h-5 text-amber-500 flex-shrink-0 mt-0.5" />
        <div>
          <h4 className="font-medium text-amber-400 text-sm">Data Quality Warning</h4>
          <p className="text-xs text-muted-foreground mt-1">
            15% of ANC records are missing gestational age. This may affect coverage calculations. 
            <button className="text-health-mint hover:underline ml-1">View affected records</button>
          </p>
        </div>
      </div>
    </div>
  );
}

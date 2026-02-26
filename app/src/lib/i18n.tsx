// Comprehensive Translation System for HealthAI
// Supports English and Swahili

import { useState, useEffect, useCallback, createContext, useContext, type ReactNode } from 'react';

export type Language = 'en' | 'sw';

export interface Translations {
  // Common
  appName: string;
  loading: string;
  error: string;
  success: string;
  cancel: string;
  save: string;
  delete: string;
  edit: string;
  create: string;
  search: string;
  filter: string;
  export: string;
  import: string;
  download: string;
  upload: string;
  close: string;
  back: string;
  next: string;
  previous: string;
  submit: string;
  confirm: string;
  dismiss: string;
  view: string;
  more: string;
  less: string;
  show: string;
  hide: string;
  all: string;
  none: string;
  or: string;
  and: string;
  
  // Navigation
  dashboard: string;
  dataUpload: string;
  datasets: string;
  aiInsights: string;
  settings: string;
  profile: string;
  logout: string;
  login: string;
  signup: string;
  
  // Dashboard
  welcome: string;
  welcomeBack: string;
  heresWhatsHappening: string;
  quickStats: string;
  recentActivity: string;
  
  // KPIs
  ancCoverage: string;
  facilityDelivery: string;
  maternalMortality: string;
  totalAncVisits: string;
  ancCoverageDesc: string;
  facilityDeliveryDesc: string;
  progressToTarget: string;
  confidence: string;
  high: string;
  medium: string;
  low: string;
  
  // Charts
  trend: string;
  comparison: string;
  distribution: string;
  performance: string;
  monthly: string;
  quarterly: string;
  yearly: string;
  byDistrict: string;
  byFacility: string;
  byAgeGroup: string;
  
  // AI Insights
  aiGeneratedInsight: string;
  generatedByAI: string;
  confidenceScore: string;
  citations: string;
  sources: string;
  wasThisHelpful: string;
  helpful: string;
  notHelpful: string;
  flagForReview: string;
  viewDetails: string;
  askYourData: string;
  askQuestionPlaceholder: string;
  analyzing: string;
  zeroHallucination: string;
  allNumbersComputed: string;
  citationsProvided: string;
  confidenceScores: string;
  
  // Data Upload
  dragDropFiles: string;
  orClickToBrowse: string;
  supportedFormats: string;
  fileSizeLimit: string;
  uploading: string;
  processing: string;
  completed: string;
  failed: string;
  qualityScore: string;
  rows: string;
  columns: string;
  issuesFound: string;
  noIssues: string;
  analyze: string;
  connect: string;
  recentUploads: string;
  review: string;
  sample: string;
  exploreDemo: string;
  
  // Data Quality
  dataQuality: string;
  completeness: string;
  validity: string;
  consistency: string;
  timeliness: string;
  uniqueness: string;
  
  // Settings
  appearance: string;
  theme: string;
  themeDescription: string;
  language: string;
  languageDescription: string;
  system: string;
  light: string;
  dark: string;
  english: string;
  swahili: string;
  notifications: string;
  notificationsDescription: string;
  emailNotifications: string;
  pushNotifications: string;
  aiInsightNotifications: string;
  dataQualityAlerts: string;
  weeklySummary: string;
  privacy: string;
  privacyDescription: string;
  usageAnalytics: string;
  shareAnonymousData: string;
  account: string;
  accountDescription: string;
  connectedServices: string;
  apiKeys: string;
  dataManagement: string;
  dataManagementDescription: string;
  exportData: string;
  deleteAccount: string;
  connected: string;
  notConnected: string;
  
  // Time
  justNow: string;
  minutesAgo: string;
  hoursAgo: string;
  daysAgo: string;
  today: string;
  yesterday: string;
  at: string;
  
  // Subscription/Pricing
  pricing: string;
  plans: string;
  starter: string;
  professional: string;
  enterprise: string;
  free: string;
  perMonth: string;
  forever: string;
  custom: string;
  mostPopular: string;
  getStarted: string;
  startFreeTrial: string;
  contactSales: string;
  features: string;
  records: string;
  basicDashboards: string;
  advancedDashboards: string;
  aiPredictions: string;
  databaseConnectors: string;
  emailSupport: string;
  prioritySupport: string;
  teamCollaboration: string;
  unlimitedRecords: string;
  customIntegrations: string;
  onPremise: string;
  dedicatedSupport: string;
  sla: string;
  training: string;
  
  // Payment (Tanzania)
  payment: string;
  paymentMethods: string;
  mpesa: string;
  airtelMoney: string;
  tigoPesa: string;
  haloPesa: string;
  zantel: string;
  ttcl: string;
  bankTransfer: string;
  cardPayment: string;
  enterPhoneNumber: string;
  enterAmount: string;
  payNow: string;
  paymentSuccessful: string;
  paymentFailed: string;
  checkYourPhone: string;
  
  // Status
  active: string;
  inactive: string;
  pending: string;
  onTarget: string;
  atRisk: string;
  belowTarget: string;
  
  // Footer
  documentation: string;
  helpCenter: string;
  contact: string;
  about: string;
  blog: string;
  careers: string;
  terms: string;
  privacyPolicy: string;
  cookies: string;
  allRightsReserved: string;
  hipaaCompliant: string;
  soc2Certified: string;
}

export const translations: Record<Language, Translations> = {
  en: {
    appName: 'HealthAI',
    loading: 'Loading...',
    error: 'Error',
    success: 'Success',
    cancel: 'Cancel',
    save: 'Save',
    delete: 'Delete',
    edit: 'Edit',
    create: 'Create',
    search: 'Search',
    filter: 'Filter',
    export: 'Export',
    import: 'Import',
    download: 'Download',
    upload: 'Upload',
    close: 'Close',
    back: 'Back',
    next: 'Next',
    previous: 'Previous',
    submit: 'Submit',
    confirm: 'Confirm',
    dismiss: 'Dismiss',
    view: 'View',
    more: 'More',
    less: 'Less',
    show: 'Show',
    hide: 'Hide',
    all: 'All',
    none: 'None',
    or: 'or',
    and: 'and',
    
    dashboard: 'Dashboard',
    dataUpload: 'Data Upload',
    datasets: 'Datasets',
    aiInsights: 'AI Insights',
    settings: 'Settings',
    profile: 'Profile',
    logout: 'Log Out',
    login: 'Log In',
    signup: 'Sign Up',
    
    welcome: 'Welcome',
    welcomeBack: 'Welcome back',
    heresWhatsHappening: "Here's what's happening in your health facilities",
    quickStats: 'Quick Stats',
    recentActivity: 'Recent Activity',
    
    ancCoverage: 'ANC Coverage',
    facilityDelivery: 'Facility Delivery',
    maternalMortality: 'Maternal Mortality',
    totalAncVisits: 'Total ANC Visits',
    ancCoverageDesc: 'Percentage of pregnant women with 4+ ANC visits',
    facilityDeliveryDesc: 'Percentage of deliveries at health facilities',
    progressToTarget: 'Progress to target',
    confidence: 'Confidence',
    high: 'High',
    medium: 'Medium',
    low: 'Low',
    
    trend: 'Trend',
    comparison: 'Comparison',
    distribution: 'Distribution',
    performance: 'Performance',
    monthly: 'Monthly',
    quarterly: 'Quarterly',
    yearly: 'Yearly',
    byDistrict: 'by District',
    byFacility: 'by Facility',
    byAgeGroup: 'by Age Group',
    
    aiGeneratedInsight: 'AI-Generated Insight',
    generatedByAI: 'Generated by AI',
    confidenceScore: 'Confidence Score',
    citations: 'Citations',
    sources: 'Sources',
    wasThisHelpful: 'Was this helpful?',
    helpful: 'Helpful',
    notHelpful: 'Not helpful',
    flagForReview: 'Flag for review',
    viewDetails: 'View details',
    askYourData: 'Ask your data',
    askQuestionPlaceholder: 'e.g., What factors are driving ANC coverage improvement?',
    analyzing: 'Analyzing...',
    zeroHallucination: 'Zero Hallucination Guarantee',
    allNumbersComputed: 'All numbers computed',
    citationsProvided: 'Citations provided',
    confidenceScores: 'Confidence scores',
    
    dragDropFiles: 'Drop your files here',
    orClickToBrowse: 'or click to browse from your computer',
    supportedFormats: 'Supported formats',
    fileSizeLimit: 'Maximum file size: 100MB',
    uploading: 'Uploading...',
    processing: 'Processing...',
    completed: 'Completed',
    failed: 'Failed',
    qualityScore: 'Quality Score',
    rows: 'rows',
    columns: 'columns',
    issuesFound: 'issues found',
    noIssues: 'No issues found',
    analyze: 'Analyze',
    connect: 'Connect Database',
    recentUploads: 'Recent Uploads',
    review: 'Review',
    sample: 'Sample Datasets',
    exploreDemo: 'Explore with demo data',
    
    dataQuality: 'Data Quality',
    completeness: 'Completeness',
    validity: 'Validity',
    consistency: 'Consistency',
    timeliness: 'Timeliness',
    uniqueness: 'Uniqueness',
    
    appearance: 'Appearance',
    theme: 'Theme',
    themeDescription: 'Choose your preferred color scheme',
    language: 'Language',
    languageDescription: 'Select your preferred language',
    system: 'System',
    light: 'Light',
    dark: 'Dark',
    english: 'English',
    swahili: 'Swahili',
    notifications: 'Notifications',
    notificationsDescription: 'Manage how you receive updates',
    emailNotifications: 'Email Notifications',
    pushNotifications: 'Push Notifications',
    aiInsightNotifications: 'AI Insights',
    dataQualityAlerts: 'Data Quality Alerts',
    weeklySummary: 'Weekly Summary',
    privacy: 'Privacy & Security',
    privacyDescription: 'Control your data and privacy settings',
    usageAnalytics: 'Usage Analytics',
    shareAnonymousData: 'Share Anonymous Data',
    account: 'Account',
    accountDescription: 'Manage your account settings',
    connectedServices: 'Connected Services',
    apiKeys: 'API Keys',
    dataManagement: 'Data Management',
    dataManagementDescription: 'Export or delete your data',
    exportData: 'Export Data',
    deleteAccount: 'Delete Account',
    connected: 'Connected',
    notConnected: 'Not Connected',
    
    justNow: 'Just now',
    minutesAgo: 'minutes ago',
    hoursAgo: 'hours ago',
    daysAgo: 'days ago',
    today: 'Today',
    yesterday: 'Yesterday',
    at: 'at',
    
    pricing: 'Pricing',
    plans: 'Plans',
    starter: 'Starter',
    professional: 'Professional',
    enterprise: 'Enterprise',
    free: 'Free',
    perMonth: '/month',
    forever: 'forever',
    custom: 'Custom',
    mostPopular: 'Most Popular',
    getStarted: 'Get Started',
    startFreeTrial: 'Start Free Trial',
    contactSales: 'Contact Sales',
    features: 'Features',
    records: 'records',
    basicDashboards: 'Basic Dashboards',
    advancedDashboards: 'Advanced Dashboards',
    aiPredictions: 'AI Predictions',
    databaseConnectors: 'Database Connectors',
    emailSupport: 'Email Support',
    prioritySupport: 'Priority Support',
    teamCollaboration: 'Team Collaboration',
    unlimitedRecords: 'Unlimited Records',
    customIntegrations: 'Custom Integrations',
    onPremise: 'On-Premise Deployment',
    dedicatedSupport: 'Dedicated Support',
    sla: 'SLA Guarantee',
    training: 'Training & Onboarding',
    
    payment: 'Payment',
    paymentMethods: 'Payment Methods',
    mpesa: 'M-Pesa',
    airtelMoney: 'Airtel Money',
    tigoPesa: 'Tigo Pesa',
    haloPesa: 'HaloPesa',
    zantel: 'Zantel',
    ttcl: 'TTCL',
    bankTransfer: 'Bank Transfer',
    cardPayment: 'Card Payment',
    enterPhoneNumber: 'Enter phone number',
    enterAmount: 'Enter amount',
    payNow: 'Pay Now',
    paymentSuccessful: 'Payment successful!',
    paymentFailed: 'Payment failed. Please try again.',
    checkYourPhone: 'Check your phone to complete payment',
    
    active: 'Active',
    inactive: 'Inactive',
    pending: 'Pending',
    onTarget: 'On Target',
    atRisk: 'At Risk',
    belowTarget: 'Below Target',
    
    documentation: 'Documentation',
    helpCenter: 'Help Center',
    contact: 'Contact',
    about: 'About',
    blog: 'Blog',
    careers: 'Careers',
    terms: 'Terms of Service',
    privacyPolicy: 'Privacy Policy',
    cookies: 'Cookie Policy',
    allRightsReserved: 'All rights reserved',
    hipaaCompliant: 'HIPAA Compliant',
    soc2Certified: 'SOC 2 Certified',
  },
  
  sw: {
    appName: 'HealthAI',
    loading: 'Inapakia...',
    error: 'Hitilafu',
    success: 'Mafanikio',
    cancel: 'Ghairi',
    save: 'Hifadhi',
    delete: 'Futa',
    edit: 'Hariri',
    create: 'Unda',
    search: 'Tafuta',
    filter: 'Chuja',
    export: 'Hamisha',
    import: 'Leta',
    download: 'Pakua',
    upload: 'Pakia',
    close: 'Funga',
    back: 'Rudi',
    next: 'Endelea',
    previous: 'Nyuma',
    submit: 'Wasilisha',
    confirm: 'Thibitisha',
    dismiss: 'Ondoa',
    view: 'Angalia',
    more: 'Zaidi',
    less: 'Punguza',
    show: 'Onyesha',
    hide: 'Ficha',
    all: 'Zote',
    none: 'Hakuna',
    or: 'au',
    and: 'na',
    
    dashboard: 'Dashibodi',
    dataUpload: 'Pakia Data',
    datasets: 'Seti za Data',
    aiInsights: 'Maarifa ya AI',
    settings: 'Mipangilio',
    profile: 'Wasifu',
    logout: 'Toka',
    login: 'Ingia',
    signup: 'Jisajili',
    
    welcome: 'Karibu',
    welcomeBack: 'Karibu tena',
    heresWhatsHappening: 'Haya ndiyo yanayotokea katika vituo vya afya vyako',
    quickStats: 'Takwimu za Haraka',
    recentActivity: 'Shughuli za Hivi Karibuni',
    
    ancCoverage: 'Ufikiaji wa ANC',
    facilityDelivery: 'Kujifungua Vituoni',
    maternalMortality: 'Vifo vya Akina Mama',
    totalAncVisits: 'Ziara za ANC Jumla',
    ancCoverageDesc: 'Asilimia ya wanawake wajawazito wenye ziara 4+ za ANC',
    facilityDeliveryDesc: 'Asilimia ya wanaojifungua katika vituo vya afya',
    progressToTarget: 'Maendeleo kuelekea lengo',
    confidence: 'Uhakika',
    high: 'Mkubwa',
    medium: 'Wastani',
    low: 'Mdogo',
    
    trend: 'Mwelekeo',
    comparison: 'Linganisha',
    distribution: 'Usambazaji',
    performance: 'Utendaji',
    monthly: 'Kila Mwezi',
    quarterly: 'Robo Mwaka',
    yearly: 'Kila Mwaka',
    byDistrict: 'kwa Wilaya',
    byFacility: 'kwa Kituo',
    byAgeGroup: 'kwa Kikundi cha Umri',
    
    aiGeneratedInsight: 'Maarifa Yaliyotokana na AI',
    generatedByAI: 'Yaliyotengenezwa na AI',
    confidenceScore: 'Kiwango cha Uhakika',
    citations: 'Marejeleo',
    sources: 'Vyanzo',
    wasThisHelpful: 'Je, hili lilikuwa la manufaa?',
    helpful: 'La manufaa',
    notHelpful: 'Sio la manufaa',
    flagForReview: 'Weka alama kwa ukaguzi',
    viewDetails: 'Angalia Maelezo',
    askYourData: 'Uliza data yako',
    askQuestionPlaceholder: 'mfano. Ni sababu zipi zinasababisha uboreshaji wa ufikiaji wa ANC?',
    analyzing: 'Inachanganua...',
    zeroHallucination: 'Hakuna Udanganyifu',
    allNumbersComputed: 'Namba zote zimehesabiwa',
    citationsProvided: 'Marejeleo yametolewa',
    confidenceScores: 'Viwango vya uhakika',
    
    dragDropFiles: 'Acha faili zako hapa',
    orClickToBrowse: 'au bofya kuchunguza kutoka kwa kompyuta yako',
    supportedFormats: 'Fomati zinazotumika',
    fileSizeLimit: 'Ukubwa wa faili wa juu: MB 100',
    uploading: 'Inapakia...',
    processing: 'Inachakata...',
    completed: 'Imekamilika',
    failed: 'Imeshindwa',
    qualityScore: 'Kiwango cha Ubora',
    rows: 'safu',
    columns: 'nguzo',
    issuesFound: 'matatizo yaliyopatikana',
    noIssues: 'Hakuna matatizo',
    analyze: 'Changanua',
    connect: 'Unganisha Hifadhidata',
    recentUploads: 'Vipakiaji Vya Hivi Karibuni',
    review: 'Kagua',
    sample: 'Seti za Data za Mfano',
    exploreDemo: 'Chunguza na data ya mfano',
    
    dataQuality: 'Ubora wa Data',
    completeness: 'Ukamilifu',
    validity: 'Uhalali',
    consistency: 'Ulinganifu',
    timeliness: 'Wakati sahihi',
    uniqueness: 'Upekee',
    
    appearance: 'Muonekano',
    theme: 'Mandhari',
    themeDescription: 'Chagua mpangilio wa rangi unaopendelea',
    language: 'Lugha',
    languageDescription: 'Chagua lugha yako unayopendelea',
    system: 'Mfumo',
    light: 'Nuru',
    dark: 'Giza',
    english: 'Kiingereza',
    swahili: 'Kiswahili',
    notifications: 'Arifa',
    notificationsDescription: 'Dhibiti jinsi unavyopokea visasisho',
    emailNotifications: 'Arifa za Barua Pepe',
    pushNotifications: 'Arifa za Moja kwa Moja',
    aiInsightNotifications: 'Maarifa ya AI',
    dataQualityAlerts: 'Arifa za Ubora wa Data',
    weeklySummary: 'Muhtasari wa Wiki',
    privacy: 'Faragha na Usalama',
    privacyDescription: 'Dhibiti data yako na mipangilio ya faragha',
    usageAnalytics: 'Takwimu za Matumizi',
    shareAnonymousData: 'Shiriki Data Isiyojulikana',
    account: 'Akaunti',
    accountDescription: 'Dhibiti mipangilio ya akaunti yako',
    connectedServices: 'Huduma Zilizounganishwa',
    apiKeys: 'Funguo za API',
    dataManagement: 'Usimamizi wa Data',
    dataManagementDescription: 'Hamisha au futa data yako',
    exportData: 'Hamisha Data',
    deleteAccount: 'Futa Akaunti',
    connected: 'Imeunganishwa',
    notConnected: 'Haijaunganishwa',
    
    justNow: 'Sasa hivi',
    minutesAgo: 'dakika zilizopita',
    hoursAgo: 'masaa yaliyopita',
    daysAgo: 'siku zilizopita',
    today: 'Leo',
    yesterday: 'Jana',
    at: 'saa',
    
    pricing: 'Bei',
    plans: 'Mipango',
    starter: 'Mwanzo',
    professional: 'Mtaalamu',
    enterprise: 'Kampuni',
    free: 'Bure',
    perMonth: '/mwezi',
    forever: 'milele',
    custom: 'Maalum',
    mostPopular: 'Maarufu Zaidi',
    getStarted: 'Anza',
    startFreeTrial: 'Anza Jaribio Bure',
    contactSales: 'Wasiliana na Mauzo',
    features: 'Vipengele',
    records: 'rekodi',
    basicDashboards: 'Dashibodi za Msingi',
    advancedDashboards: 'Dashibodi za Kina',
    aiPredictions: 'Utabiri wa AI',
    databaseConnectors: 'Unganishi wa Hifadhidata',
    emailSupport: 'Msaada wa Barua Pepe',
    prioritySupport: 'Msaada wa Kipaumbele',
    teamCollaboration: 'Ushirikiano wa Timu',
    unlimitedRecords: 'Rekodi Zisizo na Kikomo',
    customIntegrations: 'Uunganisho Maalum',
    onPremise: 'Utekelezaji Mahali',
    dedicatedSupport: 'Msaada Maalum',
    sla: 'Dhamana ya SLA',
    training: 'Mafunzo na Uanzishaji',
    
    payment: 'Malipo',
    paymentMethods: 'Njia za Malipo',
    mpesa: 'M-Pesa',
    airtelMoney: 'Airtel Money',
    tigoPesa: 'Tigo Pesa',
    haloPesa: 'HaloPesa',
    zantel: 'Zantel',
    ttcl: 'TTCL',
    bankTransfer: 'Hamisha Benki',
    cardPayment: 'Malipo kwa Kadi',
    enterPhoneNumber: 'Weka namba ya simu',
    enterAmount: 'Weka kiasi',
    payNow: 'Lipa Sasa',
    paymentSuccessful: 'Malipo yamefaulu!',
    paymentFailed: 'Malipo yameshindwa. Tafadhali jaribu tena.',
    checkYourPhone: 'Angalia simu yako kukamilisha malipo',
    
    active: 'Hai',
    inactive: 'Isiyohai',
    pending: 'Inasubiri',
    onTarget: 'Kufikia Lengo',
    atRisk: 'Hatarini',
    belowTarget: 'Chini ya Lengo',
    
    documentation: 'Nyaraka',
    helpCenter: 'Kituo cha Msaada',
    contact: 'Wasiliana',
    about: 'Kuhusu',
    blog: 'Blogu',
    careers: 'Kazi',
    terms: 'Masharti ya Huduma',
    privacyPolicy: 'Sera ya Faragha',
    cookies: 'Sera ya Vidakuzi',
    allRightsReserved: 'Haki zote zimehifadhiwa',
    hipaaCompliant: 'Inalingana na HIPAA',
    soc2Certified: 'Imethibitishwa SOC 2',
  }
};

// Hook for using translations
export function useTranslation() {
  const [language, setLanguageState] = useState<Language>('en');
  
  useEffect(() => {
    const saved = localStorage.getItem('healthai_language') as Language;
    if (saved && (saved === 'en' || saved === 'sw')) {
      setLanguageState(saved);
    }
  }, []);
  
  const setLanguage = useCallback((lang: Language) => {
    setLanguageState(lang);
    localStorage.setItem('healthai_language', lang);
    document.documentElement.lang = lang;
  }, []);
  
  const t = translations[language];
  
  // Format time with hours and minutes
  const formatTime = useCallback((date: Date | string) => {
    const d = new Date(date);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    
    if (diffMins < 1) return t.justNow;
    if (diffMins < 60) return `${diffMins} ${t.minutesAgo}`;
    if (diffHours < 24) return `${diffHours} ${t.hoursAgo}`;
    if (diffDays === 1) return t.yesterday;
    if (diffDays < 7) return `${diffDays} ${t.daysAgo}`;
    
    const dateStr = d.toLocaleDateString(language === 'sw' ? 'sw-TZ' : 'en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
    const timeStr = d.toLocaleTimeString(language === 'sw' ? 'sw-TZ' : 'en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
    
    return `${dateStr} ${t.at} ${timeStr}`;
  }, [language, t]);
  
  return { t, language, setLanguage, formatTime };
}

// Context provider for app-wide translation
interface I18nContextType {
  t: Translations;
  language: Language;
  setLanguage: (lang: Language) => void;
  formatTime: (date: Date | string) => string;
}

const I18nContext = createContext<I18nContextType | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const i18n = useTranslation();
  
  return (
    <I18nContext.Provider value={i18n}>
      {children}
    </I18nContext.Provider>
  );
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within I18nProvider');
  }
  return context;
}

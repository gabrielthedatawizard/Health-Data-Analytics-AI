import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, 
  ArrowRight, 
  BarChart3, 
  Brain, 
  CheckCircle2, 
  ChevronDown,
  Database,
  FileSpreadsheet,
  Globe,
  Lock,
  Menu,
  Shield,
  Sparkles,
  TrendingUp,
  Upload,
  X,
  Zap
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] as const } }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1, delayChildren: 0.2 }
  }
};

const scaleIn = {
  hidden: { opacity: 0, scale: 0.9 },
  visible: { opacity: 1, scale: 1, transition: { duration: 0.5 } }
};

interface LandingPageProps {
  onEnterDashboard: () => void;
}

export function LandingPage({ onEnterDashboard }: LandingPageProps) {
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
      setMobileMenuOpen(false);
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.5 }}
        className={cn(
          "fixed top-0 left-0 right-0 z-50 transition-all duration-300",
          isScrolled ? "bg-background/90 backdrop-blur-lg border-b border-border" : "bg-transparent"
        )}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 rounded-xl gradient-mint flex items-center justify-center">
                <Activity className="w-5 h-5 text-background" />
              </div>
              <span className="text-xl font-bold text-foreground">HealthAI</span>
            </div>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center gap-8">
              <button onClick={() => scrollToSection('features')} className="text-sm text-muted-foreground hover:text-foreground transition-colors">Features</button>
              <button onClick={() => scrollToSection('how-it-works')} className="text-sm text-muted-foreground hover:text-foreground transition-colors">How It Works</button>
              <button onClick={() => scrollToSection('pricing')} className="text-sm text-muted-foreground hover:text-foreground transition-colors">Pricing</button>
              <button onClick={() => scrollToSection('about')} className="text-sm text-muted-foreground hover:text-foreground transition-colors">About</button>
            </div>

            {/* CTA Buttons */}
            <div className="hidden md:flex items-center gap-3">
              <Button variant="ghost" onClick={onEnterDashboard}>
                Sign In
              </Button>
              <Button onClick={onEnterDashboard} className="gap-2 gradient-mint text-background">
                Get Started
                <ArrowRight className="w-4 h-4" />
              </Button>
            </div>

            {/* Mobile Menu Button */}
            <button 
              className="md:hidden p-2"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden bg-background border-b border-border"
            >
              <div className="px-4 py-4 space-y-3">
                <button onClick={() => scrollToSection('features')} className="block w-full text-left py-2 text-muted-foreground">Features</button>
                <button onClick={() => scrollToSection('how-it-works')} className="block w-full text-left py-2 text-muted-foreground">How It Works</button>
                <button onClick={() => scrollToSection('pricing')} className="block w-full text-left py-2 text-muted-foreground">Pricing</button>
                <button onClick={() => scrollToSection('about')} className="block w-full text-left py-2 text-muted-foreground">About</button>
                <hr className="border-border" />
                <Button onClick={onEnterDashboard} className="w-full gradient-mint text-background">
                  Get Started
                </Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.nav>

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center pt-16 overflow-hidden">
        {/* Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/4 -left-64 w-96 h-96 bg-health-mint/10 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 -right-64 w-96 h-96 bg-health-purple/10 rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Content */}
            <motion.div
              initial="hidden"
              animate="visible"
              variants={staggerContainer}
            >
              <motion.div variants={fadeInUp}>
                <Badge variant="outline" className="mb-6 gap-2 border-health-mint/30 text-health-mint">
                  <Sparkles className="w-3 h-3" />
                  AI-Powered Healthcare Analytics
                </Badge>
              </motion.div>

              <motion.h1 
                variants={fadeInUp}
                className="text-4xl sm:text-5xl lg:text-6xl font-bold text-foreground leading-tight"
              >
                Transform Health Data into{' '}
                <span className="text-gradient-mint">Actionable Insights</span>
              </motion.h1>

              <motion.p 
                variants={fadeInUp}
                className="mt-6 text-lg text-muted-foreground max-w-xl"
              >
                Upload your health data and let our AI generate comprehensive dashboards, 
                detect anomalies, and provide evidence-based recommendations — all with 
                zero hallucinations and full audit trails.
              </motion.p>

              <motion.div variants={fadeInUp} className="mt-8 flex flex-wrap gap-4">
                <Button 
                  size="lg" 
                  onClick={onEnterDashboard}
                  className="gap-2 gradient-mint text-background"
                >
                  Start Free Trial
                  <ArrowRight className="w-4 h-4" />
                </Button>
                <Button 
                  size="lg" 
                  variant="outline"
                  onClick={() => scrollToSection('how-it-works')}
                  className="gap-2"
                >
                  Watch Demo
                  <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center">
                    <div className="w-2 h-2 bg-primary rounded-full" />
                  </div>
                </Button>
              </motion.div>

              <motion.div variants={fadeInUp} className="mt-10 flex items-center gap-6">
                <div className="flex -space-x-2">
                  {[1, 2, 3, 4].map((i) => (
                    <div 
                      key={i} 
                      className="w-10 h-10 rounded-full border-2 border-background bg-muted flex items-center justify-center"
                    >
                      <span className="text-xs font-medium">U{i}</span>
                    </div>
                  ))}
                </div>
                <div>
                  <p className="text-sm font-medium text-foreground">500+ Health Facilities</p>
                  <p className="text-xs text-muted-foreground">Trust HealthAI for their analytics</p>
                </div>
              </motion.div>
            </motion.div>

            {/* Right Content - Hero Image */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              className="relative"
            >
              <div className="relative rounded-3xl overflow-hidden shadow-2xl">
                <img 
                  src="/hero-illustration.png" 
                  alt="HealthAI Dashboard" 
                  className="w-full h-auto"
                />
                {/* Floating Stats Card */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.8, duration: 0.5 }}
                  className="absolute bottom-6 left-6 right-6 glass-card p-4 rounded-xl"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs text-muted-foreground">Data Quality Score</p>
                      <p className="text-2xl font-bold text-emerald-400">94%</p>
                    </div>
                    <div className="h-10 w-px bg-border" />
                    <div>
                      <p className="text-xs text-muted-foreground">Insights Generated</p>
                      <p className="text-2xl font-bold text-health-mint">1,247</p>
                    </div>
                    <div className="h-10 w-px bg-border" />
                    <div>
                      <p className="text-xs text-muted-foreground">Time Saved</p>
                      <p className="text-2xl font-bold text-blue-400">85%</p>
                    </div>
                  </div>
                </motion.div>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Scroll Indicator */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <button 
            onClick={() => scrollToSection('features')}
            className="flex flex-col items-center gap-2 text-muted-foreground hover:text-foreground transition-colors"
          >
            <span className="text-xs">Scroll to explore</span>
            <motion.div
              animate={{ y: [0, 8, 0] }}
              transition={{ duration: 1.5, repeat: Infinity }}
            >
              <ChevronDown className="w-5 h-5" />
            </motion.div>
          </button>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="py-20 border-y border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid grid-cols-2 md:grid-cols-4 gap-8"
          >
            {[
              { value: '500+', label: 'Health Facilities', icon: Database },
              { value: '1.2M+', label: 'Records Analyzed', icon: FileSpreadsheet },
              { value: '99.9%', label: 'Uptime', icon: Zap },
              { value: '85%', label: 'Time Saved', icon: TrendingUp },
            ].map((stat, i) => (
              <motion.div key={i} variants={fadeInUp} className="text-center">
                <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-primary/10 mb-4">
                  <stat.icon className="w-6 h-6 text-primary" />
                </div>
                <p className="text-3xl font-bold text-foreground">{stat.value}</p>
                <p className="text-sm text-muted-foreground mt-1">{stat.label}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="text-center mb-16"
          >
            <motion.div variants={fadeInUp}>
              <Badge variant="outline" className="mb-4 border-health-mint/30 text-health-mint">
                Features
              </Badge>
            </motion.div>
            <motion.h2 variants={fadeInUp} className="text-3xl sm:text-4xl font-bold text-foreground">
              Everything You Need for <span className="text-gradient-mint">Health Analytics</span>
            </motion.h2>
            <motion.p variants={fadeInUp} className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
              From data upload to AI-powered insights, we've got you covered with enterprise-grade features.
            </motion.p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
          >
            {[
              {
                icon: Upload,
                title: 'Easy Data Upload',
                description: 'Upload CSV, Excel, or connect directly to your database. Support for DHIS2 and HMIS formats.',
                image: '/feature-ai-analysis.png'
              },
              {
                icon: Brain,
                title: 'AI-Powered Analysis',
                description: 'Our AI automatically profiles your data, detects anomalies, and generates insights with citations.',
                image: '/feature-dashboard.png'
              },
              {
                icon: BarChart3,
                title: 'Auto-Generated Dashboards',
                description: 'Get beautiful, interactive dashboards in under 2 minutes with recommended visualizations.',
                image: '/feature-dashboard.png'
              },
              {
                icon: Shield,
                title: 'HIPAA Compliant',
                description: 'End-to-end encryption, audit trails, and full compliance with healthcare data regulations.',
                image: '/feature-security.png'
              },
              {
                icon: Globe,
                title: 'Works Offline',
                description: 'Designed for low-resource settings. Works without internet and syncs when connected.',
                image: '/feature-team.png'
              },
              {
                icon: Sparkles,
                title: 'Zero Hallucinations',
                description: 'All numbers are computed deterministically. AI only narrates verified facts with confidence scores.',
                image: '/feature-ai-analysis.png'
              },
            ].map((feature, i) => (
              <motion.div
                key={i}
                variants={fadeInUp}
                className="group glass-card rounded-2xl overflow-hidden card-hover"
              >
                <div className="h-48 overflow-hidden">
                  <img 
                    src={feature.image} 
                    alt={feature.title}
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                  />
                </div>
                <div className="p-6">
                  <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-4">
                    <feature.icon className="w-5 h-5 text-primary" />
                  </div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 bg-muted/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="text-center mb-16"
          >
            <motion.div variants={fadeInUp}>
              <Badge variant="outline" className="mb-4 border-health-mint/30 text-health-mint">
                How It Works
              </Badge>
            </motion.div>
            <motion.h2 variants={fadeInUp} className="text-3xl sm:text-4xl font-bold text-foreground">
              From Data to Insights in <span className="text-gradient-mint">3 Simple Steps</span>
            </motion.h2>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid md:grid-cols-3 gap-8"
          >
            {[
              {
                step: '01',
                title: 'Upload Your Data',
                description: 'Drag and drop your CSV, Excel files, or connect to your database. We support DHIS2, HMIS, and standard formats.',
                icon: Upload
              },
              {
                step: '02',
                title: 'AI Analysis',
                description: 'Our AI engine automatically profiles your data, checks quality, detects patterns, and computes statistics.',
                icon: Brain
              },
              {
                step: '03',
                title: 'Get Insights',
                description: 'Receive auto-generated dashboards, AI insights with citations, and actionable recommendations.',
                icon: Sparkles
              },
            ].map((item, i) => (
              <motion.div key={i} variants={fadeInUp} className="relative">
                <div className="glass-card rounded-2xl p-8 h-full">
                  <div className="text-6xl font-bold text-primary/20 mb-4">{item.step}</div>
                  <div className="w-12 h-12 rounded-xl gradient-mint flex items-center justify-center mb-4">
                    <item.icon className="w-6 h-6 text-background" />
                  </div>
                  <h3 className="text-xl font-semibold text-foreground mb-3">{item.title}</h3>
                  <p className="text-muted-foreground">{item.description}</p>
                </div>
                {i < 2 && (
                  <div className="hidden md:block absolute top-1/2 -right-4 transform -translate-y-1/2 z-10">
                    <ArrowRight className="w-8 h-8 text-primary/30" />
                  </div>
                )}
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="text-center mb-16"
          >
            <motion.div variants={fadeInUp}>
              <Badge variant="outline" className="mb-4 border-health-mint/30 text-health-mint">
                Pricing
              </Badge>
            </motion.div>
            <motion.h2 variants={fadeInUp} className="text-3xl sm:text-4xl font-bold text-foreground">
              Choose Your <span className="text-gradient-mint">Plan</span>
            </motion.h2>
            <motion.p variants={fadeInUp} className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
              Start free and scale as your needs grow. All plans include core analytics features.
            </motion.p>
          </motion.div>

          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid md:grid-cols-3 gap-8 max-w-5xl mx-auto"
          >
            {[
              {
                name: 'Starter',
                price: 'Free',
                period: 'forever',
                description: 'Perfect for individual researchers and small clinics',
                features: [
                  'Up to 10,000 records',
                  'Basic dashboards',
                  'CSV/Excel upload',
                  'Email support',
                  'Community access'
                ],
                cta: 'Get Started',
                popular: false
              },
              {
                name: 'Professional',
                price: '$49',
                period: '/month',
                description: 'For health facilities and district offices',
                features: [
                  'Up to 1M records',
                  'Advanced dashboards',
                  'AI insights & predictions',
                  'Database connectors',
                  'Priority support',
                  'Team collaboration'
                ],
                cta: 'Start Free Trial',
                popular: true
              },
              {
                name: 'Enterprise',
                price: 'Custom',
                period: '',
                description: 'For large health systems and governments',
                features: [
                  'Unlimited records',
                  'Custom integrations',
                  'On-premise deployment',
                  'Dedicated support',
                  'SLA guarantee',
                  'Training & onboarding'
                ],
                cta: 'Contact Sales',
                popular: false
              },
            ].map((plan, i) => (
              <motion.div
                key={i}
                variants={fadeInUp}
                className={cn(
                  "relative rounded-2xl p-8",
                  plan.popular 
                    ? "glass-card border-2 border-health-mint/50" 
                    : "bg-muted/50 border border-border"
                )}
              >
                {plan.popular && (
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2">
                    <Badge className="gradient-mint text-background">Most Popular</Badge>
                  </div>
                )}
                <div className="text-center mb-6">
                  <h3 className="text-xl font-semibold text-foreground">{plan.name}</h3>
                  <div className="mt-4 flex items-baseline justify-center gap-1">
                    <span className="text-4xl font-bold text-foreground">{plan.price}</span>
                    <span className="text-muted-foreground">{plan.period}</span>
                  </div>
                  <p className="mt-2 text-sm text-muted-foreground">{plan.description}</p>
                </div>
                <ul className="space-y-3 mb-8">
                  {plan.features.map((feature, fi) => (
                    <li key={fi} className="flex items-center gap-3">
                      <CheckCircle2 className="w-5 h-5 text-emerald-400 flex-shrink-0" />
                      <span className="text-sm text-muted-foreground">{feature}</span>
                    </li>
                  ))}
                </ul>
                <Button 
                  onClick={onEnterDashboard}
                  className={cn(
                    "w-full",
                    plan.popular ? "gradient-mint text-background" : ""
                  )}
                  variant={plan.popular ? "default" : "outline"}
                >
                  {plan.cta}
                </Button>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* About/Trust Section */}
      <section id="about" className="py-24 bg-muted/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <motion.div
              initial="hidden"
              whileInView="visible"
              viewport={{ once: true }}
              variants={staggerContainer}
            >
              <motion.div variants={fadeInUp}>
                <Badge variant="outline" className="mb-4 border-health-mint/30 text-health-mint">
                  About HealthAI
                </Badge>
              </motion.div>
              <motion.h2 variants={fadeInUp} className="text-3xl sm:text-4xl font-bold text-foreground">
                Built for Healthcare, <span className="text-gradient-mint">Designed for Impact</span>
              </motion.h2>
              <motion.p variants={fadeInUp} className="mt-4 text-muted-foreground">
                HealthAI was born from a simple observation: health facilities in Tanzania and across 
                Africa collect massive amounts of data, but struggle to turn it into actionable insights.
              </motion.p>
              <motion.p variants={fadeInUp} className="mt-4 text-muted-foreground">
                Our mission is to democratize health analytics — making enterprise-grade AI accessible 
                to every health worker, regardless of technical expertise or infrastructure constraints.
              </motion.p>

              <motion.div variants={fadeInUp} className="mt-8 grid grid-cols-2 gap-4">
                {[
                  { label: 'Founded', value: '2023' },
                  { label: 'Team', value: '25+' },
                  { label: 'Countries', value: '8' },
                  { label: 'Funding', value: '$2.5M' },
                ].map((item, i) => (
                  <div key={i} className="glass-card rounded-xl p-4">
                    <p className="text-2xl font-bold text-foreground">{item.value}</p>
                    <p className="text-sm text-muted-foreground">{item.label}</p>
                  </div>
                ))}
              </motion.div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 50 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <img 
                src="/about-hero.png" 
                alt="HealthAI Team" 
                className="rounded-3xl shadow-2xl"
              />
            </motion.div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            variants={scaleIn}
            className="relative overflow-hidden rounded-3xl gradient-mint p-12 text-center"
          >
            <div className="absolute inset-0 opacity-20">
              <div className="absolute top-0 left-0 w-64 h-64 bg-white rounded-full blur-3xl" />
              <div className="absolute bottom-0 right-0 w-64 h-64 bg-white rounded-full blur-3xl" />
            </div>
            <div className="relative">
              <h2 className="text-3xl sm:text-4xl font-bold text-background mb-4">
                Ready to Transform Your Health Data?
              </h2>
              <p className="text-background/80 text-lg mb-8 max-w-xl mx-auto">
                Join 500+ health facilities already using HealthAI to make data-driven decisions.
              </p>
              <div className="flex flex-wrap justify-center gap-4">
                <Button 
                  size="lg" 
                  onClick={onEnterDashboard}
                  className="bg-background text-foreground hover:bg-background/90"
                >
                  Start Free Trial
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
                <Button 
                  size="lg" 
                  variant="outline"
                  className="border-background text-background hover:bg-background/10"
                >
                  Contact Sales
                </Button>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-8">
            <div className="col-span-2 md:col-span-1">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 rounded-lg gradient-mint flex items-center justify-center">
                  <Activity className="w-4 h-4 text-background" />
                </div>
                <span className="text-lg font-bold text-foreground">HealthAI</span>
              </div>
              <p className="text-sm text-muted-foreground">
                AI-powered healthcare analytics for everyone.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-foreground mb-4">Product</h4>
              <ul className="space-y-2">
                <li><button onClick={() => scrollToSection('features')} className="text-sm text-muted-foreground hover:text-foreground">Features</button></li>
                <li><button onClick={() => scrollToSection('pricing')} className="text-sm text-muted-foreground hover:text-foreground">Pricing</button></li>
                <li><button onClick={onEnterDashboard} className="text-sm text-muted-foreground hover:text-foreground">Dashboard</button></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-foreground mb-4">Company</h4>
              <ul className="space-y-2">
                <li><button onClick={() => scrollToSection('about')} className="text-sm text-muted-foreground hover:text-foreground">About</button></li>
                <li><a href="#" className="text-sm text-muted-foreground hover:text-foreground">Blog</a></li>
                <li><a href="#" className="text-sm text-muted-foreground hover:text-foreground">Careers</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-foreground mb-4">Support</h4>
              <ul className="space-y-2">
                <li><a href="#" className="text-sm text-muted-foreground hover:text-foreground">Documentation</a></li>
                <li><a href="#" className="text-sm text-muted-foreground hover:text-foreground">Help Center</a></li>
                <li><a href="#" className="text-sm text-muted-foreground hover:text-foreground">Contact</a></li>
              </ul>
            </div>
          </div>
          <div className="pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-sm text-muted-foreground">
              © 2024 HealthAI. All rights reserved.
            </p>
            <div className="flex items-center gap-4">
              <Lock className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">HIPAA Compliant</span>
              <span className="text-muted-foreground">|</span>
              <Shield className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">SOC 2 Certified</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

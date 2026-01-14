import { useEffect, useState } from "react";
import { Layout } from "@/components/layout/Layout";
import {
  Activity,
  TrendingUp,
  Apple,
  Clock,
  Lock,
  ArrowRight,
  BarChart3,
  Heart,
  Loader2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { cn } from "@/lib/utils";
import { getUser, isLoggedIn } from "@/services/auth";
import { getDashboardSummary } from "@/services/dashboard";

type DashboardSummary = {
  blood_sugar_status: string;
  risk_trend: string;
  diet_status: string;
  health_score: number;
  checks_this_month: number;
  streak_days: number;
  diet_adherence_percent: number;
  recent_activity: { title: string; time: string }[];
};

const colorClasses = {
  primary: "text-primary bg-primary/10 border-primary/20",
  info: "text-[hsl(200,85%,50%)] bg-[hsl(200,85%,50%)]/10 border-[hsl(200,85%,50%)]/20",
  success: "text-success bg-success/10 border-success/20",
};

export default function Dashboard() {
  const user = getUser();
  const isAuthenticated = isLoggedIn() && !!user;

  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const openLogin = () => {
    window.dispatchEvent(new Event("open-login"));
  };

  useEffect(() => {
    if (!isAuthenticated) return;

    const loadDashboard = async () => {
      try {
        setError("");
        setLoading(true);
        const data = await getDashboardSummary();
        setSummary(data);
      } catch (err: any) {
        setError(err?.message || "Failed to load dashboard");
      } finally {
        setLoading(false);
      }
    };

    loadDashboard();
  }, [isAuthenticated]);

  const summaryCards = [
    {
      icon: Activity,
      title: "Blood Sugar Status",
      value: isAuthenticated ? summary?.blood_sugar_status ?? "—" : "—",
      subtitle: isAuthenticated ? "Latest record" : "Login to view",
      color: "primary",
      href: "/blood-sugar",
      locked: !isAuthenticated,
    },
    {
      icon: TrendingUp,
      title: "Risk Trend",
      value: isAuthenticated ? summary?.risk_trend ?? "—" : "—",
      subtitle: isAuthenticated ? "Latest trend" : "Login to view",
      color: "info",
      href: "/risk-forecast",
      locked: !isAuthenticated,
    },
    {
      icon: Apple,
      title: "Diet Status",
      value: isAuthenticated ? summary?.diet_status ?? "—" : "—",
      subtitle: isAuthenticated ? "Based on activity" : "Login to view",
      color: "success",
      href: "/diet-suggestions",
      locked: !isAuthenticated,
    },
  ];

  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="mb-10">
          <h1 className="text-3xl md:text-4xl font-bold mb-2">Dashboard</h1>
          <p className="text-muted-foreground text-lg">
            {isAuthenticated
              ? `Welcome back${user?.name ? `, ${user.name}` : ""}! Here's your health overview.`
              : "Login to access your personalized health dashboard."}
          </p>
        </div>

        {/* Error */}
        {error && (
          <div className="mb-8 p-4 rounded-xl bg-destructive/10 border border-destructive/30 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Summary Cards */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          {summaryCards.map((card, index) => (
            <div
              key={card.title}
              className={cn(
                "glass-card p-6 relative overflow-hidden animate-fade-in",
                card.locked && "opacity-75"
              )}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {card.locked && (
                <div className="absolute inset-0 bg-background/60 backdrop-blur-sm z-10 flex items-center justify-center">
                  <div className="text-center">
                    <Lock className="h-8 w-8 text-muted-foreground mx-auto mb-2" />
                    <p className="text-sm text-muted-foreground mb-3">
                      Login to access
                    </p>
                    <Button variant="glow" size="sm" onClick={openLogin}>
                      Login
                    </Button>
                  </div>
                </div>
              )}

              <div className="flex items-start justify-between mb-4">
                <div
                  className={cn(
                    "w-12 h-12 rounded-xl flex items-center justify-center border",
                    colorClasses[card.color as keyof typeof colorClasses]
                  )}
                >
                  <card.icon className="h-6 w-6" />
                </div>

                <Link to={card.href}>
                  <Button variant="ghost" size="icon" className="h-8 w-8">
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </Link>
              </div>

              <h3 className="text-sm font-medium text-muted-foreground mb-1">
                {card.title}
              </h3>

              <p className="text-2xl font-bold mb-1">
                {loading && isAuthenticated ? (
                  <span className="inline-flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Loading
                  </span>
                ) : (
                  card.value
                )}
              </p>

              <p className="text-xs text-muted-foreground">{card.subtitle}</p>
            </div>
          ))}
        </div>

        {/* Stats & Activity */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Quick Stats */}
          <div className="glass-card p-6">
            <div className="flex items-center gap-2 mb-6">
              <BarChart3 className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Quick Stats</h2>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-xl bg-secondary/50 border border-border/50">
                <div className="flex items-center gap-2 mb-2">
                  <Heart className="h-4 w-4 text-destructive" />
                  <span className="text-xs text-muted-foreground">
                    Health Score
                  </span>
                </div>
                <p className="text-2xl font-bold">
                  {isAuthenticated ? summary?.health_score ?? "0" : "0"}
                </p>
              </div>

              <div className="p-4 rounded-xl bg-secondary/50 border border-border/50">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="h-4 w-4 text-primary" />
                  <span className="text-xs text-muted-foreground">
                    Checks This Month
                  </span>
                </div>
                <p className="text-2xl font-bold">
                  {isAuthenticated ? summary?.checks_this_month ?? "—" : "—"}
                </p>
              </div>

              <div className="p-4 rounded-xl bg-secondary/50 border border-border/50">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="h-4 w-4 text-success" />
                  <span className="text-xs text-muted-foreground">Streak</span>
                </div>
                <p className="text-2xl font-bold">
                  {isAuthenticated
                    ? `${summary?.streak_days ?? 0} days`
                    : "—"}
                </p>
              </div>

              <div className="p-4 rounded-xl bg-secondary/50 border border-border/50">
                <div className="flex items-center gap-2 mb-2">
                  <Apple className="h-4 w-4 text-warning" />
                  <span className="text-xs text-muted-foreground">
                    Diet Adherence
                  </span>
                </div>
                <p className="text-2xl font-bold">
                  {isAuthenticated
                    ? `${summary?.diet_adherence_percent ?? "0"}%`
                    : "—"}
                </p>
              </div>
            </div>
          </div>

          {/* Recent Activity */}
          <div className="glass-card p-6">
            <div className="flex items-center gap-2 mb-6">
              <Clock className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold">Recent Activity</h2>
            </div>

            {isAuthenticated ? (
              <div className="space-y-4">
                {(summary?.recent_activity || []).map((item, index) => (
                  <div
                    key={index}
                    className="flex items-center gap-4 p-3 rounded-lg hover:bg-secondary/30 transition-colors"
                  >
                    <div className="w-2 h-2 rounded-full bg-primary" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">{item.title}</p>
                      <p className="text-xs text-muted-foreground">{item.time}</p>
                    </div>
                  </div>
                ))}

                {!loading && summary?.recent_activity?.length === 0 && (
                  <p className="text-sm text-muted-foreground">
                    No activity yet.
                  </p>
                )}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-40 text-center">
                <Lock className="h-8 w-8 text-muted-foreground mb-3" />
                <p className="text-muted-foreground mb-4">
                  Login to view your activity
                </p>
                <Button variant="glow" size="sm" onClick={openLogin}>
                  Get Started
                </Button>
              </div>
            )}
          </div>
        </div>

        {/* CTA for non-authenticated users */}
        {!isAuthenticated && (
          <div className="mt-12 glass-card p-8 text-center">
            <h3 className="text-2xl font-bold mb-2">Unlock Your Health Insights</h3>
            <p className="text-muted-foreground mb-6 max-w-lg mx-auto">
              Sign in to access personalized blood sugar tracking, risk forecasts,
              and AI-powered diet recommendations.
            </p>
            <Button variant="glow" size="lg" onClick={openLogin}>
              Login with Google
            </Button>
          </div>
        )}
      </div>
    </Layout>
  );
}

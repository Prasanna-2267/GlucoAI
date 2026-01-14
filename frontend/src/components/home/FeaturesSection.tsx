import { Activity, TrendingUp, Apple, MessageCircle } from "lucide-react";
import { Link } from "react-router-dom";
import { cn } from "@/lib/utils";

const features = [
  {
    icon: Activity,
    title: "Blood Sugar Status",
    description: "Upload your medical reports or enter values manually to get instant AI analysis of your blood sugar levels.",
    href: "/blood-sugar",
    color: "primary",
    delay: "0s"
  },
  {
    icon: TrendingUp,
    title: "Diabetes Risk Forecast",
    description: "Answer lifestyle questions and receive AI-powered predictions about your diabetes risk trajectory.",
    href: "/risk-forecast",
    color: "info",
    delay: "0.1s"
  },
  {
    icon: Apple,
    title: "Lifestyle & Diet Tips",
    description: "Get personalized diet recommendations and lifestyle changes based on your health profile and preferences.",
    href: "/diet-suggestions",
    color: "success",
    delay: "0.2s"
  },
  {
    icon: MessageCircle,
    title: "AI Chatbot Assistant",
    description: "Chat with our specialized AI assistant for diabetes-related questions. Upload PDFs for detailed analysis.",
    href: "/chatbot",
    color: "warning",
    delay: "0.3s"
  }
];

const colorClasses = {
  primary: "from-primary/20 to-primary/5 hover:from-primary/30 hover:to-primary/10 border-primary/20 hover:border-primary/40",
  info: "from-info/20 to-info/5 hover:from-info/30 hover:to-info/10 border-info/20 hover:border-info/40",
  success: "from-success/20 to-success/5 hover:from-success/30 hover:to-success/10 border-success/20 hover:border-success/40",
  warning: "from-warning/20 to-warning/5 hover:from-warning/30 hover:to-warning/10 border-warning/20 hover:border-warning/40"
};

const iconColorClasses = {
  primary: "text-primary bg-primary/10",
  info: "text-info bg-info/10",
  success: "text-success bg-success/10",
  warning: "text-warning bg-warning/10"
};

export function FeaturesSection() {
  return (
    <section className="py-24 relative">
      <div className="container mx-auto px-4">
        {/* Section Header */}
        <div className="text-center max-w-2xl mx-auto mb-16">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Comprehensive Diabetes
            <span className="gradient-text"> Support Tools</span>
          </h2>
          <p className="text-muted-foreground text-lg">
            Everything you need to understand, monitor, and manage your diabetes risk in one intelligent platform.
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 gap-6 max-w-5xl mx-auto">
          {features.map((feature) => (
            <Link
              key={feature.title}
              to={feature.href}
              className={cn(
                "group relative p-6 rounded-2xl border bg-gradient-to-br transition-all duration-300",
                "hover:shadow-xl hover:-translate-y-1",
                colorClasses[feature.color as keyof typeof colorClasses],
                "animate-fade-in-up"
              )}
              style={{ animationDelay: feature.delay }}
            >
              {/* Icon */}
              <div className={cn(
                "w-14 h-14 rounded-xl flex items-center justify-center mb-4 transition-transform group-hover:scale-110",
                iconColorClasses[feature.color as keyof typeof iconColorClasses]
              )}>
                <feature.icon className="h-7 w-7" />
              </div>

              {/* Content */}
              <h3 className="text-xl font-semibold mb-2 group-hover:text-primary transition-colors">
                {feature.title}
              </h3>
              <p className="text-muted-foreground">
                {feature.description}
              </p>

              {/* Arrow */}
              <div className="absolute top-6 right-6 opacity-0 -translate-x-2 group-hover:opacity-100 group-hover:translate-x-0 transition-all duration-300">
                <svg className="w-6 h-6 text-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

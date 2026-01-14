import { useEffect, useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  ArrowRight,
  Check,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";

import { predictRisk } from "@/services/risk";
import { getHistoryStatus } from "@/services/history";


const questions = [
  {
    id: "medication",
    question: "How often do you take your medication as prescribed?",
    options: ["Always", "Usually", "Sometimes", "Rarely"],
  },
  {
    id: "sugar_intake",
    question: "How often do you consume sugary foods or drinks?",
    options: ["Rarely", "Sometimes", "Often", "Very Often"],
  },
  {
    id: "meals",
    question: "Do you skip meals regularly?",
    options: ["Never", "Occasionally", "Often", "Very Often"],
  },
  {
    id: "exercise",
    question: "How many times per week do you exercise?",
    options: ["4+ times", "2-3 times", "Once", "None"],
  },
  {
    id: "weight",
    question: "Have you noticed any significant weight changes recently?",
    options: ["No change", "Slight change", "Moderate change", "Significant change"],
  },
  {
    id: "smoking",
    question: "Do you smoke or consume alcohol?",
    options: ["Neither", "Occasionally", "Regularly", "Frequently"],
  },
];

interface ForecastResult {
  trend: "improving" | "stable" | "worsening";
  confidence: number;
  reasoning: string;
  outlook: string[];
}

export default function RiskForecast() {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [result, setResult] = useState<ForecastResult | null>(null);

  const [loading, setLoading] = useState(false);
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [error, setError] = useState("");

  // ✅ new/existing user status
  const [isNewUser, setIsNewUser] = useState(false);

  // ✅ glucose history inputs (ONLY if new user)
  const [g30, setG30] = useState("");
  const [g60, setG60] = useState("");
  const [g90, setG90] = useState("");

  useEffect(() => {
    const loadStatus = async () => {
      try {
        setLoadingStatus(true);
        const data = await getHistoryStatus();
        setIsNewUser(!data?.has_history);
      } catch (err: any) {
        console.error(err);
        // fallback: assume new user if status fails
        setIsNewUser(true);
      } finally {
        setLoadingStatus(false);
      }
    };

    loadStatus();
  }, []);

  const answeredCount = Object.keys(answers).length;
  const canSubmit = answeredCount >= 4;

  const handleAnswer = (questionId: string, value: string) => {
    setAnswers((prev) => ({ ...prev, [questionId]: value }));
  };

  const handleSubmit = async () => {
    if (!canSubmit) return;

    setError("");
    setLoading(true);

    try {
      // ✅ validation for new user
      if (isNewUser) {
        if (!g30 || !g60 || !g90) {
          setError("Please enter your glucose values for 30, 60 and 90 days ago.");
          setLoading(false);
          return;
        }
      }

      const payload: any = {
        lifestyle_answers: answers,
      };

      // ✅ Add glucose history only for NEW user
      if (isNewUser) {
        payload.glucose_30_days_ago = Number(g30);
        payload.glucose_60_days_ago = Number(g60);
        payload.glucose_90_days_ago = Number(g90);
      }

      const data = await predictRisk(payload);

      const apiTrend = String(data?.risk_trend || "").toLowerCase();

      const mappedTrend: ForecastResult["trend"] =
        apiTrend.includes("improv")
          ? "improving"
          : apiTrend.includes("wors")
          ? "worsening"
          : "stable";

      setResult({
        trend: mappedTrend,
        confidence: Math.round(Number(data?.confidence || 0) * 100), // if 0.78 -> 78
        reasoning: data?.reasoning || "No reasoning provided.",
        outlook: data?.future_outlook || [],
      });
    } catch (err: any) {
      console.error(err);
      setError(err?.message || "Risk prediction failed.");
    } finally {
      setLoading(false);
    }
  };

  const trendIcons = {
    improving: TrendingDown,
    stable: Minus,
    worsening: TrendingUp,
  };

  const trendColors = {
    improving: "text-success bg-success/10 border-success/30",
    stable:
      "text-[hsl(200,85%,50%)] bg-[hsl(200,85%,50%)]/10 border-[hsl(200,85%,50%)]/30",
    worsening: "text-destructive bg-destructive/10 border-destructive/30",
  };

  // ✅ Result View
  if (result) {
    const TrendIcon = trendIcons[result.trend];

    return (
      <Layout>
        <div className="container mx-auto px-4 py-12">
          <div className="max-w-2xl mx-auto">
            <div className="glass-card p-8 animate-scale-in">
              <div className={cn("p-6 rounded-xl border mb-8", trendColors[result.trend])}>
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-current/10 flex items-center justify-center">
                    <TrendIcon className="h-8 w-8" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Your Risk Trend</p>
                    <h2 className="text-3xl font-bold capitalize">{result.trend}</h2>
                    <p className="text-sm">Confidence: {result.confidence}%</p>
                  </div>
                </div>
              </div>

              <div className="mb-8">
                <h3 className="font-semibold mb-3">Analysis</h3>
                <p className="text-muted-foreground">{result.reasoning}</p>
              </div>

              {!!result.outlook.length && (
                <div className="mb-8">
                  <h3 className="font-semibold mb-3">Recommendations</h3>
                  <div className="space-y-3">
                    {result.outlook.map((point, index) => (
                      <div key={index} className="flex gap-3 p-3 rounded-lg bg-secondary/30">
                        <Check className="h-5 w-5 text-primary flex-shrink-0 mt-0.5" />
                        <p className="text-sm">{point}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex gap-4">
                <Button
                  variant="outline"
                  className="flex-1"
                  onClick={() => {
                    setResult(null);
                    setAnswers({});
                    setError("");
                  }}
                >
                  Take Again
                </Button>
                <Button variant="glow" className="flex-1" asChild>
                  <a href="/diet-suggestions">Get Diet Tips</a>
                </Button>
              </div>

              <p className="text-xs text-muted-foreground mt-6 text-center">
                ⚠️ This is an AI-assisted insight, not a medical diagnosis.
              </p>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  // ✅ Main Form
  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-3xl mx-auto mb-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-[hsl(200,85%,50%)]/10 mb-6">
            <TrendingUp className="h-8 w-8 text-[hsl(200,85%,50%)]" />
          </div>
          <h1 className="text-3xl md:text-4xl font-bold mb-4">Diabetes Risk Forecast</h1>
          <p className="text-muted-foreground text-lg">
            Answer a few lifestyle questions to get AI-powered predictions about your diabetes risk trajectory.
          </p>
        </div>

        {/* Progress */}
        <div className="max-w-2xl mx-auto mb-8">
          <div className="flex items-center justify-between text-sm mb-2">
            <span className="text-muted-foreground">Progress</span>
            <span className="font-medium">
              {answeredCount} / {questions.length} answered
            </span>
          </div>
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-[hsl(200,85%,50%)] transition-all duration-300"
              style={{ width: `${(answeredCount / questions.length) * 100}%` }}
            />
          </div>
          {!canSubmit && (
            <p className="text-xs text-muted-foreground mt-2">
              Answer at least 4 questions to get your forecast
            </p>
          )}
        </div>

        {/* Status */}
        <div className="max-w-2xl mx-auto mb-6">
          {loadingStatus ? (
            <div className="p-3 rounded-lg bg-secondary/30 text-sm text-muted-foreground flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Checking your detection history...
            </div>
          ) : (
            <div className="p-3 rounded-lg bg-secondary/30 text-sm text-muted-foreground">
              User Type:{" "}
              <span className="font-semibold text-primary">
                {isNewUser ? "New User" : "Existing User"}
              </span>
            </div>
          )}
        </div>

        {/* New user extra inputs */}
        {!loadingStatus && isNewUser && (
          <div className="max-w-2xl mx-auto mb-8">
            <div className="glass-card p-6 space-y-4">
              <h3 className="font-semibold">Glucose History (required for new users)</h3>

              <div className="grid sm:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label>30 days ago</Label>
                  <Input
                    type="number"
                    placeholder="e.g., 120"
                    value={g30}
                    onChange={(e) => setG30(e.target.value)}
                    className="bg-secondary/50 border-border/50"
                  />
                </div>

                <div className="space-y-2">
                  <Label>60 days ago</Label>
                  <Input
                    type="number"
                    placeholder="e.g., 130"
                    value={g60}
                    onChange={(e) => setG60(e.target.value)}
                    className="bg-secondary/50 border-border/50"
                  />
                </div>

                <div className="space-y-2">
                  <Label>90 days ago</Label>
                  <Input
                    type="number"
                    placeholder="e.g., 125"
                    value={g90}
                    onChange={(e) => setG90(e.target.value)}
                    className="bg-secondary/50 border-border/50"
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="max-w-2xl mx-auto mb-6 p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive">
            {error}
          </div>
        )}

        {/* Questions */}
        <div className="max-w-2xl mx-auto space-y-6">
          {questions.map((q, index) => (
            <div
              key={q.id}
              className={cn(
                "glass-card p-6 animate-fade-in",
                answers[q.id] && "border-primary/30"
              )}
              style={{ animationDelay: `${index * 0.05}s` }}
            >
              <Label className="text-base font-medium mb-4 block">
                {index + 1}. {q.question}
              </Label>
              <RadioGroup
                value={answers[q.id]}
                onValueChange={(value) => handleAnswer(q.id, value)}
                className="grid sm:grid-cols-2 gap-3"
              >
                {q.options.map((option) => (
                  <div key={option} className="flex items-center">
                    <RadioGroupItem
                      value={option}
                      id={`${q.id}-${option}`}
                      className="peer sr-only"
                    />
                    <Label
                      htmlFor={`${q.id}-${option}`}
                      className={cn(
                        "w-full p-3 rounded-lg border cursor-pointer transition-all",
                        "hover:bg-secondary/50 hover:border-primary/30",
                        answers[q.id] === option
                          ? "bg-primary/10 border-primary text-primary"
                          : "bg-secondary/30 border-border/50"
                      )}
                    >
                      {option}
                    </Label>
                  </div>
                ))}
              </RadioGroup>
            </div>
          ))}

          {/* Submit */}
          <div className="pt-6">
            <Button
              variant="glow"
              size="lg"
              className="w-full gap-2"
              disabled={!canSubmit || loading || loadingStatus}
              onClick={handleSubmit}
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Predicting...
                </>
              ) : (
                <>
                  Get Your Forecast
                  <ArrowRight className="h-5 w-5" />
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </Layout>
  );
}

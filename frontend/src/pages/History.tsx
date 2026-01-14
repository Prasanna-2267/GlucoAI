import { useEffect, useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { getUserHistory } from "@/services/history2";
import { Loader2, TrendingUp, Activity } from "lucide-react";
import { cn } from "@/lib/utils";

type DetectionItem = {
  glucose: number;
  blood_pressure: number;
  bmi: number;
  age: number;
  prediction: string;
  probability: number;
  created_at: string;
};

type RiskPredictionItem = {
  risk_trend: string;
  created_at: string;
};

export default function History() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [detections, setDetections] = useState<DetectionItem[]>([]);
  const [riskPredictions, setRiskPredictions] = useState<RiskPredictionItem[]>([]);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        setError("");

        const data = await getUserHistory();

        setDetections(data?.detections || []);
        setRiskPredictions(data?.risk_predictions || []);
      } catch (err: any) {
        setError(err?.message || "History fetch failed");
      } finally {
        setLoading(false);
      }
    };

    load();
  }, []);

  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold mb-2">History</h1>
          <p className="text-muted-foreground mb-8">
            View your past diabetes detections & risk forecasts.
          </p>

          {loading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          )}

          {error && (
            <div className="mb-6 p-4 rounded-lg bg-destructive/10 border border-destructive/30 text-destructive">
              {error}
            </div>
          )}

          {!loading && !error && (
            <div className="grid md:grid-cols-2 gap-6">
              {/* ✅ Detection History */}
              <div className="glass-card p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Activity className="h-5 w-5 text-primary" />
                  <h2 className="font-semibold">Blood Sugar Detection</h2>
                </div>

                {detections.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    No detection history yet.
                  </p>
                ) : (
                  <div className="space-y-3 max-h-[450px] overflow-y-auto pr-2">
                    {detections.map((item, index) => {
                      const isNon = item.prediction?.toLowerCase().includes("non");
                      return (
                        <div
                          key={index}
                          className={cn(
                            "p-4 rounded-xl border bg-secondary/20",
                            isNon ? "border-success/30" : "border-destructive/30"
                          )}
                        >
                          <div className="flex justify-between items-center mb-2">
                            <p className="font-semibold">
                              {item.prediction.toUpperCase()}
                            </p>
                            <p className="text-sm text-muted-foreground">
                              {(item.probability * 100).toFixed(1)}%
                            </p>
                          </div>

                          <p className="text-xs text-muted-foreground">
                            {new Date(item.created_at).toLocaleString()}
                          </p>

                          <div className="text-xs text-muted-foreground mt-2 space-y-1">
                            <p>Age: {item.age}</p>
                            <p>Glucose: {item.glucose}</p>
                            <p>BMI: {item.bmi}</p>
                            <p>BP: {item.blood_pressure}</p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* ✅ Risk Prediction History */}
              <div className="glass-card p-6">
                <div className="flex items-center gap-2 mb-4">
                  <TrendingUp className="h-5 w-5 text-[hsl(200,85%,50%)]" />
                  <h2 className="font-semibold">Risk Forecast</h2>
                </div>

                {riskPredictions.length === 0 ? (
                  <p className="text-sm text-muted-foreground">
                    No risk forecasts yet.
                  </p>
                ) : (
                  <div className="space-y-3 max-h-[450px] overflow-y-auto pr-2">
                    {riskPredictions.map((item, index) => (
                      <div
                        key={index}
                        className="p-4 rounded-xl border border-border/50 bg-secondary/20"
                      >
                        <p className="font-semibold capitalize">{item.risk_trend}</p>
                        <p className="text-xs text-muted-foreground mt-1">
                          {new Date(item.created_at).toLocaleString()}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          <p className="text-xs text-muted-foreground mt-8 text-center">
            ⚠️ This is stored securely and shown only for your account.
          </p>
        </div>
      </div>
    </Layout>
  );
}

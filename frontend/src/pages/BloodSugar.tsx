import { useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Upload,
  FileText,
  Activity,
  AlertCircle,
  CheckCircle2,
  Loader2,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

import { detectManual } from "@/services/detect";
import { detectFromReport } from "@/services/detectUpload";

export default function BloodSugar() {
  const [activeTab, setActiveTab] = useState("upload");

  const [result, setResult] = useState<null | { status: string; probability: number }>(null);

  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  // ✅ Manual states
  const [age, setAge] = useState("");
  const [glucose, setGlucose] = useState("");
  const [bloodPressure, setBloodPressure] = useState("");
  const [bmi, setBmi] = useState("");

  // ✅ Upload states
  const [uploadAge, setUploadAge] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);

  // -----------------------------
  // ✅ Manual Submit
  // -----------------------------
  const handleManualSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setResult(null);

    try {
      setLoading(true);

      const payload = {
        age: Number(age),
        glucose: Number(glucose),
        bmi: Number(bmi),
        blood_pressure: bloodPressure ? Number(bloodPressure) : undefined,
      };

      const data = await detectManual(payload);

      setResult({
        status: data.prediction,
        probability: Math.round(data.probability * 1000) / 10,
      });
    } catch (err: any) {
      setError(err?.message || "Manual detection failed");
    } finally {
      setLoading(false);
    }
  };

  // -----------------------------
  // ✅ Upload Submit
  // -----------------------------
  const handleReportSubmit = async () => {
    setError("");
    setResult(null);

    if (!uploadFile) {
      setError("Please upload a report first.");
      return;
    }

    if (!uploadAge || Number(uploadAge) <= 0) {
      setError("Age is required for report analysis.");
      return;
    }

    try {
      setLoading(true);

      // ✅ only send age (BP removed)
        const data = await detectFromReport(uploadFile, { age: Number(uploadAge) });


      setResult({
        status: data.prediction,
        probability: Math.round(data.probability * 1000) / 10,
      });

      // ✅ clear after success
      setUploadFile(null);
      setUploadAge("");
    } catch (err: any) {
      setError(err?.message || "Report analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="max-w-3xl mx-auto mb-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 mb-6">
            <Activity className="h-8 w-8 text-primary" />
          </div>

          <h1 className="text-3xl md:text-4xl font-bold mb-4">
            Blood Sugar Status
          </h1>

          <p className="text-muted-foreground text-lg">
            Upload your medical report or enter values manually to get AI-powered
            blood sugar analysis.
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-2xl mx-auto">
          <div className="glass-card p-6 md:p-8">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-2 mb-8">
                <TabsTrigger value="upload" className="gap-2">
                  <Upload className="h-4 w-4" />
                  Upload Report
                </TabsTrigger>

                <TabsTrigger value="manual" className="gap-2">
                  <FileText className="h-4 w-4" />
                  Manual Entry
                </TabsTrigger>
              </TabsList>

              {/* ✅ Error */}
              {error && (
                <div className="mb-5 p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" />
                  {error}
                </div>
              )}

              {/* ✅ Upload Tab */}
              <TabsContent value="upload" className="space-y-6">
                {/* File Upload Zone */}
                <div className="border-2 border-dashed border-border/50 rounded-xl p-8 text-center hover:border-primary/50 transition-colors">
                  <Upload className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="font-semibold mb-2">Drop your medical report here</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Supports PDF or image files up to 10MB
                  </p>

                  <Input
                    type="file"
                    accept="application/pdf,image/png,image/jpeg,image/jpg"
                    className="hidden"
                    id="reportFile"
                    onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  />

                  <div className="flex items-center justify-center gap-3">
                    <Button
                      variant="outline"
                      onClick={() => document.getElementById("reportFile")?.click()}
                      disabled={loading}
                    >
                      Browse Files
                    </Button>

                    {uploadFile && (
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setUploadFile(null)}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    )}
                  </div>

                  {uploadFile && (
                    <p className="text-xs text-muted-foreground mt-3">
                      Selected:{" "}
                      <span className="font-medium">{uploadFile.name}</span>
                    </p>
                  )}
                </div>

                {/* ✅ Age only (BP removed) */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="age-upload">Age (required)</Label>
                    <Input
                      id="age-upload"
                      type="number"
                      placeholder="Enter your age"
                      value={uploadAge}
                      onChange={(e) => setUploadAge(e.target.value)}
                      className="bg-secondary/50 border-border/50"
                      required
                    />
                  </div>
                </div>

                <Button
                  variant="glow"
                  className="w-full"
                  size="lg"
                  onClick={handleReportSubmit}
                  disabled={loading || !uploadFile || !uploadAge}
                >
                  {loading ? (
                    <span className="flex items-center gap-2">
                      <Loader2 className="h-5 w-5 animate-spin" />
                      Analyzing...
                    </span>
                  ) : (
                    "Analyze Report"
                  )}
                </Button>
              </TabsContent>

              {/* ✅ Manual Tab */}
              <TabsContent value="manual" className="space-y-6">
                <form onSubmit={handleManualSubmit} className="space-y-4">
                  <div className="grid sm:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="age">Age *</Label>
                      <Input
                        id="age"
                        type="number"
                        placeholder="Enter age"
                        value={age}
                        onChange={(e) => setAge(e.target.value)}
                        className="bg-secondary/50 border-border/50"
                        required
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="glucose">Glucose Level (mg/dL) *</Label>
                      <Input
                        id="glucose"
                        type="number"
                        placeholder="e.g., 100"
                        value={glucose}
                        onChange={(e) => setGlucose(e.target.value)}
                        className="bg-secondary/50 border-border/50"
                        required
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="bp">Blood Pressure (numeric)</Label>
                      <Input
                        id="bp"
                        type="number"
                        placeholder="e.g., 80"
                        value={bloodPressure}
                        onChange={(e) => setBloodPressure(e.target.value)}
                        className="bg-secondary/50 border-border/50"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="bmi">BMI *</Label>
                      <Input
                        id="bmi"
                        type="number"
                        step="0.1"
                        placeholder="e.g., 24.5"
                        value={bmi}
                        onChange={(e) => setBmi(e.target.value)}
                        className="bg-secondary/50 border-border/50"
                        required
                      />
                    </div>
                  </div>

                  <Button
                    variant="glow"
                    type="submit"
                    className="w-full"
                    size="lg"
                    disabled={loading}
                  >
                    {loading ? (
                      <span className="flex items-center gap-2">
                        <Loader2 className="h-5 w-5 animate-spin" />
                        Analyzing...
                      </span>
                    ) : (
                      "Analyze Values"
                    )}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>

            {/* ✅ Result */}
            {result && (
              <div className="mt-8 pt-8 border-t border-border/50 animate-fade-in">
                <h3 className="font-semibold mb-4 flex items-center gap-2">
                  <Activity className="h-5 w-5 text-primary" />
                  Analysis Result
                </h3>

                <div
                  className={cn(
                    "p-6 rounded-xl border",
                    result.status.toLowerCase().includes("non")
                      ? "bg-success/10 border-success/30"
                      : "bg-destructive/10 border-destructive/30"
                  )}
                >
                  <div className="flex items-center gap-3 mb-4">
                    {result.status.toLowerCase().includes("non") ? (
                      <CheckCircle2 className="h-8 w-8 text-success" />
                    ) : (
                      <AlertCircle className="h-8 w-8 text-destructive" />
                    )}

                    <div>
                      <p className="text-xl font-bold">{result.status}</p>
                      <p className="text-sm text-muted-foreground">
                        Confidence: {result.probability}%
                      </p>
                    </div>
                  </div>

                  <p className="text-xs text-muted-foreground mt-4 text-center">
                    ⚠️ This is an AI-assisted insight, not a medical diagnosis.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </Layout>
  );
}

import { useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  Apple,
  Utensils,
  Heart,
  AlertTriangle,
  Leaf,
  Check,
  X,
  Loader2,
  Coffee,
  Soup,
  Drumstick,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { dietRecommend } from "@/services/dietRecommend";

interface DietResult {
  diet_type: string;
  foods_to_prefer_breakfast: string[];
  foods_to_prefer_lunch: string[];
  foods_to_prefer_dinner: string[];
  foods_to_limit: string[];
  tips: string[];
  note?: string;
}

const mapEatingOut = (val: string) => {
  if (val === "rarely") return "rarely";
  if (val === "sometimes") return "weekly";
  if (val === "often") return "frequently";
  if (val === "daily") return "frequently";
  return val;
};

export default function DietSuggestions() {
  const [result, setResult] = useState<DietResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Form states
  const [dietPreference, setDietPreference] = useState("");
  const [mealsPerDay, setMealsPerDay] = useState("");
  const [eatingOutFrequency, setEatingOutFrequency] = useState("");
  const [culturalPreference, setCulturalPreference] = useState("");

  // split preferred foods (backend expects separate meals)
  const [preferredBreakfast, setPreferredBreakfast] = useState("");
  const [preferredLunch, setPreferredLunch] = useState("");
  const [preferredDinner, setPreferredDinner] = useState("");

  const [allergies, setAllergies] = useState("");

    const normalizeMeals = (val: string) => {
    if (val === "5+ meals/snacks") return "5+";
    return val;
  };


  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setResult(null);

    // ✅ basic validation
    if (!dietPreference || !mealsPerDay || !eatingOutFrequency || !culturalPreference) {
      setError("Please fill all required fields.");
      return;
    }
    if (!preferredBreakfast || !preferredLunch || !preferredDinner) {
      setError("Please provide preferred foods for breakfast, lunch and dinner.");
      return;
    }

    try {
      setLoading(true);

    const payload = {
      diet_preference: dietPreference,
      meals_per_day: parseInt(mealsPerDay, 10),
      eats_outside: mapEatingOut(eatingOutFrequency),
      cultural_preference: culturalPreference,
      preferred_food_breakfast: preferredBreakfast,
      preferred_food_lunch: preferredLunch,
      preferred_food_dinner: preferredDinner,
      allergies: allergies || "None",
    };

      const data = await dietRecommend(payload);
      setResult(data);
    } catch (err: any) {
      setError(err?.message || "Diet recommendation failed");
    } finally {
      setLoading(false);
    }
  };

  // ✅ RESULT UI
  if (result) {
    return (
      <Layout>
        <div className="container mx-auto px-4 py-12">
          <div className="max-w-3xl mx-auto">
            <div className="glass-card p-8 animate-scale-in">
              {/* Header */}
              <div className="text-center mb-8">
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-success/10 mb-4">
                  <Leaf className="h-8 w-8 text-success" />
                </div>

                <h2 className="text-2xl font-bold mb-2">
                  Your Personalized Diet Plan
                </h2>

                <p className="text-primary font-medium text-lg">
                  {result.diet_type}
                </p>
              </div>

              {/* Meal Cards */}
              <div className="grid md:grid-cols-3 gap-6 mb-8">
                {/* Breakfast */}
                <div className="p-6 rounded-xl bg-success/5 border border-success/20">
                  <div className="flex items-center gap-2 mb-4">
                    <Coffee className="h-5 w-5 text-success" />
                    <h3 className="font-semibold text-success">Breakfast</h3>
                  </div>
                  <ul className="space-y-2">
                    {result.foods_to_prefer_breakfast?.map((food, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm">
                        <div className="w-1.5 h-1.5 rounded-full bg-success mt-2 flex-shrink-0" />
                        {food}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Lunch */}
                <div className="p-6 rounded-xl bg-primary/5 border border-primary/20">
                  <div className="flex items-center gap-2 mb-4">
                    <Soup className="h-5 w-5 text-primary" />
                    <h3 className="font-semibold text-primary">Lunch</h3>
                  </div>
                  <ul className="space-y-2">
                    {result.foods_to_prefer_lunch?.map((food, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm">
                        <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 flex-shrink-0" />
                        {food}
                      </li>
                    ))}
                  </ul>
                </div>

                {/* Dinner */}
                <div className="p-6 rounded-xl bg-info/5 border border-info/20">
                  <div className="flex items-center gap-2 mb-4">
                    <Drumstick className="h-5 w-5 text-info" />
                    <h3 className="font-semibold text-info">Dinner</h3>
                  </div>
                  <ul className="space-y-2">
                    {result.foods_to_prefer_dinner?.map((food, idx) => (
                      <li key={idx} className="flex items-start gap-2 text-sm">
                        <div className="w-1.5 h-1.5 rounded-full bg-info mt-2 flex-shrink-0" />
                        {food}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>

              {/* Foods to Limit */}
              <div className="p-6 rounded-xl bg-warning/5 border border-warning/20 mb-8">
                <div className="flex items-center gap-2 mb-4">
                  <X className="h-5 w-5 text-warning" />
                  <h3 className="font-semibold text-warning">Foods to Limit</h3>
                </div>

                <ul className="grid sm:grid-cols-2 gap-2">
                  {result.foods_to_limit?.map((food, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-sm">
                      <div className="w-1.5 h-1.5 rounded-full bg-warning mt-2 flex-shrink-0" />
                      {food}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Tips */}
              <div className="p-6 rounded-xl bg-primary/5 border border-primary/20 mb-6">
                <div className="flex items-center gap-2 mb-4">
                  <Heart className="h-5 w-5 text-primary" />
                  <h3 className="font-semibold">Lifestyle Tips</h3>
                </div>

                <div className="grid gap-3">
                  {result.tips?.map((tip, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-3 p-3 rounded-lg bg-secondary/30"
                    >
                      <span className="w-6 h-6 rounded-full bg-primary/20 text-primary text-xs font-bold flex items-center justify-center flex-shrink-0">
                        {idx + 1}
                      </span>
                      <p className="text-sm">{tip}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Note */}
              <p className="text-xs text-muted-foreground text-center mb-6">
                ⚠️ {result.note || "This is an AI-assisted insight, not medical advice."}
              </p>

              <div className="flex gap-4">
                <Button variant="outline" className="flex-1" onClick={() => setResult(null)}>
                  Adjust Preferences
                </Button>
                <Button variant="glow" className="flex-1">
                  Save to Profile
                </Button>
              </div>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  // ✅ FORM UI
  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        {/* Header */}
        <div className="max-w-3xl mx-auto mb-12 text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-success/10 mb-6">
            <Apple className="h-8 w-8 text-success" />
          </div>
          <h1 className="text-3xl md:text-4xl font-bold mb-4">
            Lifestyle & Diet Suggestions
          </h1>
          <p className="text-muted-foreground text-lg">
            Get personalized diet recommendations based on your preferences and
            health profile.
          </p>
        </div>

        {/* Requirement Notice */}
        <div className="max-w-2xl mx-auto mb-8">
          <div className="p-4 rounded-xl bg-warning/10 border border-warning/30 flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-warning flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-warning">Complete Risk Forecast First</p>
              <p className="text-sm text-muted-foreground">
                For best results, complete the Diabetes Risk Forecast before
                getting diet recommendations.
              </p>
            </div>
          </div>
        </div>

        {/* Form */}
        <div className="max-w-2xl mx-auto">
          <form onSubmit={handleSubmit} className="glass-card p-6 md:p-8 space-y-6">
            {error && (
              <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/30 text-sm text-destructive flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                {error}
              </div>
            )}

            <div className="grid sm:grid-cols-2 gap-6">
              <div className="space-y-2">
                <Label>Diet Preference *</Label>
                <Select value={dietPreference} onValueChange={setDietPreference}>
                  <SelectTrigger className="bg-secondary/50 border-border/50">
                    <SelectValue placeholder="Select preference" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="omnivore">Omnivore</SelectItem>
                    <SelectItem value="vegetarian">Vegetarian</SelectItem>
                    <SelectItem value="vegan">Vegan</SelectItem>
                    <SelectItem value="pescatarian">Pescatarian</SelectItem>
                    <SelectItem value="keto">Keto</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Meals Per Day *</Label>
                <Select value={mealsPerDay} onValueChange={setMealsPerDay}>
                  <SelectTrigger className="bg-secondary/50 border-border/50">
                    <SelectValue placeholder="Select meals" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="2">2 meals</SelectItem>
                    <SelectItem value="3">3 meals</SelectItem>
                    <SelectItem value="4">4 meals</SelectItem>
                    <SelectItem value="5+">5 meals/snacks</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Eating Out Frequency *</Label>
                <Select
                  value={eatingOutFrequency}
                  onValueChange={setEatingOutFrequency}
                >
                  <SelectTrigger className="bg-secondary/50 border-border/50">
                    <SelectValue placeholder="Select frequency" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="rarely">Rarely (once a month)</SelectItem>
                    <SelectItem value="sometimes">Sometimes (weekly)</SelectItem>
                    <SelectItem value="often">Often (2-3x per week)</SelectItem>
                    <SelectItem value="daily">Daily</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Cultural Preference *</Label>
                <Select
                  value={culturalPreference}
                  onValueChange={setCulturalPreference}
                >
                  <SelectTrigger className="bg-secondary/50 border-border/50">
                    <SelectValue placeholder="Select cuisine" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="western">Western</SelectItem>
                    <SelectItem value="asian">Asian</SelectItem>
                    <SelectItem value="mediterranean">Mediterranean</SelectItem>
                    <SelectItem value="indian">Indian</SelectItem>
                    <SelectItem value="latin">Latin American</SelectItem>
                    <SelectItem value="mixed">Mixed/No preference</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Preferred foods split by meal */}
            <div className="space-y-2">
              <Label>Preferred Foods - Breakfast *</Label>
              <Textarea
                placeholder="e.g., idli, dosa, oats, fruits..."
                value={preferredBreakfast}
                onChange={(e) => setPreferredBreakfast(e.target.value)}
                className="bg-secondary/50 border-border/50 min-h-[80px]"
                required
              />
            </div>

            <div className="space-y-2">
              <Label>Preferred Foods - Lunch *</Label>
              <Textarea
                placeholder="e.g., rice, chapati, dal, chicken..."
                value={preferredLunch}
                onChange={(e) => setPreferredLunch(e.target.value)}
                className="bg-secondary/50 border-border/50 min-h-[80px]"
                required
              />
            </div>

            <div className="space-y-2">
              <Label>Preferred Foods - Dinner *</Label>
              <Textarea
                placeholder="e.g., soup, millet dosa, paneer..."
                value={preferredDinner}
                onChange={(e) => setPreferredDinner(e.target.value)}
                className="bg-secondary/50 border-border/50 min-h-[80px]"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="allergies">Allergies or Restrictions (optional)</Label>
              <Input
                id="allergies"
                placeholder="e.g., nuts, shellfish, gluten"
                value={allergies}
                onChange={(e) => setAllergies(e.target.value)}
                className="bg-secondary/50 border-border/50"
              />
            </div>

            <Button
              variant="glow"
              type="submit"
              size="lg"
              className="w-full gap-2"
              disabled={loading}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Generating Diet Plan...
                </span>
              ) : (
                <>
                  <Utensils className="h-5 w-5" />
                  Get Diet Recommendations
                </>
              )}
            </Button>
          </form>
        </div>
      </div>
    </Layout>
  );
}

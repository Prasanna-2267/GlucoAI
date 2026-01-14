import { useEffect, useMemo, useState } from "react";
import { Layout } from "@/components/layout/Layout";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  User,
  Mail,
  Calendar,
  LogOut,
  Shield,
  Activity,
  Loader2,
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { cn } from "@/lib/utils";

import { getMyProfile } from "@/services/profile";
import { logoutUser } from "@/services/session";

type ProfileResponse = {
  user_id: string;
  email: string;
  name: string;
  avatar_url?: string | null;
  provider?: string | null;
  created_at?: string; // ISO timestamp
};

export default function Profile() {
  const navigate = useNavigate();

  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    const loadProfile = async () => {
      try {
        setLoading(true);
        setError("");
        const data = await getMyProfile();
        setProfile(data);
      } catch (err: any) {
        setError(err?.message || "Failed to load profile");
      } finally {
        setLoading(false);
      }
    };

    loadProfile();
  }, []);

  const initials = useMemo(() => {
    if (!profile?.name) return "U";
    return profile.name
      .split(" ")
      .filter(Boolean)
      .slice(0, 2)
      .map((n) => n[0]?.toUpperCase())
      .join("");
  }, [profile?.name]);

  const createdAtText = useMemo(() => {
    if (!profile?.created_at) return "—";
    const dt = new Date(profile.created_at);
    return dt.toLocaleDateString(undefined, {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  }, [profile?.created_at]);

  const handleLogout = () => {
    logoutUser();
    navigate("/", { replace: true });
  };

  return (
    <Layout>
      <div className="container mx-auto px-4 py-12">
        <div className="max-w-2xl mx-auto">
          {/* Loading */}
          {loading && (
            <div className="glass-card p-8 flex items-center justify-center gap-3">
              <Loader2 className="h-5 w-5 animate-spin" />
              <p className="text-sm text-muted-foreground">Loading profile...</p>
            </div>
          )}

          {/* Error */}
          {!loading && error && (
            <div className="glass-card p-6 border border-destructive/30 bg-destructive/10">
              <p className="text-sm text-destructive">{error}</p>

              <Button
                className="mt-4"
                variant="outline"
                onClick={() => window.location.reload()}
              >
                Retry
              </Button>
            </div>
          )}

          {/* Profile */}
          {!loading && !error && profile && (
            <>
              {/* Profile Card */}
              <div className="glass-card p-8 text-center mb-8 animate-fade-in">
                <Avatar className="w-24 h-24 mx-auto mb-4 border-4 border-primary/20">
                  {/* ✅ show google profile photo */}
                  <AvatarImage
                    src={profile.avatar_url || ""}
                    alt={profile.name}
                    referrerPolicy="no-referrer"
                  />

                  {/* ✅ fallback initials */}
                  <AvatarFallback className="bg-gradient-to-br from-primary to-[hsl(200,85%,50%)] text-primary-foreground text-2xl">
                    {initials}
                  </AvatarFallback>
                </Avatar>

                <h1 className="text-2xl font-bold mb-1">{profile.name}</h1>
                <p className="text-muted-foreground mb-4">{profile.email}</p>

                <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground">
                  <Calendar className="h-4 w-4" />
                  <span>Member since {createdAtText}</span>
                </div>
              </div>

              {/* Account Info */}
              <div
                className="glass-card p-6 mb-6 animate-fade-in"
                style={{ animationDelay: "0.1s" }}
              >
                <h2 className="font-semibold mb-4 flex items-center gap-2">
                  <User className="h-5 w-5 text-primary" />
                  Account Information
                </h2>

                <div className="space-y-4">
                  <div className="flex items-center justify-between py-3 border-b border-border/50">
                    <div className="flex items-center gap-3">
                      <Mail className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm">Email</span>
                    </div>
                    <span className="text-sm text-muted-foreground">
                      {profile.email}
                    </span>
                  </div>

                  <div className="flex items-center justify-between py-3 border-b border-border/50">
                    <div className="flex items-center gap-3">
                      <Shield className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm">Login Provider</span>
                    </div>
                    <span className="text-sm text-muted-foreground capitalize">
                      {profile.provider || "google"}
                    </span>
                  </div>

                  <div className="flex items-center justify-between py-3">
                    <div className="flex items-center gap-3">
                      <Calendar className="h-4 w-4 text-muted-foreground" />
                      <span className="text-sm">Account Created</span>
                    </div>
                    <span className="text-sm text-muted-foreground">
                      {createdAtText}
                    </span>
                  </div>
                </div>
              </div>

              {/* Quick Links */}
              <div
                className="glass-card p-6 mb-6 animate-fade-in"
                style={{ animationDelay: "0.2s" }}
              >
                <h2 className="font-semibold mb-4 flex items-center gap-2">
                  <Activity className="h-5 w-5 text-primary" />
                  Quick Links
                </h2>

                <div className="grid sm:grid-cols-2 gap-3">
                  <Link
                    to="/history"
                    className={cn(
                      "p-4 rounded-xl bg-secondary/30 hover:bg-secondary/50 transition-colors flex items-center gap-3"
                    )}
                  >
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
                      <Activity className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-sm">View History</p>
                      <p className="text-xs text-muted-foreground">
                        Past detections & forecasts
                      </p>
                    </div>
                  </Link>

                  <Link
                    to="/blood-sugar"
                    className={cn(
                      "p-4 rounded-xl bg-secondary/30 hover:bg-secondary/50 transition-colors flex items-center gap-3"
                    )}
                  >
                    <div className="w-10 h-10 rounded-lg bg-success/10 flex items-center justify-center">
                      <Activity className="h-5 w-5 text-success" />
                    </div>
                    <div>
                      <p className="font-medium text-sm">Blood Sugar Check</p>
                      <p className="text-xs text-muted-foreground">New analysis</p>
                    </div>
                  </Link>
                </div>
              </div>

              {/* Logout */}
              <Button
                variant="destructive"
                className="w-full gap-2"
                size="lg"
                onClick={handleLogout}
              >
                <LogOut className="h-5 w-5" />
                Logout
              </Button>

              <p className="text-xs text-muted-foreground mt-6 text-center">
                ⚠️ Your profile details are fetched securely from database.
              </p>
            </>
          )}
        </div>
      </div>
    </Layout>
  );
}

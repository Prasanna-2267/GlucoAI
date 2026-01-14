import { useEffect, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Activity } from "lucide-react";
import { loginWithGoogle } from "@/services/auth";

interface LoginModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

declare global {
  interface Window {
    google?: any;
  }
}

export function LoginModal({ isOpen, onClose, onSuccess }: LoginModalProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");
  const googleBtnRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!isOpen) return;

    const scriptId = "google-identity-script";

    const loadScript = () => {
      return new Promise<void>((resolve, reject) => {
        // Already loaded
        if (document.getElementById(scriptId)) return resolve();

        const script = document.createElement("script");
        script.id = scriptId;
        script.src = "https://accounts.google.com/gsi/client";
        script.async = true;
        script.defer = true;

        script.onload = () => resolve();
        script.onerror = () => reject(new Error("Failed to load Google script"));
        document.body.appendChild(script);
      });
    };

    const initGoogle = async () => {
      try {
        setError("");

        await loadScript();

        if (!window.google || !googleBtnRef.current) {
          throw new Error("Google login not available");
        }

        // ✅ Initialize Google Login
        window.google.accounts.id.initialize({
          client_id: import.meta.env.VITE_GOOGLE_CLIENT_ID,
          callback: async (response: any) => {
            try {
              setError("");
              setLoading(true);

              const idToken = response?.credential;
              if (!idToken) throw new Error("Google token missing");

              // ✅ Exchange Google token -> backend JWT token
              await loginWithGoogle(idToken);

              // ✅ update state in parent
              onSuccess();
              onClose();

              // ✅ important: refresh so navbar reads localStorage + shows user
              window.location.reload();
            } catch (err: any) {
              console.error(err);
              setError(err?.message || "Login failed. Please try again.");
            } finally {
              setLoading(false);
            }
          },
        });

        // ✅ Clear old button (important for reopening modal)
        googleBtnRef.current.innerHTML = "";

        // ✅ Render official Google button
        window.google.accounts.id.renderButton(googleBtnRef.current, {
          theme: "outline",
          size: "large",
          width: 340,
          text: "continue_with",
        });
      } catch (err: any) {
        console.error(err);
        setError(err?.message || "Google login setup failed");
      }
    };

    initGoogle();
  }, [isOpen, onClose, onSuccess]);

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="glass-card border-border/50 sm:max-w-md">
        <DialogHeader className="text-center">
          <div className="mx-auto mb-4 w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/20 to-info/20 flex items-center justify-center">
            <Activity className="h-8 w-8 text-primary" />
          </div>

          <DialogTitle className="text-2xl font-bold">
            Welcome to GlucoAI
          </DialogTitle>

          <DialogDescription className="text-muted-foreground">
            Sign in to access Google-based secure login and unlock personalized
            diabetes features.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 pt-4">
          {/* ✅ Google official button will be rendered here */}
          <div className="flex justify-center">
            <div ref={googleBtnRef} />
          </div>

          {loading && (
            <p className="text-sm text-muted-foreground text-center">
              Signing in...
            </p>
          )}

          {error && <p className="text-sm text-red-500 text-center">{error}</p>}

          <p className="text-xs text-center text-muted-foreground">
            By signing in, you agree to our Terms of Service and Privacy Policy.
          </p>
        </div>

        <div className="mt-4 p-4 rounded-lg bg-primary/5 border border-primary/20">
          <p className="text-xs text-muted-foreground text-center">
            🔒 Your data is encrypted and secure. GlucoAI provides AI-assisted
            insights only and does not replace professional medical advice.
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
import { ReactNode, useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Lock, Activity } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { isLoggedIn, getUser } from "@/services/auth";
import { LoginModal } from "@/components/auth/LoginModal";

interface ProtectedRouteProps {
  children: ReactNode;
}

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const [showBlockedModal, setShowBlockedModal] = useState(false);
  const [showLoginModal, setShowLoginModal] = useState(false);

  const navigate = useNavigate();

  const authenticated = isLoggedIn();
  const user = getUser();
  const isAuthenticated = authenticated && !!user;

  useEffect(() => {
    if (!isAuthenticated) {
      setShowBlockedModal(true);
    } else {
      setShowBlockedModal(false);
      setShowLoginModal(false);
    }
  }, [isAuthenticated]);

  // ✅ if authenticated allow access
  if (isAuthenticated) return <>{children}</>;

  return (
    <>
      {/* 🔒 Block screen modal */}
      <Dialog
        open={showBlockedModal}
        onOpenChange={(open) => {
          setShowBlockedModal(open);

          // if modal closed, go home
          if (!open) navigate("/");
        }}
      >
        <DialogContent className="glass-card border-border/50 sm:max-w-md">
          <DialogHeader className="text-center">
            <div className="mx-auto mb-4 w-16 h-16 rounded-2xl bg-destructive/10 flex items-center justify-center">
              <Lock className="h-8 w-8 text-destructive" />
            </div>

            <DialogTitle className="text-2xl font-bold">
              Login Required
            </DialogTitle>

            <DialogDescription className="text-muted-foreground">
              Please login to access this feature.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 pt-4">
            {/* ✅ Open Login Modal directly */}
            <Button
              variant="glow"
              className="w-full h-12 gap-3 text-base"
              onClick={() => {
                setShowLoginModal(true);
              }}
            >
              <Activity className="h-5 w-5" />
              Go to Login
            </Button>

            <Button
              variant="ghost"
              className="w-full"
              onClick={() => navigate("/")}
            >
              Return to Home
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* ✅ Actual Login Modal */}
      <LoginModal
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onSuccess={() => {
          setShowLoginModal(false);
          setShowBlockedModal(false);
        }}
      />
    </>
  );
}

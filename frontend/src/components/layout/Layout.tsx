import { ReactNode, useEffect, useState } from "react";
import { Navbar } from "./Navbar";
import { FloatingChatButton } from "./FloatingChatButton";
import { LoginModal } from "@/components/auth/LoginModal";
import { getUser, isLoggedIn, logout } from "@/services/auth";

interface LayoutProps {
  children: ReactNode;
}

const defaultUser = {
  name: "Guest",
  email: "Not logged in",
  avatar: "",
  createdAt: "",
};

export function Layout({ children }: LayoutProps) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(defaultUser);
  const [showLoginModal, setShowLoginModal] = useState(false);

  // ✅ Load auth state from storage + listen for updates
  useEffect(() => {
    const syncAuthState = () => {
      const logged = isLoggedIn();
      const storedUser = getUser();

      setIsAuthenticated(logged);

      if (logged && storedUser) {
        setUser({ ...defaultUser, ...storedUser });
      } else {
        setUser(defaultUser);
      }
    };

    syncAuthState(); // first load

    // listen for changes triggered from session.ts
    window.addEventListener("user-updated", syncAuthState);

    return () => window.removeEventListener("user-updated", syncAuthState);
  }, []);

  // ✅ Listen for dashboard button "open-login"
  useEffect(() => {
    const openLogin = () => setShowLoginModal(true);

    window.addEventListener("open-login", openLogin);
    return () => window.removeEventListener("open-login", openLogin);
  }, []);
  const handleLogin = () => {
    setShowLoginModal(true);
  };

  const handleLoginSuccess = () => {
    // After login, state will auto update by "user-updated"
    setShowLoginModal(false);
  };

  const handleLogout = () => {
    logout();
    setIsAuthenticated(false);
    setUser(defaultUser);
    window.location.href = "/";
  };

  return (
    <div className="min-h-screen bg-background">
      <Navbar
        isAuthenticated={isAuthenticated}
        user={user}
        onLogin={handleLogin}
        onLogout={handleLogout}
      />

      <main className="pt-16">{children}</main>

      <FloatingChatButton />

      <LoginModal
        isOpen={showLoginModal}
        onClose={() => setShowLoginModal(false)}
        onSuccess={handleLoginSuccess}
      />
    </div>
  );
}

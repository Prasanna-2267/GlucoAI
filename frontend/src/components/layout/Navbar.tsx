import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Activity,
  Menu,
  X,
  User,
  History,
  LogOut,
  ChevronDown,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface NavbarProps {
  isAuthenticated?: boolean;
  user?: {
    name: string;
    email: string;
    avatar?: string;
    createdAt?: string;
  };
  onLogin?: () => void;
  onLogout?: () => void;
}

const navLinks = [
  { href: "/", label: "Home", protected: false },
  { href: "/dashboard", label: "Dashboard", protected: true },
  { href: "/blood-sugar", label: "Blood Sugar", protected: true },
  { href: "/risk-forecast", label: "Risk Forecast", protected: true },
  { href: "/diet-suggestions", label: "Diet Tips", protected: true },
];

export function Navbar({
  isAuthenticated = false,
  user,
  onLogin,
  onLogout,
}: NavbarProps) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const location = useLocation();

  const visibleLinks = navLinks.filter(
    (link) => !link.protected || isAuthenticated
  );

  // ✅ safe user name
  const displayName =
    user?.name?.trim() || (isAuthenticated ? "User" : "Guest");
  const emailText = user?.email?.trim() || "";

  // ✅ safe avatar fallback
  const avatarLetter = (displayName?.[0] || "U").toUpperCase();

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 group">
            <div className="relative">
              <Activity className="h-8 w-8 text-primary transition-transform group-hover:scale-110" />
              <div className="absolute inset-0 blur-lg bg-primary/30 opacity-0 group-hover:opacity-100 transition-opacity" />
            </div>
            <span className="text-xl font-bold gradient-text">GlucoAI</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {visibleLinks.map((link) => (
              <Link
                key={link.href}
                to={link.href}
                className={cn(
                  "px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                  location.pathname === link.href
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                )}
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* Auth Section */}
          <div className="flex items-center gap-4">
            {isAuthenticated && user ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" className="gap-2 pl-2 pr-3">
                    <Avatar className="h-8 w-8 border border-border">
                      <AvatarImage src={user.avatar} alt={displayName} />
                      <AvatarFallback className="bg-primary/20 text-primary">
                        {avatarLetter}
                      </AvatarFallback>
                    </Avatar>
                    <span className="hidden sm:inline text-sm font-medium">
                      {displayName}
                    </span>
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  </Button>
                </DropdownMenuTrigger>

                <DropdownMenuContent align="end" className="w-64 glass-card p-2">
                  <div className="px-3 py-2">
                    <p className="text-sm font-medium">{displayName}</p>
                    {!!emailText && (
                      <p className="text-xs text-muted-foreground">{emailText}</p>
                    )}
                    {user.createdAt && (
                      <p className="text-xs text-muted-foreground mt-1">
                        Joined {user.createdAt}
                      </p>
                    )}
                  </div>

                  <DropdownMenuSeparator className="bg-border/50" />

                  <DropdownMenuItem asChild className="cursor-pointer">
                    <Link to="/profile" className="flex items-center gap-2">
                      <User className="h-4 w-4" />
                      Profile
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuItem asChild className="cursor-pointer">
                    <Link to="/history" className="flex items-center gap-2">
                      <History className="h-4 w-4" />
                      History
                    </Link>
                  </DropdownMenuItem>

                  <DropdownMenuSeparator className="bg-border/50" />

                  <DropdownMenuItem
                    onClick={onLogout}
                    className="cursor-pointer text-destructive focus:text-destructive"
                  >
                    <LogOut className="h-4 w-4 mr-2" />
                    Logout
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : (
              <Button variant="glow" onClick={onLogin} className="hidden sm:flex">
                Login with Google
              </Button>
            )}

            {/* Mobile Menu Button */}
            <Button
              variant="ghost"
              size="icon"
              className="md:hidden"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? (
                <X className="h-5 w-5" />
              ) : (
                <Menu className="h-5 w-5" />
              )}
            </Button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden py-4 border-t border-border/50 animate-fade-in">
            <div className="flex flex-col gap-1">
              {visibleLinks.map((link) => (
                <Link
                  key={link.href}
                  to={link.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={cn(
                    "px-4 py-3 rounded-lg text-sm font-medium transition-all duration-200",
                    location.pathname === link.href
                      ? "text-primary bg-primary/10"
                      : "text-muted-foreground hover:text-foreground hover:bg-secondary/50"
                  )}
                >
                  {link.label}
                </Link>
              ))}

              {!isAuthenticated && (
                <Button
                  variant="glow"
                  onClick={() => {
                    setMobileMenuOpen(false);
                    onLogin?.();
                  }}
                  className="mt-2"
                >
                  Login with Google
                </Button>
              )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
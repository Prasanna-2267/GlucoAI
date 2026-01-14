import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import Index from "./pages/Index";
import Dashboard from "./pages/Dashboard";
import BloodSugar from "./pages/BloodSugar";
import RiskForecast from "./pages/RiskForecast";
import DietSuggestions from "./pages/DietSuggestions";
import Chatbot from "./pages/Chatbot";
import Profile from "./pages/Profile";
import History from "./pages/History";
import NotFound from "./pages/NotFound";

import { ProtectedRoute } from "@/components/auth/ProtectedRoute";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          {/* ✅ Public Routes */}
          <Route path="/" element={<Index />} />
          <Route path="/chatbot" element={<Chatbot />} />

          {/* 🔒 Protected Routes */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />

          <Route
            path="/blood-sugar"
            element={
              <ProtectedRoute>
                <BloodSugar />
              </ProtectedRoute>
            }
          />

          <Route
            path="/risk-forecast"
            element={
              <ProtectedRoute>
                <RiskForecast />
              </ProtectedRoute>
            }
          />

          <Route
            path="/diet-suggestions"
            element={
              <ProtectedRoute>
                <DietSuggestions />
              </ProtectedRoute>
            }
          />

          <Route
            path="/profile"
            element={
              <ProtectedRoute>
                <Profile />
              </ProtectedRoute>
            }
          />

          <Route
            path="/history"
            element={
              <ProtectedRoute>
                <History />
              </ProtectedRoute>
            }
          />

          {/* 404 */}
          <Route path="*" element={<NotFound />} />
          <Route path="/history" element={<History />} />

        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
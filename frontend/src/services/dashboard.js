import { authFetch } from "./api";

export async function getDashboardSummary() {
  const res = await authFetch("/dashboard/summary", {
    method: "GET",
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data?.detail || "Failed to load dashboard summary");
  }

  return data;
}

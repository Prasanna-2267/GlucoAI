import { authFetch } from "./api";

export async function getHistoryStatus() {
  const res = await authFetch("/me/history-status", {
    method: "GET",
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to get history status");

  return data; // { has_history: true/false }
}

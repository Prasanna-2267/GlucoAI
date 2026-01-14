import { authFetch } from "./api";

export async function getUserHistory() {
  const res = await authFetch("/history", {
    method: "GET",
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to fetch history");

  return data; // { detections: [...], risk_predictions: [...] }
}

import { authFetch } from "./api";

// ✅ Add this function
export async function predictRisk(payload) {
  const res = await authFetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Risk prediction failed");

  return data;
}

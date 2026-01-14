import { authFetch } from "./api";

export async function dietRecommend(payload) {
  const res = await authFetch("/diet-recommend", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data?.detail || "Failed to get diet recommendation");
  }

  return data;
}

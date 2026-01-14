import { authFetch } from "./api";

export async function getDietRecommend(payload) {
  const res = await authFetch("/diet-recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Diet recommendation failed");

  return data;
}

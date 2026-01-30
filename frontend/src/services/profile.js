import { authFetch } from "./api";

export async function getMyProfile() {
  const res = await authFetch("/me", { method: "GET" });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Failed to fetch profile");
  return data;
}

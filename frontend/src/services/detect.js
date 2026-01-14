import { authFetch } from "./api";

export async function detectManual(payload) {
  const res = await authFetch("/detect/manual", {
    method: "POST",
    body: JSON.stringify(payload),
  });

  const data = await res.json().catch(() => null);

  if (!res.ok) {
    const errMsg =
      data?.detail?.message ||
      data?.detail ||
      data?.message ||
      "Detection failed";
    throw new Error(errMsg);
  }

  return data;
}
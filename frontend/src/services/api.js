import { API_BASE_URL } from "../config";
import { getAccessToken, logoutUser } from "./session";

export function getToken() {
  return getAccessToken();
}

export async function authFetch(endpoint, options = {}) {
  const token = getToken();

  // ✅ If body is FormData, don't set JSON content-type
  const isFormData = options.body instanceof FormData;

  const res = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (res.status === 401) logoutUser();

  return res;
}

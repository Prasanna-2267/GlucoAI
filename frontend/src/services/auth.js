import { API_BASE_URL } from "@/config";
import { getAccessToken, removeAccessToken, setAccessToken, setStoredUser, getStoredUser, removeStoredUser } from "./session";

/**
 * ✅ Email + Password Login
 * Backend should return:
 * {
 *   access_token: "...",
 *   user_id: "...",
 *   email: "...",
 *   name: "..."
 * }
 */
export async function login(email, password) {
  const res = await fetch(`${API_BASE_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data?.detail || "Login failed");
  }

  // ✅ store token + user
  setAccessToken(data.access_token);
  setStoredUser({
    user_id: data.user_id,
    email: data.email,
    name: data.name,
  });

  return data;
}

/**
 * ✅ Google login
 */
export async function loginWithGoogle(idToken) {
  const res = await fetch(`${API_BASE_URL}/auth/google`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id_token: idToken }),
  });

  const data = await res.json();

  if (!res.ok) {
    throw new Error(data?.detail || "Google login failed");
  }

  // ✅ store token + user
  setAccessToken(data.access_token);
  setStoredUser({
    user_id: data.user_id,
    email: data.email,
    name: data.name,
  });

  return data;
}

/**
 * ✅ Logout
 */
export function logout() {
  removeAccessToken();
  removeStoredUser();
}

/**
 * ✅ Auth helpers
 */
export function isLoggedIn() {
  return !!getAccessToken();
}

export function getUser() {
  return getStoredUser();
}
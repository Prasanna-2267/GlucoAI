export type StoredUser = {
  user_id: string;
  email: string;
  name: string;
  avatar?: string;
  createdAt?: string;
};

const USER_KEY = "user";
const TOKEN_KEY = "access_token";

/**
 * ✅ User functions
 */
export function getStoredUser(): StoredUser | null {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;

  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function setStoredUser(user: StoredUser) {
  localStorage.setItem(USER_KEY, JSON.stringify(user));
  window.dispatchEvent(new Event("user-updated"));
}

export function removeStoredUser() {
  localStorage.removeItem(USER_KEY);
  window.dispatchEvent(new Event("user-updated"));
}

/**
 * ✅ Token functions
 */
export function getAccessToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAccessToken(token: string) {
  localStorage.setItem(TOKEN_KEY, token);
  window.dispatchEvent(new Event("user-updated"));
}

export function removeAccessToken() {
  localStorage.removeItem(TOKEN_KEY);
  window.dispatchEvent(new Event("user-updated"));
}

/**
 * ✅ Logout
 */
export function logoutUser() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
  window.dispatchEvent(new Event("user-updated"));
}
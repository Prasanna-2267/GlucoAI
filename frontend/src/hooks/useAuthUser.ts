import { useEffect, useState } from "react";
import { getStoredUser, StoredUser } from "@/services/session";

export function useAuthUser() {
  const [user, setUser] = useState<StoredUser | null>(getStoredUser());

  useEffect(() => {
    const update = () => setUser(getStoredUser());

    window.addEventListener("user-updated", update);
    return () => window.removeEventListener("user-updated", update);
  }, []);

  return user;
}

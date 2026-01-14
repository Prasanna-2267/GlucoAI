import { API_BASE_URL } from "../config";
import { getAccessToken } from "./session";

export async function detectFromReport(file, extra = {}) {
  const token = getAccessToken();

  const formData = new FormData();
  formData.append("file", file);

  // ✅ correct check
  if (extra.age !== undefined && extra.age !== null && String(extra.age).trim() !== "") {
    formData.append("age", String(extra.age).trim());
  }

  if (
    extra.blood_pressure !== undefined &&
    extra.blood_pressure !== null &&
    String(extra.blood_pressure).trim() !== ""
  ) {
    formData.append("blood_pressure", String(extra.blood_pressure).trim());
  }

  // DEBUG (keep for 1 test)
  // for (const pair of formData.entries()) console.log("FormData:", pair[0], pair[1]);

  const res = await fetch(`${API_BASE_URL}/model-upload`, {
    method: "POST",
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: formData,
  });

  const data = await res.json();

  if (!res.ok) {
    console.error("Upload error response:", data);
    throw new Error(
      data?.detail?.message || data?.detail || "Upload detection failed"
    );
  }

  return data;
}

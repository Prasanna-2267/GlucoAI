import { API_BASE_URL } from "../config";
import { getToken } from "./api";

export async function uploadReportForDetection(file, extra = {}) {
  const token = getToken();
  const formData = new FormData();
  formData.append("file", file);

  // optional extra fields:
  if (extra.age) formData.append("age", extra.age);
  if (extra.blood_pressure) formData.append("blood_pressure", extra.blood_pressure);

  const res = await fetch(`${API_BASE_URL}/model-upload`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) throw new Error(JSON.stringify(data.detail || data));

  return data;
}

import { API_BASE_URL } from "../config";

/**
 * Sends chatbot question + optional PDF to backend.
 * Backend: POST /chatbot (FormData)
 */
export async function askChatbot(question, pdfFile = null) {
  const formData = new FormData();
  formData.append("question", question);

  if (pdfFile) {
    formData.append("pdf", pdfFile);
  }

  const res = await fetch(`${API_BASE_URL}/chatbot`, {
    method: "POST",
    body: formData,
  });

  let data = null;

  // ✅ safe JSON parsing
  try {
    data = await res.json();
  } catch {
    data = null;
  }

  // ✅ handle backend errors safely
  if (!res.ok) {
    const errMsg =
      data?.detail ||
      data?.message ||
      "Chatbot error. Please try again.";
    throw new Error(errMsg);
  }

  // ✅ expected: { answer: "...", source: "rag"|"llm" }
  return data;
}
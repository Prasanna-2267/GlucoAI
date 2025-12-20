import os
import pickle
import numpy as np
import re
from typing import Optional, Dict
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing in environment variables")

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="Diabetes AI Chatbot API",
    version="1.0.0",
    description="Gemini + LangChain based diabetes-only chatbot"
)

# --------------------------------------------------
# Request / Response Schemas
# --------------------------------------------------
class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

    

# --------------------------------------------------
# Gemini LLM (LangChain 2025)
# --------------------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3,
    max_output_tokens=512
)

# --------------------------------------------------
# Diabetes-only Guardrail Prompt
# --------------------------------------------------
DIABETES_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a medical AI assistant specialized ONLY in diabetes.

STRICT RULES:
- Answer ONLY diabetes-related questions.
- Allowed topics: Type 1, Type 2, gestational diabetes, symptoms, causes,
  diagnosis, HbA1c, blood glucose, insulin, oral medications,
  complications, diet, exercise, lifestyle management, prevention.
- If the question is NOT related to diabetes, politely refuse.

IMPORTANT:
- Educational purpose only
- Do NOT provide medical diagnosis
- Use simple, clear language

If unrelated, reply exactly:
"I'm designed to answer only diabetes-related questions."
        """
    ),
    ("human", "{question}")
])

# --------------------------------------------------
# Health Check
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

# --------------------------------------------------
# Chat Endpoint
# --------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
async def chat_diabetes(request: ChatRequest):
    try:
        chain = DIABETES_PROMPT | llm
        response = chain.invoke({"question": request.question})

        return ChatResponse(answer=response.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
MODEL_PATH = "models/diabetes_xgboost_model.json"
SCALER_PATH = "models/scaler.pkl"

model = XGBClassifier()
model.load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler: StandardScaler = pickle.load(f)



DEFAULT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Fill missing diabetes features with safe median-like values. "
     "Return ONLY JSON with keys: glucose, blood_pressure, skin_thickness, insulin, bmi."),
    ("human", "{data}")
])

# -------------------------------------------------
# FASTAPI APP
# -------------------------------------------------
app = FastAPI(title="Diabetes Detection API")

# -------------------------------------------------
# OCR EXTRACTION (placeholder)
# -------------------------------------------------
def run_ocr(file: UploadFile) -> dict:
    """
    Extracts diabetes-related values from a medical report image or PDF.
    This is a demo OCR implementation for resume projects.
    """

    # -------- Step 1: Read file --------
    content = file.file.read()

    text = ""

    # -------- Step 2: Handle PDF or Image --------
    if file.filename.lower().endswith(".pdf"):
        images = convert_from_bytes(content)
        for img in images:
            text += pytesseract.image_to_string(img)
    else:
        image = Image.open(file.file)
        text = pytesseract.image_to_string(image)

    text = text.lower()

    # -------- Step 3: Regex-based extraction --------
    def extract(pattern):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    extracted_data = {
        "glucose": extract(r"glucose[:\s]*([0-9]+\.?[0-9]*)"),
        "blood_pressure": extract(r"blood pressure[:\s]*([0-9]+\.?[0-9]*)"),
        "skin_thickness": extract(r"skin thickness[:\s]*([0-9]+\.?[0-9]*)"),
        "insulin": extract(r"insulin[:\s]*([0-9]+\.?[0-9]*)"),
        "bmi": extract(r"bmi[:\s]*([0-9]+\.?[0-9]*)")
    }

    return extracted_data

# -------------------------------------------------
# REQUEST SCHEMA (manual / hybrid)
# -------------------------------------------------


class DetectRequest(BaseModel):
    age: int

    glucose: Optional[float] = None
    blood_pressure: Optional[float] = None
    skin_thickness: Optional[float] = None
    insulin: Optional[float] = None
    bmi: Optional[float] = None

    pregnancies: Optional[int] = None
    diabetes_pedigree: Optional[float] = None

    # values coming from OCR / frontend prefill
    extracted: Optional[dict] = None



# -------------------------------------------------
# RESPONSE SCHEMA
# -------------------------------------------------
class DetectResponse(BaseModel):
    prediction: str
    probability: float
    values_used: Dict[str, float]

# -------------------------------------------------
# FILL MISSING VALUES
# -------------------------------------------------


@app.post("/model-upload", response_model=DetectResponse)
async def detect_from_report_with_model(
    file: UploadFile = File(...),

    # User-provided fallback values
    age: Optional[int] = None,
    blood_pressure: Optional[float] = None,

    pregnancies: int = 0,
    diabetes_pedigree: float = 0.5
):
    """
    Upload report -> OCR -> ask user if critical fields missing -> model prediction
    """

    # 1️⃣ OCR extraction
    extracted = run_ocr(file)

    # 2️⃣ Resolve AGE (OCR doesn't extract age → must come from user)
    final_age = age

    # 3️⃣ Resolve BLOOD PRESSURE (OCR → user)
    final_bp = (
        extracted.get("blood_pressure")
        if extracted.get("blood_pressure") is not None
        else blood_pressure
    )

    # 4️⃣ ❗ Validate critical fields
    missing_required = []
    if final_age is None:
        missing_required.append("age")
    if final_bp is None:
        missing_required.append("blood_pressure")

    if missing_required:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Please provide the missing required fields to continue prediction.",
                "missing_fields": missing_required,
                "ocr_extracted": extracted
            }
        )

    # 5️⃣ Fill remaining values safely
    final_data = {
        "pregnancies": pregnancies,
        "glucose": extracted.get("glucose") or 120.0,
        "blood_pressure": float(final_bp),
        "skin_thickness": extracted.get("skin_thickness") or 20.0,
        "insulin": extracted.get("insulin") or 80.0,
        "bmi": extracted.get("bmi") or 25.0,
        "diabetes_pedigree": diabetes_pedigree,
        "age": int(final_age)
    }

    # 6️⃣ Model input (exact training order)
    X = np.array([[
        final_data["pregnancies"],
        final_data["glucose"],
        final_data["blood_pressure"],
        final_data["skin_thickness"],
        final_data["insulin"],
        final_data["bmi"],
        final_data["diabetes_pedigree"],
        final_data["age"]
    ]])

    # 7️⃣ Scale + Predict
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]

    # 8️⃣ Return result
    return DetectResponse(
        prediction="Diabetic" if prob >= 0.5 else "Non-Diabetic",
        probability=round(float(prob), 4),
        values_used=final_data
    )


# -------------------------------------------------
# 2️⃣ MANUAL / HYBRID ENDPOINT
# -------------------------------------------------
@app.post("/detect/manual", response_model=DetectResponse)
def detect_manual(req: DetectRequest):
    data = {
        "glucose": req.glucose,
        "blood_pressure": req.blood_pressure,
        "skin_thickness": req.skin_thickness,
        "insulin": req.insulin,
        "bmi": req.bmi
    }

    final_data = {
        "pregnancies": req.pregnancies if req.pregnancies is not None else 0,
        "glucose": float(req.glucose),
        "blood_pressure": float(req.blood_pressure) if req.blood_pressure is not None else 70.0,
        "skin_thickness": float(req.skin_thickness) if req.skin_thickness is not None else 20.0,
        "insulin": float(req.insulin) if req.insulin is not None else 80.0,
        "bmi": float(req.bmi),
        "diabetes_pedigree": float(req.diabetes_pedigree) if req.diabetes_pedigree is not None else 0.5,
        "age": int(req.age)
    }


    X = np.array([[
        final_data["pregnancies"],
        final_data["glucose"],
        final_data["blood_pressure"],
        final_data["skin_thickness"],
        final_data["insulin"],
        final_data["bmi"],
        final_data["diabetes_pedigree"],
        final_data["age"]
    ]])


    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]

    return DetectResponse(
        prediction="Diabetic" if prob >= 0.5 else "Non-Diabetic",
        probability=round(float(prob), 4),
        values_used=final_data
    )
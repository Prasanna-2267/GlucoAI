import os
import pickle
import numpy as np
import json
import re
import uuid
from typing import Optional, Dict
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from fastapi.middleware.cors import CORSMiddleware

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from models import DiabetesRecord

from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    user_id: str 

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
    user_id: str = Form(...), 
    # User-provided fallback values
    age: Optional[int] = None,
    blood_pressure: Optional[float] = None,

    pregnancies: int = 0,
    diabetes_pedigree: float = 0.5
):
    """
    Upload report -> OCR -> ask user if critical fields missing -> model prediction
    """
    db = SessionLocal()

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
    prediction = "diabetic" if prob >= 0.5 else "non-diabetic"

    record = DiabetesRecord(
        user_id=uuid.UUID(user_id),
            glucose=final_data["glucose"],
            blood_pressure=final_data["blood_pressure"],
            skin_thickness=final_data["skin_thickness"],
            insulin=final_data["insulin"],
            bmi=final_data["bmi"],
            age=final_data["age"],
            prediction=prediction,
            probability=float(prob)
        )
    db.add(record)
    db.commit()

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
    db = SessionLocal()
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
    prediction = "diabetic" if prob >= 0.5 else "non-diabetic"

    record = DiabetesRecord(
        user_id=uuid.UUID(req.user_id),
            glucose=final_data["glucose"],
            blood_pressure=final_data["blood_pressure"],
            skin_thickness=final_data["skin_thickness"],
            insulin=final_data["insulin"],
            bmi=final_data["bmi"],
            age=final_data["age"],
            prediction=prediction,
            probability=float(prob)
        )
    db.add(record)
    db.commit()

    return DetectResponse(
        prediction="Diabetic" if prob >= 0.5 else "Non-Diabetic",
        probability=round(float(prob), 4),
        values_used=final_data
    )

# --------------- GOOGLE AUTHENTICATION -------------------

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GOOGLE_CLIENT_ID:
    raise RuntimeError("GOOGLE_CLIENT_ID not set")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")

# --------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# --------------------------------------------------
# USER MODEL
# --------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# --------------------------------------------------
# AUTO CREATE TABLES
# --------------------------------------------------
Base.metadata.create_all(bind=engine)



# --------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# --------------------------------------------------
class GoogleAuthRequest(BaseModel):
    id_token: str

class GoogleAuthResponse(BaseModel):
    user_id: str
    email: str
    name: str
    is_new_user: bool

# --------------------------------------------------
# GOOGLE AUTH ENDPOINT
# --------------------------------------------------
@app.post("/auth/google", response_model=GoogleAuthResponse)
def google_auth(request: GoogleAuthRequest):
    try:
        # 1️⃣ Verify Google ID Token
        idinfo = id_token.verify_oauth2_token(
            request.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        email = idinfo.get("email")
        name = idinfo.get("name")

        if not email or not name:
            raise HTTPException(status_code=400, detail="Invalid Google token")

        db = SessionLocal()

        # 2️⃣ Check if user exists
        user = db.query(User).filter(User.email == email).first()

        if user:
            return GoogleAuthResponse(
                user_id=str(user.id),
                email=user.email,
                name=user.name,
                is_new_user=False
            )

        # 3️⃣ Create new user
        new_user = User(
            id=uuid.uuid4(),
            email=email,
            name=name
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return GoogleAuthResponse(
            user_id=str(new_user.id),
            email=new_user.email,
            name=new_user.name,
            is_new_user=True
        )

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or expired Google token")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# --------------------------------------------------
# PREDICT ENDPOINT
# --------------------------------------------------

class PredictRiskRequest(BaseModel):
    user_id: str

    # Only required if NO history exists
    glucose_30_days_ago: Optional[float] = None
    glucose_60_days_ago: Optional[float] = None
    glucose_90_days_ago: Optional[float] = None

    # Lifestyle questions (answers as text)
    lifestyle_answers: Dict[str, str]

class PredictRiskResponse(BaseModel):
    risk_trend: str          # Improving / Stable / Worsening
    confidence: str          # Low / Medium / High
    reasoning: list[str]
    future_outlook: list[str]
    note: str

RISK_PREDICTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an AI assistant performing diabetes RISK TREND ESTIMATION.

STRICT RULES:
- This is NOT a medical diagnosis.
- Do NOT predict glucose numbers.
- Do NOT suggest medications.
- Output must be qualitative only.

You must classify risk trend as one of:
- Improving
- Stable
- Worsening

You must also provide:
- confidence: Low / Medium / High
- reasoning: 3-4 bullet points
- future_outlook: EXACTLY 3 bullet points

Field meanings (STRICT):
- reasoning:
  Explain WHY this risk trend was assigned based on patterns, history, and lifestyle.
- future_outlook:
  Explain WHAT MAY HAPPEN if the user continues the same lifestyle without changes.
  (Do NOT give medical advice, numbers, or treatments.)

If historical data is limited, confidence MUST be Low.

Return ONLY valid JSON in this exact format:
{{
  "risk_trend": "",
  "confidence": "",
  "reasoning": [],
  "future_outlook": []
}}

Do NOT include explanations, markdown, or text outside JSON.
        """
    ),
    ("human", "{input_data}")
])



def get_user_detection_history(user_id: str):
    db = SessionLocal()
    records = (
        db.query(DiabetesRecord)
        .filter(DiabetesRecord.user_id == uuid.UUID(user_id))
        .order_by(DiabetesRecord.created_at)
        .all()
    )
    db.close()

    return [
        {
            "date": r.created_at.isoformat(),
            "glucose": r.glucose,
            "bmi": r.bmi,
            "prediction": r.prediction
        }
        for r in records
    ]


def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group())

@app.post("/predict", response_model=PredictRiskResponse)
def predict(req: PredictRiskRequest):
    history = get_user_detection_history(req.user_id)

    if history:
        input_data = {
            "user_type": "existing",
            "detection_history": history,
            "lifestyle_answers": req.lifestyle_answers
        }
    else:
        input_data = {
            "user_type": "new",
            "reported_glucose_history": {
                "30_days_ago": req.glucose_30_days_ago,
                "60_days_ago": req.glucose_60_days_ago,
                "90_days_ago": req.glucose_90_days_ago
            },
            "lifestyle_answers": req.lifestyle_answers
        }

    chain = RISK_PREDICTION_PROMPT | llm
    response = chain.invoke({"input_data": input_data})

    import json
    result = extract_json(response.content)  # safe because output is forced JSON

    return PredictRiskResponse(
        risk_trend=result["risk_trend"],
        confidence=result["confidence"],
        reasoning=result["reasoning"],
        future_outlook=result["future_outlook"],
        note="This is an AI-assisted risk trend estimation, not a medical diagnosis."
    )


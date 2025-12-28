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
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
from jose import jwt, JWTError
from datetime import datetime, timedelta
from models import User, DiabetesRecord, RiskTrend
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60

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


# --------------- GOOGLE AUTHENTICATION -------------------
#------Add security dependency-------
security = HTTPBearer()
#------------------------------------

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM],options={"verify_exp": True})
        user_id = payload.get("user_id")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")

        db = SessionLocal()
        user = db.query(User).filter(User.id == uuid.UUID(user_id)).first()
        db.close()

        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        return user

    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")



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
# AUTO CREATE TABLES
# --------------------------------------------------
Base.metadata.create_all(bind=engine)



# --------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# --------------------------------------------------
class GoogleAuthRequest(BaseModel):
    id_token: str

class GoogleAuthResponse(BaseModel):
    access_token: str
    token_type: str
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
        user = db.query(User).filter(User.email == email).first()

        if user:
            access_token = create_access_token({
                "user_id": str(user.id),
                "email": user.email
            })

            return GoogleAuthResponse(
                access_token=access_token,
                token_type="bearer",
                user_id=str(user.id),
                email=user.email,
                name=user.name,
                is_new_user=False
            )

        new_user = User(
            id=uuid.uuid4(),
            email=email,
            name=name
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        access_token = create_access_token({
            "user_id": str(new_user.id),
            "email": new_user.email
        })

        return GoogleAuthResponse(
            access_token=access_token,
            token_type="bearer",
            user_id=str(new_user.id),
            email=new_user.email,
            name=new_user.name,
            is_new_user=True
        )

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or expired Google token")

    finally:
        db.close()

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


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
MODEL_PATH = "models/diabetes_xgboost_model.json"
SCALER_PATH = "models/scaler.pkl"

import xgboost as xgb

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
    current_user: User = Depends(get_current_user),
    age: Optional[int] = None,
    blood_pressure: Optional[float] = None,
    pregnancies: int = 0,
    diabetes_pedigree: float = 0.5
):
    """
    Upload report → OCR → validate missing critical fields → model prediction
    """

    db = SessionLocal()

    # 1️⃣ OCR extraction
    ocr_data = run_ocr(file)

    # 2️⃣ Resolve AGE (must come from user)
    final_age = age

    # 3️⃣ Resolve BLOOD PRESSURE (OCR has priority)
    if ocr_data.get("blood_pressure") is not None:
        final_bp = ocr_data["blood_pressure"]
    else:
        final_bp = blood_pressure

    # 4️⃣ Validate ONLY after resolution
    missing_fields = []
    if final_age is None:
        missing_fields.append("age")
    if final_bp is None:
        missing_fields.append("blood_pressure")

    if missing_fields:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Please provide missing required fields to continue prediction.",
                "missing_fields": missing_fields,
                "ocr_extracted": ocr_data
            }
        )

    # 5️⃣ Merge all features safely
    final_data = {
        "pregnancies": pregnancies,
        "glucose": ocr_data.get("glucose") if ocr_data.get("glucose") is not None else 120.0,
        "blood_pressure": float(final_bp),
        "skin_thickness": ocr_data.get("skin_thickness") if ocr_data.get("skin_thickness") is not None else 20.0,
        "insulin": ocr_data.get("insulin") if ocr_data.get("insulin") is not None else 80.0,
        "bmi": ocr_data.get("bmi") if ocr_data.get("bmi") is not None else 25.0,
        "diabetes_pedigree": diabetes_pedigree,
        "age": int(final_age)
    }

    # 6️⃣ Model input (training order preserved)
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
    prediction = "Diabetic" if prob >= 0.5 else "Non-Diabetic"

    # 8️⃣ Store result
    record = DiabetesRecord(
        user_id=current_user.id,
        glucose=final_data["glucose"],
        blood_pressure=final_data["blood_pressure"],
        skin_thickness=final_data["skin_thickness"],
        insulin=final_data["insulin"],
        bmi=final_data["bmi"],
        age=final_data["age"],
        prediction=prediction.lower(),
        probability=float(prob)
    )

    db.add(record)
    db.commit()
    db.close()

    # 9️⃣ Return response
    return DetectResponse(
        prediction=prediction,
        probability=round(float(prob), 4),
        values_used=final_data
    )

# -------------------------------------------------
# 2️⃣ MANUAL / HYBRID ENDPOINT
# -------------------------------------------------
@app.post("/detect/manual")
def detect_manual(
    req: DetectRequest,
    current_user: User = Depends(get_current_user)
):
    # current_user is already verified via JWT

    final_data = {
        "pregnancies": req.pregnancies or 0,
        "glucose": req.glucose,
        "blood_pressure": req.blood_pressure or 83.0,
        "skin_thickness": req.skin_thickness or 20.0,
        "insulin": req.insulin or 80.0,
        "bmi": req.bmi or 25.0,
        "diabetes_pedigree": req.diabetes_pedigree or 0.5,
        "age": req.age
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

    prediction = "Diabetic" if prob >= 0.5 else "Non-Diabetic"

    # STORE USING JWT USER
    record = DiabetesRecord(
        user_id=current_user.id,
        glucose=final_data["glucose"],
        blood_pressure=final_data["blood_pressure"],
        skin_thickness=final_data["skin_thickness"],
        insulin=final_data["insulin"],
        bmi=final_data["bmi"],
        age=final_data["age"],
        prediction=prediction,
        probability=float(prob)
    )

    db = SessionLocal()
    db.add(record)
    db.commit()
    db.close()

    return {
        "prediction": prediction,
        "probability": round(float(prob), 4),
        "values_used": final_data
    }


# --------------------------------------------------
# PREDICT ENDPOINT
# --------------------------------------------------
class PredictRiskRequest(BaseModel):
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
def predict(
    req: PredictRiskRequest,
    current_user: User = Depends(get_current_user)
):
    # 🔐 user identity comes from JWT, NOT request body
    user_id = str(current_user.id)

    history = get_user_detection_history(user_id)

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

    try:
        result = extract_json(response.content)
    except ValueError:
        raise HTTPException(500, "LLM output invalid")
    
    risk_trend_value = result["risk_trend"]

    db = SessionLocal()
    record = RiskTrend(
        user_id=current_user.id,
        risk_trend=risk_trend_value
    )
    db.add(record)
    db.commit()
    db.close()


    return PredictRiskResponse(
        risk_trend=result["risk_trend"],
        confidence=result["confidence"],
        reasoning=result["reasoning"],
        future_outlook=result.get("future_outlook", []),
        note="This is an AI-assisted risk trend estimation, not a medical diagnosis."
    )

#--------------------------------------------------
# DIETARY RECOMMENDATIONS ENDPOINT
#--------------------------------------------------
class DietRecommendRequest(BaseModel):
    diet_preference: str          # vegetarian / non-vegetarian / eggetarian
    meals_per_day: int            # 2 / 3 / 4
    eats_outside: str             # rarely / weekly / frequently
    cultural_preference: str      # indian / mixed / no preference
    allergies: Optional[str] = None
    

class DietRecommendResponse(BaseModel):
    diet_type: str
    foods_to_prefer: list[str]
    foods_to_limit: list[str]
    tips: list[str]
    note: str

def get_latest_risk_trend(user_id: uuid.UUID):
    db = SessionLocal()
    record = (
        db.query(RiskTrend)
        .filter(RiskTrend.user_id == user_id)
        .order_by(RiskTrend.created_at.desc())
        .first()
    )
    db.close()
    return record.risk_trend if record else None

DIET_RECOMMEND_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an AI assistant providing qualitative DIET RECOMMENDATIONS.

STRICT RULES:
- This is NOT medical advice.
- Do NOT suggest medications or supplements.
- Do NOT give numeric calorie values.
- Keep recommendations general and lifestyle-based.

You must provide:
- diet_type: short label (e.g., Vegetarian, Balanced, Low Sugar)
- foods_to_prefer: EXACTLY 5 items
- foods_to_limit: EXACTLY 5 items
- tips: EXACTLY 3 general lifestyle tips

Return ONLY valid JSON in this exact format:
{{
  "diet_type": "",
  "foods_to_prefer": [],
  "foods_to_limit": [],
  "tips": []
}}

Do NOT include explanations, markdown, or text outside JSON.
        """
    ),
    ("human", "{input_data}")
])



@app.post("/diet-recommend", response_model=DietRecommendResponse)
def diet_recommend(
    req: DietRecommendRequest,
    current_user: User = Depends(get_current_user)
):
    # 🔐 user identity from JWT
    user_id = current_user.id

    # 1️⃣ Fetch latest risk trend
    risk_trend = get_latest_risk_trend(user_id)

    if not risk_trend:
        raise HTTPException(
            status_code=400,
            detail="Risk trend not available. Please run prediction first."
        )

    # 2️⃣ Prepare input for Gemini
    input_data = {
        "risk_trend": risk_trend,
        "diet_preference": req.diet_preference,
        "meals_per_day": req.meals_per_day,
        "eats_outside": req.eats_outside,
        "cultural_preference": req.cultural_preference,
        "allergies": req.allergies
    }

    # 3️⃣ Invoke Gemini
    chain = DIET_RECOMMEND_PROMPT | llm
    response = chain.invoke({"input_data": input_data})

    result = extract_json(response.content)
    diet_type = result.get("diet_type", "General diabetic-friendly diet")
    foods_to_prefer = result.get("foods_to_prefer", [])
    foods_to_limit = result.get("foods_to_limit", [])
    tips = result.get("tips", [])

    # Enforce exact counts (as per prompt contract)
    foods_to_prefer = foods_to_prefer[:5]
    foods_to_limit = foods_to_limit[:5]
    tips = tips[:3]


    return DietRecommendResponse(
    diet_type=diet_type,
    foods_to_prefer=foods_to_prefer,
    foods_to_limit=foods_to_limit,
    tips=tips,
    note="This is an AI-assisted dietary suggestion, not medical advice."
)
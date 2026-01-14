# GlucoAI 🩺🤖  
**AI-Powered Diabetes Support Platform** (Detection • Risk Forecast • Chatbot • Diet Tips)

GlucoAI is a full-stack web application built to help users analyze diabetes-related health data, classify diabetic/non-diabetic risk, forecast risk trends, and get AI-powered guidance through a chatbot.  
It supports **manual entry** and **report upload (PDF/Image OCR)**, stores user history securely, and provides personalized risk insights.

---

## ✨ Features

### ✅ Authentication
- Google OAuth Login
- JWT-based secure authentication
- Auto logout on invalid token / 401 response

### ✅ Blood Sugar Detection
- Manual Entry (Age, Glucose, BMI, BP etc.)
- Upload report (PDF / image)
- OCR extraction from reports
- Model prediction using ML (XGBoost + SMOTE + Scaler)
- Stores detection results in database

### ✅ Diabetes Risk Forecast
- Lifestyle-based questions
- Forecast trend: **Improving / Stable / Worsening**
- Confidence score + reasoning + recommendations
- Uses detection history automatically for existing users

### ✅ Chatbot (Diabetes Assistant)
- Chat UI + floating chatbot button
- Messages sent to backend chatbot endpoint
- Supports optional PDF upload for analysis

### ✅ Profile Page
- Shows real user details from DB
- Displays Google avatar/profile photo
- Logout support

### ✅ History Page
- Combined history of:
  - Detection history (Diabetic/Non-diabetic)
  - Risk forecast history
- Dates & time fetched from database

---

## 🛠 Tech Stack

### Frontend
- React + TypeScript (Vite)
- Tailwind CSS
- shadcn/ui components
- Lucide Icons
- React Router DOM

### Backend
- FastAPI (Python)
- JWT Authentication
- Google OAuth token verification
- OCR processing for reports
- ML Model prediction
- LLM-based risk forecast prompt chain

### Database
- PostgreSQL
- Tables used:
  - `users`
  - `diabetes_records`
  - `risk_trends`


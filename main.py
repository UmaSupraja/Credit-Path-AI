import logging
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Dict
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import requests # Required for proxying
from datetime import datetime, timedelta
from jose import JWTError, jwt

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("predictions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration (IMPORTANT: REPLACE THESE WITH ENV VARS IN RAILWAY) ---

# Replace with a long, random string. Store this securely!
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "YOUR_SECRET_KEY_HERE")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Get the Gemini Key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
SYSTEM_PROMPT = """
You are an expert AI Credit Coach for the "CREDIT PATH-AI" application. 
Your role is to answer user questions about credit scores, loans, debt, and general financial health.
Be helpful, encouraging, and provide clear, simple explanations. 
Do NOT answer questions that are off-topic (e.g., about history, science, celebrities, etc.). 
If asked an off-topic question, politely decline and remind the user of your purpose.
Keep your answers concise and easy to understand.
"""


# --- Initialize FastAPI App ---
app = FastAPI(
    title="CreditPathAI Loan Recovery System",
    description="An AI-based system to predict loan default risk, recommend actions, and provide a financial AI Coach.",
    version="1.0.0"
)

# --- Add CORS Middleware (Allows all domains to connect) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DEMO User Database (In a real application, use PostgreSQL/MongoDB) ---
# This dictionary stores simple email/password pairs
DEMO_USER_DB: Dict[str, str] = {}


# --- Pydantic Models ---

# Borrower Data Model (Same as before)
class Borrower(BaseModel):
    homeOwn: str
    annualIncome: float = Field(..., gt=0, description="Annual Income in dollars")
    yearsInCo: int = Field(..., ge=0, description="Years in current job")
    yearsOfCredit: int = Field(..., ge=0, description="Years of credit history")
    loanPurpose: str
    loanTerm: str
    monthlyDt: float = Field(..., ge=0, description="Total monthly debt payments")
    creditSc: int = Field(..., ge=300, le=850, description="Credit Score")

class BatchPredictionRequest(BaseModel):
    instances: List[Borrower]

# Auth Models
class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    token: str

class ChatPrompt(BaseModel):
    prompt: str

# --- JWT Token Generation & Verification ---

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15) # Default 15 mins
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        return email
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

# Dependency to check for valid token in Authorization header
def get_current_user(auth_header: str = Depends(lambda header: header.get("Authorization"))):
    if not auth_header:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")
    
    parts = auth_header.split()
    if parts[0].lower() != "bearer" or len(parts) != 2:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token scheme")
    
    token = parts[1]
    return verify_token(token)


# --- API Endpoints ---

# 1. AUTHENTICATION ENDPOINTS

@app.post("/register", response_model=Token, tags=["Authentication"])
async def register_user(user: UserCreate):
    if user.email in DEMO_USER_DB:
        raise HTTPException(status_code=400, detail="Account already exists.")
    
    # In a real app, hash the password before storing
    DEMO_USER_DB[user.email] = user.password 
    
    # Generate token immediately upon successful registration (optional)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"token": access_token}


@app.post("/login", response_model=Token, tags=["Authentication"])
async def login_user(user: UserLogin):
    stored_password = DEMO_USER_DB.get(user.email)
    
    # In a real app, verify the hashed password
    if not stored_password or stored_password != user.password:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"token": access_token}


# 2. CHAT PROXY ENDPOINT (Protected)

@app.post("/chat", tags=["AI Coach (Protected)"])
async def chat_proxy(chat_prompt: ChatPrompt, email: str = Depends(get_current_user)):
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        logger.error("GEMINI_API_KEY not set!")
        raise HTTPException(status_code=500, detail="AI Service not configured. Please set the API key.")
    
    # Payload for the Gemini API
    payload = {
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {"parts": [{"text": chat_prompt.prompt}]}
        ]
    }
    
    try:
        # Securely call the external API from the server side
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()
        candidate = result.get("candidates", [{}])[0]
        text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "The coach could not generate a response.")
        
        return {"response": text}
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API Request Failed: {e}")
        raise HTTPException(status_code=502, detail="External AI service failed to respond.")
    except Exception as e:
        logger.error(f"Chat Proxy Error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during chat processing.")


# 3. PREDICTION ENDPOINT (Protected)

# --- ML Model & Recommendation Logic (Same as your provided code) ---
def predict_default_probability(data: Borrower) -> float:
    risk_score = 0.0
    risk_score += (850 - data.creditSc) / 850
    
    monthly_income = data.annualIncome / 12
    if monthly_income > 0:
        dti = data.monthlyDt / monthly_income
        risk_score += min(dti * 0.5, 0.5)
        
    if data.homeOwn == 'RENT':
        risk_score += 0.1
        
    probability = min(max(risk_score / 2.0, 0.05), 0.95)
    return probability

def map_to_recommendation(probability: float) -> (str, str):
    if probability >= 0.65:
        return "High Risk", "Priority Collection / Loan Restructure"
    elif 0.35 <= probability < 0.65:
        return "Medium Risk", "Personalized Call / Email"
    else:
        return "Low Risk", "Standard Reminder"

# API Endpoint for Batch Prediction (Now protected)
@app.post("/predict/batch", tags=["Prediction (Protected)"])
async def batch_predict(request: BatchPredictionRequest, email: str = Depends(get_current_user)):
    try:
        predictions = []
        if not request.instances:
            raise HTTPException(status_code=400, detail="Input list cannot be empty.")
            
        for i, borrower_data in enumerate(request.instances):
            probability = predict_default_probability(borrower_data)
            risk_level, action = map_to_recommendation(probability)
            
            result = {
                "borrower_index": i,
                "probability": probability,
                "risk_level": risk_level,
                "recommended_action": action
            }
            predictions.append(result)
            
            logger.info({
                "status": "SUCCESS",
                "input_data": borrower_data.dict(),
                "prediction": result,
                "user": email # Log the user who made the prediction
            })
            
        return {"predictions": predictions}
        
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error({"status": "FAILURE", "error": str(e)})
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


# 4. UTILITY ENDPOINTS

@app.get("/health", tags=["Utility"])
async def health_check():
    return {"status": "OK", "message": "CreditPathAI API is running."}

@app.get("/", response_class=FileResponse, tags=["Utility"])
async def read_index():
    # This assumes your FastAPI app is set up to look for a static directory
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    html_path = os.path.join(static_dir, "index.html")
    
    if not os.path.exists(html_path):
        # Fallback for deployment environments where the index.html is served externally
        # The user's current setup relies on GitHub Pages, so this is mainly a placeholder.
        return {"error": "index.html not found on this server instance."}
    return html_path

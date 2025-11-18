# --- final_main.py ---
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

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

# --- Initialize FastAPI App ---
app = FastAPI(
    title="CreditPathAI Loan Recovery System",
    description="An AI-based system to predict loan default risk and recommend actions.",
    version="1.0.0"
)

# --- Add CORS Middleware (FIXED: ALLOW ALL ORIGINS) ---
# This allows your frontend (GitHub Pages) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # <--- FINAL FIX: Allows all domains to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Model for Input Validation (The 8 Metrics) ---
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

# --- ML Model & Recommendation Logic ---
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

# --- Serve Frontend (Removed database dependency) ---
@app.get("/", response_class=FileResponse)
async def read_index():
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    html_path = os.path.join(static_dir, "index.html")
    
    if not os.path.exists(html_path):
         return {"error": "index.html not found. Please open the HTML file directly in your browser."}
    return html_path

# --- NEW HEALTH CHECK ENDPOINT ---
@app.get("/health")
async def health_check():
    return {"status": "OK", "message": "CreditPathAI API is running."}

# --- API Endpoint for Batch Prediction ---
@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
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
                "prediction": result
            })
            
        return {"predictions": predictions}
        
    except HTTPException as http_err:
        raise http_err
    except Exception as e:
        logger.error({"status": "FAILURE", "error": str(e)})
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

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

# --- Add CORS Middleware ---
# This allows your frontend (running on a different port) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Model for Input Validation (Using the 8 Metrics) ---
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
    """Simulates a model prediction based on a simplified risk score."""
    risk_score = 0.0
    # Higher score = more risk
    risk_score += (850 - data.creditSc) / 850  # Risk from credit score
    
    monthly_income = data.annualIncome / 12
    if monthly_income > 0:
        # Calculate Debt-to-Income Ratio
        dti = data.monthlyDt / monthly_income
        # Add DTI to risk, capping it to avoid extreme values
        risk_score += min(dti * 0.5, 0.5)
        
    if data.homeOwn == 'RENT':
        risk_score += 0.1  # Renting is slightly higher risk than owning
        
    # Normalize the final score to a probability (0.05 to 0.95)
    probability = min(max(risk_score / 2.0, 0.05), 0.95)
    return probability

def map_to_recommendation(probability: float) -> (str, str):
    """Maps a prediction probability to a recommended action."""
    if probability >= 0.65:
        return "High Risk", "Priority Collection / Loan Restructure"
    elif 0.35 <= probability < 0.65:
        return "Medium Risk", "Personalized Call / Email"
    else:
        return "Low Risk", "Standard Reminder"

# --- Serve Frontend ---
# This endpoint is no longer strictly needed if you open index.html directly,
# but it's good practice.
@app.get("/", response_class=FileResponse)
async def read_index():
    # This assumes your index.html is in a folder named 'static'
    # If index.html is in the *same* folder as main.py, this will fail.
    # For simplicity, just open the index.html file directly from your folder.
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    html_path = os.path.join(static_dir, "index.html")
    
    if not os.path.exists(html_path):
         return {"error": "index.html not found. Please open the HTML file directly in your browser."}
    return html_path

# --- NEW HEALTH CHECK ENDPOINT ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint to confirm the API is running."""
    # This is a standard endpoint for monitoring systems to check if the API is alive.
    return {"status": "OK", "message": "CreditPathAI API is running."}

# --- API Endpoint for Batch Prediction ---
@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Accepts a batch of borrower data, predicts risk, and logs the transaction."""
    try:
        predictions = []
        if not request.instances:
            # Check for empty input list
            raise HTTPException(status_code=400, detail="Input list cannot be empty.")
            
        for i, borrower_data in enumerate(request.instances):
            # 1. Get Prediction
            probability = predict_default_probability(borrower_data)
            
            # 2. Get Recommendation
            risk_level, action = map_to_recommendation(probability)
            
            # 3. Format the result
            result = {
                "borrower_index": i,
                "probability": probability,
                "risk_level": risk_level,
                "recommended_action": action
            }
            predictions.append(result)
            
            # 4. Log the successful transaction
            logger.info({
                "status": "SUCCESS",
                "input_data": borrower_data.dict(),
                "prediction": result
            })
            
        # 5. Return the list of predictions
        return {"predictions": predictions}
        
    except HTTPException as http_err:
        # Re-raise HTTP exceptions (like the 400 bad request)
        raise http_err
    except Exception as e:
        # Catch any other unexpected server errors
        logger.error({"status": "FAILURE", "error": str(e)})
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
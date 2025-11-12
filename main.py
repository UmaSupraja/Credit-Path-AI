import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, EmailStr
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# --- NEW Database & Auth Imports ---
import databases
from passlib.context import CryptContext

# --- Database Setup ---
# This will read the "DATABASE_URL" from your Vercel Environment Variables
# If it's not found (like when testing locally), it will use a temporary in-memory-only database.
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")
database = databases.Database(DATABASE_URL)

# This is for hashing passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Database Table Setup ---
# This defines what our 'users' table will look like
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

# Use SQLAlchemy (required by 'databases' library) to define the table structure
# We need to add a check for "sqlite" for local testing
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # This is for Postgres
    engine = create_engine(DATABASE_URL)

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String(120), unique=True, index=True),
    Column("hashed_password", String(255)),
)
# Create the table if it doesn't exist
metadata.create_all(engine)


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

# --- NEW: App Lifespan (Connect/Disconnect DB) ---
# This code runs when your app starts and stops
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup:
    try:
        await database.connect()
        logger.info("Database connection established.")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        # In a real app, you might want to raise this to stop the app
    yield
    # On shutdown:
    await database.disconnect()
    logger.info("Database connection closed.")


# --- Initialize FastAPI App ---
# We have REMOVED root_path="/api" to fix the Vercel 404 error
app = FastAPI(
    title="CreditPathAI Loan Recovery System",
    description="An AI-based system to predict loan default risk and recommend actions.",
    version="1.0.0",
    lifespan=lifespan
)

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

# Model for the Risk Calculator
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

# NEW: Models for User Login & Creation
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

# --- ML Model & Recommendation Logic ---
def predict_default_probability(data: Borrower) -> float:
    """Simulates a model prediction based on a simplified risk score."""
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
    """Maps a prediction probability to a recommended action."""
    if probability >= 0.65:
        return "High Risk", "Priority Collection / Loan Restructure"
    elif 0.35 <= probability < 0.65:
        return "Medium Risk", "Personalized Call / Email"
    else:
        return "Low Risk", "Standard Reminder"

# --- Serve Frontend (No longer needed, Vercel handles this) ---
# We remove the @app.get("/") that serves index.html
# Vercel's "routes" in vercel.json handles this now.

# NEW: Root path for API (for testing)
@app.get("/api")
async def read_root():
    return {"message": "CreditPathAI Backend is running. Access the frontend at its separate URL."}


# --- NEW HEALTH CHECK ENDPOINT ---
@app.get("/health")
async def health_check():
    """Confirms the API is running and the database is connected."""
    try:
        # Perform a simple query to check DB connection
        await database.execute("SELECT 1")
        return {"status": "OK", "message": "CreditPathAI API is running and database is connected."}
    except Exception as e:
        logger.error(f"Health check failed: Database connection error: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Service Unavailable: Cannot connect to the database. Error: {e}"
        )


# --- NEW: User Creation Endpoint ---
@app.post("/create-account")
async def create_account(user: UserCreate):
    """Creates a new user in the database."""
    # Check if user already exists
    query = users.select().where(users.c.email == user.email)
    existing_user = await database.fetch_one(query)
    if existing_user:
        logger.warning(f"Account creation failed: Email already registered - {user.email}")
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash the password
    hashed_password = pwd_context.hash(user.password)
    
    # Insert new user into database
    query = users.insert().values(email=user.email, hashed_password=hashed_password)
    try:
        await database.execute(query)
        logger.info(f"New account created: {user.email}")
        return {"message": "Account created successfully"}
    except Exception as e:
        logger.error(f"Database error during account creation: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while creating the account.")


# --- NEW: User Login Endpoint ---
@app.post("/login")
async def login(user: UserLogin):
    """Logs a user in by verifying their email and password."""
    # Find the user by email
    query = users.select().where(users.c.email == user.email)
    db_user = await database.fetch_one(query)
    
    # Check if user exists and password is correct
    if not db_user:
        logger.warning(f"Login failed: User not found - {user.email}")
        raise HTTPException(status_code=400, detail="Incorrect email or password")
        
    if not pwd_context.verify(user.password, db_user["hashed_password"]):
        logger.warning(f"Login failed: Incorrect password for user - {user.email}")
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    # Login successful
    logger.info(f"User logged in: {user.email}")
    return {"message": "Login successful"}


# --- API Endpoint for Batch Prediction ---
@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Accepts a batch of borrower data, predicts risk, and logs the transaction."""
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
            
            # Log the successful prediction
            logger.info({
                "status": "SUCCESS",
                "input_data": borrower_data.model_dump(), # Use .model_dump() for Pydantic v2
                "prediction": result
            })
            
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error({"status": "FAILURE", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

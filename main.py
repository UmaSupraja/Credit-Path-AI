# main.py
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, EmailStr
from typing import List, Tuple
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# --- NEW Database & Auth Imports ---
import databases
from passlib.context import CryptContext

# --- Database Setup ---
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./test.db")
database = databases.Database(DATABASE_URL)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Database Table Setup ---
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String(120), unique=True, index=True),
    Column("hashed_password", String(255)),
)

# create tables if missing (safe to call in many environments)
try:
    metadata.create_all(engine)
except Exception:
    # In some readonly/protected envs this may fail; we'll continue so the app can start and surface DB errors at runtime.
    pass

# --- Setup Logging (robust for serverless / read-only FS) ---
log_handlers = [logging.StreamHandler()]

# Try adding a FileHandler if writable; ignore failures to prevent crashes during startup
try:
    fh = logging.FileHandler("predictions.log")
    log_handlers.insert(0, fh)
except Exception:
    # file logging unavailable in this environment, continue with stream only
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=log_handlers,
)
logger = logging.getLogger(__name__)

# --- App Lifespan (DB connect/disconnect) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        await database.connect()
        logger.info("Database connection established.")
    except Exception as e:
        logger.error(f"Failed to connect to database during startup: {e}")
        # don't raise here so the app can start and return a controlled error on health endpoints
    yield
    # shutdown
    try:
        if database.is_connected:
            await database.disconnect()
            logger.info("Database connection closed.")
    except Exception as e:
        logger.warning(f"Error while disconnecting database: {e}")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="CreditPathAI Loan Recovery System",
    description="An AI-based system to predict loan default risk and recommend actions.",
    version="1.0.0",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
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
    # credit score contribution
    risk_score += (850 - data.creditSc) / 850
    # debt-to-income
    monthly_income = data.annualIncome / 12 if data.annualIncome else 0
    if monthly_income > 0:
        dti = data.monthlyDt / monthly_income
        risk_score += min(dti * 0.5, 0.5)
    # renting penalty
    if data.homeOwn.upper() == "RENT":
        risk_score += 0.1
    probability = min(max(risk_score / 2.0, 0.05), 0.95)
    return probability

def map_to_recommendation(probability: float) -> Tuple[str, str]:
    if probability >= 0.65:
        return "High Risk", "Priority Collection / Loan Restructure"
    elif 0.35 <= probability < 0.65:
        return "Medium Risk", "Personalized Call / Email"
    else:
        return "Low Risk", "Standard Reminder"

# --- API Prefix: /api/* to match Vercel routing conventions --- #
@app.get("/api")
async def read_root():
    return {"message": "CreditPathAI Backend is running. Use /api/* endpoints."}

@app.get("/api/health")
async def health_check():
    """Confirms the API is running and (attempts to) check the database connection."""
    # Quick check using database.is_connected
    try:
        if database.is_connected:
            return {"status": "OK", "message": "Database connection appears active."}
        # Attempt a lightweight query to validate DB (works for SQLAlchemy-backed DBs)
        try:
            # Using a raw SQL that works on SQLite/Postgres
            result = await database.fetch_one("SELECT 1")
            return {"status": "OK", "message": "Database reachable", "result": result}
        except Exception as q_err:
            # Return an informative 503
            logger.error(f"Health check DB query failed: {q_err}")
            raise HTTPException(status_code=503, detail=f"Service Unavailable: DB query failed: {q_err}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check general failure: {e}")
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {e}")

# --- User Endpoints ---
@app.post("/api/create-account")
async def create_account(user: UserCreate):
    """Creates a new user in the database."""
    # Check if user already exists
    query = users.select().where(users.c.email == user.email)
    try:
        existing_user = await database.fetch_one(query)
    except Exception as e:
        logger.error(f"DB error checking existing user: {e}")
        raise HTTPException(status_code=500, detail="Database error while checking user existence.")
    if existing_user:
        logger.warning(f"Account creation failed: Email already registered - {user.email}")
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = pwd_context.hash(user.password)
    insert_query = users.insert().values(email=user.email, hashed_password=hashed_password)
    try:
        await database.execute(insert_query)
        logger.info(f"New account created: {user.email}")
        return {"message": "Account created successfully"}
    except Exception as e:
        logger.error(f"Database error during account creation: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while creating the account.")

@app.post("/api/login")
async def login(user: UserLogin):
    """Logs a user in by verifying their email and password."""
    query = users.select().where(users.c.email == user.email)
    try:
        db_user = await database.fetch_one(query)
    except Exception as e:
        logger.error(f"DB error during login lookup: {e}")
        raise HTTPException(status_code=500, detail="Database error during login.")

    if not db_user:
        logger.warning(f"Login failed: User not found - {user.email}")
        raise HTTPException(status_code=400, detail="Incorrect email or password")

    try:
        if not pwd_context.verify(user.password, db_user["hashed_password"]):
            logger.warning(f"Login failed: Incorrect password for user - {user.email}")
            raise HTTPException(status_code=400, detail="Incorrect email or password")
    except Exception as e:
        logger.error(f"Error verifying password for {user.email}: {e}")
        raise HTTPException(status_code=500, detail="Internal error verifying credentials.")

    logger.info(f"User logged in: {user.email}")
    return {"message": "Login successful"}

# --- Prediction Endpoint (batch) ---
@app.post("/api/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Accepts a batch of borrower data, predicts risk, and logs the transaction."""
    if not request.instances:
        raise HTTPException(status_code=400, detail="Input list cannot be empty.")

    predictions = []
    try:
        for i, borrower_data in enumerate(request.instances):
            probability = predict_default_probability(borrower_data)
            risk_level, action = map_to_recommendation(probability)

            result = {
                "borrower_index": i,
                "probability": probability,
                "risk_level": risk_level,
                "recommended_action": action,
            }
            predictions.append(result)

            # Use model_dump (Pydantic v2) safely
            try:
                input_dump = borrower_data.model_dump()
            except Exception:
                # fallback if model_dump not present
                input_dump = borrower_data.dict() if hasattr(borrower_data, "dict") else str(borrower_data)

            logger.info(
                {
                    "status": "SUCCESS",
                    "input_data": input_dump,
                    "prediction": result,
                }
            )

        return {"predictions": predictions}
    except HTTPException:
        raise
    except Exception as e:
        logger.error({"status": "FAILURE", "error": str(e)})
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

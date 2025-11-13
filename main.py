# --- NEW CODE: COPY ALL OF THIS ---
import logging
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, EmailStr
from typing import List
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os

# --- NEW: Database Connection Imports ---
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# --- NEW: Password Hashing Imports ---
from passlib.context import CryptContext

# --- NEW: User Authentication Imports ---
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta

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

# ---
# 1. DATABASE SETUP
# ---

# THIS IS THE CRITICAL CHANGE
# It now reads the URL from the Railway "Variables" tab
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
if not SQLALCHEMY_DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set!")
    # We can set a default for local testing, but it will fail on Railway if not set
    SQLALCHEMY_DATABASE_URL = "postgresql://user:pass@host/db" 

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get a DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---
# 2. PASSWORD & SECURITY SETUP
# ---

# It now reads the SECRET_KEY from the Railway "Variables" tab
SECRET_KEY = os.getenv("SECRET_KEY", "a-default-secret-key-for-local-testing")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# ---
# 3. DATABASE MODEL (The User Table)
# ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

# ---
# 4. PYDANTIC SCHEMAS (for validation)
# ---
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserInDB(BaseModel):
    email: EmailStr
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

# --- Initialize FastAPI App ---
app = FastAPI(
    title="CreditPathAI Loan Recovery System",
    description="An AI-based system to predict loan default risk and recommend actions.",
    version="1.0.0"
)

# --- Create Database Tables ---
@app.on_event("startup")
def on_startup():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created (if they didn't exist).")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

# --- Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    # This should be your GitHub Pages URL
    allow_origins=["https://UmaSupraja.github.io"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---
# 5. NEW AUTHENTICATION ENDPOINTS
# ---

# --- Security Helper Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# --- Endpoint to Create a User ---
@app.post("/create-account", response_model=UserInDB)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password)
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Could not create user.")

# --- Endpoint to Log In ---
@app.post("/login", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), 
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


# ---
# YOUR EXISTING ENDPOINTS
# ---

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

# --- Serve Frontend ---
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

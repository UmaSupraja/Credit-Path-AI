# CreditPath-AI: Loan Default Risk Prediction 

An AI-powered loan default risk prediction system leveraging machine learning to help banks make data-driven lending decisions. Built with LightGBM, achieving 99.07% AUC-ROC accuracy in the theoretical model.

### 🚀 Live Deployment & Overview

This project is configured as a stable, split full-stack web application.

Deployment Status

Component

Host

URL Status

Frontend (UI & Logic)

GitHub Pages

Live (Hosts the login pages and calculator interface)

Backend (API/Calculator)

Railway.app

Live (Runs the risk calculation logic)

Backend (API/Calculator)

Railway.app

Live (Runs the risk calculation logic)

Live Website Link: https://UmaSupraja.github.io/Credit-Path-AI/

(Note: The live API uses a lightweight Python model for stability and speed, as the full-scale model files are not deployed.)

Key Features

Mandatory Demo Login Flow: The application starts with a user authentication page featuring Create Account and Sign In. (Login uses simple browser memory storage for demonstration purposes.)

AI Risk Prediction: The 8-field form sends borrower data to the live FastAPI backend, which returns an instant Risk Prediction (Low, Medium, or High) and an Actionable Recommendation.

AI Credit Coach: A chat interface that connects to the Google Gemini API to provide general financial and credit health guidance.

####  Data and Technical Specification

Dataset

The model's theoretical performance metrics are based on the Loan Default Dataset sourced from Kaggle.

Source: Link to the dataset on Kaggle

Location: The dataset is stored in the repository at Data/loan_data.csv.

Technology Stack

Component

Technology

Role

Backend

FastAPI

High-performance asynchronous API framework.

Model (Theor.)

LightGBM

Primary gradient boosting model (99.07% AUC).

Deployment

Railway / Uvicorn

ASGI server providing a stable, scalable host for the API.

Frontend

HTML/JS/Tailwind

Single-page application (SPA) with responsive styling.

Model Performance (Theoretical Metrics)

The production model was trained using 45,000+ loan records and analyzed 24 features including derived metrics like LTI Ratio and Credit Stability Index.

Metric

Value

Interpretation

AUC-ROC

99.07%

Outstanding discrimination between defaulters and non-defaulters.

Accuracy

95%

Correctly predicts 19 out of 20 cases overall.

Recall

94%

Effectively catches 94% of actual defaulters (minimizing costly missed defaults).

Project Structure (Live Code)

Credit-Path-AI/
├── main.py                 # FastAPI application and simplified risk logic.
├── requirements.txt        # Python dependencies (fastapi, uvicorn, pydantic).
└── index.html              # Frontend UI, Demo Login/Logout logic, and API call configuration.


### 🛠️ How to Run This Project Locally

1. Prepare the Python Environment

#### Clone the Repository
git clone [https://github.com/UmaSupraja/Credit-Path-AI.git](https://github.com/UmaSupraja/Credit-Path-AI.git)
cd Credit-Path-AI

#### Create and Activate the Environment
python -m venv venv
source venv/bin/activate
#### Install the required libraries
pip install -r requirements.txt


2. Run the Backend Server

uvicorn main:app --reload


(The server will be running at http://127.0.0.1:8000.)

3. Access the Frontend

Open the user interface:

#### In your file explorer, double-click this file:
open index.html


(The frontend automatically uses the local API address when running the file directly.)



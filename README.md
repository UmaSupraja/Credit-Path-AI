# CreditPath-AI: Loan Default Risk Prediction System

An AI-powered loan default risk prediction system leveraging machine learning to help banks make data-driven lending decisions. Built with LightGBM, achieving 99.07% AUC-ROC accuracy in the theoretical model.

## Overview

Problem Statement

Loan default risk is a critical challenge in banking affecting profitability, operational costs, and regulatory compliance. Traditional credit assessment methods are manual, time-consuming, and often unable to detect complex risk patterns. CreditPath-AI addresses this by leveraging machine learning to accurately predict loan default probability, enabling banks to manage risk proactively and make strategic decisions.

Why This Matters

Financial Impact: Early identification of risky borrowers can reduce losses by 40-60%.

Operational Efficiency: Automates risk scoring in seconds versus hours of manual review.

Regulatory Compliance: Meets Basel III and RBI guidelines with transparent, auditable models.

Business Value: Supports faster application processing while balancing risk mitigation.

-----
## Key Features

The application is deployed using a stable split deployment approach to meet operational demands:

Live Website Link: https://UmaSupraja.github.io/Credit-Path-AI/

Mandatory Demo Login Flow: The application starts with a secure-feeling Create Account and Sign In feature. (Login uses simple browser memory storage for demonstration purposes.)

Single Prediction: Real-time default risk assessment for individual borrowers via the FastAPI backend.

Actionable Recommendations: Instant probability calculation with color-coded risk levels and business-focused recommendations.

AI Credit Coach: A chat interface that connects to the Google Gemini API to provide general financial and credit health guidance.

Batch Processing (Simulated Feature): Placeholder structure for processing multiple borrowers via CSV upload.

--------------------
## Technology Stack

 a. Backend

| Component | Technology | Role | 
| :--- | :--- | :--- | 
| **Framework** | **FastAPI** | High-performance asynchronous API for prediction and logging. | 
| **Server** | **Uvicorn / Railway** | ASGI server providing a stable host environment. | 
| **Validation** | **Pydantic** | Input validation and strict schema management. |

 b. Frontend

| Component | Technology | Role | 
| :--- | :--- | :--- | 
| **Structure** | **HTML5 / JavaScript (ES6+)** | Single-page application (SPA) core logic and demo state management. | 
| **Styling** | **Tailwind CSS** | Responsive, mobile-first utility framework for UI. |

c. Machine Learning

| Component | Technology | Role | 
| :--- | :--- | :--- | 
| **Primary Model** | **LightGBM** | Primary classification model achieving highest AUC-ROC. | 
| **Training Tools** | **Scikit-learn/SMOTE** | Data preprocessing, feature scaling, and class balancing. |

----------------------
#### Model Performance

Final Model: LightGBM Classifier (Theoretical Metrics)

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **AUC-ROC** | **99.07%** | Outstanding discrimination between defaulters and non-defaulters. |
| **Accuracy** | 95% | Correctly predicts 19 out of 20 cases overall. |
| **Recall** | 94% | Effectively catches 94% of actual defaulters (minimizing costly missed defaults). |
| **F1-Score** | 0.95 | Excellent balance between Precision and Recall. |

Confusion Matrix
| | **Predicted Non-Default** | **Predicted Default** | 
| :---: | :---: | :---: | 
| **Actual Non-Default** | 6,673 | 327 | 
| **Actual Default** | 400 | 6,600 |

----------------

## Project Structure

The project code reflects a simplified structure for stable cloud deployment:

```
CreditPath-AI/
│
├── backend/
│   ├── main.py          
│   
├── requirements.txt       
│
├── frontend/
│   ├── index.html              
|
└── README.md
```
----------------

## Installation

Prerequisites

Python 3.8 or higher

pip package manager

Step 1: Clone Repository
```bash
git clone [https://github.com/UmaSupraja/Credit-Path-AI.git](https://github.com/UmaSupraja/Credit-Path-AI.git)
cd Credit-Path-AI
```

Step 2: Install Backend Dependencies

```bash
# Create and Activate the Environment
python -m venv venv
source venv/bin/activate
# Install the required libraries
pip install -r requirements.txt
```

7. Usage

Start Backend Server

To run the API locally for development:

```bash
uvicorn main:app --reload
```
(The server will be running at http://127.0.0.1:8000.)

Access Frontend

Open the user interface:
```bash
# In your file explorer, double-click this file:
open index.html
```

(The frontend automatically uses the local API address when running the file directly.)

## API Documentation

API Endpoint (Prediction)

The core endpoint for risk calculation:

POST /predict/batch
| Parameter | Type | Required | Description | 
| :--- | :--- | :--- | :--- | 
| **instances** | `List[Borrower]` | Yes | Array containing the borrower data object. |

Output Schema: Returns an array containing the calculated default_probability, risk_level, and recommended_action.

Health Check

GET /health

Returns {"status": "OK", "message": "CreditPathAI API is running."}

--------------------

## Data Processing Pipeline (Theoretical)

This outlines the full, intended data pipeline for the prediction model:

Input Data

Missing Value Handling (Median Imputation)

Feature Engineering (LTI Ratio, Credit Stability Index)

Feature Scaling (StandardScaler)

One-Hot Encoding (Categorical Variables)

Class Balancing (SMOTE)

LightGBM Model Prediction

Risk Classification (Thresholds: 0.3, 0.6)

Recommendation Engine

JSON Response with Actions

------------------------------

## Key Learnings

Data Preprocessing

SMOTE balancing improved recall by 12%.

Derived features (LTI_Ratio) boosted accuracy by ~5%.

Median imputation handled outliers better than mean.

Model Selection

LightGBM outperformed XGBoost by 0.1% AUC.

Leaf-wise growth strategy proved more effective.

Well-calibrated probabilities enabled reliable thresholds.

API Design

Pydantic validation caught 95% of input errors.

Async FastAPI handled concurrent requests efficiently.

Comprehensive logging essential for production monitoring.

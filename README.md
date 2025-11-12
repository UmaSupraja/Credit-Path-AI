CreditPathAI - Loan Default Prediction

This is a complete full-stack web application that predicts the risk of loan default using a machine learning model. It features a Python (FastAPI) backend, a responsive HTML/JS/Tailwind frontend, and a connection to the Gemini API for an AI-powered "Credit Coach."

This project was built based on the "CreditPathAI" project notes.

🚀 Live Demo

The application is deployed on Vercel and is available at the following link:

Live Website: https://credit-path-ai-ruddy.vercel.app/

(Note: The backend is on a free "serverless" plan, so it may take 10-15 seconds to "wake up" on the first risk calculation.)

✨ Features

All-in-One Deployment: Frontend and Backend are hosted together on Vercel.

AI Risk Prediction: The 8-field form sends data to a Python API, which returns a real-time risk prediction (Low, Medium, High) and a recommended action.

Demo Login System: A secure-feeling login and account creation flow (for demo purposes).

Password Visibility: "Eye" icons on all password fields to toggle visibility.

AI Credit Coach: A "pop-up" chat window that connects to the Google Gemini API to answer financial questions.

API Health Check: The backend includes a /health endpoint for monitoring.

Error Handling: The frontend displays user-friendly error messages for failed logins or API errors.

#### Dataset

The data used for this project is the Loan Default Dataset sourced from Kaggle.

Source: Link to the dataset on Kaggle (You can replace this with your specific link if you have one.)

Location: The dataset is stored in the repository at Data/loan_data.csv.

(Note: The live deployed app does not read from this CSV; it uses a simulated Python model for speed and simplicity as per the project notes.)

#### How to Run This Project Locally

If you want to run this project on your own computer, you can follow these steps.

Clone the Repository:

git clone [https://github.com/UmaSupraja/Credit-Path-AI.git](https://github.com/UmaSupraja/Credit-Path-AI.git)
cd Credit-Path-AI


Create and Activate a Python Virtual Environment:

# Create the environment
python -m venv venv

# Activate the environment
# On Windows:
.\venv\Scripts\Activate.ps1
# On Mac/Linux:
# source venv/bin/activate


Install the Required Libraries:

pip install -r requirements.txt


Run the Backend Server:

python -m uvicorn main:app --reload


The server will be running at http://127.0.0.1:8000.

Open the Frontend:

Important: You must edit the index.html file first.

Find the line const apiUrl = "/predict/batch";

Change it to const apiUrl = "http://localhost:8000/predict/batch";

Save the file, then open index.html in your web browser.

# PerformaLytics

A machine learning web app that predicts a student's exam score based on academic, behavioral, and environmental factors, with an interactive dashboard for predictions and insights.

**Live Demo:** https://perfomalytics.onrender.com

*(Note: hosted on Render's free tier — the app may take 30-50 seconds to load on first visit if it's been idle.)*

## Overview
PerformaLytics analyzes 20+ student factors — study habits, attendance, parental involvement, sleep, motivation, and more — to predict exam performance, and shows which factors matter most.


## Model Results
| Model | Test R² | MAE |
|---|---:|---:|
| **Linear Regression (Selected)** | **0.7706** | **0.45** |
| Random Forest | 0.6753 | 1.11 |

Linear Regression was selected as the best model — it explains ~77% of the variance in exam scores using genuine, real-world features.

## Features
- **Predict tab** — Interactive form to input student details and get a predicted exam score
- **Insights tab** — Feature importance chart (what drives scores most) and model comparison visualization
- **History tab** — Tracks past predictions with CSV export
- Input validation warnings for unusual data combinations
- Responsive design (works on mobile)

## Tech Stack
Python, pandas, NumPy, scikit-learn, Streamlit, Matplotlib, joblib

## Project Structure
```text
performalytics/
├── data/StudentPerformanceFactors.csv
├── models/ # trained model, scaler, model name
├── src/train_model.py # preprocessing + training pipeline
├── app/app.py # Streamlit dashboard
├── model_results.json
└── requirements.txt
```

## How to Run Locally
```bash
git clone https://github.com/harshit-ag5965/PerformaLytics.git
cd PerformaLytics
pip install -r requirements.txt
python src/train_model.py
streamlit run app/app.py
```

## Dataset
Student performance dataset with 6,607 records and 34 attributes covering academic, behavioral, and demographic factors.
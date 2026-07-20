import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib, json, os

# Paths (relative to project root — run this script from project root, not from src/)
DATA_PATH = 'data/StudentPerformanceFactors.csv'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Drop identifier columns AND the leaked column (Cumulative_GPA = Exam_Score/10 + 1, don't use it!)
drop_cols = ['Student_ID', 'Student_Name', 'Enrollment_Number', 'Academic_Year',
             'Admission_Date', 'Data_Entry_Date', 'Enrollment_Status',
             'Previous_Scores_Semester_Wise', 'Cumulative_GPA']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

maps = {
    'Parental_Involvement': {'Low': 0, 'Medium': 1, 'High': 2},
    'Access_to_Resources': {'Low': 0, 'Medium': 1, 'High': 2},
    'Motivation_Level': {'Low': 0, 'Medium': 1, 'High': 2},
    'Family_Income': {'Low': 0, 'Medium': 1, 'High': 2},
    'Teacher_Quality': {'Low': 0, 'Medium': 1, 'High': 2},
    'Parental_Education_Level': {'High School': 0, 'College': 1, 'Postgraduate': 2},
    'Distance_from_Home': {'Near': 0, 'Moderate': 1, 'Far': 2},
    'Extracurricular_Activities': {'No': 0, 'Yes': 1},
    'Internet_Access': {'No': 0, 'Yes': 1},
    'Learning_Disabilities': {'No': 0, 'Yes': 1},
    'School_Type': {'Public': 0, 'Private': 1},
    'Gender': {'Male': 0, 'Female': 1},
    'Peer_Influence': {'Negative': 0, 'Neutral': 1, 'Positive': 2},
    'Section': {'A': 0, 'B': 1, 'C': 2, 'D': 3},
}
for col, m in maps.items():
    if col in df.columns:
        df[col] = df[col].map(m)

for col in ['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

df['Study_Efficiency'] = df['Hours_Studied'] * (df['Attendance'] / 100)
df['Support_Index'] = df['Parental_Involvement'] + df['Internet_Access'] + df['Family_Income']

X = df.drop(columns=['Exam_Score'])
y = df['Exam_Score']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, f'{MODELS_DIR}/scaler.pkl')

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
}

results = {}
best_r2 = -np.inf
best_model_name = None
best_model = None

for name, model in models.items():
    if name == 'Linear Regression':
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    results[name] = {'r2': r2, 'mae': mae}
    print(f"{name}: R2={r2:.4f}  MAE={mae:.2f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model = model

print(f"\nBest model: {best_model_name} (R2={best_r2:.4f})")

joblib.dump(best_model, f'{MODELS_DIR}/student_performance_model.pkl')
joblib.dump(best_model_name, f'{MODELS_DIR}/model_name.pkl')

with open('model_results.json', 'w') as f:
    json.dump({'results': results, 'best_model': best_model_name,
               'feature_names': feature_names}, f, indent=2)

print("Saved model, scaler, and results.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime

st.set_page_config(page_title="PerfoMetrics", page_icon="📈", layout="wide")

# ---- Custom CSS ----
st.markdown("""
<style>
    div[data-testid="stMetric"] {
        background: #1A2E2A;
        border: 1px solid #2A5C50;
        border-radius: 8px;
        padding: 14px;
    }
    div[data-testid="stMetricValue"] { color: #4FA891; }
    .stButton>button {
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
    }

    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
        div[data-testid="stHorizontalBlock"] > div {
            width: 100% !important;
        }
        .main .block-container { padding-left: 1rem; padding-right: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

model = joblib.load('models/student_performance_model.pkl')
scaler = joblib.load('models/scaler.pkl')
model_name = joblib.load('models/model_name.pkl')
with open('model_results.json') as f:
    results = json.load(f)

# ---- Header ----
st.markdown("<h1 style='text-align: center;'>PerfoMetrics</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='text-align: center;'>Predicts exam scores using {model_name} — Test R²: {results['results'][model_name]['r2']:.3f}</p>",
    unsafe_allow_html=True
)
st.divider()

tab1, tab2, tab3 = st.tabs(["Predict", "Insights", "History"])

maps = {'Low': 0, 'Medium': 1, 'High': 2}
peer_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
edu_map = {'High School': 0, 'College': 1, 'Postgraduate': 2}
dist_map = {'Near': 0, 'Moderate': 1, 'Far': 2}
yn_map = {'No': 0, 'Yes': 1}
section_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# Defaults used for both initial values and reset
DEFAULTS = {
    'hours_studied': 15, 'attendance': 80, 'class_participation': 50,
    'previous_scores': 60, 'sleep_hours': 7, 'tutoring': 2, 'physical_activity': 3,
    'motivation': "Low", 'parental_involvement': "Low", 'resources': "Low",
    'family_income': "Low", 'teacher_quality': "Low", 'peer_influence': "Neutral",
    'internet': "Yes", 'extracurricular': "No", 'learning_disability': "No",
    'school_type': "Public", 'gender': "Male", 'parental_edu': "High School",
    'distance': "Near", 'grade_level': 2, 'current_semester': 4, 'age': 20, 'section': "A",
}

# Initialize session state with defaults (only once)
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

def reset_form():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val

# ============ TAB 1: PREDICT ============
with tab1:
    header_col, reset_col = st.columns([4, 1])
    with header_col:
        st.subheader("Enter Student Details")
    with reset_col:
        st.button("Reset Form", on_click=reset_form, use_container_width=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**Study Habits**")
        hours_studied = st.slider("Hours Studied per week", 0, 50, key='hours_studied')
        attendance = st.slider("Attendance (%)", 60, 100, key='attendance')
        class_participation = st.slider("Class Participation Score", 0, 100, key='class_participation')
        previous_scores = st.slider("Previous Exam Score", 0, 100, key='previous_scores')
        sleep_hours = st.slider("Sleep Hours", 4, 10, key='sleep_hours')
        tutoring = st.slider("Tutoring Sessions per month", 0, 10, key='tutoring')
        physical_activity = st.slider("Physical Activity (hrs/week)", 0, 10, key='physical_activity')

    with col2:
        st.markdown("**Background & Support**")
        motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"], key='motivation')
        parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"], key='parental_involvement')
        resources = st.selectbox("Access to Resources", ["Low", "Medium", "High"], key='resources')
        family_income = st.selectbox("Family Income", ["Low", "Medium", "High"], key='family_income')
        teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"], key='teacher_quality')
        peer_influence = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"], key='peer_influence')
        internet = st.selectbox("Internet Access", ["Yes", "No"], key='internet')
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"], key='extracurricular')
        learning_disability = st.selectbox("Learning Disabilities", ["No", "Yes"], key='learning_disability')
        school_type = st.selectbox("School Type", ["Public", "Private"], key='school_type')
        gender = st.selectbox("Gender", ["Male", "Female"], key='gender')
        parental_edu = st.selectbox("Parental Education", ["High School", "College", "Postgraduate"], key='parental_edu')
        distance = st.selectbox("Distance from Home", ["Near", "Moderate", "Far"], key='distance')
        grade_level = st.slider("Grade Level", 1, 4, key='grade_level')
        current_semester = st.slider("Current Semester", 1, 8, key='current_semester')
        age = st.slider("Age", 15, 25, key='age')
        section = st.selectbox("Section", ["A", "B", "C", "D"], key='section')

    st.divider()
    predict_clicked = st.button("Predict Exam Score", type="primary", use_container_width=True)

    if predict_clicked:
        warnings_list = []
        if hours_studied > 40 and sleep_hours < 5:
            warnings_list.append("⚠️ Very high study hours with very low sleep — results may be less reliable (unusual combination in training data).")
        if attendance < 65 and hours_studied > 35:
            warnings_list.append("⚠️ Low attendance with very high study hours is a rare combination in the data — prediction confidence may be lower.")
        for w in warnings_list:
            st.warning(w)

        row = {
            'Grade_Level': grade_level,
            'Current_Semester': current_semester,
            'Age': age,
            'Section': section_map[section],
            'Class_Participation_Score': class_participation,
            'Hours_Studied': hours_studied,
            'Attendance': attendance,
            'Parental_Involvement': maps[parental_involvement],
            'Access_to_Resources': maps[resources],
            'Extracurricular_Activities': yn_map[extracurricular],
            'Sleep_Hours': sleep_hours,
            'Previous_Scores': previous_scores,
            'Motivation_Level': maps[motivation],
            'Internet_Access': yn_map[internet],
            'Tutoring_Sessions': tutoring,
            'Family_Income': maps[family_income],
            'Teacher_Quality': maps[teacher_quality],
            'School_Type': 0 if school_type == 'Public' else 1,
            'Peer_Influence': peer_map[peer_influence],
            'Physical_Activity': physical_activity,
            'Learning_Disabilities': yn_map[learning_disability],
            'Parental_Education_Level': edu_map[parental_edu],
            'Distance_from_Home': dist_map[distance],
            'Gender': 0 if gender == 'Male' else 1,
        }
        row['Study_Efficiency'] = hours_studied * (attendance / 100)
        row['Support_Index'] = maps[parental_involvement] + yn_map[internet] + maps[family_income]

        feature_names = results['feature_names']
        X_input = pd.DataFrame([row])[feature_names]

        if model_name == 'Linear Regression':
            X_input = scaler.transform(X_input)

        pred = model.predict(X_input)[0]
        pred = max(0, min(100, pred))

        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric("Predicted Exam Score", f"{pred:.1f} / 100")
        with res_col2:
            if pred >= 80:
                st.success("Excellent performance predicted!")
            elif pred >= 60:
                st.info("Good performance — some room to improve.")
            else:
                st.warning("At risk — consider more study hours and attendance.")

        history_file = 'prediction_history.csv'
        new_entry = row.copy()
        new_entry['Predicted_Score'] = round(pred, 1)
        new_entry['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            history_df = pd.concat([history_df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            history_df = pd.DataFrame([new_entry])
        history_df.to_csv(history_file, index=False)

# ============ TAB 2: INSIGHTS ============
with tab2:
    st.subheader("What Drives Exam Scores?")

    if model_name == 'Linear Regression':
        importances = np.abs(model.coef_)
    else:
        importances = model.feature_importances_

    feature_names = results['feature_names']
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    imp_df = imp_df.sort_values('Importance', ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.barh(imp_df['Feature'], imp_df['Importance'], color='#2A9D8F')
    ax.set_xlabel("Relative Importance", color='white')
    ax.set_title("Top 10 Factors Influencing Exam Score", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    st.pyplot(fig)

    st.caption("Higher bars = stronger influence on the predicted exam score.")

    st.divider()
    st.subheader("Model Comparison")

    model_comp = pd.DataFrame([
        {'Model': name, 'R2': v['r2'], 'MAE': v['mae']}
        for name, v in results['results'].items()
    ])

    fig2, (ax_r2, ax_mae) = plt.subplots(1, 2, figsize=(9, 4))
    for f in [fig2]:
        f.patch.set_alpha(0)

    colors = ['#2A9D8F' if m == model_name else '#555555' for m in model_comp['Model']]

    ax_r2.set_facecolor('none')
    ax_r2.bar(model_comp['Model'], model_comp['R2'], color=colors)
    ax_r2.set_title("Test R² (higher is better)", color='white', fontsize=10)
    ax_r2.tick_params(colors='white', labelsize=8)
    ax_r2.set_ylim(0, 1)
    for spine in ax_r2.spines.values():
        spine.set_color('white')

    ax_mae.set_facecolor('none')
    ax_mae.bar(model_comp['Model'], model_comp['MAE'], color=colors)
    ax_mae.set_title("Test MAE (lower is better)", color='white', fontsize=10)
    ax_mae.tick_params(colors='white', labelsize=8)
    for spine in ax_mae.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    st.pyplot(fig2)

    st.caption(f"Highlighted bar ({model_name}) is the model currently used for predictions.")

# ============ TAB 3: HISTORY ============
with tab3:
    st.subheader("Prediction History")

    history_file = 'prediction_history.csv'
    if os.path.exists(history_file):
        history_df = pd.read_csv(history_file)
        st.dataframe(history_df.tail(10)[::-1], use_container_width=True)

        csv_data = history_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Full History (CSV)", csv_data, "prediction_history.csv", "text/csv")
    else:
        st.caption("No predictions made yet — try one in the Predict tab!")
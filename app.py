import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="🎓 Student GPA Predictor", page_icon="📚")

st.title("🎓 Student GPA Predictor")
st.markdown("Predict GPA based on sleep, screen time, and mental health")

# โหลดโมเดล
@st.cache_resource
def load_model():
    model = joblib.load('student_gpa_model.joblib')
    columns = joblib.load('feature_columns.joblib')
    return model, columns

try:
    model, expected_columns = load_model()
except:
    st.error("❌ Model files not found! Please run train_model.py first.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Student Information")

age = st.sidebar.number_input("Age", min_value=10, max_value=50, value=18)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
education_level = st.sidebar.selectbox("Education Level", ["High School", "Undergraduate", "Graduate"])
avg_sleep_hours = st.sidebar.slider("Avg Sleep Hours", 3.0, 12.0, 7.0, 0.1)
screen_time_hours = st.sidebar.slider("Screen Time Hours", 0.0, 15.0, 5.0, 0.1)
social_media_hours = st.sidebar.slider("Social Media Hours", 0.0, 12.0, 2.0, 0.1)
study_hours_per_day = st.sidebar.slider("Study Hours/Day", 0.0, 12.0, 3.0, 0.1)
exercise_hours_per_week = st.sidebar.slider("Exercise Hours/Week", 0.0, 15.0, 3.0, 0.1)
caffeine_drinks_per_day = st.sidebar.number_input("Caffeine Drinks/Day", min_value=0, max_value=10, value=1)
stress_level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 5)
anxiety_score = st.sidebar.slider("Anxiety Score (1-10)", 1, 10, 5)
uses_sleep_app = st.sidebar.selectbox("Uses Sleep App", ["No", "Yes"]) == "Yes"
feels_burned_out = st.sidebar.selectbox("Feels Burned Out", ["No", "Yes"]) == "Yes"

if st.sidebar.button("🔮 Predict GPA", type="primary"):
    input_data = pd.DataFrame([{
        'age': age, 'gender': gender, 'education_level': education_level,
        'avg_sleep_hours': avg_sleep_hours, 'screen_time_hours': screen_time_hours,
        'social_media_hours': social_media_hours, 'study_hours_per_day': study_hours_per_day,
        'exercise_hours_per_week': exercise_hours_per_week, 'caffeine_drinks_per_day': caffeine_drinks_per_day,
        'stress_level': stress_level, 'anxiety_score': anxiety_score,
        'uses_sleep_app': uses_sleep_app, 'feels_burned_out': feels_burned_out
    }])
    
    input_data = input_data[expected_columns]
    prediction = model.predict(input_data)[0]
    
    st.success(f"### Predicted GPA: **{prediction:.2f}**")
    
    if prediction >= 3.5:
        st.info("🌟 Excellent! Keep it up!")
    elif prediction >= 3.0:
        st.info("👍 Good job!")
    elif prediction >= 2.5:
        st.info(" Room for improvement")
    else:
        st.info("💪 Consider adjusting study habits")
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="🎓 Student GPA Predictor", page_icon="📚")

st.title(" Student GPA Predictor")
st.markdown("Predict GPA based on sleep, screen time, and mental health")

# โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        # ตรวจสอบว่าไฟล์มีอยู่จริง
        model_path = 'student_gpa_model.joblib'
        columns_path = 'feature_columns.joblib'
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None, None
            
        if not os.path.exists(columns_path):
            st.error(f"Columns file not found: {columns_path}")
            return None, None
        
        model = joblib.load(model_path)
        columns = joblib.load(columns_path)
        return model, columns
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, expected_columns = load_model()

if model is None:
    st.stop()

# ส่วนที่เหลือของโค้ด...
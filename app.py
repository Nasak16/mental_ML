import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="🎓 Student GPA Predictor", page_icon="📚", layout="centered")

st.title("🎓 ระบบทำนายเกรดเฉลี่ย (GPA)")
st.markdown("""
ทำนาย GPA จากพฤติกรรมการนอน การใช้หน้าจอ และสุขภาพจิต  
ใช้โมเดล **Support Vector Regression (SVR)**
""")
st.markdown("---")

# โหลดโมเดล
@st.cache_resource
def load_model():
    try:
        model = joblib.load('student_gpa_model.joblib')
        columns = joblib.load('feature_columns.joblib')
        return model, columns
    except Exception as e:
        st.error(f"⚠️ ไม่สามารถโหลดโมเดลได้: {str(e)}")
        return None, None

model, expected_columns = load_model()

if model is None:
    st.stop()

# Sidebar สำหรับกรอกข้อมูล
st.sidebar.header("📝 กรอกข้อมูลของนักเรียน")

# รับข้อมูลจากผู้ใช้
age = st.sidebar.number_input("อายุ (ปี)", min_value=10, max_value=50, value=18, step=1)
gender = st.sidebar.selectbox("เพศ", ["Male", "Female", "Non-binary", "Prefer not to say"])
education_level = st.sidebar.selectbox("ระดับการศึกษา", ["High School", "Undergraduate", "Graduate"])
avg_sleep_hours = st.sidebar.slider("ชั่วโมงนอนเฉลี่ย/วัน", 3.0, 12.0, 7.0, 0.1)
screen_time_hours = st.sidebar.slider("ชั่วโมงใช้หน้าจอ/วัน", 0.0, 15.0, 5.0, 0.1)
social_media_hours = st.sidebar.slider("ชั่วโมงโซเชียลมีเดีย/วัน", 0.0, 12.0, 2.0, 0.1)
study_hours_per_day = st.sidebar.slider("ชั่วโมงเรียน/วัน", 0.0, 12.0, 3.0, 0.1)
exercise_hours_per_week = st.sidebar.slider("ชั่วโมงออกกำลังกาย/สัปดาห์", 0.0, 15.0, 3.0, 0.1)
caffeine_drinks_per_day = st.sidebar.number_input("เครื่องดื่มคาเฟอีน/วัน", min_value=0, max_value=10, value=1, step=1)
stress_level = st.sidebar.slider("ระดับความเครียด (1-10)", 1, 10, 5, 1)
anxiety_score = st.sidebar.slider("คะแนนความกังวล (1-10)", 1, 10, 5, 1)
uses_sleep_app = st.sidebar.selectbox("ใช้แอปติดตามการนอน", ["No", "Yes"]) == "Yes"
feels_burned_out = st.sidebar.selectbox("รู้สึกหมดไฟ (Burnout)", ["No", "Yes"]) == "Yes"

# ปุ่มทำนาย
if st.sidebar.button("🔮 ทำนาย GPA", type="primary"):
    # สร้าง DataFrame
    input_data = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'education_level': education_level,
        'avg_sleep_hours': avg_sleep_hours,
        'screen_time_hours': screen_time_hours,
        'social_media_hours': social_media_hours,
        'study_hours_per_day': study_hours_per_day,
        'exercise_hours_per_week': exercise_hours_per_week,
        'caffeine_drinks_per_day': caffeine_drinks_per_day,
        'stress_level': stress_level,
        'anxiety_score': anxiety_score,
        'uses_sleep_app': uses_sleep_app,
        'feels_burned_out': feels_burned_out
    }])
    
    # เรียงคอลัมน์ให้ถูกต้อง
    input_data = input_data[expected_columns]
    
    # ทำนาย
    prediction = model.predict(input_data)[0]
    
    # แสดงผล
    st.success(f"### 🎉 GPA ที่ทำนายได้: **{prediction:.2f}**")
    
    # แปลงเป็นเกรด
    if prediction >= 3.5:
        grade = "A / A-"
        emoji = "🌟"
    elif prediction >= 3.0:
        grade = "B+ / B"
        emoji = "👍"
    elif prediction >= 2.5:
        grade = "C+ / C"
        emoji = "📚"
    elif prediction >= 2.0:
        grade = "D+ / D"
        emoji = "️"
    else:
        grade = "F"
        emoji = ""
    
    st.info(f"{emoji} เกรดที่คาดการณ์: **{grade}**")
    
    with st.expander("📋 ดูข้อมูลที่กรอก"):
        st.dataframe(input_data.T, column_config={"0": "ค่า"})

# ข้อมูลเพิ่มเติม
st.markdown("---")
st.markdown("### 📊 เกี่ยวกับโมเดล")
st.markdown("""
- **อัลกอริทึม**: Support Vector Regression (SVR)
- **Features**: 13 ตัวแปร (อายุ, การนอน, การใช้หน้าจอ, ความเครียด, ฯลฯ)
- **ข้อมูลฝึก**: Student Sleep & Mental Health Dataset (3,000 records)
""")
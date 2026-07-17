import streamlit as st
import pandas as pd
import joblib

# โหลดโมเดลและรายการคอลัมน์
@st.cache_resource
def load_model():
    model = joblib.load('student_gpa_model.joblib')
    columns = joblib.load('feature_columns.joblib')
    return model, columns

model, expected_columns = load_model()

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="Student GPA Predictor", page_icon="🎓", layout="centered")
st.title("🎓 ระบบทำนายเกรดเฉลี่ย (GPA) ของนักเรียน/นักศึกษา")
st.markdown("ทำนาย GPA จากพฤติกรรมการนอน การใช้หน้าจอ และสุขภาพจิต โดยใช้โมเดล **Support Vector Regression (SVR)**")
st.markdown("---")

# --- สร้างฟอร์มรับข้อมูลด้านข้าง (Sidebar) ---
st.sidebar.header("📊 กรุณากรอกข้อมูลของนักเรียน")

age = st.sidebar.number_input("อายุ (Age)", min_value=10, max_value=40, value=18, step=1)
gender = st.sidebar.selectbox("เพศ (Gender)", ["Male", "Female", "Non-binary", "Prefer not to say"])
education_level = st.sidebar.selectbox("ระดับการศึกษา (Education Level)", ["High School", "Undergraduate", "Graduate"])
avg_sleep_hours = st.sidebar.slider("ชั่วโมงนอนเฉลี่ยต่อวัน (Avg Sleep Hours)", 4.0, 12.0, 7.0, 0.1)
screen_time_hours = st.sidebar.slider("เวลาใช้หน้าจอต่อวัน (Screen Time Hours)", 0.0, 15.0, 5.0, 0.1)
social_media_hours = st.sidebar.slider("เวลาใช้โซเชียลมีเดียต่อวัน (Social Media Hours)", 0.0, 15.0, 2.0, 0.1)
study_hours_per_day = st.sidebar.slider("ชั่วโมงเรียน/อ่านหนังสือต่อวัน (Study Hours)", 0.0, 12.0, 3.0, 0.1)
exercise_hours_per_week = st.sidebar.slider("ชั่วโมงออกกำลังกายต่อสัปดาห์ (Exercise Hours)", 0.0, 15.0, 3.0, 0.1)
caffeine_drinks_per_day = st.sidebar.number_input("จำนวนเครื่องดื่มคาเฟอีนต่อวัน (Caffeine Drinks)", min_value=0, max_value=10, value=1, step=1)
stress_level = st.sidebar.slider("ระดับความเครียด (1-10) (Stress Level)", 1, 10, 5, 1)
anxiety_score = st.sidebar.slider("คะแนนความวิตกกังวล (1-10) (Anxiety Score)", 1, 10, 5, 1)
uses_sleep_app = st.sidebar.selectbox("ใช้แอปติดตามการนอนหรือไม่? (Uses Sleep App)", ["Yes", "No"]) == "Yes"
feels_burned_out = st.sidebar.selectbox("รู้สึกหมดไฟ/Burnout หรือไม่? (Feels Burned Out)", ["Yes", "No"]) == "Yes"

# --- ปุ่มทำนาย ---
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 ทำนาย GPA")

# --- ประมวลผลและแสดงผล ---
if predict_button:
    # สร้าง DataFrame จากข้อมูลที่ผู้ใช้กรอก (ต้องเรียงชื่อคอลัมน์ให้ตรงกับตอนเทรน)
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
    
    # เรียงคอลัมน์ให้ตรงกับที่โมเดลคาดหวัง
    input_data = input_data[expected_columns]
    
    # ทำนายผล
    prediction = model.predict(input_data)[0]
    
    # แสดงผลลัพธ์
    st.success(f"### 🎉 GPA ที่ทำนายได้: {prediction:.2f}")
    
    # แปลง GPA เป็นเกรดอักษรคร่าวๆ (ตามระบบสากล/ไทย)
    if prediction >= 3.5:
        grade = "A / A-"
    elif prediction >= 3.0:
        grade = "B+ / B"
    elif prediction >= 2.5:
        grade = "C+ / C"
    elif prediction >= 2.0:
        grade = "D+ / D"
    else:
        grade = "F"
        
    st.info(f"💡 เทียบเท่าเกรดอักษรประมาณ: **{grade}**")
    
    with st.expander("🔍 ดูข้อมูลที่คุณกรอก"):
        st.dataframe(input_data.T, column_config={"0": "Value"})
else:
    st.warning("👈 กรุณากรอกข้อมูลที่แถบด้านข้างแล้วกดปุ่มทำนาย")
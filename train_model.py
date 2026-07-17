import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score

print("📊 Loading data...")
df = pd.read_csv('student_sleep_mental_health_2026.csv')

# แยก Features และ Target
X = df.drop(columns=['student_id', 'gpa'])
y = df['gpa']

# กำหนด categorical และ numerical columns
categorical_features = ['gender', 'education_level', 'uses_sleep_app', 'feels_burned_out']
numerical_features = ['age', 'avg_sleep_hours', 'screen_time_hours', 'social_media_hours', 
                      'study_hours_per_day', 'exercise_hours_per_week', 'caffeine_drinks_per_day', 
                      'stress_level', 'anxiety_score']

# สร้าง Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้าง Pipeline
print("🤖 Training SVR model...")
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf', C=1.0, epsilon=0.1))
])

# ฝึกโมเดล
model_pipeline.fit(X_train, y_train)

# ทดสอบโมเดล
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n✅ Model Performance:")
print(f"   MAE: {mae:.4f}")
print(f"   R²:  {r2:.4f}")

# บันทึกโมเดล
joblib.dump(model_pipeline, 'student_gpa_model.joblib')
joblib.dump(list(X.columns), 'feature_columns.joblib')

print("\n💾 Model saved successfully!")
print("   - student_gpa_model.joblib")
print("   - feature_columns.joblib")
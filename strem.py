import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ====================
# Load Pipeline & Model
# ====================
def bin_experience(x):
    bins = np.array(x).astype(int).ravel()
    labels = []
    for v in bins:
        if v <=2:
            labels.append("Junior")
        elif v <=5:
            labels.append("Mid")
        else:
            labels.append("Senior")
    return np.array(labels).reshape(-1,1)

loaded_pipeline = joblib.load("preprocesor.pkl")

loaded_model = joblib.load("best_xgboost_model.pkl")

# ====================
# Judul Aplikasi
# ====================
st.title("Hiring Prediction App")
st.write("Masukkan data kandidat, lalu sistem akan memprediksi apakah kandidat akan diterima atau tidak.")

# ====================
# Form Input
# ====================

# (Isi sesuai kolom X_train Anda, selain Age, DistanceFromCompany, Gender, PreviousCompanies, HiringDecision)
# Misal fitur yang masih tersisa: ["EducationLevel", "SkillScore", "InterviewScore", "ExperienceYears"]

RecruitmentStrategy = st.selectbox(
    "Recruitment Strategy", 
    options=[1, 2, 3], 
    format_func=lambda x: {1:"Agresif", 2:"Moderat", 3:"Konservatif"}[x]
)
EducationLevel = st.selectbox(
    "Education Level",
    options=[1, 2, 3, 4],
    format_func=lambda x: {1:"High School", 2:"Bachelor", 3:"Master", 4:"PhD"}[x]
)
experience_years = st.number_input("Experience (Years)", min_value=0, max_value=40, value=2,step=1)
interview_score = st.slider("Interview Score", 0, 100, 50)
skill_score = st.slider("Skill Score", 0, 100, 50)
PersonalityScore = st.slider("Personality Score", 0, 100, 50)


# ====================
# Dataframe Input Baru
# ====================
new_data = pd.DataFrame({
    "RecruitmentStrategy": [RecruitmentStrategy],
    "EducationLevel": [EducationLevel],
    "ExperienceYears": [experience_years],    
    "InterviewScore": [interview_score],
    "SkillScore": [skill_score],
    "PersonalityScore": [PersonalityScore]
    
   
})

st.write("### Data Kandidat")
st.dataframe(new_data)

# ====================
# Prediksi
# ====================
if st.button("Prediksi"):
    # Transformasi dengan pipeline
    new_data_transformed = loaded_pipeline.transform(new_data)

    # Prediksi
    prediction = loaded_model.predict(new_data_transformed)[0]
    prediction_proba = loaded_model.predict_proba(new_data_transformed)[0][1]

    # Tampilkan Hasil
    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.success(f"Kandidat **DITERIMA** ✅ (Probabilitas: {prediction_proba:.2f})")
    else:
        st.error(f"Kandidat **TIDAK DITERIMA** ❌ (Probabilitas: {prediction_proba:.2f})")

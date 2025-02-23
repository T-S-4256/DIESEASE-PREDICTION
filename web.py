import os
import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Prediction Of Disease Outbreak", layout="wide", page_icon="doctor"
)

# Load the trained models
diabetes_model = pickle.load(open(r"saved_demo_model\diabetes_model.sav", "rb"))
heart_model = pickle.load(open(r"saved_demo_model\heart_model.sav", "rb"))
parkinsons_model = pickle.load(open(r"saved_demo_model\parkinsons_model.sav", "rb"))

# Load scalers
diabetes_scaler = pickle.load(open(r"saved_demo_model\Di_scaler.sav", "rb"))
heart_scaler = pickle.load(open(r"saved_demo_model\heart_scaler.sav", "rb"))  # FIXED
parkinsons_scaler=pickle.load(open(r"saved_demo_model\parkinsons_scaler.sav","rb"))
# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        "Prediction Of Disease Outbreak",
        ["Diabetes Prediction", "Heart Disease Prediction", "Parkinsons Disease Prediction"],
        menu_icon="hospital-fill",
        icons=["activity", "heart", "person"],
        default_index=0,
    )

# ******************************************************************
# Diabetes Prediction
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction Using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input("No Of Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure Value")
    with col1:
        SkinThickness = st.text_input("Skin Thickness Value")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        BMI = st.text_input("BMI Value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    with col2:
        Age = st.text_input("Age Of The Person")

    Diabetes_diagnosis = ""

    if st.button("Diabetes Test Result"):
        try:
            input_data = np.array([
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age),
            ]).reshape(1, -1)

            # Apply StandardScaler (IMPORTANT FIX)
            input_data_scaled = diabetes_scaler.transform(input_data)  

            # Make prediction
            diabetes_prediction = diabetes_model.predict(input_data_scaled)

            Diabetes_diagnosis = "The Person Has Diabetes" if diabetes_prediction[0] == 1 else "The Person Does Not Have Diabetes"

        except ValueError:
            Diabetes_diagnosis = "Invalid input! Please enter numerical values."

    st.success(Diabetes_diagnosis)


# ******************************************************************
# Heart Disease Prediction
if selected == "Heart Disease Prediction":  # FIXED SPELLING
    st.title("Heart Disease Prediction Using ML")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input("Age Of The Person")
    with col2:
        sex = st.text_input("Sex (0 = Female, 1 = Male)")
    with col3:
        cp = st.text_input("Chest Pain Type (0-3)")
    with col1:
        trestbps = st.text_input("Resting Blood Pressure")
    with col2:
        chol = st.text_input("Serum Cholesterol (mg/dl)")
    with col3:
        fbs = st.text_input("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)")
    with col1:
        restecg = st.text_input("Resting ECG Results (0-2)")
    with col2:
        thalach = st.text_input("Maximum Heart Rate Achieved")
    with col3:
        exang = st.text_input("Exercise-Induced Angina (1 = Yes, 0 = No)")
    with col1:
        oldpeak = st.text_input("ST Depression Induced by Exercise")
    with col2:
        slope = st.text_input("Slope of the Peak Exercise ST Segment (0-2)")
    with col3:
        ca = st.text_input("Number of Major Vessels Colored by Fluoroscopy (0-3)")
    with col1:
        thal = st.text_input("Thalassemia (0-3)")

    heart_diagnosis = ""  # FIXED VARIABLE NAME

    if st.button("Heart Disease Test Result"):
        try:
            user_input = np.array([
                float(age),
                float(sex),
                float(cp),
                float(trestbps),
                float(chol),
                float(fbs),
                float(restecg),
                float(thalach),
                float(exang),
                float(oldpeak),
                float(slope),
                float(ca),
                float(thal),
            ]).reshape(1, -1)

            # Scale the input using heart_scaler (FIXED)
            user_input_scaled = heart_scaler.transform(user_input)

            # Predict using the heart disease model
            heart_prediction = heart_model.predict(user_input_scaled)

            heart_diagnosis = "The Person Has Heart Disease" if heart_prediction[0] == 1 else "The Person Does Not Have Heart Disease"

        except ValueError:
            heart_diagnosis = "Invalid input! Please enter numerical values."

    st.success(heart_diagnosis)


# *******************************************************************
# Parkinson's Disease Prediction

if selected == "Parkinsons Disease Prediction":
    st.title("Parkinson's Disease Prediction Using ML")


    col1, col2, col3 = st.columns(3)
    
    with col1:
        MDVP_Fo = st.text_input("MDVP:Fo (Fundamental Frequency in Hz)")
    with col2:
        MDVP_Fhi = st.text_input("MDVP:Fhi (Highest Frequency in Hz)")
    with col3:
        MDVP_Flo = st.text_input("MDVP:Flo (Lowest Frequency in Hz)")
    with col1:
        MDVP_Jitter = st.text_input("MDVP:Jitter (Absolute Jitter in %)")
    with col2:
        MDVP_Jitter_Abs = st.text_input("MDVP:Jitter (Absolute Jitter in ms)")
    with col3:
        MDVP_RAP = st.text_input("MDVP:RAP (Relative Amplitude Perturbation)")
    with col1:
        MDVP_PPQ = st.text_input("MDVP:PPQ (Five-Point Period Perturbation Quotient)")
    with col2:
        Jitter_DDP = st.text_input("Jitter:DDP (Three-Point Period Perturbation Quotient)")
    with col3:
        MDVP_Shimmer = st.text_input("MDVP:Shimmer (Shimmer in %)")
    with col1:
        MDVP_Shimmer_dB = st.text_input("MDVP:Shimmer (Shimmer in dB)")
    with col2:
        Shimmer_APQ3 = st.text_input("Shimmer:APQ3 (Amplitude Perturbation Quotient)")
    with col3:
        Shimmer_APQ5 = st.text_input("Shimmer:APQ5 (Amplitude Perturbation Quotient)")
    with col1:
        MDVP_APQ = st.text_input("MDVP:APQ (Amplitude Perturbation Quotient)")
    with col2:
        Shimmer_DDA = st.text_input("Shimmer:DDA (Three-Point Amplitude Perturbation Quotient)")
    with col3:
        NHR = st.text_input("NHR (Noise-to-Harmonics Ratio)")
    with col1:
        HNR = st.text_input("HNR (Harmonics-to-Noise Ratio)")
    with col2:
        RPDE = st.text_input("RPDE (Recurrence Period Density Entropy)")
    with col3:
        DFA = st.text_input("DFA (Detrended Fluctuation Analysis)")
    with col1:
        spread1 = st.text_input("Spread1 (Nonlinear Measures)")
    with col2:
        spread2 = st.text_input("Spread2 (Nonlinear Measures)")
    with col3:
        D2 = st.text_input("D2 (Correlation Dimension)")
    with col1:
        PPE = st.text_input("PPE (Pitch Period Entropy)")

    parkinsons_diagnosis = ""

    if st.button("Parkinson's Test Result"):
        try:
            input_data = np.array([
                float(MDVP_Fo), float(MDVP_Fhi), float(MDVP_Flo),
                float(MDVP_Jitter), float(MDVP_Jitter_Abs), float(MDVP_RAP),
                float(MDVP_PPQ), float(Jitter_DDP), float(MDVP_Shimmer),
                float(MDVP_Shimmer_dB), float(Shimmer_APQ3), float(Shimmer_APQ5),
                float(MDVP_APQ), float(Shimmer_DDA), float(NHR),
                float(HNR), float(RPDE), float(DFA),
                float(spread1), float(spread2), float(D2), float(PPE)
            ]).reshape(1, -1)

            # Apply StandardScaler before making prediction
            input_data_scaled = parkinsons_scaler.transform(input_data)

            # Predict using the trained Parkinson’s model
            parkinsons_prediction = parkinsons_model.predict(input_data_scaled)

            parkinsons_diagnosis = (
                "The Person Has Parkinson's Disease" if parkinsons_prediction[0] == 1 
                else "The Person Does Not Have Parkinson's Disease"
            )

        except ValueError:
            parkinsons_diagnosis = "Invalid input! Please enter numerical values."

    st.success(parkinsons_diagnosis)

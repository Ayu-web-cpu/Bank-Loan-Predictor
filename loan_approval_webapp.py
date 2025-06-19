import streamlit as st
import numpy as np
import joblib

# Load model, scaler, encoder
rf = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('education_encoder.pkl')

st.title("Bank Loan Approval Predictor")

st.write("Enter applicant details:")

age = st.slider("Age", 21, 60, 30)
income = st.number_input("Annual Income", min_value=20000, max_value=100000000, value=50000, step=1000)  # Up to 10 crore
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
credit_score = st.slider("Credit Score", 300, 900, 600)

if st.button("Predict Loan Approval"):
    education_encoded = le.transform([education])[0]
    X_new = np.array([[age, income, education_encoded, credit_score]])
    X_new_scaled = scaler.transform(X_new)
    prediction = rf.predict(X_new_scaled)[0]
    prob = rf.predict_proba(X_new_scaled)[0][1]
    if prediction == 1:
        st.success(f"Loan Approved! (Probability: {prob:.2f})")
    else:
        st.error(f"Loan Not Approved. (Probability: {prob:.2f})")
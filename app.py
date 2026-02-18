import streamlit as st
import joblib
import numpy as np
import os

# Load model and features with better error handling
try:
    model = joblib.load("hustler_fund_default_model.pkl")
    features = joblib.load("model_features.pkl")
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("Hustler Fund Loan Default Risk Predictor")

st.write("Enter borrower details below:")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=70)
income = st.number_input("Monthly Income (KES)", min_value=1000)
loan_amount = st.number_input("Loan Amount (KES)", min_value=500)
repayment_period = st.number_input("Repayment Period (Days)", min_value=7)
previous_default = st.selectbox("Previous Default", [0, 1])

# Create input array
input_data = np.array([[age, income, loan_amount, repayment_period, previous_default]])

if st.button("Predict Default Risk"):
    try:
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        if prediction[0] == 1:
            st.error(f"High Risk of Default ⚠️ (Probability: {probability:.2f})")
        else:
            st.success(f"Low Risk of Default ✅ (Probability: {probability:.2f})")
    except Exception as e:
        st.error(f"Prediction error: {e}")

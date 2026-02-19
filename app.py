import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and features with better error handling
try:
    model = joblib.load(os.path.join(BASE_DIR, "hustler_fund_default_model.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "model_features.pkl"))
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

# Create input dataframe with the 5 raw features
input_df = pd.DataFrame({
    'age': [age],
    'income': [income],
    'loan_amount': [loan_amount],
    'repayment_period': [repayment_period],
    'previous_default': [previous_default]
})

if st.button("Predict Default Risk"):
    try:
        # Get expected features from the model
        expected_features = features if isinstance(features, list) else list(features)
        
        # Debug info
        st.write(f"Model expects {len(expected_features)} features: {expected_features}")
        st.write(f"Input data has {len(input_df.columns)} features: {list(input_df.columns)}")
        
        # Create array with all expected features
        # If features are missing, fill with 0
        input_array = np.zeros((1, len(expected_features)))
        
        for i, feature in enumerate(expected_features):
            if feature in input_df.columns:
                input_array[0, i] = input_df[feature].values[0]
        
        # Make prediction
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)[0][1]

        if prediction[0] == 1:
            st.error(f"High Risk of Default ⚠️ (Probability: {probability:.2f})")
        else:
            st.success(f"Low Risk of Default ✅ (Probability: {probability:.2f})")
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write(f"Debug - Features loaded: {features}")
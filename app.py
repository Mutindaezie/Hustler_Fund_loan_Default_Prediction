import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "hustler_fund_default_model.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "model_features.pkl"))
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.title("Hustler Fund Loan Default Risk Predictor")
st.write("Enter borrower details below:")

# Create two columns for better UI
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    county = st.text_input("County", value="Nairobi")
    monthly_income = st.number_input("Monthly Income (KES)", min_value=0, value=50000)
    loan_amount = st.number_input("Loan Amount (KES)", min_value=100, value=10000)

with col2:
    loan_term_days = st.number_input("Loan Term (Days)", min_value=7, value=30)
    loan_purpose = st.selectbox("Loan Purpose", ["Business", "Personal", "Education", "Other"])
    mpesa_transactions = st.number_input("M-Pesa Transactions (Monthly)", min_value=0, value=5)
    mpesa_volume = st.number_input("M-Pesa Volume (KES)", min_value=0, value=50000)
    previous_loans = st.number_input("Previous Loans", min_value=0, value=0)

# Bottom section
previous_defaults = st.number_input("Previous Defaults", min_value=0, value=0)
repayment_score = st.slider("Repayment Score (0-100)", min_value=0, max_value=100, value=75)
credit_score = st.slider("Credit Score (0-1000)", min_value=0, max_value=1000, value=600)

if st.button("Predict Default Risk", key="predict_btn"):
    try:
        # Create DataFrame with categorical columns for OneHotEncoder
        input_df = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'County': [county],
            'Monthly_Income': [monthly_income],
            'Loan_Amount': [loan_amount],
            'Loan_Term_Days': [loan_term_days],
            'Loan_Purpose': [loan_purpose],
            'Mpesa_Transactions': [mpesa_transactions],
            'Mpesa_Volume': [mpesa_volume],
            'Previous_Loans': [previous_loans],
            'Previous_Defaults': [previous_defaults],
            'Repayment_Score': [repayment_score],
            'Credit_Score': [credit_score]
        })

        # Expected features after OneHotEncoding
        expected_features = features if isinstance(features, list) else list(features)

        # Identify categorical and numeric columns based on expected features
        categorical_features = ['Gender', 'County', 'Loan_Purpose']
        numeric_features = ['Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Term_Days',
                           'Mpesa_Transactions', 'Mpesa_Volume', 'Previous_Loans',
                           'Previous_Defaults', 'Repayment_Score', 'Credit_Score']

        # Separate categorical and numeric data
        cat_data = input_df[categorical_features]
        num_data = input_df[numeric_features]

        # Apply OneHotEncoding to categorical features
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(cat_data)

        # Get encoded feature names
        cat_feature_names = encoder.get_feature_names_out(categorical_features)

        # Create DataFrame with encoded categorical features
        cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names)

        # Combine numeric and encoded categorical features
        processed_df = pd.concat([num_data.reset_index(drop=True), cat_df], axis=1)

        # Reorder columns to match model's expected features
        processed_df = processed_df.reindex(columns=expected_features, fill_value=0)

        # Convert to array for prediction
        input_array = processed_df.values

        # Make prediction
        prediction = model.predict(input_array)
        probability = model.predict_proba(input_array)[0][1]

        st.divider()
        if prediction[0] == 1:
            st.error(f"⚠️ HIGH RISK OF DEFAULT")
            st.write(f"Default Probability: {probability:.1%}")
        else:
            st.success(f"✅ LOW RISK OF DEFAULT")
            st.write(f"Default Probability: {probability:.1%}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.write(traceback.format_exc())

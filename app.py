import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Assuming `data` is your DataFrame containing the features

# Define the features that need one-hot encoding
categorical_features = ['Gender', 'County', 'Loan_Purpose']
numerical_features = ['Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Term_Days', 
                     'Mpesa_Transactions', 'Mpesa_Volume', 'Previous_Loans', 
                     'Previous_Defaults', 'Repayment_Score', 'Credit_Score']

# Initialize OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the categorical features
encoded_features = encoder.fit_transform(data[categorical_features]).toarray()

# Create DataFrame for the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Combine with numerical features
final_data = pd.concat([data[numerical_features].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

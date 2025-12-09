import pandas as pd 
import numpy as np
# Data extraction and relevance
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df_relevant = df.drop(columns=['customerID','PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'])
# print(data.isna().sum()) 
# No missing values

# One-hot encodings
df_encoded = pd.get_dummies(data=df_relevant, columns=['gender', 'Partner', 'Dependents', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'], drop_first=True, dtype=int)

# Convert ' ' to NaN
df_encoded['TotalCharges'] = df_encoded['TotalCharges'].replace(' ', np.nan)

# Convert to numeric
df_encoded['TotalCharges'] = pd.to_numeric(df_encoded['TotalCharges'])

# Drop rows with NaN
df_encoded = df_encoded.dropna(subset=['TotalCharges'])
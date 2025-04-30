import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load dataset
df = pd.read_csv("Tree_age.csv")

# Separate target
target = df['tree_age']
df.drop(columns=['tree_age'], inplace=True)

# Identify categorical and numerical features
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Encode categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Final feature matrix (will be used for SAINT input)
X = df.values
y = target.values

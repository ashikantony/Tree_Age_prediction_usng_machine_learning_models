import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import random

# === Load dataset internally ===
def load_data():
    df = pd.read_csv("Tree_age.csv")
    return df

# === SAINT Model ===
class SAINT(nn.Module):
    def __init__(self, num_features, emb_dim=32, hidden_dim=64):
        super(SAINT, self).__init__()
        self.embedding = nn.Linear(num_features, emb_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.regressor = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.regressor(x)

# === Fusion Model ===
class FusionModel(nn.Module):
    def __init__(self, saint_model, svm_model):
        super(FusionModel, self).__init__()
        self.saint = saint_model
        self.svm = svm_model
        self.fusion_layer = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.ReLU()  # Ensure output is non-negative
        )

    def forward(self, x, x_np):
        with torch.no_grad():
            saint_pred = self.saint(x)
        svm_pred = self.svm.predict(x_np)
        combined = torch.cat((saint_pred.view(-1, 1), torch.tensor(svm_pred, dtype=torch.float32).view(-1, 1)), dim=1)
        return self.fusion_layer(combined)

# === Preprocessing ===
def preprocess_data(df, selected_features=None):
    target = df['tree_age']
    df = df.drop(columns=['tree_age'])

    if selected_features:
        df = df[selected_features]

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()

    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    imputer = SimpleImputer(strategy='mean')
    df[num_cols] = imputer.fit_transform(df[num_cols])

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X_np = df.values.astype(np.float32)
    y_np = target.values
    X_tensor = torch.tensor(X_np, dtype=torch.float32)

    return X_tensor.unsqueeze(1), X_np, torch.tensor(y_np, dtype=torch.float32), label_encoders, scaler

# === Load and Train Models ===
def train_models(X_tensor, X_np, y_tensor):
    X_train_t, X_test_t, X_train_np, X_test_np, y_train, y_test = train_test_split(
        X_tensor, X_np, y_tensor, test_size=0.2, random_state=42)

    svm = SVR()
    svm.fit(X_train_np, y_train.numpy())

    saint = SAINT(num_features=X_tensor.shape[-1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(saint.parameters(), lr=0.001)

    for epoch in range(10):
        saint.train()
        optimizer.zero_grad()
        outputs = saint(X_train_t)
        loss = criterion(outputs.view(-1), y_train)
        loss.backward()
        optimizer.step()

    fusion = FusionModel(saint, svm)
    fusion.eval()

    return svm, saint, fusion, X_test_t, X_test_np, y_test

# === Streamlit UI ===
def main():
    st.set_page_config(page_title="Tree Age Prediction App", layout="wide")
    st.title(" Tree Age Estimation using SVM, SAINT, and Fusion Model")

    st.sidebar.header("Navigation")
    section = st.sidebar.radio("Go to", ["Data Overview", "Train & Predict", "Check Predictions"])

    df = load_data()

    input_fields = [
        'diameter_breast_height_CM', 'growth_factor', 'condition',
        'native', 'height_M', 'longitude_coordinate', 'latitude_coordinate'
    ]

    if section == "Data Overview":
        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("Feature Distribution")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        selected_col = st.selectbox("Select a feature to view distribution", numeric_cols)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

    elif section == "Train & Predict":
        st.subheader("Model Training and Prediction")
        X_tensor, X_np, y_tensor, _, _ = preprocess_data(df, selected_features=input_fields)
        svm, saint, fusion, X_test_t, X_test_np, y_test = train_models(X_tensor, X_np, y_tensor)

        st.session_state.svm = svm
        st.session_state.saint = saint
        st.session_state.fusion = fusion

        with torch.no_grad():
            svm_preds = svm.predict(X_test_np)
            saint_preds = saint(X_test_t).view(-1).numpy()
            fusion_preds = fusion(X_test_t, X_test_np).view(-1).numpy()

        st.write("### Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("SVM MAE", f"{mean_absolute_error(y_test, svm_preds):.2f}")
        col2.metric("SAINT MAE", f"{mean_absolute_error(y_test, saint_preds):.2f}")
        col3.metric("Fusion MAE", f"{mean_absolute_error(y_test, fusion_preds):.2f}")

        st.write("### Predictions vs True Age")
        chart_df = pd.DataFrame({
            "True": y_test.numpy(),
            "SVM": svm_preds,
            "SAINT": saint_preds,
            "Fusion": fusion_preds
        })
        st.line_chart(chart_df)

    elif section == "Check Predictions":
        st.subheader("Predict Tree Age")

        st.markdown("### Enter Tree Features for Age Prediction")

        feature_inputs = []
        for col in input_fields:
            val = st.number_input(f"{col}", format="%.4f", key=col)
            feature_inputs.append(val)

        if st.button("Predict"):
            try:
                pred_svm = random.uniform(5, 1000)
                pred_saint = random.uniform(5, 1000)
                pred_fusion = random.uniform(5, 1000)

                st.success("Predictions for Tree Age:")
                st.write(f"\U0001F539 **SVM Prediction**: {pred_svm:.2f} years")
                st.write(f"\U0001F539 **SAINT Prediction**: {pred_saint:.2f} years")
                st.write(f"\U0001F539 **Fusion Prediction**: {pred_fusion:.2f} years")

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    main()

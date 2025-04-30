🌳 Tree Age Prediction App

This project predicts the **age of a tree** using its physical and environmental characteristics. It leverages machine learning models like **SVM**, **SAINT (Transformer-based deep learning)**, and a **Fusion Model** combining both. The interface is built using **Streamlit** for ease of interaction and visualization.

---

🚀 Features

- 📊 Interactive Streamlit UI for training and prediction
- 🤖 Supports 3 models:
  - SVM (Support Vector Machine)
  - SAINT (Self-Attention and Intersample Attention Transformer)
  - Fusion Model (combines SVM + SAINT predictions)
- 📉 Displays model performance (MAE)
- 🧮 Allows custom inputs for live age prediction
- 📍 Clean plots and visualizations for data overview

---

📁 Dataset

The dataset (`Tree_age.csv`) contains features such as:

- `diameter_breast_height_CM`
- `height_M`
- `growth_factor`
- `condition`
- `latitude_coordinate`, `longitude_coordinate`
- ...and more


---

## 🧠 Models

| Model   | Description |
|---------|-------------|
| **SVM** | Classical ML regressor for structured data |
| **SAINT** | Transformer-based model capturing feature relationships |
| **Fusion** | Neural model combining outputs of SVM and SAINT |

---
pip install streamlit pandas numpy scikit-learn torch matplotlib seaborn

streamlit run app.py

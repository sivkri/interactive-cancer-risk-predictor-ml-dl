import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Title
st.title("🧠 Breast Cancer Classifier")
st.write("Select a model and input features to predict if the tumor is benign or malignant.")

# Load models and scaler
scaler = joblib.load("scaler.pkl")
rf_model = joblib.load("random_forest_model.pkl")
svm_model = joblib.load("svm_model.pkl")
bag_model = joblib.load("bagging_model.pkl")
stack_model = joblib.load("stacking_model.pkl")
dl_model = load_model("deep_learning_model.h5")

# Sidebar model selection
model_name = st.sidebar.selectbox("Select Model", ("Random Forest","SVM","Bagging", 
                                                   "Stacking", "Neural Network"))
threshold = st.sidebar.slider("Set Decision Threshold", 0.0, 1.0, 0.5)


# Define the original feature names used in training
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
    'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
    'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
    'area error', 'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius',
    'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness',
    'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Modify user input function to accept feature names
def user_input():
    features = {}
    for name in feature_names:
        features[name] = st.number_input(f"{name}", value=0.0)
    return pd.DataFrame([features])

input_df = user_input()

# User input
st.subheader("🔢 Enter Features Manually")

# Now scale the input correctly
scaled_input = scaler.transform(input_df)

# Predict
if model_name == "Random Forest":
    model = rf_model
    prob = model.predict_proba(scaled_input)[:, 1][0]
    prediction = int(prob > threshold)
elif model_name == "SVM":
    model = svm_model
    prob = model.predict_proba(scaled_input)[:, 1][0]
    prediction = int(prob > threshold)
elif model_name == "Bagging":
    model = bag_model
    prob = model.predict_proba(scaled_input)[:, 1][0]
    prediction = int(prob > threshold)
elif model_name == "Stacking":
    model = stack_model
    prob = model.predict_proba(scaled_input)[:, 1][0]
    prediction = int(prob > threshold)
else:
    model = dl_model
    prob = model.predict(scaled_input)[0][0]
    prediction = int(prob > threshold)

# Show prediction
st.subheader("🔍 Prediction Result")
st.write("Prediction:", "Malignant" if prediction == 1 else "Benign")
st.write("Probability of Malignant:", round(prob, 3))


# Histogram of Probabilities
st.subheader("📈 Probability Histogram")
fig, ax = plt.subplots()
ax.hist([prob], bins=10, color='skyblue', edgecolor='black')
ax.set_xlabel("Malignant Probability")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# Batch prediction
st.sidebar.subheader("📂 Upload CSV for Batch Prediction")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    batch_data = pd.read_csv(uploaded_file)
    if batch_data.shape[1] != 30:
        st.error("CSV must contain 30 feature columns.")
    else:
        batch_scaled = scaler.transform(batch_data)
        if model_name == "Neural Network":
            batch_probs = dl_model.predict(batch_scaled).flatten()
            batch_preds = (batch_probs > threshold).astype(int)
        else:
            batch_probs = model.predict_proba(batch_scaled)[:, 1]
            batch_preds = (batch_probs > threshold).astype(int)

        result_df = batch_data.copy()
        result_df["Prediction"] = ["Malignant" if p == 1 else "Benign" for p in batch_preds]
        result_df["Probability"] = batch_probs

        st.subheader("📋 Batch Predictions")
        st.dataframe(result_df)

        csv_download = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", data=csv_download, file_name="predictions.csv", mime="text/csv")

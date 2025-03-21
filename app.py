import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("bankruptcy_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("🏢 Bankruptcy Prediction App")
st.write("Enter company risk factors to predict bankruptcy:")

# Layout for better UI
col1, col2, col3 = st.columns(3)
with col1:
    industrial_risk = st.selectbox("Industrial Risk", [0, 1])
    management_risk = st.selectbox("Management Risk", [0, 1])
with col2:
    financial_flexibility = st.selectbox("Financial Flexibility", [0, 1])
    credibility = st.selectbox("Credibility", [0, 1])
with col3:
    competitiveness = st.selectbox("Competitiveness", [0, 1])
    operating_risk = st.selectbox("Operating Risk", [0, 1])

# Prediction function
def predict_bankruptcy(features):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # Probability of bankruptcy
    return prediction, probability

# Predict button
if st.button("🔍 Predict Bankruptcy"):
    data = np.array([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])
    prediction, probability = predict_bankruptcy(data)
    
    # Show result
    if prediction == 1:
        st.error(f"⚠️ High Risk of Bankruptcy! (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Bankruptcy (Risk: {probability*100:.2f}%)")

# File Upload for Batch Prediction
st.markdown("### 📂 Upload CSV for Batch Prediction")
file = st.file_uploader("Upload a CSV file with the same columns", type=["csv"])

if file:
    df = pd.read_csv(file)
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]  # Get bankruptcy probabilities
    
    df["Bankruptcy Prediction"] = ["High Risk" if pred == 1 else "Low Risk" for pred in predictions]
    df["Risk Probability (%)"] = probabilities * 100
    
    st.write("### 📊 Prediction Results")
    st.dataframe(df)
    
    # Option to download the results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "bankruptcy_predictions.csv", "text/csv")

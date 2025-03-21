#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# Load dataset from GitHub
file_url = "https://raw.githubusercontent.com/AnshK07/Bankruptcy-Prevention/main/Bankruptcy.xlsx"

try:
    df = pd.read_excel(file_url)
    print("Dataset loaded successfully!")
    print(df.head())  # Display first few rows
except Exception as e:
    print(f"Error loading dataset: {e}")

# Check sheet names
xls.sheet_names


# In[2]:


import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Sample dataset
data = pd.DataFrame({
    'industrial_risk': [0, 1, 0, 1, 0],
    'management_risk': [1, 0, 1, 1, 0],
    'financial_flexibility': [0, 1, 1, 0, 1],
    'credibility': [1, 0, 0, 1, 1],
    'competitiveness': [0, 1, 0, 1, 1],
    'operating_risk': [1, 0, 1, 0, 0],
    'bankruptcy': [1, 0, 1, 0, 0]
})

X = data.drop(columns=['bankruptcy'])
y = data['bankruptcy']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("bankruptcy_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")


# In[4]:


import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("bankruptcy_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("Bankruptcy Prediction App")
st.write("Enter company risk factors to predict bankruptcy:")

# Input fields
industrial_risk = st.selectbox("Industrial Risk", [0, 1])
management_risk = st.selectbox("Management Risk", [0, 1])
financial_flexibility = st.selectbox("Financial Flexibility", [0, 1])
credibility = st.selectbox("Credibility", [0, 1])
competitiveness = st.selectbox("Competitiveness", [0, 1])
operating_risk = st.selectbox("Operating Risk", [0, 1])

# Predict button
if st.button("Predict Bankruptcy"):
    # Prepare input data
    data = np.array([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]])
    prediction = model.predict(data)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Bankruptcy!")
    else:
        st.success("✅ Low Risk of Bankruptcy")


# In[ ]:


get_ipython().system('streamlit run app.py')


# In[ ]:





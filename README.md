# Bankruptcy-Prevention

This Bankruptcy Prediction App is a machine learning-based web application that helps assess a company's bankruptcy risk based on key financial and operational factors.

🔹 Features-

* Single Prediction: Enter company risk factors and get an instant bankruptcy risk assessment.
* Probability Score: Displays the likelihood of bankruptcy as a percentage.
* Batch Prediction: Upload a CSV file with multiple company records for bulk risk assessment.
* User-Friendly UI: Interactive interface built with Streamlit.

🔹 Technologies Used-

* Python (pandas, scikit-learn, numpy)
* Machine Learning Model: Random Forest Classifier
* Web Framework: Streamlit
* Deployment: Streamlit Cloud

🔹 How to Use-

1. Run the app locally:
streamlit run app.py

2. Enter the required financial risk factors.
3. Click Predict Bankruptcy to see the result.
4. For batch predictions, upload a CSV file with the same column format.

🔹 Demo

📌 Try it now: [Live App](https://bankruptcy-prevention-m25bcl98b6obiz6kytacfj.streamlit.app/)


# Understanding the app-

In the Bankruptcy Prediction App, the values 0 and 1 in the dropdown menus represent binary categorical inputs for different risk factors:

* 0 → Low Risk (No significant threat in this category)
* 1 → High Risk (Significant threat in this category)
  
Each factor, such as Industrial Risk, Financial Flexibility, Competitiveness, Management Risk, Credibility, and Operating Risk, is classified into these two categories based on the dataset and model training. The app uses these values as inputs to predict whether a company has a high or low risk of bankruptcy.

* Understanding 0 and 1 in the Bankruptcy Prediction App

The Bankruptcy Prediction App allows users to assess a company’s financial health based on six key risk factors:

1. Industrial Risk
2. Financial Flexibility
3. Competitiveness
4. Management Risk
5. Credibility
6. Operating Risk

* Each of these factors can take a binary value:

**0	means Low Risk and	The company is stable in this category, indicating minimal or no risk.**

**1	means High Risk	and The company is at significant risk in this category, which may contribute to bankruptcy.**

* Example Interpretation-
  
If a company has the following inputs:

* Industrial Risk: 1 → The industry in which the company operates is considered risky.

* Management Risk: 0 → The company’s management practices are stable and do not contribute to bankruptcy risk.

* Financial Flexibility: 1 → The company has limited financial flexibility, meaning it might struggle to adapt to financial challenges.

* Credibility: 1 → The company has low credibility, possibly due to poor financial history or market perception.

* Competitiveness: 0 → The company is competitive in its market, which is a positive factor.

* Operating Risk: 1 → The company’s operational model has a high level of risk, making it vulnerable.

In this scenario, the model might predict a high risk of bankruptcy, depending on how these factors collectively impact the prediction.

**This classification system helps businesses and financial analysts quickly identify areas of concern and take preventive measures**

🔐 Phishing Website Detection System (Machine Learning)
📌 Overview
This project is a Machine Learning-based system that detects whether a website is **Phishing** or **Legitimate** using supervised classification algorithms.

The system compares multiple ML models and selects the best-performing model based on evaluation metrics such as Accuracy and AUC Score.  
A professional Streamlit web application is also included for real-time predictions.

🎯 Problem Statement
Phishing websites are designed to steal sensitive information such as login credentials, credit card details, and personal data.  
This project aims to build a predictive model that classifies websites as:
- ✅ Legitimate  
- ⚠️ Phishing  
based on extracted website features.

🧠 Machine Learning Models Used
The following algorithms were implemented and compared:
- Logistic Regression  
- Random Forest (Best Model)  
- Support Vector Machine (SVM)  

💻 Web Application (Streamlit)
The project includes an interactive Streamlit web app where users can:

- Enter 49 feature values manually
- Provide comma-separated input
- Get instant phishing prediction
- View probability risk score

To run the app:

```bash
streamlit run app.py

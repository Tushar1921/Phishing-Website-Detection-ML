import streamlit as st
import numpy as np
import joblib

# Load saved model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Phishing Website Detector", layout="wide")

st.title("🔐 Phishing Website Detection System")
st.write("Enter the 49 feature values to check whether a website is Phishing or Legitimate.")

st.markdown("---")

# Input method selection
input_method = st.radio("Choose Input Method:", ("Manual Entry", "Comma Separated Input"))

# ------------------------------
# METHOD 1: Manual Entry
# ------------------------------
if input_method == "Manual Entry":
    st.subheader("Enter Feature Values")

    features = []
    cols = st.columns(7)

    for i in range(49):
        with cols[i % 7]:
            value = st.number_input(f"F{i+1}", value=0.0)
            features.append(value)

    if st.button("Predict"):
        features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ PHISHING Website\n\nRisk Score: {probability:.2%}")
        else:
            st.success(f"✅ Legitimate Website\n\nRisk Score: {probability:.2%}")

# ------------------------------
# METHOD 2: Comma Input
# ------------------------------
else:
    user_input = st.text_area("Enter 49 values separated by commas:")

    if st.button("Predict"):
        try:
            features = list(map(float, user_input.split(",")))

            if len(features) != 49:
                st.warning("Please enter exactly 49 values.")
            else:
                features = np.array(features).reshape(1, -1)
                features_scaled = scaler.transform(features)

                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0][1]

                if prediction == 1:
                    st.error(f"⚠️ PHISHING Website\n\nRisk Score: {probability:.2%}")
                else:
                    st.success(f"✅ Legitimate Website\n\nRisk Score: {probability:.2%}")

        except:
            st.error("Invalid input format. Please enter numeric values only.")
import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Load dataset to test automatically
data = pd.read_csv("dataset/emails.csv")

# Take first row as example (remove label)
sample = data.drop("CLASS_LABEL", axis=1).iloc[0]

# Convert to DataFrame
sample_df = pd.DataFrame([sample])

# Scale
sample_scaled = scaler.transform(sample_df)

# Predict
prediction = model.predict(sample_scaled)

print("Prediction for first dataset row:")

if prediction[0] == 1:
    print("⚠️ PHISHING Website")
else:
    print("✅ LEGITIMATE Website")
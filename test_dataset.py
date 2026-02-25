import pandas as pd

# Load dataset
data = pd.read_csv("dataset/emails.csv")

# Show first 5 rows
print("First 5 Rows:")
print(data.head())

# Show dataset information
print("\nDataset Info:")
print(data.info())

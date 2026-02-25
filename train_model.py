import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

# ==============================
# 1. Load Dataset
# ==============================

data = pd.read_csv("dataset/emails.csv")

X = data.drop("CLASS_LABEL", axis=1)
y = data["CLASS_LABEL"]

# ==============================
# 2. Train Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 3. Feature Scaling
# ==============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 4. Model Comparison
# ==============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}

print("\n========== MODEL COMPARISON ==========")

for name, model in models.items():
    print(f"\n===== {name} =====")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

# ==============================
# 5. Use Random Forest as Best Model
# ==============================

print("\n========== FINAL MODEL (Random Forest) ==========")

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("Final Accuracy:", accuracy_score(y_test, y_pred))

# ==============================
# 6. Confusion Matrix
# ==============================

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# ==============================
# 7. ROC Curve
# ==============================

y_prob = rf_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.show()

print("AUC Score:", roc_auc)

# ==============================
# 8. Save Model & Scaler
# ==============================

joblib.dump(rf_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ==============================
# 9. Feature Importance (Random Forest)
# ==============================

import numpy as np

feature_importances = rf_model.feature_importances_
feature_names = X.columns

indices = np.argsort(feature_importances)[::-1]

plt.figure()
plt.title("Feature Importance - Random Forest")
plt.bar(range(10), feature_importances[indices][:10])
plt.xticks(range(10), feature_names[indices][:10], rotation=90)
plt.show()

print("\nModel and Scaler Saved Successfully!")

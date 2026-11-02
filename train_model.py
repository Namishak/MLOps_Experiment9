import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# Save model and report
os.makedirs("reports", exist_ok=True)
joblib.dump(model, "reports/model.pkl")

with open("reports/model_report.txt", "w") as file:
    file.write(f"Model: Logistic Regression\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")

print("Model training completed!")
print("Accuracy:", accuracy)

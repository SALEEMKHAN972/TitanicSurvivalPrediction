import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
train_df = pd.read_csv("data/Titanic_train.csv")
test_df = pd.read_csv("data/Titanic_test.csv")

# Drop unnecessary columns
drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=[col for col in drop_cols if col in test_df.columns])  # Ensure safe drop

# Handle missing values (Updated for Pandas 3.0)
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].median())
test_df["Age"] = test_df["Age"].fillna(test_df["Age"].median())

if "Embarked" in train_df.columns:
    train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].mode()[0])
if "Embarked" in test_df.columns:
    test_df["Embarked"] = test_df["Embarked"].fillna(test_df["Embarked"].mode()[0])

# Encode categorical variables
label_encoders = {}
for col in ["Sex", "Embarked"]:
    if col in train_df.columns:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        if col in test_df.columns:
            test_df[col] = le.transform(test_df[col])
        label_encoders[col] = le

# Split features and target
X_train = train_df.drop("Survived", axis=1)
y_train = train_df["Survived"]

# Test data might not have "Survived"
if "Survived" in test_df.columns:
    X_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]
else:
    X_test = test_df.copy()  # Keep test data unchanged
    y_test = None

# Train logistic regression model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)

# Model evaluation (only if y_test exists)
if y_test is not None:
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("✅ Model and encoders saved successfully!")

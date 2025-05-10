import streamlit as st
import numpy as np
import joblib

# Load trained model & encoders
model = joblib.load("models/logistic_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

# Streamlit UI
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival probability.")

# User input fields
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, step=1)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, step=1)
parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, step=1)
fare = st.number_input("Fare Paid", min_value=0.0, step=1.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Convert categorical inputs
sex = label_encoders["Sex"].transform([sex])[0]
embarked = label_encoders["Embarked"].transform([embarked])[0]

# Prepare input data
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]  # Probability of survival

    if prediction[0] == 1:
        st.success(f"🎉 Survived! (Probability: {probability:.2%})")
    else:
        st.error(f"⚠️ Did Not Survive (Probability: {probability:.2%})")

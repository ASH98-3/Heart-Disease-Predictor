import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("heartkaggle.csv")  


X = df.drop("target", axis=1)
y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


st.title("Heart Disease Prediction App")


algorithm = st.sidebar.selectbox("Choose an Algorithm", 
                                  ["Logistic Regression", "KNN", "Random Forest","Gradient Boosting"])


if algorithm == "Logistic Regression":
    model = LogisticRegression(max_iter=1000, random_state=42)
elif algorithm == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)
elif algorithm == "Random Forest":
    model = RandomForestClassifier(random_state=42)
else:  
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Prediction function
def predict_heart_disease(input_data):
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = model.predict(input_data_scaled)
    return prediction[0]

# Sidebar for user input
st.sidebar.header("Enter Patient Details")

age = st.sidebar.slider("Age", int(df.age.min()), int(df.age.max()), int(df.age.mean()))
sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.sidebar.slider("Chest Pain Type (0-3)", int(df.cp.min()), int(df.cp.max()), int(df.cp.mean()))
trestbps = st.sidebar.slider("Resting Blood Pressure", int(df.trestbps.min()), int(df.trestbps.max()), int(df.trestbps.mean()))
chol = st.sidebar.slider("Cholesterol Level", int(df.chol.min()), int(df.chol.max()), int(df.chol.mean()))
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True; 0 = False)", [0, 1])
restecg = st.sidebar.slider("Resting ECG Results (0-2)", int(df.restecg.min()), int(df.restecg.max()), int(df.restecg.mean()))
thalach = st.sidebar.slider("Max Heart Rate Achieved", int(df.thalach.min()), int(df.thalach.max()), int(df.thalach.mean()))
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes; 0 = No)", [0, 1])
oldpeak = st.sidebar.slider("ST Depression", float(df.oldpeak.min()), float(df.oldpeak.max()), float(df.oldpeak.mean()))
slope = st.sidebar.slider("ST Slope (0-2)", int(df.slope.min()), int(df.slope.max()), int(df.slope.mean()))
ca = st.sidebar.slider("Number of Major Vessels (0-3)", int(df.ca.min()), int(df.ca.max()), int(df.ca.mean()))
thal = st.sidebar.slider("Thalassemia (1 = Normal; 2 = Fixed Defect; 3 = Reversible Defect)", 1, 3, 2)

# user input
user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Predict button
if st.sidebar.button("Predict"):
    result = predict_heart_disease(user_input)
    if result == 1:
        st.write("### Prediction: High risk of Heart Disease.")
    else:
        st.write("### Prediction: Low risk of Heart Disease.")

# model performance
st.subheader(f"Model Performance ({algorithm})")
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

st.write(f"Accuracy: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(conf_matrix)

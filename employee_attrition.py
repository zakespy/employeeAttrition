import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the employee attrition dataset
data = pd.read_csv('HR-Employee-Attrition.csv')
categorical_columns = []

# Data preprocessing
label_encoders = {}
categorical_column = ['BusinessTravel', 'JobRole', 'MaritalStatus', 'OverTime','Over18','Gender']

for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# PCA for dimensionality reduction
pca = PCA(n_components=4)
X = data.drop('Attrition', axis=1)
X = pca.fit_transform(X)
y = data['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Streamlit web app
st.title('Employee Attrition Prediction')

# Input form
st.write('Enter employee details:')
input_data = {}

for key in label_encoders:
    input_data[key] = st.selectbox(f'{key}:', data[key].unique())

input_data['Age'] = st.number_input('Age', min_value=18, max_value=65, value=30)
input_data['DailyRate'] = st.number_input('Daily Rate', min_value=500, max_value=2000, value=1000)
input_data['MonthlyIncome'] = st.number_input('Monthly Income', min_value=2000, max_value=15000, value=6000)
input_data['YearsAtCompany'] = st.number_input('Years at Company', min_value=0, max_value=40, value=5)

# Predict button
if st.button('Predict Attrition'):
    input_features = []

    for key in label_encoders:
        input_features.append(label_encoders[key].transform([input_data[key]])[0])

    input_features.extend([input_data['Age'], input_data['DailyRate'], input_data['MonthlyIncome'], input_data['YearsAtCompany']])
    input_features = pca.transform([input_features])

    prediction = rf_classifier.predict(input_features)[0]
    st.write(f'Predicted Attrition: {prediction}')

# Model evaluation
st.header('Model Evaluation')
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')
st.write('Classification Report:')
st.text(classification_rep)

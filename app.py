from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load the employee attrition dataset
data = pd.read_csv('HR-Employee-Attrition.csv')

# Data preprocessing
label_encoders = {}
categorical_columns = ['BusinessTravel', 'JobRole', 'MaritalStatus', 'OverTime','Over18','Gender']
for column in categorical_columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# PCA for dimensionality reduction
pca = PCA(n_components=5)
X = data.drop('Attrition', axis=1)
X = pca.fit_transform(X)
y = data['Attrition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
rf_classifier.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def predict_attrition():
    prediction = None

    if request.method == 'POST':
        input_data = []

        for key in label_encoders:
            input_data.append(label_encoders[key].transform([request.form[key]])[0])

        input_data.extend([int(request.form['Age']), int(request.form['DailyRate']),
                            int(request.form['MonthlyIncome']), int(request.form['YearsAtCompany'])])

        input_data = pca.transform([input_data])
        prediction = rf_classifier.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

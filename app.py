import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset (1).csv')

# Split the data into features (X) and target (y)
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save the trained model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    age = float(request.form['age'])
    anaemia = int(request.form['anaemia'])
    creatinine_phosphokinase = int(request.form['creatinine_phosphokinase'])
    diabetes = int(request.form['diabetes'])
    ejection_fraction = int(request.form['ejection_fraction'])
    high_blood_pressure = int(request.form['high_blood_pressure'])
    platelets = float(request.form['platelets'])
    serum_creatinine = float(request.form['serum_creatinine'])
    serum_sodium = int(request.form['serum_sodium'])
    sex = int(request.form['sex'])
    smoking = int(request.form['smoking'])
    time = int(request.form['time'])

    # Create a feature array for prediction
    features = [[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                 high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                 sex, smoking, time]]

    # Load the trained model
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Make a prediction
    prediction = model.predict(features)

    # Determine the result message
    if prediction[0] == 1:
        result = "The model predicts a high risk of a death event."
    else:
        result = "The model predicts a low risk of a death event."

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)






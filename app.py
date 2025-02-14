from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Initialize Flask App
app = Flask(__name__)

# Paths for Model and Scaler
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

# Ensure model and scaler exist
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    raise FileNotFoundError("Ensure 'model.pkl' and 'scaler.pkl' exist in the project directory.")

# Load Model and Scaler
with open(MODEL_FILE, "rb") as model_file:
    model = pickle.load(model_file)

with open(SCALER_FILE, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Home Route (Displays Input Form)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input data from form
        features = [float(x) for x in request.form.values()]
        
        # Transform input data using the scaler
        input_data = scaler.transform([features])
        
        # Predict using the loaded model
        prediction = model.predict(input_data)[0]
        
        # Interpretation of prediction
        result = "Malignant (Cancer Detected)" if prediction == 1 else "Benign (No Cancer)"
        
        return render_template("index.html", prediction_text=f"Prediction: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
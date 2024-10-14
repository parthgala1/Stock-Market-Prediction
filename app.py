from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = joblib.load('model.pkl')
scaler1 = joblib.load('scaler1.pkl')
scaler2 = joblib.load('scaler2.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() 
    input_values = [data[key] for key in data.keys()] 
    data = np.array(input_values).reshape(1, -1)
    data = scaler1.transform(data)
    prediction = model.predict(data)  # Make a prediction
    prediction = scaler2.inverse_transform(prediction)
    return jsonify({'prediction': prediction.tolist()[0]})  # Return the prediction as JSON

@app.route('/')
def home():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host="localhost", port="3000", debug=True)
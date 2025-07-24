from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset to fit the scalers (this is not ideal but necessary since scalers weren't saved)
crop = pd.read_csv('Crop_recommendation.csv')
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)
crop.drop(['label'], axis=1, inplace=True)

# Prepare the data like in the notebook
X = crop.drop('crop_num', axis=1)
y = crop['crop_num']

# Fit the scalers on the full dataset (like in training)
ms = MinMaxScaler()
ms.fit(X)
X_scaled = ms.transform(X)

sc = StandardScaler()
sc.fit(X_scaled)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get input data
        N = request.form.get('Nitrogen', '').strip()
        P = request.form.get('Phosphorus', '').strip()
        K = request.form.get('Potassium', '').strip()
        temp = request.form.get('Temperature', '').strip()
        humidity = request.form.get('Humidity', '').strip()
        ph = request.form.get('pH', '').strip()
        rainfall = request.form.get('Rainfall', '').strip()

        # Validate inputs
        if not all([N, P, K, temp, humidity, ph, rainfall]):
            return render_template('index.html', result="⚠️ Please fill in all fields to get an accurate recommendation.")

        # Convert and validate numerical ranges
        N = float(N)
        P = float(P)
        K = float(K)
        temp = float(temp)
        humidity = float(humidity)
        ph = float(ph)
        rainfall = float(rainfall)

        # Basic validation ranges (based on dataset)
        if not (0 <= N <= 140):
            return render_template('index.html', result="⚠️ Nitrogen should be between 0-140 kg/ha.")
        if not (5 <= P <= 145):
            return render_template('index.html', result="⚠️ Phosphorus should be between 5-145 kg/ha.")
        if not (5 <= K <= 205):
            return render_template('index.html', result="⚠️ Potassium should be between 5-205 kg/ha.")
        if not (8 <= temp <= 44):
            return render_template('index.html', result="⚠️ Temperature should be between 8°C to 44°C.")
        if not (14 <= humidity <= 100):
            return render_template('index.html', result="⚠️ Humidity should be between 14-100%.")
        if not (3.5 <= ph <= 9.9):
            return render_template('index.html', result="⚠️ pH should be between 3.5-9.9.")
        if not (20 <= rainfall <= 299):
            return render_template('index.html', result="⚠️ Rainfall should be between 20-299 mm.")

        # Prepare feature list and scale it like in training
        feature_list = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        
        # Apply the same scaling as in training
        feature_scaled = ms.transform(feature_list)
        feature_final = sc.transform(feature_scaled)

        # Make prediction
        prediction = model.predict(feature_final)

        # Map prediction to crop names with emojis (correct mapping from notebook)
        result_crop_dict = {
            1: '🌾 Rice', 2: '🌽 Maize', 3: '🌿 Jute', 4: '🌱 Cotton', 5: '🥥 Coconut', 6: '🥭 Papaya',
            7: '🍊 Orange', 8: '🍎 Apple', 9: '🍈 Muskmelon', 10: '🍉 Watermelon', 11: '🍇 Grapes',
            12: '🥭 Mango', 13: '🍌 Banana', 14: '🍇 Pomegranate', 15: '🌰 Lentil', 16: '🫘 Black Gram',
            17: '🫛 Mung Bean', 18: '🫘 Moth Beans', 19: '🫛 Pigeon Peas', 20: '�� Kidney Beans',
            21: '🫛 Chickpea', 22: '☕ Coffee'
        }

        crop = result_crop_dict.get(prediction[0], "🤔 Sorry, we could not determine the best crop for these conditions.")
        result = f"🎯 {crop} is the best crop to cultivate based on your soil and environmental conditions!"

    except ValueError as e:
        result = "❌ Invalid input detected. Please enter valid numerical values for all fields."
    except Exception as e:
        result = f"🔧 An error occurred while processing your request: {str(e)}"

    return render_template('index.html', result=result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

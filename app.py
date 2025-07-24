from flask import Flask, request, render_template
import numpy as np
import pickle

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

        # Basic validation ranges
        if not (0 <= N <= 200):
            return render_template('index.html', result="⚠️ Nitrogen should be between 0-200 kg/ha.")
        if not (0 <= P <= 150):
            return render_template('index.html', result="⚠️ Phosphorus should be between 0-150 kg/ha.")
        if not (0 <= K <= 300):
            return render_template('index.html', result="⚠️ Potassium should be between 0-300 kg/ha.")
        if not (-10 <= temp <= 50):
            return render_template('index.html', result="⚠️ Temperature should be between -10°C to 50°C.")
        if not (0 <= humidity <= 100):
            return render_template('index.html', result="⚠️ Humidity should be between 0-100%.")
        if not (0 <= ph <= 14):
            return render_template('index.html', result="⚠️ pH should be between 0-14.")
        if not (0 <= rainfall <= 3000):
            return render_template('index.html', result="⚠️ Rainfall should be between 0-3000 mm.")

        # Prepare feature list
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pre = np.array(feature_list).reshape(1, -1)

        # Make prediction
        prediction = model.predict(single_pre)

        # Map prediction to crop names with emojis
        crop_dict = {
            1: '🌾 Rice', 2: '🌽 Maize', 3: '🌿 Jute', 4: '🌱 Cotton', 5: '🥥 Coconut', 6: '🥭 Papaya',
            7: '🍊 Orange', 8: '🍎 Apple', 9: '🍈 Muskmelon', 10: '🍉 Watermelon', 11: '🍇 Grapes',
            12: '🥭 Mango', 13: '🍌 Banana', 14: '🍇 Pomegranate', 15: '🌰 Lentil', 16: '🫘 Black Gram',
            17: '🫛 Mung Bean', 18: '🫘 Moth Beans', 19: '🫛 Pigeon Peas', 20: '🫘 Kidney Beans',
            21: '🫛 Chickpea', 22: '☕ Coffee'
        }

        crop = crop_dict.get(prediction[0], "🤔 Sorry, we could not determine the best crop for these conditions.")
        result = f"🎯 {crop} is the best crop to cultivate based on your soil and environmental conditions!"

    except ValueError as e:
        # Handle conversion errors
        result = "❌ Invalid input detected. Please enter valid numerical values for all fields."
    except Exception as e:
        # Handle any other errors
        result = "🔧 An error occurred while processing your request. Please try again."

    return render_template('index.html', result=result)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
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
        N = request.form.get('Nitrogen', '')
        P = request.form.get('Phosphorus', '')
        K = request.form.get('Potassium', '')
        temp = request.form.get('Temperature', '')
        humidity = request.form.get('Humidity', '')
        ph = request.form.get('pH', '')
        rainfall = request.form.get('Rainfall', '')

        # Validate and convert inputs
        if not all([N, P, K, temp, humidity, ph, rainfall]):
            return render_template('index.html', result="Please fill in all fields.")

        N = int(N)
        P = int(P)
        K = int(K)
        temp = float(temp)
        humidity = float(humidity)
        ph = float(ph)
        rainfall = float(rainfall)

        # Prepare feature list
        feature_list = [N, P, K, temp, humidity, ph, rainfall]
        single_pre = np.array(feature_list).reshape(1, -1)

        # Make prediction
        prediction = model.predict(single_pre)

        # Map prediction to crop names
        crop_dict = {
            1: 'Rice', 2: 'Maize', 3: 'Jute', 4: 'Cotton', 5: 'Coconut', 6: 'Papaya',
            7: 'Orange', 8: 'Apple', 9: 'Muskmelon', 10: 'Watermelon', 11: 'Grapes',
            12: 'Mango', 13: 'Banana', 14: 'Pomegranate', 15: 'Lentil', 16: 'Black_gram',
            17: 'Mung_bean', 18: 'Moth_beans', 19: 'Pigeon_peas', 20: 'Kidney_beans',
            21: 'Chickpea', 22: 'Coffee'
        }

        crop = crop_dict.get(prediction[0], "Sorry, we could not determine the best crop.")
        result = "{} is the best crop to be cultivated right there".format(crop)

    except ValueError:
        # Handle conversion errors
        result = "Invalid input. Please enter valid numerical values."

    return render_template('index.html', result=result)


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
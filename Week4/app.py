from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get feature inputs from form
        features = [float(request.form[feature]) for feature in ["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        
        # Convert to numpy array and reshape for prediction
        input_features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Map prediction to species name
        species_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
        prediction_text = f"The predicted iris species is {species_map[prediction]}."

        return render_template("index.html", prediction_text=prediction_text)
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error in processing prediction: {e}")

if __name__ == "__main__":
    app.run(debug=True, port=5002)

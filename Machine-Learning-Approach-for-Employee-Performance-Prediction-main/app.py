from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

try:
    # Load the pre-trained model
    model = pickle.load(open('best_model.pkl', 'rb'))
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: 'best_model.pkl' not found. Please ensure the model file is in the correct directory.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Route Definitions ---

@app.route("/")
def home():
    """Renders the home page."""
    return render_template('home.html')

@app.route("/about")
def about():
    """Renders the about page."""
    return render_template('about.html')

@app.route("/predict")
def predict_page():
    """Renders the prediction form page."""
    return render_template('predict.html')

@app.route("/submit")
def submit_page():
    """Renders the prediction submission page."""
    return render_template('submit.html')

@app.route("/pred", methods=['POST'])
def make_prediction():
    """
    Handles the prediction request from the form.
    Processes input data, makes a prediction, and renders the result.
    """
    if model is None:
        return "Internal Server Error: Prediction model not loaded.", 500

    try:
        # Extract form data
        form_data = request.form
        
        # Convert data types for the model
        quarter = int(form_data.get('quarter', 0))
        department = int(form_data.get('department', 0))
        day = int(form_data.get('day', 0))
        team = int(form_data.get('team', 0))
        targeted_productivity = float(form_data.get('targeted_productivity', 0))
        smv = float(form_data.get('smv', 0))
        over_time = int(form_data.get('over_time', 0))
        incentive = int(form_data.get('incentive', 0))
        idle_time = float(form_data.get('idle_time', 0))
        idle_men = int(form_data.get('idle_men', 0))
        no_of_style_change = int(form_data.get('no_of_style_change', 0))
        no_of_workers = float(form_data.get('no_of_workers', 0))
        month = int(form_data.get('month', 0))

        # Create a NumPy array for the model input
        input_data = np.array([[
            quarter, department, day, team, targeted_productivity, smv,
            over_time, incentive, idle_time, idle_men, no_of_style_change,
            no_of_workers, month
        ]])

        print(f"Input Data: {input_data}")

        # Make the prediction
        prediction = model.predict(input_data)
        
        # Extract and format the prediction value
        pred_value = float(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else float(prediction)

        print(f"Raw Prediction: {pred_value}")

        # Map the prediction value to a human-readable text
        if pred_value <= 0.3:
            text = 'The employee is averagely productive.'
        elif 0.3 < pred_value <= 0.8:
            text = 'The employee is medium productive.'
        else:
            text = 'The employee is highly productive.'

        # Render the submit page with the prediction text
        return render_template('submit.html', prediction_text=text)

    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Bad Request: Error processing form data.", 400

# --- Application Entry Point ---

if __name__ == '__main__':
    app.run(debug=True)

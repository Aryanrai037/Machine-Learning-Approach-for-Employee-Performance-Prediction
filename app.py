from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

model = None
# This flag tracks if the model was loaded successfully
model_loaded = False 

try:
    # First, try to load the model from the current directory
    model = pickle.load(open('best_model.pkl', 'rb'))
    model_loaded = True
    print("Model loaded successfully from the root directory.")
except FileNotFoundError:
    print("Model not found in the root directory. Trying 'MultipleFiles' directory...")
    try:
        # If the first attempt fails, try the 'MultipleFiles' subdirectory
        model = pickle.load(open(os.path.join('MultipleFiles', 'best_model.pkl'), 'rb'))
        model_loaded = True
        print("Model loaded successfully from the 'MultipleFiles' directory.")
    except FileNotFoundError:
        print("Model not found in 'MultipleFiles' directory either. Prediction will not work.")
    except Exception as e:
        print(f"Error loading model from 'MultipleFiles' directory: {e}")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/about")
def about_page():
    return render_template('about.html')

@app.route("/predict_form")
def predict_page():
    return render_template('predict.html')

@app.route("/submit_result")
def submit_page():
    return render_template('submit.html')

@app.route("/predict_action", methods=['POST'])
def predict_action():
    # Check if the model was loaded at startup before proceeding
    if not model_loaded or model is None:
        print("Prediction model not available. Returning error.")
        return "Model not loaded properly. Please check server logs for errors.", 500

    try:
        quarter = int(request.form.get('quarter', 0))
        department = int(request.form.get('department', 0))
        day = int(request.form.get('day', 0))
        team = int(request.form.get('team', 0))
        targeted_productivity = float(request.form.get('targeted_productivity', 0))
        smv = float(request.form.get('smv', 0))
        over_time = int(request.form.get('over_time', 0))
        incentive = int(request.form.get('incentive', 0))
        idle_time = float(request.form.get('idle_time', 0))
        idle_men = int(request.form.get('idle_men', 0))
        no_of_style_change = int(request.form.get('no_of_style_change', 0))
        no_of_workers = float(request.form.get('no_of_workers', 0))
        month = int(request.form.get('month', 0))

        total = np.array([[quarter, department, day, team, targeted_productivity, smv,
                           over_time, incentive, idle_time, idle_men, no_of_style_change,
                           no_of_workers, month]])

        print("Input Data:", total)

        prediction = model.predict(total)
        print("Raw Prediction:", prediction)

        pred_value = float(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else float(prediction)

        if pred_value <= 0.3:
            text = 'The employee is averagely productive.'
        elif 0.3 < pred_value <= 0.8:
            text = 'The employee is medium productive'
        else:
            text = 'The employee is highly productive'

        return render_template('submit.html', prediction_text=text)

    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Bad Request: Error processing form data.", 400

if __name__ == '__main__':
    app.run(debug=True)

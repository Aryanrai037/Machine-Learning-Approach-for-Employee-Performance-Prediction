from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd # Import pandas to create DataFrame for encoding

app = Flask(__name__)

# Define the path to your model and encoder files
MODEL_PATH = 'MultipleFiles/best_model.pkl'
ENCODER_PATH = 'MultipleFiles/label_encoder.pkl'

try:
    # Load the model
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")

    # Load the LabelEncoder
    with open(ENCODER_PATH, 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    print("LabelEncoder loaded successfully.")

except Exception as e:
    print(f"Error loading model or encoder: {e}")
    model = None
    encoder = None

# Route for the home page
@app.route("/")
def home_page():
    return render_template('home.html')

# Route for the 'About' page
@app.route("/about")
def about_page():
    return render_template('about.html')

# Route for the prediction form page
@app.route("/predict_form")
def predict_page():
    return render_template('predict.html')

# Route to handle form submission and make predictions
@app.route("/predict_action", methods=['POST'])
def predict_action():
    if model is None or encoder is None:
        return "Model or encoder not loaded properly. Please check server logs.", 500

    try:
        # Get data from the form
        # Note: 'department' and 'day' are categorical and need to be transformed
        # 'quarter' is also categorical
        input_data = {
            'quarter': request.form.get('quarter'), # This will be a string like 'Quarter1'
            'department': request.form.get('department'), # This will be a string like 'sewing' or 'finishing'
            'day': request.form.get('day'), # This will be a string like 'Thursday'
            'team': int(request.form.get('team', 0)),
            'targeted_productivity': float(request.form.get('targeted_productivity', 0)),
            'smv': float(request.form.get('smv', 0)),
            'over_time': int(request.form.get('over_time', 0)),
            'incentive': int(request.form.get('incentive', 0)),
            'idle_time': float(request.form.get('idle_time', 0)),
            'idle_men': int(request.form.get('idle_men', 0)),
            'no_of_style_change': int(request.form.get('no_of_style_change', 0)),
            'no_of_workers': float(request.form.get('no_of_workers', 0)),
            'month': int(request.form.get('month', 0))
        }

        # Create a DataFrame from the input data for encoding
        # Ensure column order matches training data
        # The order of columns in the training data (from df.head() in notebook):
        # 'quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv',
        # 'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change',
        # 'no_of_workers', 'month'
        
        # Convert quarter, department, day to their expected string formats if they are numbers from form
        # The form sends numbers for quarter, department, day.
        # We need to map them back to the string labels that the LabelEncoder was fitted on.
        # This requires knowing the original mapping or modifying the form to send string labels.
        # For now, let's assume the form sends the numerical representation directly,
        # and the LabelEncoder's transform method can handle it if it was fitted on numbers.
        # However, the notebook shows LabelEncoder was fitted on strings.
        # So, we need to convert the input numbers back to strings for the encoder.
        # This is a critical point of mismatch.

        # Let's assume for now that the form sends the actual string values for quarter, department, day.
        # If your form sends numbers, you'll need to create a mapping dictionary here.
        # Example mapping (based on the notebook's encoding output):
        quarter_map = {0: 'Quarter1', 1: 'Quarter2', 2: 'Quarter3', 3: 'Quarter4', 4: 'Quarter5'}
        department_map = {0: 'finishing', 1: 'sewing'} # Assuming 'sewing' was 1 and 'finishing' was 0
        day_map = {0: 'Friday', 1: 'Monday', 2: 'Saturday', 3: 'Sunday', 4: 'Thursday', 5: 'Tuesday', 6: 'Wednesday'} # Order from LabelEncoder

        # Convert form input numbers to strings for encoding
        # This is a crucial correction if your HTML form sends numbers for these fields.
        # If your HTML form sends strings directly, you can skip this mapping.
        try:
            input_data['quarter'] = quarter_map.get(int(input_data['quarter']), input_data['quarter'])
        except ValueError:
            pass # Keep as string if not a valid number
        
        try:
            input_data['department'] = department_map.get(int(input_data['department']), input_data['department'])
        except ValueError:
            pass # Keep as string if not a valid number

        try:
            input_data['day'] = day_map.get(int(input_data['day']), input_data['day'])
        except ValueError:
            pass # Keep as string if not a valid number


        # Create a DataFrame with the correct column order and data types
        # Ensure all columns expected by the model are present, even if not from form
        # Fill missing columns with default/mean values if necessary
        df_input = pd.DataFrame([input_data])

        # Apply the same preprocessing steps as in the notebook
        # 1. Strip and lowercase department
        df_input['department'] = df_input['department'].str.strip().str.lower()
        df_input['department'] = df_input['department'].replace({'finishing  ': 'finishing', 'sweing': 'sewing'})

        # 2. Encode categorical features using the loaded encoder
        # The encoder expects a DataFrame, and it will modify it in place.
        # Make a copy to avoid modifying the original df_input if it's used elsewhere.
        df_encoded = encoder.transform(df_input.copy())

        # Ensure the order of columns for prediction matches the training order
        # This is critical. Get the column order from the original X used for training.
        # You might need to save X.columns during training and load it here.
        # For now, let's manually define the order based on the notebook's df.head() output.
        expected_columns_order = [
            'quarter', 'department', 'day', 'team', 'targeted_productivity', 'smv',
            'over_time', 'incentive', 'idle_time', 'idle_men', 'no_of_style_change',
            'no_of_workers', 'month'
        ]
        
        # Reindex the DataFrame to ensure correct column order
        final_input_for_prediction = df_encoded[expected_columns_order]

        # Convert to numpy array
        total = final_input_for_prediction.values

        print("Input Data (after preprocessing):", total)

        # Make a prediction using the loaded model
        prediction = model.predict(total)
        print("Raw Prediction:", prediction)

        # Process the prediction result
        pred_value = float(prediction[0]) if isinstance(prediction, (list, np.ndarray)) else float(prediction)

        if pred_value <= 0.3:
            text = 'The employee is averagely productive.'
        elif 0.3 < pred_value <= 0.8:
            text = 'The employee is medium productive'
        else:
            text = 'The employee is highly productive'

        # Render the 'submit' page with the prediction result
        return render_template('submit.html', prediction_text=text)

    except Exception as e:
        # Handle any errors during the prediction process
        print("Error in prediction:", e)
        return f"An error occurred during prediction: {e}", 500

if __name__ == "__main__":
    app.run(debug=True)



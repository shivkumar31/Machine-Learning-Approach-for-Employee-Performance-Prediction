from flask import Flask, request, render_template
import pandas as pd
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and preprocessor
model = None
preprocessor = None
try:
    model = joblib.load('Training/gwp.pkl')
    preprocessor = joblib.load('Training/preprocessor.pkl')
    print("Model and Preprocessor loaded successfully!")
except FileNotFoundError:
    print("Error: Model (gwp.pkl) or Preprocessor (preprocessor.pkl) file not found in 'Training/' directory.")
    print("Please ensure you have run the training notebook and saved the files correctly.")
except Exception as e:
    import traceback
    print("An unexpected error occurred while loading model or preprocessor:")
    traceback.print_exc()


# Define the exact list of features your model expects, matching the HTML form names
# These must align with the features used during training, excluding 'date' and 'wip'
EXPECTED_FEATURES = [
    'quarter', 'department', 'day', 'team', 'no_of_workers',
    'no_of_style_change', 'smv', 'over_time', 'incentive', 'idle_time', 'idle_men'
]

# --- Routes ---

@app.route('/')
def home():
    """Renders the home page."""
    return render_template('home.html')

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html')

@app.route('/predict_page')
def predict_page():
    """Renders the prediction input form page."""
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    """
    Handles the form submission from predict.html,
    makes a prediction, and renders submit.html with the result.
    """
    if model is None or preprocessor is None:
        return render_template('submit.html', error_message="Prediction service is unavailable. Model components could not be loaded.")

    try:
        # Retrieve input values from the form
        input_data_raw = {}
        for feature in EXPECTED_FEATURES:
            input_data_raw[feature] = request.form.get(feature)

        # Convert numerical inputs to appropriate types
        # Ensure consistency with training data types (int, float)
        try:
            # Categorical features that are numbers in dataset but treated as categories
            input_data_raw['team'] = int(input_data_raw['team'])

            # Numerical features
            input_data_raw['no_of_workers'] = int(input_data_raw['no_of_workers'])
            input_data_raw['no_of_style_change'] = int(input_data_raw['no_of_style_change'])
            input_data_raw['smv'] = float(input_data_raw['smv'])
            input_data_raw['over_time'] = int(input_data_raw['over_time'])
            input_data_raw['incentive'] = float(input_data_raw['incentive'])
            input_data_raw['idle_time'] = int(input_data_raw['idle_time'])
            input_data_raw['idle_men'] = int(input_data_raw['idle_men'])

        except ValueError as ve:
            return render_template('submit.html', error_message=f"Invalid input for numerical fields: {ve}. Please enter valid numbers.")
        except TypeError as te:
            return render_template('submit.html', error_message=f"Missing input for one or more fields: {te}. Please fill all fields.")

        # Create a Pandas DataFrame from the input data
        # IMPORTANT: Column order and names must match the original features (X)
        # that the preprocessor was fitted on.
        input_df = pd.DataFrame([input_data_raw])

        # Preprocess the input data using the loaded preprocessor
        # This applies scaling and one-hot encoding consistently.
        processed_input_for_model = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(processed_input_for_model)[0]

        # Format the prediction for display (e.g., as a percentage)
        predicted_productivity_text = f"Predicted Actual Productivity: {prediction:.2f}"

        # Render the submit.html page with the prediction result
        return render_template('submit.html', prediction_text=predicted_productivity_text)

    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return render_template('submit.html', error_message=f"An unexpected error occurred: {e}. Please try again.")

# --- Main Function to Run the Flask App ---
if __name__ == '__main__':
    # Run the Flask development server
    # debug=True provides helpful error messages and auto-reloads on code changes.
    # host='0.0.0.0' makes the server accessible from other devices on your network.
    app.run(debug=True, host='0.0.0.0')

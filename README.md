# Machine-Learning-Approach-for-Employee-Performance-Prediction


# 1. Overview
This project provides a Machine Learning Approach for Garment Worker Productivity Prediction. It's a web application that analyzes operational and team data to predict actual_productivity, aiding in resource allocation, performance improvement, and operational planning.

# 2. Key Features & Scenarios
Predictive Analytics: Forecasts actual_productivity using features like quarter, department, day, team size, SMV, overtime, incentives, and idle time.
Resource Optimization: Helps allocate workers and teams effectively.
Performance Insight: Identifies factors impacting productivity for targeted improvements.
User Interface: A simple Flask web app for input and instant predictions.

# 3. Dataset
Uses the "Garments Worker Productivity" dataset, including:
Categorical: quarter, department, day, team
Numerical: no_of_workers, no_of_style_change, smv, over_time, incentive, idle_time, idle_men
Target: actual_productivity
The dataset is in the Training/ directory.

# 4. Model Explanation
The prediction engine is an XGBoost Regressor. This advanced ensemble model combines many decision trees, learning sequentially from errors to accurately predict actual_productivity by capturing complex relationships in the data.

# 5. Technologies Used
Python: Core programming language
Flask: Web application framework
Pandas, NumPy: Data handling
Scikit-learn: Preprocessing and evaluation
XGBoost: Predictive model
Joblib: Model persistence
Jupyter Notebook: Development workflow
HTML & Tailwind CSS: User interface

# 6. Project Structure
Garment_Productivity_Predictor/
├── templates/
│   ├── about.html
│   ├── home.html
│   ├── predict.html
│   └── submit.html
├── Training/
│   ├── garments_worker_productivity.csv
│   ├── Employee_Productivity_Prediction.ipynb
│   ├── gwp.pkl
│   └── preprocessor.pkl
├── app.py
├── requirements.txt
└── README.md


# 7. Getting Started
7.1. Prerequisites
Python 3.8+
Git (optional)

7.2. Clone the Repository
git clone https://github.com/shivkumar31/Machine-Learning-Approach-for-Employee-Performance-Prediction.git
cd Garment_Productivity_Predictor


7.3. Setup Virtual Environment & Install Dependencies
python -m venv venv
Windows: .\venv\Scripts\activate

pip install -r requirements.txt


# 8. How to Use the Project
8.1. Step 1: Train the Model
Open Jupyter Notebook:
jupyter notebook

Navigate to and open Training/Employee_Productivity_Prediction.ipynb.
Run All Cells: Execute all cells in the notebook. This will load data, preprocess it, train the XGBoost model, evaluate it, and save gwp.pkl and preprocessor.pkl to Training/.

8.2. Step 2: Run the Web Application
Navigate to Project Root: Ensure your terminal is in the Garment_Productivity_Predictor/ directory with the virtual environment activated.
Start Flask Application:
python app.py

The server will run on http://127.0.0.1:5000/.
Access Web Interface: Open http://127.0.0.1:5000/ in your browser.
Interact: Navigate to "Predict Now", fill the form with team details, click "Submit", and view the predicted productivity.

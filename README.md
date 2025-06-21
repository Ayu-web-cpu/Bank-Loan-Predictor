🏦 Bank Loan Eligibility Predictor
This is a simple and interactive machine learning web app that predicts whether a loan should be approved based on user input such as age, income, education level, and credit score.

Built using Streamlit, scikit-learn, and Joblib, this project is designed to simulate a real-world fintech loan eligibility system.

🌐 Live App
🚀 Try it live:
👉 Bank Loan Predictor Web App

🎯 Objective
To develop a machine learning classifier that predicts the likelihood of a loan approval using key applicant details.

🧠 How It Works
✳️ Inputs (via UI):
Age: Slider between 21 to 60

Annual Income: Numeric input (₹20,000 to ₹10 Cr)

Education: Dropdown (Graduate / Not Graduate)

Credit Score: Slider between 300 and 900

⚙️ Processing:
Education is encoded using LabelEncoder (education_encoder.pkl)

Input features are scaled using StandardScaler (scaler.pkl)

Prediction is made using a trained Random Forest model (rf_model.pkl)

Probabilities are shown for transparency

✅ Output:
If prediction is 1 → ✔️ Loan Approved!

If prediction is 0 → ❌ Loan Not Approved

📁 Project Structure
graphql
Copy
Edit
Bank-Loan-Predictor/
├── main.py                    # ✅ Main file
├── rf_model.pkl               # Trained Random Forest model
├── scaler.pkl                 # Scaler used during training
├── education_encoder.pkl      # LabelEncoder for Education feature
├── train_and_save_model.py    # Script to train and export model + encoders
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview and documentation
🛠️ Tech Stack
Python 3.8+

Streamlit – for building the web interface

scikit-learn – for model building and preprocessing

Joblib – for saving/loading model and preprocessors

NumPy – for numerical transformations

📊 Model Summary
Model: Random Forest Classifier

Input Features: Age, Annual Income, Education, Credit Score

Target Variable: Loan Approval (Yes/No)

Model Artifacts: Saved using joblib for easy deployment

✅ Expected Outcome
Predict loan approval status with confidence scores

Realistic simulation of a bank’s credit approval frontend

A base app that can be extended to include more features like employment history, debt ratio, etc.

🙌 Acknowledgements
Developed under the RISE Internship Program






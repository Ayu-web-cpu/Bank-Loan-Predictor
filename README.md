# 🏦 Bank Loan Eligibility Predictor

A simple, interactive machine learning web app that predicts whether a bank loan should be **approved** based on user input such as age, income, education level, and credit score.

Built with **Streamlit**, **scikit-learn**, and **Joblib**, this project simulates a real-world fintech loan eligibility system.

---

## 🌐 Live App

🚀 **Try it live:**  
👉 [Bank Loan Predictor Web App](#) <!-- Add deployed URL here -->https://bank-loan-predictor-o7pssjesfcqx63bgfnuk3w.streamlit.app/

---

## 🎯 Objective

Develop a machine learning classifier to predict the likelihood of loan approval using key applicant details.

---

## 🧠 How It Works

### ✳️ User Inputs (via UI)
- **Age:** Slider (21 to 60)
- **Annual Income:** Numeric input (₹20,000 to ₹10 Cr)
- **Education:** Dropdown (`Graduate` / `Not Graduate`)
- **Credit Score:** Slider (300 to 900)

### ⚙️ Processing
- **Education** is encoded using `LabelEncoder` (`education_encoder.pkl`)
- Input features are **scaled** using `StandardScaler` (`scaler.pkl`)
- Prediction is made using a trained **Random Forest** model (`rf_model.pkl`)
- **Probabilities** are shown for transparency

### ✅ Output
- **Prediction 1:** ✔️ Loan Approved!
- **Prediction 0:** ❌ Loan Not Approved

---

## 📁 Project Structure

```
Bank-Loan-Predictor/
├── main.py                    # Main file
├── rf_model.pkl               # Trained Random Forest model
├── scaler.pkl                 # Scaler used during training
├── education_encoder.pkl      # LabelEncoder for Education feature
├── train_and_save_model.py    # Script to train and export model & encoders
├── requirements.txt           # Project dependencies
└── README.md                  # Project overview and documentation
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- [Streamlit](https://streamlit.io/) – Web interface
- [scikit-learn](https://scikit-learn.org/) – Model building & preprocessing
- [Joblib](https://joblib.readthedocs.io/) – Model & preprocessor serialization
- [NumPy](https://numpy.org/) – Numerical transformations

## Output Example

Here’s a screenshot of the output:

![Screenshot 2025-07-09 101424](https://github.com/user-attachments/assets/9b7ddeaf-8b51-47a0-a963-f9807788cd11)


---

## 📊 Model Summary

- **Model:** Random Forest Classifier
- **Input Features:** Age, Annual Income, Education, Credit Score
- **Target Variable:** Loan Approval (Yes/No)
- **Artifacts:** Saved using `joblib` for easy deployment

---

## ✅ Expected Outcome

- Predict loan approval status with confidence scores
- Realistic simulation of a bank’s credit approval frontend
- Base app can be extended to include features like employment history, debt ratio, etc.

---

## 🙌 Acknowledgements

- Developed under the **RISE Internship Program**


---






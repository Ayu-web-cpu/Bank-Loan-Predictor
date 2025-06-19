# Bank Loan Approval Predictor

A simple, interactive web app that predicts whether a bank loan will be approved based on applicant details like age, income (up to ₹10 crore), education, and credit score. Built with Python, scikit-learn, and Streamlit.

---

##  Live Demo

Try the app instantly:  
[https://bank-loan-predictor-o7pssjesfcqx63bgfnuk3w.streamlit.app/](https://bank-loan-predictor-o7pssjesfcqx63bgfnuk3w.streamlit.app/)

---
##  Features

- **Instant Prediction:** Enter applicant details and get a loan approval prediction with probability.
- **Modern Web Interface:** Easy-to-use, responsive UI built with Streamlit.
- **Handles High Incomes:** Supports annual incomes up to ₹10 crore.
- **Open Source:** All code and model training steps are included.

---
## How It Works

1. **Data Generation:**  
   Synthetic applicant data is generated for model training.

2. **Preprocessing:**  
   - Handles missing values
   - Encodes education as numbers
   - Scales features for better model performance

3. **Model Training:**  
   - Trains a Random Forest Classifier to predict loan approval
   - Saves the model and preprocessing objects for later use

4. **Web App:**  
   - Users enter applicant details
   - The app preprocesses the input and predicts approval with probability

---

## 📝 Example Usage

- Open the web app in your browser.
- Enter values for age, income , education and credit score.
- Click "Predict Loan Approval" to see if the loan is likely to be approved.

---

##  Technologies Used

- **Python 3**
- **scikit-learn** 
- **pandas, numpy** 
- **Streamlit** 
- **joblib** 

---


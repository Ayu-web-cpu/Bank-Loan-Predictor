import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import matplotlib.pyplot as plt

DATA_FILE = 'loan_data.csv'
if not os.path.exists(DATA_FILE):
    data = {
        'age': np.random.randint(21, 60, 200),
        'income': np.random.randint(20000, 100000, 200),
        'education': np.random.choice(['Graduate', 'Not Graduate'], 200),
        'credit_score': np.random.randint(300, 900, 200),
        'loan_approved': np.random.choice([0, 1], 200)
    }
    df_sample = pd.DataFrame(data)
    df_sample.to_csv(DATA_FILE, index=False)
    print("Sample data created as loan_data.csv")

df = pd.read_csv(DATA_FILE)
# Add some missing values for realism
for col in ['age', 'income', 'credit_score']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

imputer = SimpleImputer(strategy='mean')
df[['age', 'income', 'credit_score']] = imputer.fit_transform(df[['age', 'income', 'credit_score']])

le = LabelEncoder()
df['education'] = le.fit_transform(df['education'].astype(str))

X = df[['age', 'income', 'education', 'credit_score']]
y = df['loan_approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lr = LogisticRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

def evaluate_model(y_test, y_pred, y_prob, model_name):
    print(f"\n{model_name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    return roc_auc

plt.figure(figsize=(8,6))
evaluate_model(y_test, y_pred_lr, y_prob_lr, "Logistic Regression")
evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


def get_input(prompt, cast_func, valid_func=None):
    while True:
        try:
            value = cast_func(input(prompt))
            if valid_func and not valid_func(value):
                print("Invalid value. Try again.")
                continue
            return value
        except Exception:
            print("Invalid input. Try again.")

print("\n--- Loan Approval Prediction for New Applicant ---")
age = get_input("Age (21-60): ", float, lambda x: 21 <= x <= 60)
income = get_input("Annual Income (20000-100000): ", float, lambda x: 20000 <= x <= 100000)
education = ""
while education not in ['Graduate', 'Not Graduate']:
    education = input("Education (Graduate/Not Graduate): ").strip()
    if education not in ['Graduate', 'Not Graduate']:
        print("Please enter 'Graduate' or 'Not Graduate'.")
credit_score = get_input("Credit Score (300-900): ", float, lambda x: 300 <= x <= 900)

education_encoded = le.transform([education])[0]
X_new = np.array([[age, income, education_encoded, credit_score]])
X_new_scaled = scaler.transform(X_new)

prediction = rf.predict(X_new_scaled)[0]
prob = rf.predict_proba(X_new_scaled)[0][1]

if prediction == 1:
    print(f"\nLoan Approved! (Probability: {prob:.2f})")
else:
    print(f"\nLoan Not Approved. (Probability: {prob:.2f})")

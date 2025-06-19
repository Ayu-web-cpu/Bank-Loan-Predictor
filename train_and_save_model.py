import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Create sample data with income up to 10 crore
data = {
    'age': np.random.randint(21, 60, 200),
    'income': np.random.randint(20000, 100000001, 200),  # Up to 10 crore
    'education': np.random.choice(['Graduate', 'Not Graduate'], 200),
    'credit_score': np.random.randint(300, 900, 200),
    'loan_approved': np.random.choice([0, 1], 200)
}
df = pd.DataFrame(data)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df[['age', 'income', 'credit_score']] = imputer.fit_transform(df[['age', 'income', 'credit_score']])

# Encode categorical feature
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'].astype(str))

# Features and target
X = df[['age', 'income', 'education', 'credit_score']]
y = df['loan_approved']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)

# Save model, scaler, encoder
joblib.dump(rf, 'rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'education_encoder.pkl')
print("Model, scaler, and encoder saved!")
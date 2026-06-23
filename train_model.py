import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib  # Digunakan untuk menyimpan objek .pkl

print("Memulai proses training model...")

# 1. Membaca dataset
df = pd.read_csv('alzheimers_disease_data.csv')

# 2. Pra-pemrosesan Data
cols_to_drop = [col for col in ['PatientID', 'DoctorInCharge'] if col in df.columns]
X = df.drop(columns=['Diagnosis'] + cols_to_drop)
y = df['Diagnosis']

# 3. Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Standarisasi Fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 5. Pelatihan Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Menyimpan model dan scaler ke dalam file .pkl
joblib.dump(model, 'model_alzheimer.pkl')
joblib.dump(scaler, 'scaler_alzheimer.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

print("Proses selesai! File model_alzheimer.pkl, scaler_alzheimer.pkl, dan feature_columns.pkl berhasil dibuat.")
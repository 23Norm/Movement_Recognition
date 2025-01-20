import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import json
import os

# Paths
DATA_PATH = "data/processed/filtered_data.csv"
MODEL_DIR = "model/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load the dataset
data = pd.read_csv(DATA_PATH)

# Separate features and labels
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save the label encoder
with open(os.path.join(MODEL_DIR, "encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

# Save model metadata
model_info = {
    "model_type": "Random Forest",
    "input_shape": X.shape[1],
    "output_labels": label_encoder.classes_.tolist(),
    "framework": "scikit-learn"
}
with open(os.path.join(MODEL_DIR, "model_info.json"), "w") as f:
    json.dump(model_info, f)

print("Model, scaler, and encoder saved successfully!")

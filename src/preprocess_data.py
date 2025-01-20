import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

input_file = os.path.join("data", "processed", "filtered_data.csv")
output_features = os.path.join("data", "processed", "features.csv")
output_labels = os.path.join("data", "processed", "labels.csv")

# Load the combined dataset
data = pd.read_csv(input_file)

# Filter for walking (label=1) and jumping (label=2) activities
filtered_data = data[data['label'].isin([1, 2])]

# Separate features (sensor readings) and labels (activity type)
X = filtered_data.iloc[:, :-1]  # All columns except the last one
y = filtered_data.iloc[:, -1]   # Last column (label)

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the processed features and labels
pd.DataFrame(X_scaled).to_csv(output_features, index=False)
y.to_csv(output_labels, index=False)

print(f"Features saved to: {output_features}")
print(f"Labels saved to: {output_labels}")

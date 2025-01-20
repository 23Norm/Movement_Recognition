import pandas as pd
import os

folder_path = '../raw/'

log_files = [f for f in os.listdir(folder_path) if '_self.log' in f]

log_files = [f for f in os.listdir(folder_path) if '_ideal.log' in f]

data_frames = []
for file_name in log_files:
    file_path = os.path.join(folder_path, file_name)
    print(f"Reading file: {file_name}")

    df = pd.read_csv(file_path, delim_whitespace=True, header=None)

    df['source_file'] = file_name

    data_frames.append(df)

combined_data = pd.concat(data_frames, ignore_index=True)

column_names = [
    "timestamp_sec", "timestamp_microsec",
]

sensors = [f"S{i}" for i in range(1, 10)]
sensor_features = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "MagX", "MagY", "MagZ", "Q1", "Q2", "Q3", "Q4"]

for sensor in sensors:
    for feature in sensor_features:
        column_names.append(f"{sensor}_{feature}")

column_names.append("label")

combined_data.columns = column_names + ['source_file']
print(f"Combined Data Shape: {combined_data.shape}")

combined_data.to_csv('combined_data.csv', index=False)

filtered_data = combined_data[combined_data['label'].isin([1, 2])]

print(f"Filtered Data Shape (walking & jumping): {filtered_data.shape}")

filtered_data.to_csv('filtered_data.csv', index=False)


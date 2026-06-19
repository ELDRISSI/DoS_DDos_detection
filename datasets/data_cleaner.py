import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Clean data (Handle Inf and NaN values)
def clean_dataset(df):
    # Strip whitespace from columns which is common in CIC datasets
    df.columns = df.columns.str.strip()
    
    # Drop columns that cause metadata overfitting (Data Leakage)
    metadata_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Timestamp']
    df = df.drop(columns=[col for col in metadata_cols if col in df.columns])
    
    # Replace inf with NaN and drop them
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

# 2. Your Stratified Downsampling Strategy (Excellent for reducing data size)
def stratified_downsample(df, target_col, max_samples_per_class=100000):
    classes = df[target_col].unique()
    sampled_dfs = []
    
    for c in classes:
        df_class = df[df[target_col] == c]
        if len(df_class) > max_samples_per_class:
            df_class = df_class.sample(n=max_samples_per_class, random_state=42)
        sampled_dfs.append(df_class)
        
    return pd.concat(sampled_dfs, axis=0).reset_index(drop=True)

# --- Process and Merge Your 4 Raw Parquet Files ---
files = [
    'DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet',
    'DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet',
    'DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet',
    'DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter.parquet'
]

print("Merging and Cleaning datasets...")
combined_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
combined_df = clean_dataset(combined_df)

# Standardize labels (strip any hidden spaces in class names)
combined_df['Label'] = combined_df['Label'].astype(str).str.strip()

# Cap large classes at 100,000 rows
print("Applying downsampling to large classes...")
downsampled_df = stratified_downsample(combined_df, target_col='Label', max_samples_per_class=100000)

# Encode Labels so they are numeric for all models (Random forest, LightGBM, etc.)
encoder = LabelEncoder()
downsampled_df['Label'] = encoder.fit_transform(downsampled_df['Label'])

# Save the encoder mapping so your friends know which number means which attack
mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print("Class Mapping for the team:", mapping)

# Split into Features and Target
X = downsampled_df.drop(columns=['Label'])
y = downsampled_df['Label']

# Stratified Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Create New Folder and Save the Baseline Dataset ---
output_folder = "cleaned_project_dataset"
os.makedirs(output_folder, exist_ok=True)

# Combine features and labels back together for saving
train_data = X_train.copy()
train_data['Label'] = y_train

test_data = X_test.copy()
test_data['Label'] = y_test

# Save as parquet (much faster and smaller compression than CSV)
train_data.to_parquet(f"{output_folder}/train_baseline.parquet", index=False)
test_data.to_parquet(f"{output_folder}/test_baseline.parquet", index=False)

print(f"✅ Success! Cleaned baseline datasets saved into the folder: '{output_folder}/'")
print("Zip this folder and send it to your friends.")
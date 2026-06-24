import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

# 1. Define the specific input files
file_names = [
    "DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter.parquet",
    "DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet",
    "DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet",
    "DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet"
]

def main():
    print("1. Loading datasets...")
    dfs = []
    for file in file_names:
        if os.path.exists(file):
            print(f"   -> Loaded {file}")
            dfs.append(pd.read_parquet(file))
        else:
            print(f"   -> WARNING: {file} not found! Please ensure it is in the same folder.")

    if not dfs:
        raise ValueError("No files were loaded. Please check your file paths.")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"   -> Total combined rows: {len(combined_df)}\n")

    # 2. Clean data & Prevent Data Leakage
    print("2. Cleaning dataset and removing leakage features...")
    combined_df.columns = combined_df.columns.str.strip()
    
    # FIX: Added 'Dst Port' and 'Protocol' to prevent the model from memorizing port rules
    metadata_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port', 'Protocol', 'Timestamp','Init Fwd Win Bytes', 'Init Bwd Win Bytes']
    cols_to_drop = [col for col in metadata_cols if col in combined_df.columns]
    combined_df = combined_df.drop(columns=cols_to_drop)

    # Handle Inf and NaN values
    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.dropna(inplace=True)
    print(f"   -> Rows after cleaning: {len(combined_df)}\n")

    # 3. Stratified Downsampling Strategy (Shrinks huge classes like 'Benign' to save memory)
    print("3. Downsampling exceptionally large classes...")
    def stratified_downsample(df, target_col, max_samples_per_class=100000):
        classes = df[target_col].unique()
        sampled_dfs = []
        for c in classes:
            df_class = df[df[target_col] == c]
            if len(df_class) > max_samples_per_class:
                df_class = df_class.sample(n=max_samples_per_class, random_state=42)
            sampled_dfs.append(df_class)
        return pd.concat(sampled_dfs, ignore_index=True)

    downsampled_df = stratified_downsample(combined_df, target_col='Label', max_samples_per_class=100000)
    print(f"   -> Rows after downsampling: {len(downsampled_df)}\n")

    # NOTE: Label encoding has been REMOVED here to preserve your original string class names.

    # 4. Split into Features and Target
    X = downsampled_df.drop(columns=['Label'])
    y = downsampled_df['Label']

    # 5. Stratified Train/Test Split (80% Train, 20% Test)
    print("4. Splitting into Train and Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"   -> Training samples (before oversampling): {len(X_train)}")
    print(f"   -> Testing samples: {len(X_test)}\n")

    # 6. Apply Random Oversampling ONLY to Training Set
    print("5. Applying Random Oversampling to Training data...")
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    print(f"   -> Training samples (after oversampling): {len(X_train_resampled)}\n")

    # 7. Recombine and Save
    print("6. Saving ready-to-use Parquet files...")
    output_folder = "ready_project_dataset"
    os.makedirs(output_folder, exist_ok=True)

    # Save Training Data (Balanced/Oversampled)
    train_data = X_train_resampled.copy()
    train_data['Label'] = y_train_resampled
    train_file = os.path.join(output_folder, "train_ready.parquet")
    train_data.to_parquet(train_file, index=False)
    print(f"   -> Saved: {train_file}")

    # Save Testing Data (Original Real-World Distribution)
    test_data = X_test.copy()
    test_data['Label'] = y_test
    test_file = os.path.join(output_folder, "test_ready.parquet")
    test_data.to_parquet(test_file, index=False)
    print(f"   -> Saved: {test_file}")

    print("\n✅ Data processing complete! Your files are ready for the notebook.")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import os

# Define the dataset directory
DATASET_DIR = "datasets"

def run_health_check(directory):
    print("🕵️‍♂️ Starting Automated Dataset Integrity & Health Check (Parquet Edition)...\n" + "="*50)
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"❌ Error: The folder '{directory}' does not exist in this directory!")
        return
        
    # Automatically grab all Parquet files inside the folder
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    
    if not files:
        print(f"❌ No .parquet files found inside the '{directory}' folder!")
        print(f"Current folder contents: {os.listdir(directory)}")
        return
        
    print(f"🔍 Found {len(files)} Parquet files to analyze.\n" + "-"*50)

    for file in files:
        file_path = os.path.join(directory, file)
        print(f"📊 Analyzing: {file}")
        
        try:
            # Read the parquet file
            df = pd.read_parquet(file_path)
            print(f"  • Shape: {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # 1. Check for Whitespaces in Column Names
            spaced_cols = [col for col in df.columns if col.strip() != col]
            if spaced_cols:
                print(f"  • ⚠️ Column Name Alert: {len(spaced_cols)} columns have hidden spaces!")
                print(f"    Examples: {spaced_cols[:3]}")
            else:
                print("  • ✅ Column Names: Clean (No hidden whitespaces)")
                
            # 2. Check for Missing Values (NaN)
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                nan_cols = df.columns[df.isna().any()].tolist()
                print(f"  • ⚠️ Missing Values (NaN): Found {nan_count:,} total missing values.")
                print(f"    Affected columns: {nan_cols[:3]}")
            else:
                print("  • ✅ Missing Values: Clean (0 NaN values)")
                
            # 3. Check for Infinite Values (inf / -inf)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_count = np.isinf(df[numeric_cols]).sum().sum()
            if inf_count > 0:
                inf_cols = df[numeric_cols].columns[np.isinf(df[numeric_cols]).any()].tolist()
                print(f"  • ⚠️ Infinite Values (inf): Found {inf_count:,} infinity values.")
                print(f"    Affected columns: {inf_cols}")
            else:
                print("  • ✅ Infinite Values: Clean (0 inf values)")
                
            # 4. Check for Data Type Anomalies
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            if 'Label' in text_cols:
                text_cols.remove('Label')
            if text_cols:
                print(f"  • ⚠️ Type Mismatch Alert: Numeric features loaded as text/object: {text_cols}")
            else:
                print("  • ✅ Data Types: Clean (All features are correctly numeric)")
                
            # 5. Label Distribution Inspection
            if 'Label' in df.columns:
                print("  • 🎯 Label Distribution:")
                label_counts = df['Label'].value_counts()
                for label, count in label_counts.items():
                    print(f"    - {str(label).strip()}: {count:,} rows")
            else:
                print("  • ❌ Error: No 'Label' column detected!")
                
        except Exception as e:
            print(f"  • ❌ Failed to read file. Error: {e}")
            
        print("-" * 50)

# Run the health check
run_health_check(DATASET_DIR)
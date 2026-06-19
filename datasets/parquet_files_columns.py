import os
import pyarrow.parquet as pq

# 1. Define the directory path
folder_path = r"c:\Users\01\Documents\GitHub\DoS_DDos_detection\datasets"

# 2. Find the first parquet file in that folder automatically
parquet_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]

if not parquet_files:
    print(f"No .parquet files found in {folder_path}")
else:
    # Pick the first parquet file found
    target_file = parquet_files[0]
    full_path = os.path.join(folder_path, target_file)
    
    # 3. Read metadata and print columns
    metadata = pq.read_metadata(full_path)
    print(f"Successfully opened: '{target_file}'")
    print(f"The Parquet file has {metadata.num_columns} columns.")

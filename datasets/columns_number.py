import pandas as pd

# Use the full absolute path to your dataset
file_path = r"c:\Users\01\Documents\GitHub\DoS_DDos_detection\datasets\Wednesday-workingHours.pcap_ISCX.csv"

df = pd.read_csv(file_path)
num_columns = df.shape[1]  # Added [1] to get just the column count integer

print(f"The CSV file has {num_columns} columns.")

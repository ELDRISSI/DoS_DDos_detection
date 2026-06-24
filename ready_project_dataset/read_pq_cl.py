import pyarrow.parquet as pq

# Load only the metadata of the Parquet file
parquet_file = pq.ParquetFile('./test_ready.parquet')

# Extract and print column names
columns = parquet_file.schema.names
print(columns)

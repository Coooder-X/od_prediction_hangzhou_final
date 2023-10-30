import h5py
import numpy as np
import h5py
import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import dask.dataframe as dd
import time
file_path = r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\hangzhou\hangzhou_202005.h5'
h5_file = h5py.File(file_path, 'r')
trips_group = h5_file['trips']
timestamps_group = h5_file['timestamps']
print(len(timestamps_group))
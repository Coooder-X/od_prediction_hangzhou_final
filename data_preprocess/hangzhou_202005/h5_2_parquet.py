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
csv_file_path = r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\hangzhou\train.csv'  # 替换为您想要保存CSV文件的路径
columns = ['TIMESTAMP', 'dropoff_TIMESTAMP', 'POLYLINE']
df = pd.DataFrame(columns=columns)
with open(csv_file_path, 'w', newline='') as csv_file:
    num = 0
    for trip_data, timestamp_data in tqdm.tqdm(zip(trips_group.values(), timestamps_group.values())):
        trip_data = trip_data[:].tolist() # 获取数据集的值
        start_time = timestamp_data[0]
        end_time = timestamp_data[-1]
        df.loc[num] = [start_time, end_time, trip_data]
        num = num + 1


t1 = time.time()
df.to_feather(r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\hangzhou\merged_data.feather')
data_parquet = pd.read_feather(r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\hangzhou\merged_data.feather')
print(time.time() - t1)
del data_parquet
t2 = time.time()
df.to_parquet(r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\hangzhou\merged_data_engine.parquet', engine='pyarrow')
data_parquet = pd.read_parquet(r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\hangzhou\merged_data_engine.parquet')
print(time.time() - t2)
del data_parquet
t3 = time.time()
df.to_parquet(r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\hangzhou\merged_data.parquet', engine='pyarrow')
data_parquet = pd.read_parquet(r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\hangzhou\merged_data.parquet', engine='pyarrow')
print(time.time() - t3)





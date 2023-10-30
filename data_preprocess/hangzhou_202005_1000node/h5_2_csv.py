import h5py
import numpy as np
import h5py
import csv
import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time
import ast
import tqdm
csv_file_path = r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\hangzhou\train.csv'  # 替换为您想要保存CSV文件的路径
file_path = r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\hangzhou\hangzhou_202005.h5'
h5_file = h5py.File(file_path, 'r')
trips_group = h5_file['trips']
timestamps_group = h5_file['timestamps']


with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['TIMESTAMP','dropoff_TIMESTAMP', 'POLYLINE'])
    # num = 0
    for trip_data, timestamp_data in tqdm.tqdm(zip(trips_group.values(), timestamps_group.values())):
        trip_data = trip_data[:].tolist() # 获取数据集的值
        start_time = timestamp_data[0]
        end_time = timestamp_data[-1]
        csv_writer.writerow([start_time, end_time, trip_data])
        # num = num + 1
        # if num > 100000:
        #     break
# 关闭HDF5文件
h5_file.close()
# 先保存为CSV文件，然后在保存为其他的文件，不然速度太慢。
print("CSV file saved successfully.")
df = pd.read_csv(csv_file_path)
t0 = time.time()
data_parquet = df
# data_parquet = pd.read_feather(r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\train.feather')
print(time.time() - t0)

data_parquet['pickup_longitude'] = None
data_parquet['pickup_latitude'] = None
data_parquet['dropoff_longitude'] = None
data_parquet['dropoff_latitude'] = None
data_parquet['TRIP_ID'] = None
for i in tqdm.tqdm(range(len(data_parquet['TIMESTAMP']))):
    data_list = ast.literal_eval(data_parquet['POLYLINE'].values[i])
    data_parquet['pickup_longitude'].values[i] = data_list[0][0]
    data_parquet['pickup_latitude'].values[i] = data_list[0][1]
    data_parquet['dropoff_longitude'].values[i] = data_list[-1][0]
    data_parquet['dropoff_latitude'].values[i] = data_list[-1][1]
    data_parquet['dropoff_latitude'].values[i] = data_list[-1][1]
    data_parquet['TRIP_ID'].values[i] = i+1
t1 = time.time()
data_parquet = data_parquet[['TRIP_ID','TIMESTAMP',"dropoff_TIMESTAMP", 'POLYLINE','pickup_longitude',
           'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']]
data_parquet.to_feather(r'E:\WST_code\zhijiang_data\hangzhou_202005\raw_data\train.feather')

print(time.time() - t1)
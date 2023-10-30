import h5py
import pandas as pd
import numpy as np
# 打开并读取h5文件
with h5py.File(r'..\data\complete_data\all-trj.h5', 'r') as f:
    # 在文件中查找名为'dataset'的数据集
    vec_data = f['layer3'][:].tolist()
# 指定 CSV 文件的路径
file_path = r'../data/complete_data/train.csv'
# 使用 pd.read_csv() 函数读取 CSV 文件并转换为 DataFrame
row_data = pd.read_csv(file_path)
print(row_data.shape)
file_path = r'../data/complete_data/trj_ID'
datas = pd.read_csv(file_path, sep='\t', header=None, names=["TRIP_ID","vec"])
datas["vec"] = vec_data #f['layer1']
print(datas.shape)
merged_df = pd.merge(row_data, datas, on='TRIP_ID', how='inner')
print(merged_df.shape)
print(merged_df)
file_path = '../data/complete_data/train_with_vec.csv'  # 请替换为你想要保存的文件路径
merged_df.to_csv(file_path, index=False)  # 设置 index=False 来避免写入行索引



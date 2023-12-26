#!/usr/bin/env python
# coding: utf-8
import torch
import pandas as pd
from datetime import datetime
import time
import numpy as np
import os
import math
import tqdm
import sys
path = r'..\data\temp_data'
select_columns = ['TAXI_ID', 'TIMESTAMP', 'MISSING_DATA', 'POLYLINE','dropoff_TIMESTAMP', 'pickup_longitude',
                  'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','pickup_time_bin','dropoff_time_bin','vec']
df = pd.DataFrame(columns=select_columns)
filename = 'train.csv'
tmp_df = pd.read_csv(os.path.join(path, filename), header=0)
print(eval(tmp_df['vec'][0]))
df = pd.concat([df, tmp_df], axis=0)
df1 = df[df['MISSING_DATA'] == False]

for i in tqdm.tqdm(range(len(df1['TIMESTAMP']))):
    r = np.array(eval(df1['POLYLINE'].values[i]))
    df1['POLYLINE'].values[i] = r
    df1['vec'].values[i] = np.array(eval(df1['vec'][i]))
    if len(r) == 0:
        df1['MISSING_DATA'].values[i] = True
    else:
        df1['dropoff_TIMESTAMP'].values[i] = df1['TIMESTAMP'].values[i] + len(r)*15
        df1['pickup_longitude'].values[i] = r[0,0]
        df1['pickup_latitude'].values[i] = r[0,1]
        df1['dropoff_longitude'].values[i] = r[-1,0]
        df1['dropoff_latitude'].values[i] = r[-1,1]
print(np.stack(df1['vec']).shape)
print(df1['MISSING_DATA'].shape)
df2 = df1[df1['MISSING_DATA'] == False]
# MIN_LON = min(df2['pickup_longitude'].values.min(),df2['dropoff_longitude'].values.min())
# MAX_LON = max(df2['pickup_longitude'].values.max(),df2['dropoff_longitude'].values.max())
# MIN_LAT = min(df2['pickup_latitude'].values.min(),df2['dropoff_latitude'].values.min())
# MAX_LAT = max(df2['pickup_latitude'].values.max(),df2['dropoff_latitude'].values.max())
# print(MIN_LON,MAX_LON,MIN_LAT,MAX_LAT)
MIN_LONGITUDE = -8.735152
MAX_LONGITUDE = -8.156309
MIN_LATITUDE = 40.953673
MAX_LATITUDE = 41.307945
print('before:', df1.shape)
df2 = df2[
    (df2['pickup_longitude'] >= MIN_LONGITUDE) \
    & (df2['pickup_longitude'] <= MAX_LONGITUDE) \
    & (df2['pickup_latitude'] >= MIN_LATITUDE) \
    & (df2['pickup_latitude'] <= MAX_LATITUDE) \
    & (df2['dropoff_longitude'] >= MIN_LONGITUDE) \
    & (df2['dropoff_longitude'] <= MAX_LONGITUDE) \
    & (df2['dropoff_latitude'] >= MIN_LATITUDE) \
    & (df2['dropoff_latitude'] <= MAX_LATITUDE) \
    ]
print('after:', df2.shape)
print('removed:{}'.format(df1.shape[0] - df2.shape[0]))

min_TIMESTAMP = df2['TIMESTAMP'].values.min()
max_TIMESTAMP = df2['TIMESTAMP'].values.max()
min_dropoff_TIMESTAMP = df2['dropoff_TIMESTAMP'].values.min()
max_dropoff_TIMESTAMP = df2['dropoff_TIMESTAMP'].values.max()
df2['pickup_time_bin'] = ((df2['TIMESTAMP'] - min_TIMESTAMP)/(60*60)).astype(int)
df2['dropoff_time_bin'] = ((df2['dropoff_TIMESTAMP'] - min_TIMESTAMP)/(60*60)).astype(int)

df2.sort_values(by='pickup_time_bin', axis=0, ascending=True, inplace=True)
df3 = df2[['TAXI_ID', 'TIMESTAMP', 'MISSING_DATA', 'POLYLINE','dropoff_TIMESTAMP', 'pickup_longitude',
'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','pickup_time_bin','dropoff_time_bin','vec']]
print(df3.head())

# df3.to_csv('../data/temp_data/train_temp.csv', header=True,index=False)

# df4 = pd.read_csv('../data/temp_data/train_temp.csv', header=0)
# df4 = df3
# 按一小时计算2015 年1月有744个时间片
df4 = df3
time_steps = len(np.unique(df4["pickup_time_bin"]))
print("There should be ((24*60)/60)*31 unique 60 minute time bins for the month of January 2015: ", str(time_steps))
df4.head()
# # 3. 划分格子 按小时统计进出车流量
num_rows, num_cols = 16, 16
def get_grid(df):
    '''
    将上下车坐标映射到网格
    '''
    def mapping_location_grid(longitude, latitude):
        lon = round(longitude, 6)
        lat = round(latitude, 6)
        flag, grid = 0, 0
        if lon < MIN_LONGITUDE:
            flag = 0
            grid = 0
        elif lon > MAX_LONGITUDE:
            flag = 1
            grid = num_rows * num_cols - 1
        elif lat < MIN_LATITUDE:
            flag = 2
            grid = 0
        elif lat > MAX_LATITUDE:
            flag = 3
            grid = num_rows * num_cols - 1
        if (flag > 0):
            print('outliers:{},{}'.format(lon, lat))
        lon_column = (MAX_LONGITUDE - MIN_LONGITUDE) / num_rows
        lat_row = (MAX_LATITUDE - MIN_LATITUDE) / num_cols
        grid = int((lon - MIN_LONGITUDE) / lon_column) + int((lat - MIN_LATITUDE) / lat_row) * num_cols
        return grid
    vfunc = np.vectorize(mapping_location_grid)
    df['pickup_grid'] = vfunc(df['pickup_longitude'], df['pickup_latitude'])
    df['dropoff_grid'] = vfunc(df['dropoff_longitude'], df['dropoff_latitude'])
    return df

def get_inout_flow(pickup_df, dropoff_df, time_steps, row_num, col_num):
    '''
    获取每个格子每个小时的进出车流量
    '''
    image_list = []
    node_list = []
    timestep = []

    for timeslot in range(time_steps):
        pickup_tmpt = pickup_df[pickup_df['pickup_time_bin'] == timeslot]
        dropoff_tmpt = dropoff_df[dropoff_df['dropoff_time_bin'] == timeslot]

        image_feat = np.zeros(shape=(2, row_num, col_num))
        node_feat = np.zeros(shape=(row_num * col_num, 2))
        # 这里有点问题，OD矩阵的求解方法，不太对
        if (pickup_tmpt.shape[0] > 0):  # 该时间片内有上车记录
            for j in range(pickup_tmpt.shape[0]):
                pickup_row, pickup_col = int(pickup_tmpt.iloc[j]['pickup_grid'] / row_num), int(
                    pickup_tmpt.iloc[j]['pickup_grid'] % col_num)
                if (pickup_row >= row_num or pickup_col >= col_num):
                    continue
                image_feat[0, pickup_row, pickup_col] = pickup_tmpt.iloc[j]['pickup_order_num']
                node_feat[pickup_tmpt.iloc[j]['pickup_grid'], 0] = pickup_tmpt.iloc[j]['pickup_order_num']

        if (dropoff_tmpt.shape[0] > 0):  # 该时间片内有下车记录
            for j in range(dropoff_tmpt.shape[0]):
                dropoff_row, dropoff_col = int(dropoff_tmpt.iloc[j]['dropoff_grid'] / row_num), int(
                    dropoff_tmpt.iloc[j]['dropoff_grid'] % col_num)
                if (dropoff_row > row_num or dropoff_col > col_num):
                    continue
                image_feat[1, dropoff_row, dropoff_col] = dropoff_tmpt.iloc[j]['dropoff_order_num']
                node_feat[dropoff_tmpt.iloc[j]['dropoff_grid'], 1] = dropoff_tmpt.iloc[j]['dropoff_order_num']

        image_list.append(image_feat)
        node_list.append(node_feat)
        timestep.append(timeslot)

    image_list = np.stack(image_list)
    node_list = np.stack(node_list)

    print('image_list:{},node_list:{},timestep:{}'.format(image_list.shape, node_list.shape, len(timestep)))

    return image_list, node_list, np.array(timestep)

# 定义一个自定义函数来合并数列
def merge_lists(lst):
    data = np.stack(lst)
    return np.stack([data.max(axis=0),data.mean(axis=0),data.min(axis=0)])

df10 = get_grid(df4)
df10.head()

df10.to_csv('../data/temp_data/train_temp.csv', header=True,index=False)
del df, df1, df2, df3,df4,df10
df10 = pd.read_csv('../data/temp_data/train_temp.csv', header=0)
# 每个时间片上车统计
pickup_order = df10[["pickup_time_bin", "pickup_grid","vec"]].groupby(by=["pickup_time_bin", "pickup_grid"])
pickup_order_records = df10[["pickup_time_bin", "pickup_grid"]].groupby(by=["pickup_time_bin", "pickup_grid"])['pickup_time_bin'].size()
pickup_order_records = pickup_order_records.reset_index(name='pickup_order_num')
out = pd.merge(pickup_order_records, df10, on=["pickup_time_bin", "pickup_grid"])
dropoff_order_records = df10[["dropoff_time_bin", "dropoff_grid"]].groupby(by=["dropoff_time_bin", "dropoff_grid"])['dropoff_time_bin'].size()
dropoff_order_records = dropoff_order_records.reset_index(name='dropoff_order_num')

# vec_future = df10.groupby(["pickup_time_bin", "pickup_grid","dropoff_grid"],as_index=False)['vec'].apply(merge_lists)
vec_future = df10.groupby(["pickup_time_bin", "pickup_grid","dropoff_grid"],as_index=False)
pickup_order_records.head()
dropoff_order_records.head()

import gc
gc.collect()
# TODO 以后可以进行一定的修改
def get_od_matrix(start_time, end_time, file_dir, order_records, grid_num, vec_future):
    print('def get_od_matrix()!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    time_slot = start_time
    density = 0
    m_list = []
    while time_slot < end_time:
        tmpt = order_records[order_records['pickup_time_bin'] == time_slot]
        density += tmpt.shape[0] / (grid_num * grid_num)
        matrix = pd.DataFrame(np.zeros(shape=(grid_num, grid_num)))
        for j in range(tmpt.shape[0]):
            matrix[tmpt.iloc[j]['pickup_grid']][tmpt.iloc[j]['dropoff_grid']] = tmpt.iloc[j]['order_num']
            # vec = np.stack(vec_future.get_group((time_slot, tmpt.iloc[j]['pickup_grid'],tmpt.iloc[j]['dropoff_grid']))["vec"])

        m_list.append(matrix)
        print('Time slot:', time_slot, ' density:', density)
        time_slot += 1

    matrix_all = pd.DataFrame(pd.concat(m_list, ignore_index=True, axis=0).reset_index(drop=True))
    print('Start writing files:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    density = density / 48.0
    print('Finish finally, ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return matrix_all, density

def get_select_od_matrix(start_time, end_time, file_dir, order_records, grid_num, vec_future, index_select, len_index_list):
    print('def get_od_matrix()!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    time_slot = start_time
    density = 0
    m_list = []
    v_list = []
    while time_slot < end_time:
        tmpt = order_records[order_records['pickup_time_bin'] == time_slot]
        density += tmpt.shape[0] / (grid_num * grid_num)
        matrix = pd.DataFrame(np.zeros(shape=(len_index_list)))
        vec_maxtrix = pd.DataFrame(np.zeros(shape=(len_index_list,256)))
        for j in range(tmpt.shape[0]):
            new_index = tmpt.iloc[j]['pickup_grid']*256 + tmpt.iloc[j]['dropoff_grid']
            if dic[new_index] != -1:
                matrix.iloc[dic[new_index]] = tmpt.iloc[j]['order_num']
                vec_maxtrix.iloc[dic[new_index]] = np.stack(vec_future.get_group((time_slot, tmpt.iloc[j]['pickup_grid'],tmpt.iloc[j]['dropoff_grid']))["vec"])[0]
        m_list.append(matrix.values)
        v_list.append(vec_maxtrix.values)
        print('Time slot:', time_slot, ' density:', density)
        time_slot += 1

    matrix_all = np.stack(m_list)
    vec_maxtrix = np.stack(v_list)
    print('Start writing files:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    density = density / 48.0
    print('Finish finally, ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return matrix_all, vec_maxtrix


order_records = df10[["pickup_time_bin", "pickup_grid", "dropoff_grid"]].groupby(by=["pickup_time_bin", "pickup_grid", "dropoff_grid"])[
    'pickup_time_bin'].size()
order_records = order_records.reset_index(name='order_num')
matrix_all, density = get_od_matrix(start_time=0, end_time=time_steps, file_dir='../data/temp_data/od_matrix.csv',
                                    order_records=order_records, grid_num=num_rows * num_cols, vec_future = vec_future)
grid_num = num_rows * num_cols
matrix_all_od = np.reshape(matrix_all.values, (-1, grid_num, grid_num))
print(matrix_all_od.shape)
# np.savez('../data/temp_data/od_flow_image.npz', data=matrix_all_od)



OD_data = torch.from_numpy(matrix_all_od.reshape(matrix_all_od.shape[0], -1))
p_num = OD_data[:, :].mean(dim=0)
selet = np.where(p_num > 0.02, True, False)
index = torch.linspace(0, len(selet) - 1, len(selet)).int()
OD_node = OD_data.permute(1, 0)[selet].permute(1, 0)
index_list = list(index[selet].numpy())
from collections import defaultdict
dic = defaultdict(lambda: -1)
for i in range(len(index_list)):
    dic[index_list[i]] = i

index_select = set(index_list)

matrix_all, vec_maxtrix = get_select_od_matrix(start_time=0, end_time=time_steps, file_dir='../data/temp_data/od_matrix.csv',
                                    order_records=order_records, grid_num=num_rows * num_cols, vec_future = vec_future, index_select= dic,len_index_list = len(index_list) )





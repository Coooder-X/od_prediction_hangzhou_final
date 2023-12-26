#!/usr/bin/env python
# coding: utf-8

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

# 经纬度范围
# MIN_LONGITUDE = MIN_LON-0.0000001
# MAX_LONGITUDE = MAX_LON+0.0000001
#
# MIN_LATITUDE = MIN_LAT-0.0000001
# MAX_LATITUDE = MAX_LAT+0.0000001
df2.sort_values(by='pickup_time_bin', axis=0, ascending=True, inplace=True)
df3 = df2[['TAXI_ID', 'TIMESTAMP', 'MISSING_DATA', 'POLYLINE','dropoff_TIMESTAMP', 'pickup_longitude',
'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','pickup_time_bin','dropoff_time_bin','vec']]
print(df3.head())

# df3.to_csv('../data/temp_data/train_temp.csv', header=True,index=False)
# del df, df1, df2, df3
# df4 = pd.read_csv('../data/temp_data/train_temp.csv', header=0)

df4 = df3
# 按一小时计算2015 年1月有744个时间片
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
    # for i in range(len(lst)):
    #     lst[i] = eval(lst[i])
    data = np.stack(lst)
    return np.stack([data.max(axis=0),data.mean(axis=0),data.min(axis=0)])


df10 = get_grid(df4)
df10.head()

pickup_order = df10[["pickup_time_bin", "pickup_grid","vec"]].groupby(by=["pickup_time_bin", "pickup_grid"])
print(pickup_order)
# 每个时间片上车统计
pickup_order_records = df10[["pickup_time_bin", "pickup_grid"]].groupby(by=["pickup_time_bin", "pickup_grid"])[
    'pickup_time_bin'].size()

print(pickup_order_records)
pickup_order_records = pickup_order_records.reset_index(name='pickup_order_num')

df10['vec_col3'] = df10.groupby(["pickup_time_bin", "pickup_grid"],as_index=False)['vec'].apply(merge_lists)
df10.drop('vec', axis=1, inplace=True)
df10 = df10.dropna(axis=0, how='any')
out = pd.merge(pickup_order_records, df10, on=["pickup_time_bin", "pickup_grid"])
print(out.shape)

# 每个时间片下车统计
dropoff_order_records = df10[["dropoff_time_bin", "dropoff_grid"]].groupby(by=["dropoff_time_bin", "dropoff_grid"])[
    'dropoff_time_bin'].size()
dropoff_order_records = dropoff_order_records.reset_index(name='dropoff_order_num')
pickup_order_records.head()
dropoff_order_records.head()

import gc
# del df4, df5, df6, df7, df8
gc.collect()
# 将上下车记录映射到网格
image_list, node_list, timestep = get_inout_flow(pickup_order_records, dropoff_order_records, time_steps, num_rows,
                                                 num_cols)

node_list[1].sum(axis=0)
image_list[1].sum(axis=2).sum(axis=1)
# inout流量的处理完毕
np.savez('../data/temp_data/in_out_image.npz', data=image_list)
np.savez('../data/temp_data/inout_node_feat.npz', data=node_list)
del image_list, node_list, timestep
gc.collect()
m_1 = np.load('../data/temp_data/in_out_image.npz')
print(m_1['data'].shape)

m_list = m_1['data']

img = m_list[0, 1, :, :]

import matplotlib.pyplot as plt

# def show_image(img):
#     plt.imshow(img)
#     plt.colorbar()
#     plt.axis('off')
#     plt.show()
#
# show_image(m_list[0, 0, :, :])
#
# show_image(m_list[1, 0, :, :])
#
# show_image(m_list[1, 1, :, :])
#
# show_image(m_list[2, 0, :, :])
#
# show_image(m_list[2, 1, :, :])
#
# show_image(m_list[21, 1, :, :])
#
# def show_flow(arr, i, j):
#     x = list(np.arange(120))
#     y1 = arr[:120, 0, i, j]
#     y2 = arr[:120, 1, i, j]
#     plt.plot(x, y1, color='blue', label='inflow')
#     plt.plot(x, y2, color='red', label='outflow')
#     plt.xlabel('time')
#     plt.xlabel('num')
#     plt.title('flow of {},{}'.format(i, j))
#     plt.legend(loc=1)
#     plt.show()
#
# show_flow(m_list, 14, 8)
#
# show_flow(m_list, 14, 9)
#
# show_flow(m_list, 0, 0)
#
# show_flow(m_list, 12, 3)
#
# show_flow(m_list, 15, 0)
#
# show_flow(m_list, 15, 3)
#
# show_flow(m_list, 15, 4)

# 4.按小时统计OD车流量

# 在文献中使用的是完整的OD矩阵
# TODO 以后可以进行一定的修改
def get_od_matrix(start_time, end_time, file_dir, order_records, grid_num):
    print('def get_od_matrix()!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    time_slot = start_time
    density = 0
    m_list = []

    while time_slot < end_time:
        tmpt = order_records[order_records['pickup_time_bin'] == time_slot]
        #         print('#####################################################################')
        #         print('Order Records in this hour:', tmpt['order_num'].sum())
        density += tmpt.shape[0] / (grid_num * grid_num)
        matrix = pd.DataFrame(np.zeros(shape=(grid_num, grid_num)))

        for j in range(tmpt.shape[0]):
            matrix[tmpt.iloc[j]['pickup_grid']][tmpt.iloc[j]['dropoff_grid']] = tmpt.iloc[j]['order_num']

        m_list.append(matrix)
        print('Time slot:', time_slot, ' density:', density)
        time_slot += 1

    matrix_all = pd.DataFrame(pd.concat(m_list, ignore_index=True, axis=0).reset_index(drop=True))
    print('Start writing files:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print(matrix_all.shape)
    matrix_all.to_csv(file_dir, header=False, index=False)
    density = density / 48.0

    #     print'data_density = ', density
    print('Finish finally, ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    return matrix_all, density

order_records = \
df10[["pickup_time_bin", "pickup_grid", "dropoff_grid"]].groupby(by=["pickup_time_bin", "pickup_grid", "dropoff_grid"])[
    'pickup_time_bin'].size()

order_records = order_records.reset_index(name='order_num')

matrix_all, density = get_od_matrix(start_time=0, end_time=time_steps, file_dir='../data/temp_data/od_matrix.csv',
                                    order_records=order_records, grid_num=num_rows * num_cols)

grid_num = num_rows * num_cols
matrix_all_od = np.reshape(matrix_all.values, (-1, grid_num, grid_num))

# for i in range(1):
#     show_image(matrix_all_od[i:(i + 1)].squeeze(0))

def show_od_flow(matrix, start, end, i, j):
    x = list(np.arange(start, end, 1))
    y = matrix_all_od[start:end, i, j]

    plt.plot(x, y, color='blue')
    plt.show()

# show_od_flow(matrix_all_od, 0, 120, 148, 148)
#
# show_od_flow(matrix_all_od, 0, 120, 0, 0)
#
# show_od_flow(matrix_all_od, 0, 120, 220, 221)

def get_od_image(matrix_all):
    od_image_all = []
    for i in range(time_steps):
        out_od_matrix = matrix_all_od[i].reshape((-1, num_rows, num_cols))
        in_od_matrix = out_od_matrix.T.reshape((-1, num_rows, num_cols))
        all_od = np.concatenate([out_od_matrix, in_od_matrix], axis=0)
        od_image_all.append(all_od)
    od_img_all = np.stack(od_image_all, axis=0)

    return od_img_all



od_img_all = get_od_image(matrix_all_od)
np.savez('../data/temp_data/od_flow_image.npz', data=od_img_all)

m_2 = np.load('../data/temp_data/od_flow_image.npz')
print(m_2['data'].shape)
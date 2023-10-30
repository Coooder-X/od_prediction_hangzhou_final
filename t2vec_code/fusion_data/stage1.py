#!/usr/bin/env python
# coding: utf-8
import gc
gc.collect()
import pandas as pd
import numpy as np
import os
import tqdm
import gc
R_NUM = 6
C_NUM = 6
NODE_NUM = R_NUM * C_NUM
num_rows, num_cols = R_NUM, C_NUM
gc.collect()

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








if __name__ == '__main__':
    path = r'..\data\complete_data'
    select_columns = ['TAXI_ID', 'TIMESTAMP', 'MISSING_DATA', 'POLYLINE','dropoff_TIMESTAMP', 'pickup_longitude',
                      'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude','pickup_time_bin','dropoff_time_bin','vec']
    df = pd.DataFrame(columns=select_columns)
    filename = 'train_with_vec.csv'
    tmp_df = pd.read_csv(os.path.join(path, filename), header=0)
    print(eval(tmp_df['vec'][0]))
    df = pd.concat([df, tmp_df], axis=0)
    df1 = df[df['MISSING_DATA'] == False]
    for i in tqdm.tqdm(range(len(df1['TIMESTAMP']))):
        r = np.array(eval(df1['POLYLINE'].values[i]))
        df1['POLYLINE'].values[i] = r
        df1['vec'].values[i] = list(eval(df1['vec'][i]))
        if len(r) == 0:
            df1['MISSING_DATA'].values[i] = True
        else:
            df1['dropoff_TIMESTAMP'].values[i] = df1['TIMESTAMP'].values[i] + len(r)*15
            df1['pickup_longitude'].values[i] = r[0,0]
            df1['pickup_latitude'].values[i] = r[0,1]
            df1['dropoff_longitude'].values[i] = r[-1,0]
            df1['dropoff_latitude'].values[i] = r[-1,1]

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

    df4 = df3
    time_steps = len(np.unique(df4["pickup_time_bin"]))
    print("There should be ((24*60)/60)*31 unique 60 minute time bins for the month of January 2015: ", str(time_steps))
    df4.head()
    # # 3. 划分格子 按小时统计进出车流量
    num_rows, num_cols = R_NUM, C_NUM


    df10 = get_grid(df4)
    df10.head()

    df10.to_csv("../data/temp_data/train_all_temp_" + str(NODE_NUM) + ".csv", header=True,index=False)

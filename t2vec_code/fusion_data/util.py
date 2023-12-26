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
import torch
import torch.nn.functional as F
MIN_LONGITUDE = -8.735152
MAX_LONGITUDE = -8.156309
MIN_LATITUDE = 40.953673
MAX_LATITUDE = 41.307945
num_rows = 16
num_cols = 16
def get_pre_data():
    path = r'..\data\temp_data'
    select_columns = ['TAXI_ID', 'TIMESTAMP', 'MISSING_DATA', 'POLYLINE', 'dropoff_TIMESTAMP', 'pickup_longitude',
                      'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_time_bin', 'dropoff_time_bin',
                      'vec']
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
            df1['dropoff_TIMESTAMP'].values[i] = df1['TIMESTAMP'].values[i] + len(r) * 15
            df1['pickup_longitude'].values[i] = r[0, 0]
            df1['pickup_latitude'].values[i] = r[0, 1]
            df1['dropoff_longitude'].values[i] = r[-1, 0]
            df1['dropoff_latitude'].values[i] = r[-1, 1]
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
    df2['pickup_time_bin'] = ((df2['TIMESTAMP'] - min_TIMESTAMP) / (60 * 60)).astype(int)
    df2['dropoff_time_bin'] = ((df2['dropoff_TIMESTAMP'] - min_TIMESTAMP) / (60 * 60)).astype(int)

    # 经纬度范围
    # MIN_LONGITUDE = MIN_LON-0.0000001
    # MAX_LONGITUDE = MAX_LON+0.0000001
    #
    # MIN_LATITUDE = MIN_LAT-0.0000001
    # MAX_LATITUDE = MAX_LAT+0.0000001
    df2.sort_values(by='pickup_time_bin', axis=0, ascending=True, inplace=True)
    df3 = df2[['TAXI_ID', 'TIMESTAMP', 'MISSING_DATA', 'POLYLINE', 'dropoff_TIMESTAMP', 'pickup_longitude',
               'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_time_bin', 'dropoff_time_bin',
               'vec']]
    print(df3.head())

    df3.to_csv('../data/temp_data/train_temp.csv', header=True,index=False)
    del df, df1, df2 #,  df3
    # df4 = pd.read_csv('../data/temp_data/train_temp.csv', header=0)
    return df3


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

def get_select_od_matrix(start_time, end_time, file_dir, order_records, grid_num, vec_future, dic, len_index_list):
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


def get_index(row,column,all_column):
    index = row * all_column + column
    return index
def get_line_graph_with_vec(OD,index, flow, vec_maxtrix, R_NUM, C_NUM, NODE_NUM):
    # 将图进行转化为线图的节点特征
    A = torch.zeros(10,OD.shape[1],OD.shape[1])
    for i in tqdm.tqdm(range(OD.shape[1])):
        for j in range(OD.shape[1]):
            start_index1 = index[i] % NODE_NUM
            end_index1 = index[i] // NODE_NUM
            start_x1 = start_index1 % C_NUM
            start_y1 = start_index1 % C_NUM
            end_x1 = end_index1 % C_NUM
            end_y1 = end_index1 % C_NUM
            start_index2 = index[j] % NODE_NUM
            end_index2 = index[j] // NODE_NUM
            start_x2 = start_index2 % C_NUM
            start_y2 = start_index2 % C_NUM
            end_x2 = end_index2 % C_NUM
            end_y2 = end_index2 % C_NUM
            start_dis = 1/(((start_x1-start_x2)**2 + (start_y1-start_y2)**2)**0.5 + 0.001)
            end_dis = 1/(((end_x1-end_x2)**2 + (end_y1-end_y2)**2)**0.5 + 0.001)
            A[0,i,j] = start_dis
            A[1, i, j] = end_dis
            # 求相似度
            x_1 = OD[::18,i,0]
            y_1 = OD[::18,j,0]
            corr_1 = np.corrcoef(x_1, y_1)[0, 1]
            x_start_flow = flow[::, start_index1]
            y_start_flow = flow[::, start_index2]
            x_end_flow = flow[::, end_index1]
            y_end_flow = flow[::, end_index2]
            corr_start = np.corrcoef(x_start_flow, y_start_flow)[0, 1]
            corr_end = np.corrcoef(x_end_flow, y_end_flow)[0, 1]
            A[2, i, j] = corr_start
            A[3, i, j] = corr_end
            if (abs(start_x1 - start_x2) <=1 and abs(start_y1 - start_y2) <=0) or (abs(start_x1 - start_x2) <=0 and abs(start_y1 - start_y2) <=1):
                start_neig = 1
                A[4, i, j] = start_neig
            if (abs(end_x1 - end_x2) <=1 and abs(end_y1 - end_y2) <=0) or (abs(end_x1 - start_x2) <=0 and abs(end_y1 - end_y2) <=1):
                end_neig = 1
                A[5, i, j] = end_neig
            A[6, i, j] = corr_1

            vec1 = torch.from_numpy(vec_maxtrix[:,i, :].mean(axis=0))
            vec2 = torch.from_numpy(vec_maxtrix[:, j, :].mean(axis=0))
            cos_sim = F.cosine_similarity(vec1, vec2, dim=0)
            A[7, i, j] = cos_sim


    return A
def get_line_graph(OD,index, flow):
    # 将图进行转化为线图的节点特征
    A = torch.zeros(10,OD.shape[1],OD.shape[1])
    for i in tqdm.tqdm(range(OD.shape[1])):
        for j in range(OD.shape[1]):
            start_index1 = index[i] % 256
            end_index1 = index[i] // 256
            start_x1 = start_index1 % 16
            start_y1 = start_index1 % 16
            end_x1 = end_index1 % 16
            end_y1 = end_index1 % 16
            start_index2 = index[j] % 256
            end_index2 = index[j] // 256
            start_x2 = start_index2 % 16
            start_y2 = start_index2 % 16
            end_x2 = end_index2 % 16
            end_y2 = end_index2 % 16
            start_dis = 1/(((start_x1-start_x2)**2 + (start_y1-start_y2)**2)**0.5 + 0.001)
            end_dis = 1/(((end_x1-end_x2)**2 + (end_y1-end_y2)**2)**0.5 + 0.001)
            # 求相似度
            # corr =
            x_1 = OD[::18,i,0]
            y_1 = OD[::18,j,0]
            corr_1 = np.corrcoef(x_1, y_1)[0, 1]
            A[0,i,j] = start_dis
            A[5, i, j] = end_dis
            x_2 = OD[::18, i, 1]
            y_2 = OD[::18, j, 1]
            corr_2 = np.corrcoef(x_2, y_2)[0, 1]
            x_start_flow = flow[::18, start_index1]
            y_start_flow = flow[::18, start_index2]
            x_end_flow = flow[::18, end_index1]
            y_end_flow = flow[::18, end_index2]
            corr_start = np.corrcoef(x_start_flow, y_start_flow)[0, 1]
            corr_end = np.corrcoef(x_end_flow, y_end_flow)[0, 1]
            A[1, i, j] = corr_1
            A[6, i, j] = corr_2
            A[2, i, j] = corr_start
            A[7, i, j] = corr_end
            # 求相邻
            # if start_index1 == start_index2:
            if (abs(start_x1 - start_x2) <=1 and abs(start_y1 - start_y2) <=0) or (abs(start_x1 - start_x2) <=0 and abs(start_y1 - start_y2) <=1):
                start_neig = 1
                A[3, i, j] = start_neig
            #if end_index1 == end_index2:
            if (abs(end_x1 - end_x2) <=1 and abs(end_y1 - end_y2) <=0) or (abs(end_x1 - start_x2) <=0 and abs(end_y1 - end_y2) <=1):
                end_neig = 1
                A[8, i, j] = end_neig
            if abs(end_x1 - start_x2) == 1 and abs(end_y1 - start_y2) == 0:
                end_neig = 1
                A[4, i, j] = end_neig
            if abs(start_x1 - end_x2) == 1 and abs(start_y1 - end_y2) == 0:
                end_neig = 1
                A[9, i, j] = end_neig
                # TODO
                # pass
    return A
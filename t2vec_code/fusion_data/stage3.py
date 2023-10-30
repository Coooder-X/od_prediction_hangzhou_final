#!/usr/bin/env python
# coding: utf-8
import gc
gc.collect()
import pandas as pd
import time
import numpy as np
from collections import defaultdict
R_NUM = 10
C_NUM = 10
NODE_NUM = R_NUM * C_NUM
num_rows, num_cols = R_NUM, C_NUM


# TODO 以后可以进行一定的修改
def get_select_od_matrix(start_time, end_time, file_dir, order_records, grid_num, vec_future, index_select,
                         len_index_list):
    print('def get_od_matrix()！', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    time_slot = start_time
    density = 0
    m_list = []
    v_list = []
    while time_slot < end_time:
        tmpt = order_records[order_records['pickup_time_bin'] == time_slot]
        density += tmpt.shape[0] / (grid_num * grid_num)
        matrix = pd.DataFrame(np.zeros(shape=(len_index_list)))
        vec_maxtrix = pd.DataFrame(np.zeros(shape=(len_index_list, 256)))
        for j in range(tmpt.shape[0]):
            new_index = tmpt.iloc[j]['pickup_grid'] * num_cols*num_cols + tmpt.iloc[j]['dropoff_grid']
            if dic[new_index] != -1:
                matrix.iloc[dic[new_index]] = tmpt.iloc[j]['order_num']
                vec_maxtrix.iloc[dic[new_index]] = np.stack(
                    vec_future.get_group((time_slot, tmpt.iloc[j]['pickup_grid'], tmpt.iloc[j]['dropoff_grid']))[
                        "vec"]).mean(axis=0)
        m_list.append(matrix.values)
        v_list.append(vec_maxtrix.values)
        print(tmpt.shape[0])
        print('Time slot:', time_slot, ' density:', density)
        time_slot += 1

    matrix_all = np.stack(m_list)
    vec_maxtrix = np.stack(v_list)
    print('Start writing files:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    density = density / 48.0
    print('Finish finally, ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return matrix_all, vec_maxtrix

if __name__ == '__main__':
    # from t2vec_code.fusion_data.util import get_pre_data,get_grid,merge_lists,get_inout_flow,get_od_matrix, get_select_od_matrix,get_line_graph
    index_list = np.load('../data/temp_data/index_' + str(NODE_NUM) + '.npz')["data"]
    p_num = np.load('../data/temp_data/p_num_' + str(NODE_NUM) + '.npz')["data"]
    df10 = pd.read_csv('../data/temp_data/train_all_temp_' + str(NODE_NUM) + '.csv', header=0)
    df10["vec"] = df10["vec"].apply(lambda x: eval(x))
    time_steps = len(np.unique(df10["pickup_time_bin"]))
    # 每个时间片上车统计
    pickup_order = df10[["pickup_time_bin", "pickup_grid","vec"]].groupby(by=["pickup_time_bin", "pickup_grid"])
    pickup_order_records = df10[["pickup_time_bin", "pickup_grid"]].groupby(by=["pickup_time_bin", "pickup_grid"])['pickup_time_bin'].size()
    pickup_order_records = pickup_order_records.reset_index(name='pickup_order_num')
    out = pd.merge(pickup_order_records, df10, on=["pickup_time_bin", "pickup_grid"])
    dropoff_order_records = df10[["dropoff_time_bin", "dropoff_grid"]].groupby(by=["dropoff_time_bin", "dropoff_grid"])['dropoff_time_bin'].size()
    dropoff_order_records = dropoff_order_records.reset_index(name='dropoff_order_num')
    vec_future = df10.groupby(["pickup_time_bin", "pickup_grid","dropoff_grid"],as_index=False)
    pickup_order_records.head()
    dropoff_order_records.head()
    print(pickup_order_records)
    print(dropoff_order_records)

    order_records = df10[["pickup_time_bin", "pickup_grid", "dropoff_grid"]].groupby(by=["pickup_time_bin", "pickup_grid", "dropoff_grid"])[
        'pickup_time_bin'].size()
    order_records = order_records.reset_index(name='order_num')

    grid_num = num_rows * num_cols


    index_list = list(index_list)
    print(len(index_list))
    dic = defaultdict(lambda: -1)
    for i in range(len(index_list)):
        dic[index_list[i]] = i

    index_select = set(index_list)

    matrix_all, vec_maxtrix = get_select_od_matrix(start_time=0, end_time=time_steps, file_dir='../data/temp_data/od_matrix.csv',
                                        order_records=order_records, grid_num=num_rows * num_cols, vec_future = vec_future, index_select= dic,len_index_list = len(index_list) )
    print("*******************************")
    print(matrix_all.max())
    print(matrix_all.min())
    print(matrix_all.shape)
    print(vec_maxtrix.max())
    print(vec_maxtrix.min())
    print(vec_maxtrix.shape)
    np.savez('../data/temp_data/matrix_all_' + str(NODE_NUM) + '.npz', data=matrix_all)
    np.savez('../data/temp_data/vec_maxtrix_' + str(NODE_NUM) + '.npz', data=vec_maxtrix)



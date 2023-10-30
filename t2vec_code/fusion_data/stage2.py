#!/usr/bin/env python
# coding: utf-8
import gc
gc.collect()
import pandas as pd
import time
import numpy as np
import torch
R_NUM = 6
C_NUM = 6
NODE_NUM = R_NUM * C_NUM
num_rows, num_cols = R_NUM, C_NUM

# TODO 以后可以进行一定的修改
def get_od_matrix(start_time, end_time, file_dir, order_records, grid_num, vec_future):
    print('def get_od_matrix()！', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
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


if __name__ == '__main__':
    # from t2vec_code.fusion_data.util import get_pre_data,get_grid,merge_lists,get_inout_flow,get_od_matrix, get_select_od_matrix,get_line_graph
    df10 = pd.read_csv('../data/temp_data/train_all_temp_' + str(NODE_NUM) + '.csv', header=0)
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

    order_records = df10[["pickup_time_bin", "pickup_grid", "dropoff_grid"]].groupby(by=["pickup_time_bin", "pickup_grid", "dropoff_grid"])[
        'pickup_time_bin'].size()
    order_records = order_records.reset_index(name='order_num')
    matrix_all, density = get_od_matrix(start_time=0, end_time=time_steps, file_dir='../data/temp_data/od_matrix.csv',
                                        order_records=order_records, grid_num=num_rows * num_cols, vec_future = vec_future)
    grid_num = num_rows * num_cols
    matrix_all_od = np.reshape(matrix_all.values, (-1, grid_num, grid_num))
    print(matrix_all_od.shape)
    OD_data = torch.from_numpy(matrix_all_od.reshape(matrix_all_od.shape[0], -1))
    p_num = OD_data[:, :].mean(dim=0)
    print(torch.sort(p_num,dim=0).values)
    selet = np.where(p_num > np.sort(p_num,axis=0)[-100], True, False)
    index = torch.linspace(0, len(selet) - 1, len(selet)).int()
    OD_node = OD_data.permute(1, 0)[selet].permute(1, 0)
    matrix_flow = matrix_all_od.sum(axis=-1)
    np.savez('../data/temp_data/p_num_' + str(NODE_NUM) + '.npz', data=p_num)
    np.savez('../data/temp_data/matrix_flow_' + str(NODE_NUM) + '.npz', data=matrix_flow)
    np.savez('../data/temp_data/index_' + str(NODE_NUM) + '.npz', data=index[selet].numpy())



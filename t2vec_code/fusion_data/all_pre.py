#!/usr/bin/env python
# coding: utf-8
import gc
gc.collect()
import pandas as pd
from datetime import datetime
import time
import numpy as np
import os
import math
import tqdm
import sys
import gc
import matplotlib.pyplot as plt
import torch
num_rows, num_cols = 16, 16
gc.collect()
from t2vec_code.fusion_data.util import get_pre_data,get_grid,merge_lists,get_inout_flow,get_od_matrix, get_select_od_matrix,get_line_graph
path = r'..\data\temp_data'
df3 = get_pre_data()
# # df4 = pd.read_csv('../data/temp_data/train_temp.csv', header=0)
df4 = df3
# 按一小时计算2015 年1月有744个时间片
time_steps = len(np.unique(df4["pickup_time_bin"]))
print("There should be ((24*60)/60)*31 unique 60 minute time bins for the month of January 2015: ", str(time_steps))
df4.head()
# # 3. 划分格子 按小时统计进出车流量
df10 = get_grid(df4)
df10.head()
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
# TODO 以后可以进行一定的修改
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
                                    order_records=order_records, grid_num=num_rows * num_cols, vec_future = vec_future, dic= dic,len_index_list = len(index_list) )
matrix_flow = matrix_all_od.sum(axis=-1)
matrix_all = np.concatenate((matrix_all,matrix_all),axis=-1)
A = get_line_graph(matrix_all, np.array(index_list), matrix_flow)

np.savez("../data/line_graph_data/in_out_image.npz", data=matrix_all)
np.savez("../data/line_graph_data/od_flow_image.npz", data=A)


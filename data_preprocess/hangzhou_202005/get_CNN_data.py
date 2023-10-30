import numpy as np
import os
DATA_ROOT = r"E:\WST_code\zhijiang_data\hangzhou_202005_complete"
GRID_NUM = 100
hour_num = 0.25
matrix_all = np.load(os.path.join(DATA_ROOT,'temp_data/matrix_all_' + str(GRID_NUM) + '.npz'))["data"]
vec_maxtrix = np.load(os.path.join(DATA_ROOT,'temp_data/vec_maxtrix_' + str(GRID_NUM) + '.npz'))["data"]
num_list = []
vec_list = []
for i in range(4,len(matrix_all)+1):
    num = matrix_all[i-4:i]
    sum_num = num.sum(axis = 0)
    sum_num[sum_num == 0] = 1
    num = num.sum(axis = 0)
    num_list.append(num)


matrix_all = np.stack(num_list)
vec_maxtrix = np.stack(vec_list)
matrix_all = np.concatenate((matrix_all,matrix_all),axis=-1)
np.savez(os.path.join(DATA_ROOT,'split_data_0.25_hour/OD_node_' + str(GRID_NUM) + '.npz'), data=matrix_all)
np.savez(os.path.join(DATA_ROOT,'split_data_0.25_hour/vec_input_' + str(GRID_NUM) + '.npz'), data=vec_maxtrix)

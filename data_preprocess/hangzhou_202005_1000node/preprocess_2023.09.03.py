import h5py
import gc
gc.collect()
import sys
import os
import pickle
MIN_LONGITUDE = 120.1088
MAX_LONGITUDE = 120.1922
MIN_LATITUDE = 30.2335
MAX_LATITUDE = 30.3015
R_NUM = 10
C_NUM = 10
hour_num = 0.25
node_num = 10000
GRID_NUM = R_NUM * C_NUM
num_rows, num_cols = R_NUM, C_NUM
from ..hangzhou_202005.preprocess_unit import *
DATA_ROOT = r"/app/project/wst/data"
start_time = time.time()

'''
可以调整为科学模式一步一步的完成结果。
'''

#%%
'''
第一步，轨迹特征向量加在原始数据上
'''
# 打开并读取h5文件
# 指定 CSV 文件的路径
file_path = os.path.join(DATA_ROOT,"raw_data/train.feather")
row_data = pd.read_feather(file_path)

ID_file_path = os.path.join(DATA_ROOT,"raw_data/trj_ID")
df = pd.DataFrame(columns=["TRIP_ID","vec"])

datas = pd.read_csv(ID_file_path, sep='\t', header=None, names=["TRIP_ID","vec"])
with h5py.File(os.path.join(DATA_ROOT,"raw_data/all-trj.h5"), 'r') as f:
    vec = f['layer3'][:].tolist()
# vec = [np.zeros(256)]*len(datas)
datas["vec"] = vec
merged_df = pd.merge(row_data, datas, on='TRIP_ID', how='inner')
file_path = os.path.join(DATA_ROOT,"raw_data/train_with_vec.feather")
# merged_df.reset_index(drop=True, inplace=True)  # 去除索引
merged_df.to_feather(file_path)
print(merged_df.head)
del df, merged_df, datas, vec

#%%
'''
第二步，划分网格，上下车时间
'''
# path = os.path.join(DATA_ROOT,"row_data/")
select_columns = ['trj_ID', 'TIMESTAMP', 'MISSING_DATA', 'dropoff_TIMESTAMP', 'pickup_longitude',
                  'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_time_bin', 'dropoff_time_bin',
                  'vec']
df = pd.DataFrame(columns=select_columns)
filename = 'train_with_vec.feather'
tmp_df = pd.read_feather(os.path.join(DATA_ROOT, "raw_data" ,filename))
print(time.time() - start_time)
df2 = pd.concat([df, tmp_df], axis=0)

print('after:', df2.shape)
print('length:{}'.format(df2.shape[0]))
min_TIMESTAMP = df2['TIMESTAMP'].values.min()
max_TIMESTAMP = df2['TIMESTAMP'].values.max()
min_dropoff_TIMESTAMP = df2['dropoff_TIMESTAMP'].values.min()
max_dropoff_TIMESTAMP = df2['dropoff_TIMESTAMP'].values.max()
df2['pickup_time_bin'] = ((df2['TIMESTAMP'] - min_TIMESTAMP) / (60 * 60 * hour_num)).astype(int)
df2['dropoff_time_bin'] = ((df2['dropoff_TIMESTAMP'] - min_TIMESTAMP) / (60 * 30 * hour_num)).astype(int)
df2.sort_values(by='pickup_time_bin', axis=0, ascending=True, inplace=True)
df3 = df2[['trj_ID', 'TIMESTAMP','dropoff_TIMESTAMP', 'pickup_longitude',
           'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_time_bin', 'dropoff_time_bin', 'vec']]
df4 = df3
time_steps = len(np.unique(df4["pickup_time_bin"]))
print("There should be ((24*60)/60)*31 unique 60 minute time bins for the month of January 2015: ", str(time_steps))
df4.head()
# # 3. 划分格子 按小时统计进出车流量
df10 = get_grid_pandding(df4,MIN_LONGITUDE,MAX_LONGITUDE, MIN_LATITUDE, MAX_LATITUDE,num_rows-2, num_cols-2)
df10.head()
df10 = df10.reset_index()
df10.to_feather(os.path.join(DATA_ROOT,"temp_data/train_all_temp_" + str(GRID_NUM) + ".feather"))
del df,df2,df3,df4


#%%
'''
第三步，筛选节点
'''
# from data_preprocess.porto_data_preprocess.preprocess_unit import get_od_matrix
# 不需要保存时，注释掉
df10 = pd.read_feather(os.path.join(DATA_ROOT,'temp_data/train_all_temp_' + str(GRID_NUM) + '.feather'))
time_steps = len(np.unique(df10["pickup_time_bin"]))
# 每个时间片上车统计
pickup_order = df10[["pickup_time_bin", "pickup_grid", "vec"]].groupby(by=["pickup_time_bin", "pickup_grid"])
pickup_order_records = df10[["pickup_time_bin", "pickup_grid"]].groupby(by=["pickup_time_bin", "pickup_grid"])[
    'pickup_time_bin'].size()
pickup_order_records = pickup_order_records.reset_index(name='pickup_order_num')
out = pd.merge(pickup_order_records, df10, on=["pickup_time_bin", "pickup_grid"])
dropoff_order_records = df10[["dropoff_time_bin", "dropoff_grid"]].groupby(by=["dropoff_time_bin", "dropoff_grid"])[
    'dropoff_time_bin'].size()
dropoff_order_records = dropoff_order_records.reset_index(name='dropoff_order_num')
vec_future = df10.groupby(["pickup_time_bin", "pickup_grid", "dropoff_grid"], as_index=False)
pickup_order_records.head()
dropoff_order_records.head()
order_records = \
df10[["pickup_time_bin", "pickup_grid", "dropoff_grid"]].groupby(by=["pickup_time_bin", "pickup_grid", "dropoff_grid"])[
    'pickup_time_bin'].size()
order_records = order_records.reset_index(name='order_num')
matrix_all, density = get_od_matrix(start_time=0, end_time=time_steps,
                                    order_records=order_records, grid_num=num_rows * num_cols)
print(matrix_all.shape)
grid_num = num_rows * num_cols
# matrix_all_od = np.reshape(matrix_all.values, (-1, grid_num, grid_num))
matrix_all_od = matrix_all
OD_data = torch.from_numpy(matrix_all_od.reshape(matrix_all_od.shape[0], -1))
p_num = OD_data[:, :].mean(dim=0)
selet = np.where(p_num >= np.sort(p_num, axis=0)[-node_num], True, False)
index = torch.linspace(0, len(selet) - 1, len(selet)).int()
OD_node = OD_data.permute(1, 0)[selet].permute(1, 0)
matrix_flow = matrix_all_od.sum(axis=-1)
np.savez(os.path.join(DATA_ROOT,'temp_data/od_matrix_' + str(GRID_NUM) + '.npz'), data=matrix_all)
np.savez(os.path.join(DATA_ROOT,'temp_data/p_num_' + str(GRID_NUM) + '.npz'), data=p_num)
np.savez(os.path.join(DATA_ROOT,'temp_data/matrix_flow_' + str(GRID_NUM) + '.npz'), data=matrix_flow)
np.savez(os.path.join(DATA_ROOT,'temp_data/index_' + str(GRID_NUM) + '.npz'), data=index[selet].numpy())



#%%
'''
第四步，根据筛选的节点，生成OD数据，以及轨迹特征
'''
from collections import defaultdict
index_list = np.load(os.path.join(DATA_ROOT,'temp_data/index_' + str(GRID_NUM) + '.npz'))["data"]
p_num = np.load(os.path.join(DATA_ROOT,'temp_data/p_num_' + str(GRID_NUM) + '.npz'))["data"]
df10 = pd.read_feather(os.path.join(DATA_ROOT,'temp_data/train_all_temp_' + str(GRID_NUM) + '.feather'))
time_steps = len(np.unique(df10["pickup_time_bin"]))
# 每个时间片上车统计
pickup_order = df10[["pickup_time_bin", "pickup_grid", "vec"]].groupby(
               by=["pickup_time_bin", "pickup_grid"])
pickup_order_records = df10[["pickup_time_bin", "pickup_grid"]].groupby(
                       by=["pickup_time_bin", "pickup_grid"])['pickup_time_bin'].size()
pickup_order_records = pickup_order_records.reset_index(name='pickup_order_num')
out = pd.merge(pickup_order_records, df10, on=["pickup_time_bin", "pickup_grid"])
dropoff_order_records = df10[["dropoff_time_bin", "dropoff_grid"]].groupby(
                      by=["dropoff_time_bin", "dropoff_grid"])['dropoff_time_bin'].size()
dropoff_order_records = dropoff_order_records.reset_index(name='dropoff_order_num')
vec_future = df10.groupby(["pickup_time_bin", "pickup_grid", "dropoff_grid"], as_index=False)
pickup_order_records.head()
dropoff_order_records.head()
order_records = df10[["pickup_time_bin", "pickup_grid", "dropoff_grid"]].groupby(
                by=["pickup_time_bin", "pickup_grid", "dropoff_grid"])['pickup_time_bin'].size()
order_records = order_records.reset_index(name='order_num')
grid_num = num_rows * num_cols
index_list = list(index_list)
dic = defaultdict(lambda: -1)
for i in range(len(index_list)):
    dic[index_list[i]] = i
index_select = set(index_list)
# 可能需要修改
matrix_all, vec_maxtrix = get_select_od_matrix(start_time=0, end_time=time_steps, order_records=order_records,
                                               grid_num=num_rows * num_cols,vec_future=vec_future, dic=dic,
                                               len_index_list=len(index_list),num_cols = num_cols)
np.savez(os.path.join(DATA_ROOT,'temp_data/matrix_all_' + str(GRID_NUM) + '.npz'), data=matrix_all)
np.savez(os.path.join(DATA_ROOT,'temp_data/vec_maxtrix_' + str(GRID_NUM) + '.npz'), data=vec_maxtrix)


#%%
'''
第五步，生成图数据
'''
from data_preprocess.hangzhou_202005.preprocess_unit import get_line_graph_with_vec
matrix_flow = np.load(os.path.join(DATA_ROOT,'temp_data/matrix_flow_' + str(GRID_NUM) + '.npz'))["data"]
matrix_all = np.load(os.path.join(DATA_ROOT,'temp_data/matrix_all_' + str(GRID_NUM) + '.npz'))["data"]
index_list = np.load(os.path.join(DATA_ROOT,'temp_data/index_' + str(GRID_NUM) + '.npz'))["data"]
vec_maxtrix = np.load(os.path.join(DATA_ROOT,'temp_data/vec_maxtrix_' + str(GRID_NUM) + '.npz'))["data"]
matrix_all = np.concatenate((matrix_all,matrix_all),axis=-1)
A = get_line_graph_with_vec(matrix_all, np.array(index_list), matrix_flow,vec_maxtrix, C_NUM, GRID_NUM).numpy()
A = np.nan_to_num(A, copy=True)
matrices_to_used = ["start_dis", "end_dis", "start_flow_coor",
                    "end_flow_coor", "start_neighbor",
                    "end_neighbor", "od_flow_coor","vec_similar"]
dic = {}
for i in range(len(matrices_to_used)):
    dic[matrices_to_used[i]] = A[i]
np.savez(os.path.join(DATA_ROOT,'line_graph_data/OD_node_' + str(GRID_NUM) + '.npz'), data=matrix_all)
np.savez(os.path.join(DATA_ROOT,'line_graph_data/vec_input_' + str(GRID_NUM) + '.npz'), data=vec_maxtrix)
f_save = open(os.path.join(DATA_ROOT,'split_data_0.25_hour/A' + str(GRID_NUM) + '.pkl'), 'wb')
pickle.dump(dic, f_save)
f_save.close()



# %%
# '''
# 第六步，划分数据
# '''
matrix_all = np.load(os.path.join(DATA_ROOT,'temp_data/matrix_all_' + str(GRID_NUM) + '.npz'))["data"]
vec_maxtrix = np.load(os.path.join(DATA_ROOT,'temp_data/vec_maxtrix_' + str(GRID_NUM) + '.npz'))["data"]
num_list = []
vec_list = []
for i in range(4,len(matrix_all)+1):

    num = matrix_all[i-4:i]
    vec = vec_maxtrix[i-4:i]
    sum_num = num.sum(axis = 0)
    sum_num[sum_num == 0] = 1
    vec = (num*vec).sum(axis = 0)/ sum_num
    num = num.sum(axis = 0)
    num_list.append(num)
    vec_list.append(vec)


matrix_all = np.stack(num_list)
vec_maxtrix = np.stack(vec_list)
matrix_all = np.concatenate((matrix_all,matrix_all),axis=-1)
np.savez(os.path.join(DATA_ROOT,'split_data_0.25_hour/OD_node_' + str(GRID_NUM) + '.npz'), data=matrix_all)
np.savez(os.path.join(DATA_ROOT,'split_data_0.25_hour/vec_input_' + str(GRID_NUM) + '.npz'), data=vec_maxtrix)

from data_preprocess.hangzhou_202005.preprocess_unit import get_dataloader
from train import get_dataset_args
import os
# args = get_dataset_args()
read_path = os.path.join(DATA_ROOT,"split_data_0.25_hour")
save_path = os.path.join(DATA_ROOT,"split_data_0.25_hour")
get_dataloader(1, 1, 2, 0.8, 0.2, read_path, save_path, GRID_NUM,if_vec=False, hour_num = hour_num )
get_dataloader(1, 1, 2, 0.8, 0.2, read_path, save_path, GRID_NUM,if_vec=True , hour_num = hour_num)



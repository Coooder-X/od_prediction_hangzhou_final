import numpy as np
import time
import pandas as pd
import torch
import tqdm
import torch.nn.functional as F
import  os
import random
def get_grid_pandding(df, MIN_LONGITUDE,MAX_LONGITUDE, MIN_LATITUDE, MAX_LATITUDE,num_rows, num_cols):
    '''
    将上下车坐标映射到网格
    '''
    def mapping_location_grid(longitude, latitude):
        lon = round(longitude, 6)
        lat = round(latitude, 6)
        # flag, grid = 0, 0
        lon_column = (MAX_LONGITUDE - MIN_LONGITUDE) / num_cols
        lat_row = (MAX_LATITUDE - MIN_LATITUDE) / num_rows
        if lon < MIN_LONGITUDE:
            lon_grid = 0
        elif lon > MAX_LONGITUDE:
            lon_grid = num_cols+1
        else:
            lon_grid = int((lon - MIN_LONGITUDE) / lon_column) + 1
        if lat < MIN_LATITUDE:
            lat_grid = 0
        elif lat > MAX_LATITUDE:
            lat_grid = num_cols+1
        else:
            lat_grid =  int((lat - MIN_LATITUDE) / lat_row) + 1
        grid = lon_grid + lat_grid * (num_cols+2)
        return grid
    vfunc = np.vectorize(mapping_location_grid)
    df['pickup_grid'] = vfunc(df['pickup_longitude'], df['pickup_latitude'])
    df['dropoff_grid'] = vfunc(df['dropoff_longitude'], df['dropoff_latitude'])
    return df






def get_grid(df, MIN_LONGITUDE,MAX_LONGITUDE, MIN_LATITUDE, MAX_LATITUDE,num_rows, num_cols ):
    '''
    将上下车坐标映射到网格
    '''
    def mapping_location_grid(longitude, latitude):
        lon = round(longitude, 6)
        lat = round(latitude, 6)
        flag, grid = 0, 0
        if lon < MIN_LONGITUDE:
            flag = 1
            grid = 0
        elif lon > MAX_LONGITUDE:
            flag = 2
            grid = num_rows * num_cols - 1
        elif lat < MIN_LATITUDE:
            flag = 3
            grid = 0
        elif lat > MAX_LATITUDE:
            flag = 4
            grid = num_rows * num_cols - 1
        if (flag > 0):
            pass
            # print('outliers:{},{}'.format(lon, lat))
        lon_column = (MAX_LONGITUDE - MIN_LONGITUDE) / num_cols
        lat_row = (MAX_LATITUDE - MIN_LATITUDE) / num_rows
        # x是纬度，y是经度，和我的项目一致
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


def get_od_matrix(start_time, end_time, order_records, grid_num):
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
        m_list.append(matrix.values)
        print('Time slot:', time_slot, ' density:', density)
        time_slot += 1
    # matrix_all = pd.DataFrame(pd.concat(m_list, ignore_index=True, axis=0).reset_index(drop=True))
    matrix_all = np.stack(m_list)
    print(matrix_all.shape)
    print('Start writing files:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    density = density / 48.0
    print('Finish finally, ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    return matrix_all, density


def get_line_graph_with_vec(OD,index, flow, vec_maxtrix, C_NUM, NODE_NUM):
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


def get_select_od_matrix(start_time, end_time, order_records, grid_num, vec_future, dic, len_index_list, num_cols):
    print('def get_od_matrix()!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    time_slot = start_time
    density = 0
    m_list = []
    v_list = []
    while time_slot < end_time:
        tmpt = order_records[order_records['pickup_time_bin'] == time_slot]
        density += tmpt.shape[0] / (grid_num * grid_num)
        matrix = pd.DataFrame(np.zeros(shape=(len_index_list)))
        vec_maxtrix = pd.DataFrame(np.zeros(shape=(len_index_list, 256*3)))
        for j in range(tmpt.shape[0]):

            new_index = tmpt.iloc[j]['pickup_grid'] * grid_num + tmpt.iloc[j]['dropoff_grid']
            if dic[new_index] != -1:
                matrix.iloc[dic[new_index]] = tmpt.iloc[j]['order_num']
                vec_f = np.stack(vec_future.get_group((time_slot, tmpt.iloc[j]['pickup_grid'], tmpt.iloc[j]['dropoff_grid']))[
                        "vec"])
                # vec_maxtrix.iloc[dic[new_index]] = np.concatenate(vec_f.mean(axis=0),vec_f.min(axis=0),vec_f.max(axis=0) )
                vec_maxtrix.iloc[dic[new_index]] = np.concatenate(vec_f.mean(axis=0))
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

def create_seq(len_trend, len_period, len_closeness,read_path, NODE_NUM, if_vec, hour_num=1 ):
    # in_out_file = r"../t2vec_code/data/line_graph_data/OD_node_" + str(NODE_NUM) +".npz"
    if if_vec:
        in_out_file = os.path.join(read_path,"vec_input_" + str(NODE_NUM) +".npz")
    else:
        in_out_file = os.path.join(read_path,"OD_node_" + str(NODE_NUM) +".npz")
    inout_flow = np.load(in_out_file)['data']
    total_time_steps = inout_flow.shape[0]
    start = max((24 * 7 * len_trend)*int(1/hour_num), max(24 * len_period * int(1/hour_num), len_closeness))
    flow_data_arr, od_flow_data_arr = [], []
    flow_label_arr, od_flow_label_arr = [], []
    for i in range(start, total_time_steps):
        len1, len2, len3 = len_trend, len_period, len_closeness
        flow_list = []
        od_flow_list = []
        while len1 > 0:
            flow_trend = inout_flow[i - 24 * 7 * len1* int(1/hour_num)]  # i-24*7*3, i-24*7*2 i-24*7*1
            flow_list.append(flow_trend)
            len1 = len1 - 1
        while len2 > 0:
            flow_peroid = inout_flow[i - 24 * len2* int(1/hour_num)]  # i-24*3, i-24*2 i-24*1
            flow_list.append(flow_peroid)
            len2 = len2 - 1
        while len3 > 0:
            flow_closeness = inout_flow[i - len3 * int(1/hour_num)]  # i-3, i-2 i-1
            flow_list.append(flow_closeness)
            len3 = len3 - 1
        flow_label = inout_flow[i:i + 1]
        flow_data_arr.append(flow_list)
        flow_label_arr.append(flow_label)
    if if_vec:
        flow_data_arr = np.array(flow_data_arr)[:,:,:,:].swapaxes(1,2)
        flow_label_arr = np.array(flow_label_arr)[:,:,:,:].swapaxes(1,2)
    else:
        flow_data_arr = np.array(flow_data_arr)[:,:,:,0].swapaxes(1,2)
        flow_label_arr = np.array(flow_label_arr)[:,:,:,0].swapaxes(1,2)
    # 此处可以保存成npy
    return flow_data_arr, flow_label_arr

def get_dataloader(len_trend, len_period, len_closeness, train_prop, val_prop, read_path, save_path, NODE_NUM, if_vec, hour_num=1 ):
    #1.构建数据集
    flow_data,  flow_label = create_seq(len_trend, len_period, len_closeness, read_path, NODE_NUM, if_vec, hour_num=1)
    #2.划分训练集,验证集,测试集 = 8:1:1
    num_samples = flow_data.shape[0]
    num_train = int(num_samples * train_prop)
    num_val = int(num_samples * val_prop)
    num_test = num_samples - num_train - num_val
    train_flow_data,  train_flow_label = flow_data[:num_train], flow_label[:num_train]
    val_flow_data,  val_flow_label = flow_data[num_train:num_train + num_val], flow_label[num_train:num_train + num_val]
    if if_vec:
        print("train_vec.shape:",train_flow_data.shape)
        print("val_vec.shape:", val_flow_data.shape)
        np.savez(os.path.join(save_path,"train_vec" + str(NODE_NUM) +".npz"), data = train_flow_data, lable=train_flow_label)
        np.savez(os.path.join(save_path,"val_vec" + str(NODE_NUM) +".npz"), data=val_flow_data, lable=val_flow_label)
    else:
        print("train_node.shape:",train_flow_data.shape)
        print("val_node.shape:", val_flow_data.shape)
        np.savez(os.path.join(save_path,"train" + str(NODE_NUM) +".npz"), data = train_flow_data, lable=train_flow_label)
        np.savez(os.path.join(save_path,"val" + str(NODE_NUM) +".npz"), data=val_flow_data, lable=val_flow_label)


def get_random_dataloader(len_trend, len_period, len_closeness, train_prop, val_prop, read_path, save_path, NODE_NUM, if_vec,
                   hour_num=1):
    # 1.构建数据集
    flow_data, flow_label = create_seq(len_trend, len_period, len_closeness, read_path, NODE_NUM, False, hour_num=1)
    vec_data, vec_label = create_seq(len_trend, len_period, len_closeness, read_path, NODE_NUM, True, hour_num=1)
    # 因为数据较少，且处于春节期间，所以将其打乱后进行预测，所以将其打乱后进行训练
    random.seed(2023)  # 设置随机数生成器的种子
    combined_data = list(zip(flow_data, flow_label, vec_data, vec_label))
    random.shuffle(combined_data)  # 洗牌
    flow_data, flow_label, vec_data, vec_label = zip(*combined_data)
    flow_data, flow_label, vec_data, vec_label = np.array(flow_data), np.array(flow_label), np.array(vec_data), np.array(vec_label)
    # 2.划分训练集,验证集,测试集 = 8:1:1
    num_samples = flow_data.shape[0]
    num_train = int(num_samples * train_prop)
    num_val = int(num_samples * val_prop)

    # 2.划分训练集,验证集,测试集 = 8:1:1
    train_flow_data, train_flow_label = flow_data[:num_train], flow_label[:num_train]
    val_flow_data, val_flow_label = flow_data[num_train:num_train + num_val], flow_label[num_train:num_train + num_val]
    train_vec_data, train_vec_label = vec_data[:num_train], vec_label[:num_train]
    val_vec_data, val_vec_label = vec_data[num_train:num_train + num_val], vec_label[num_train:num_train + num_val]

    print("train_vec.shape:", train_vec_data.shape)
    print("val_vec.shape:", val_vec_data.shape)
    np.savez(os.path.join(save_path, "train_vec" + str(NODE_NUM) + ".npz"), data=train_vec_data, lable=train_vec_label)
    np.savez(os.path.join(save_path, "val_vec" + str(NODE_NUM) + ".npz"), data=val_vec_data, lable=val_vec_label)

    print("train_node.shape:", train_flow_data.shape)
    print("val_node.shape:", val_flow_data.shape)
    np.savez(os.path.join(save_path, "train" + str(NODE_NUM) + ".npz"), data=train_flow_data,lable=train_flow_label)
    np.savez(os.path.join(save_path, "val" + str(NODE_NUM) + ".npz"), data=val_flow_data, lable=val_flow_label)




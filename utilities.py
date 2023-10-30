import numpy as np
import pandas as pd
from copy import deepcopy
import pickle
import random
from datetime import datetime, timedelta
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import gc
import os
import torch
def seed_torch(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as e:
            print(f"Error creating directory: {e}")
    else:
        print(f"Directory '{directory_path}' already exists.")
def creat_directory(args):
    print(os.path.basename(os.getcwd()))
    if os.path.basename(os.getcwd()) in ["train_files", "train_files_final", "train_files_final_2", "train_files_final_ST"]:
    # if os.path.basename(os.getcwd()) == "train_files" or os.path.basename(os.getcwd()) == "train_files_final" or os.path.basename(os.getcwd()) =="train_files_final_2":
        args.save_folder = os.path.join(os.path.dirname(os.getcwd()),"result")
    dir_mane = "result_"
    if args.use_adj:
        dir_mane = dir_mane + "adj_"
    if args.use_vec:
        dir_mane = dir_mane + "vec_"
    if args.use_three_branch:
        dir_mane = dir_mane + "three_branch_"
    if args.use_dynamic_graph:
        dir_mane = dir_mane + "dynamic_graph_"
    dir_mane = dir_mane + str(args.grid_num)
    if args.self_naming != "":
        dir_mane = dir_mane + "_" + args.self_naming
    create_directory_if_not_exists(args.save_folder)
    args.save_folder = os.path.join(args.save_folder,dir_mane)
    create_directory_if_not_exists(args.save_folder)
    tensorboard_folder = os.path.join(args.save_folder, 'tensorboard')
    create_directory_if_not_exists(tensorboard_folder)
    args.save_trained_model = os.path.join(args.save_folder, "save_trained_model")
    create_directory_if_not_exists(args.save_trained_model)
    return tensorboard_folder


def reset_arg(train_data, args):
    if args.use_vec:
        args.node_num = train_data.vec.shape[1]
        args.node_dim = train_data.vec.shape[2]
        args.vec_dim = train_data.vec.shape[3]
        # args.num_filters = int(train_data.new_adj_matrices.shape[0]/train_data.new_adj_matrices.shape[1])
        args.num_filters = len(train_data.new_adj_matrices)
        args.network_structure_e[0] = args.node_dim
        args.network_structure_l[0] = args.node_num
    else:
        args.node_num = train_data.data.shape[1]
        args.node_dim = train_data.data.shape[2]
        # args.num_filters = int(train_data.new_adj_matrices.shape[0]/train_data.new_adj_matrices.shape[1])
        args.num_filters = len(train_data.new_adj_matrices)
        args.network_structure_e[0] = args.node_dim
        args.network_structure_l[0] = args.node_num
    return 0


def inverse_transform(scaled_data, norml_max, norml_min):
    original_data = scaled_data * (norml_max - norml_min) + norml_min
    return original_data


def preprocess_adj_tensor(adj_tensor, symmetric=True):
    adj_out_tensor = []
    # print(adj_tensor.shape)
    for i in range(adj_tensor.shape[0]):
        adj = adj_tensor[i]
        adj_count = int(adj.shape[0] / adj.shape[1])
        adj_list = []
        for m in range(0, adj_count):
            sub_adj = adj[int(m * adj.shape[1]): int((m+1) * adj.shape[1]), :]
            sub_adj = sub_adj + np.eye(sub_adj.shape[0])
            sub_adj = normalize_adj_numpy(sub_adj, symmetric)
            adj_list.append(sub_adj)
        adj = np.concatenate(adj_list, axis=0)
        adj_out_tensor.append(adj)
    adj_out_tensor = np.array(adj_out_tensor)
    return adj_out_tensor

# features 正规化
# def scale_features(s_train_x,s_targets, dn_range, up_range):
#     # remove abnormal data (convert to 0)
#     s_train_x = s_train_x.astype("float")
#     s_train_x[np.isnan(s_train_x)] = 0
#     s_train_x[np.isinf(s_train_x)] = 0
#     n_train_x = s_train_x.reshape(s_train_x.shape[0] * s_train_x.shape[1], s_train_x.shape[-1])
#     scaler = MinMaxScaler(feature_range=(dn_range, up_range))
#     i_train_x = scaler.fit_transform(n_train_x)
#     b_train_x = i_train_x.reshape(s_train_x.shape[0], s_train_x.shape[1], s_train_x.shape[2])
#
#
#     del n_train_x, s_train_x, i_train_x
#     gc.collect()
#     return b_train_x,None, scaler

def scale_features(s_train_x,s_targets, dn_range, up_range):
    # 计算每个维度的最小值和最大值
    batch_min = s_train_x.min(dim=-1).values.min(dim=-1).values.reshape(s_train_x.shape[0],1,1)
    batch_max = s_train_x.max(dim=-1).values.max(dim=-1).values.reshape(s_train_x.shape[0],1,1)
    # 计算缩放范围
    batch_range = batch_max - batch_min
    # 将数据缩放到0和1之间
    scaled_data = (s_train_x - batch_min) / (batch_range.reshape(batch_range.shape[0],1,1))
    return scaled_data, batch_range, batch_min


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def reg_eval(y_true, y_pred):
    """
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mmape = sum(abs(2 * (y_true - y_pred) / (y_true + y_pred))) / len(y_true)
    return mse, mae, mmape

def reg_eval_y_true(y_true, y_pred):
    """
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mmape = sum(abs(y_true - y_pred) / (y_true + 10e-5)) / len(y_true)
    return mse, mae, mmape

def reg_eval_y_pred(y_true, y_pred):
    """
    """
    mse = metrics.mean_squared_error(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mmape = sum(abs((y_true - y_pred) / y_pred)) / len(y_true)
    return mse, mae, mmape

def reg_eval_with_threshold(y_true, y_pred):
    df_local = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    thresholds = [0, 1, 2, 5, 10, 20, 50]
    mmape_list = []
    for threshold in thresholds:
        _, _, mmape = reg_eval(
            df_local[df_local["y_true"] >= threshold]["y_true"],
            df_local[df_local["y_true"] >= threshold]["y_pred"]
        )
        mmape_list.append(mmape)
    return mmape_list

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def normalize_adj_numpy(adj, symmetric=True):
    if symmetric:
        d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d)
    else:
        d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj)
    return a_norm

def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[1])
    adj = normalize_adj(adj, symmetric)
    return adj

def preprocess_adj_numpy(adj, symmetric=True):
    adj = adj + np.eye(adj.shape[0])
    adj = normalize_adj_numpy(adj, symmetric)
    return adj


# def preprocess_adj_tensor(adj_tensor, symmetric=True):
#     adj_out_tensor = []
#     # print(adj_tensor.shape)
#     for i in range(adj_tensor.shape[0]):
#         adj = adj_tensor[i]
#         adj_count = int(adj.shape[0] / adj.shape[1])
#         adj_list = []
#         for m in range(0, adj_count):
#             sub_adj = adj[int(m * adj.shape[1]): int((m+1) * adj.shape[1]), :]
#             sub_adj = sub_adj + np.eye(sub_adj.shape[0])
#             sub_adj = normalize_adj_numpy(sub_adj, symmetric)
#             adj_list.append(sub_adj)
#         adj = np.concatenate(adj_list, axis=0)
#         adj_out_tensor.append(adj)
#     adj_out_tensor = np.array(adj_out_tensor)
#     return adj_out_tensor

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian

# def rescale_laplacian(laplacian):
#     try:
#         print('Calculating largest eigenvalue of normalized graph Laplacian...')
#         largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
#     except ArpackNoConvergence:
#         print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
#         largest_eigval = 2
#
#     scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
#     return scaled_laplacian

def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k

def get_features_by_looking_back(graph_od_demand, train_date_list, test_date_list, lag=1):
    graph_od_demand_t1 = deepcopy(graph_od_demand)
    graph_od_demand_t1[["date", "hour"]] = graph_od_demand_t1[["date", "hour"]].shift(-lag)
    graph_od_demand_t1 = graph_od_demand_t1[graph_od_demand_t1["date"].notnull()]

    train_graph_t1 = graph_od_demand_t1.loc[graph_od_demand_t1["date"].isin(pd.to_datetime(train_date_list)), :]
    test_graph_t1 = graph_od_demand_t1.loc[graph_od_demand_t1["date"].isin(pd.to_datetime(test_date_list)), :]

    train_graph_t1_arr = train_graph_t1.values[..., np.newaxis]
    test_graph_t1_arr = test_graph_t1.values[..., np.newaxis]

    return train_graph_t1, test_graph_t1, train_graph_t1_arr, test_graph_t1_arr

def select_od_pairs(graph_od_demand, od_list_df, num_selected_od_pairs):
    od_names = od_list_df["name"].values
    od_mean_values = graph_od_demand[od_names].mean()
    od_mean_values_sorted = od_mean_values.sort_values(ascending=False)
    selected_od_pairs = list(od_mean_values_sorted[0: num_selected_od_pairs].index)
    print("minimum of mean demand of selected od pairs:", od_mean_values_sorted[num_selected_od_pairs])
    return selected_od_pairs

def load_feature_data():
    """
    graph od demand & od list
    :return:
    """
    pic2 = open(r"nyc_data\line_graph_data\A.pkl", 'rb')
    adj_matrices = pickle.load(pic2)
    return None, None, adj_matrices

def create_seq(len_trend, len_period, len_closeness):
    in_out_file = r"nyc_data\line_graph_data\OD_node.npz"
    inout_flow = np.load(in_out_file)['data']
    print("## data.shape ##"*30)
    print(inout_flow.shape)

    total_time_steps = inout_flow.shape[0]
    start = max(24 * 7 * len_trend, max(24 * len_period, len_closeness))
    flow_data_arr, od_flow_data_arr = [], []
    flow_label_arr, od_flow_label_arr = [], []


    for i in range(start, total_time_steps):
        len1, len2, len3 = len_trend, len_period, len_closeness
        flow_list = []
        od_flow_list = []

        while len1 > 0:
            flow_trend = inout_flow[i - 24 * 7 * len1]  # i-24*7*3, i-24*7*2 i-24*7*1
            flow_list.append(flow_trend)
            len1 = len1 - 1

        while len2 > 0:
            flow_peroid = inout_flow[i - 24 * len2]  # i-24*3, i-24*2 i-24*1
            flow_list.append(flow_peroid)
            len2 = len2 - 1

        while len3 > 0:
            flow_closeness = inout_flow[i - len3]  # i-3, i-2 i-1
            flow_list.append(flow_closeness)
            len3 = len3 - 1
        flow_label = inout_flow[i:i + 1]

        flow_data_arr.append(flow_list)

        flow_label_arr.append(flow_label)

    flow_data_arr = np.array(flow_data_arr)[:,:,:,0].swapaxes(1,2)
    flow_label_arr = np.array(flow_label_arr)[:,:,:,0].swapaxes(1,2)
    # 此处可以保存成npy
    return flow_data_arr, flow_label_arr

def get_dataloader(len_trend, len_period, len_closeness, train_prop, val_prop):

    #1.构建数据集
    flow_data,  flow_label = create_seq(len_trend, len_period, len_closeness)

    #2.划分训练集,验证集,测试集 = 8:1:1
    num_samples = flow_data.shape[0]
    num_train = int(num_samples * train_prop)
    num_val = int(num_samples * val_prop)
    num_test = num_samples - num_train - num_val

    train_flow_data,  train_flow_label = flow_data[:num_train], flow_label[:num_train]

    val_flow_data,  val_flow_label = flow_data[num_train:num_train + num_val], flow_label[num_train:num_train + num_val]
    np.savez("data_set/split_data/train.npz", datat = train_flow_data, lable=train_flow_label)
    np.savez("data_set/split_data/train.npz", datat=val_flow_data, lable=val_flow_label)
    return train_flow_data,  val_flow_data, train_flow_label,   val_flow_label #, test_flow_data,  test_flow_label

def prepare_train_and_test_samples(train_start, train_end, test_start, test_end, num_selected_od_pairs=1000):
    """
    obtain
    1) train dataset and test dataset
    2) adjacent matrices (with selected od pairs)
    :param graph_od_demand:
    :param od_list_df:
    :param num_selected_od_pairs:
    :return:
    """
    _, _, adj_matrices = load_feature_data()
    print(adj_matrices.keys())
    # load adjacent matrices
    new_adj_matrices = {}
    matrices_to_used = ["OD_based_corr", "OD_based_ori_eucli_rev", "OD_based_dest_eucli_rev",
                        "OD_based_ori_neighbor", "OD_based_dest_neighbor",
                        "OD_based_ori_dist_rev", "OD_based_dest_dist_rev"]
    # print(adj_matrices["OD_based_ori_dist_rev"])
    for key in matrices_to_used:
        A = adj_matrices[key]
        A = A / A.max().max()
        np.fill_diagonal(A, 0)
        A_ = np.nan_to_num(A)
        A_array = A_[np.newaxis, ...]
        new_adj_matrices[key] = A_array
        print(A_array.max(), A_array.min())
    A_array = np.eye(A_array.shape[1])
    A_array = A_array[np.newaxis, ...]
    new_adj_matrices["identity"] = A_array
    train_x, test_x, train_y, test_y = get_dataloader(1, 1, 2, 0.8, 0.2)
    return train_x, test_x, train_y, test_y, new_adj_matrices


if __name__ == "__main__":
    train_start = "08-01-2018"
    train_end = "09-01-2018"
    test_start = "23-04-2018"
    test_end = "24-04-2018"
    prepare_train_and_test_samples(train_start, train_end, test_start, test_end, num_selected_od_pairs=1000)


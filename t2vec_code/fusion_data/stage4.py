import numpy as np
from t2vec_code.fusion_data.util import get_line_graph, get_line_graph_with_vec
import pickle
R_NUM = 10
C_NUM = 10
NODE_NUM = R_NUM * C_NUM
if __name__ == '__main__':
    matrix_flow = np.load('../data/temp_data/matrix_flow_' + str(NODE_NUM) + '.npz')["data"]
    matrix_all = np.load('../data/temp_data/matrix_all_' + str(NODE_NUM) + '.npz')["data"]
    index_list = np.load('../data/temp_data/index_' + str(NODE_NUM) + '.npz')["data"]
    vec_maxtrix = np.load('../data/temp_data/vec_maxtrix_' + str(NODE_NUM) + '.npz')["data"]
    print("************************")
    print(matrix_all.shape)
    print(matrix_all.max())
    print(matrix_all.min())

    # matrix_flow = matrix_all_od.sum(axis=-1)
    matrix_all = np.concatenate((matrix_all,matrix_all),axis=-1)
    A = get_line_graph_with_vec(matrix_all, np.array(index_list), matrix_flow,vec_maxtrix, R_NUM, C_NUM, NODE_NUM).numpy()
    A = np.nan_to_num(A, copy=True)
    matrices_to_used = ["OD_based_corr", "OD_based_ori_eucli_rev", "OD_based_dest_eucli_rev",
                        "OD_based_ori_neighbor", "OD_based_dest_neighbor",
                        "OD_based_ori_dist_rev", "OD_based_dest_dist_rev","trj_coor"]
    dic = {}
    for i in range(len(matrices_to_used)):
        dic[matrices_to_used[i]] = A[i]
    np.savez('../data/line_graph_data/OD_node_' + str(NODE_NUM) + '.npz', data=matrix_all)
    np.savez('../data/line_graph_data/vec_input_' + str(NODE_NUM) + '.npz', data=vec_maxtrix)
    f_save = open(r'../../nyc_data/split_data/A' + str(NODE_NUM) + '.pkl', 'wb')

    print("^^^^^^^^^^^^^^^^^^^^^")
    print(A.shape)
    print(A.max())
    print(A.min())
    pickle.dump(dic, f_save)
    f_save.close()
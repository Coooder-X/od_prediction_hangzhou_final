import torch
import os
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import pickle
import torch.nn.functional as F
class MyDataset(Dataset):
    def __init__(self, root, dataset_name,grid_num, use_adj= False , dynamic_graph = False, norml_max = None,norml_min = None):
        path = os.path.join(root,dataset_name+ str(grid_num) +".npz")
        self.dynamic_graph = dynamic_graph
        self.use_adj = use_adj
        vec_path = os.path.join(root, dataset_name + "_vec"+ str(grid_num) + ".npz")
        row_data = np.load(path)
        pic2 = open(os.path.join(root,"A"+ str(grid_num) +".pkl"), 'rb')
        adj_matrices = pickle.load(pic2)
        vec_data = np.load(vec_path)

        new_adj_matrices = {}
        if use_adj:
            # print("************************")
            self.matrices_to_used = ["start_dis", "end_dis",  "start_neighbor",
                                "end_neighbor", "od_flow_coor", "vec_similar"]
        else:
            self.matrices_to_used = ["start_dis", "end_dis",  "start_neighbor",
                                "end_neighbor", "od_flow_coor"]

        adj_matrix_list = []
        for key in self.matrices_to_used:
            A = adj_matrices[key]
            A = A / A.max().max()
            np.fill_diagonal(A, 0)
            A_ = np.nan_to_num(abs(A))
            A_array = A_[np.newaxis, ...]
            new_adj_matrices[key] = A_array
            adj_matrix_list.append(new_adj_matrices[key])
        A_array = np.eye(adj_matrix_list[0].shape[1])
        A_array = A_array[np.newaxis, ...]
        adj_matrix_list = [A_array] + adj_matrix_list
        # adj_matrix_list = np.concatenate(adj_matrix_list, axis=1).squeeze(0)
        self.data = row_data["data"]
        self.targets = row_data["lable"]
        self.vec = vec_data["data"]
        self.new_adj_matrices = adj_matrix_list
        if norml_max == None or norml_min == None:
            self.max = np.max(self.data)
            self.min = np.min(self.data)
        else:
            self.max = norml_max
            self.min = norml_min
        self.norm_data = (self.data - self.min) / (self.max - self.min)
        self.norm_targets = (self.targets - self.min) / (self.max - self.min)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        dynamic_key = ["od_flow_coor", "vec_similar"]
        x = self.data[index]
        y = self.targets[index]
        norm_y = self.norm_targets[index]
        vec = self.vec[index]
        if self.dynamic_graph:
            # index = self.matrices_to_used.index("od_flow_coor")
            # self.new_adj_matrices[index] = abs(np.corrcoef(x, rowvar=True)[np.newaxis, ...])
            if self.use_adj:
                index_trj_sim = self.matrices_to_used.index("vec_similar")
                self.new_adj_matrices[index_trj_sim] = abs(cosine_similarity(vec[:,-1,:], vec[:,-1,:])[np.newaxis, ...])
        adj_matrix_list = np.concatenate(self.new_adj_matrices, axis=1).squeeze(0)
        return x, y,norm_y,vec,adj_matrix_list


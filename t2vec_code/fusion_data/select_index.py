import torch
import numpy as np
matrix_all_od = torch.from_numpy(np.load(r"../data/temp_data/od_flow_image.npz")["data"][:,:,:])
flow = matrix_all_od.sum(axis=-1)
OD_data = matrix_all_od.reshape(matrix_all_od.shape[0], -1)
p_num = OD_data[:, :].mean(dim=0)
selet = np.where(p_num > 0.02, True, False)
index = torch.linspace(0, len(selet) - 1, len(selet)).int()
OD_node = OD_data.permute(1, 0)[selet].permute(1, 0)
index = index[selet]
print(OD_data.shape)
print(OD_node.shape)
print(set(index.numpy()))
A = get_line_graph(OD_node, index, flow)

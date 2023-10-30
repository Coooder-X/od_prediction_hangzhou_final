import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
import os
def show_image(img):
    plt.imshow(img)
    plt.colorbar()
    plt.axis('off')
    plt.show()

def get_index(row,column,all_column):
    index = row * all_column + column
    return index


def convert_OD_to_line_graph(OD,all_column):
    # 将图进行转化为线图的节点特征
    OD_flow = OD.reshape(OD.shape[0],OD.shape[1],-1)
    OD_flow = OD_flow.reshape(OD.shape[0],-1,2,OD_flow.shape[-1]).permute(0,2,1,3)
    # TODO: 边的特征不好表示，因为边的特征是固定的，如果使用多任务学习，则无意义
    # TODO: 可以尝试使用轨迹特征，使用动态的轨迹特征应该是可以形成一个多任务学习的
    # TODO: 即使不是动态的图，也可以使用编码器进行编码，只是解码时不进行解码。
    OD_line_in = torch.zeros(OD.shape[0],1,(all_column**2)**2,(all_column**2)**2)
    for l in range(OD_flow.shape[0]):
        for i in range(all_column**2):
            for j in range(all_column**2):
                if OD_flow[l,0,i,j] > 0:
                    index_x =  get_index(i,j,all_column**2)
                    for z in range(all_column**2):
                        if OD_flow[l,0,j,z] > 0:
                            index_y = get_index(j,z,all_column**2)
                            OD_line_in[l,0,index_x,index_y] = 1
    return OD_flow, torch.cat((OD_line_in,OD_line_in.transpose(2,3)),dim=1)

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

def get_coor():
    X = np.array([65, 72, 78, 65, 72, 70, 65, 68])
    Y = np.array([72, 69, 79, 69, 84, 75, 60, 73])
    print(np.corrcoef(X, Y)[0,1])
    return 0



if __name__ == '__main__':
    OD_data = torch.from_numpy(np.load(r"../data/temp_data/od_flow_image.npz")["data"][:,:,:])
    OD_data = OD_data.unsqueeze(dim=-1).repeat(1,1,1,2)
    flow = OD_data.reshape(-1,256,2,16*16).permute(0,1,3,2)[...,0].sum(dim=-1)
    OD_data = OD_data.reshape(-1,256,2,16*16).permute(0,1,3,2).reshape(OD_data.shape[0],-1,2)
    print("##############################################################")
    p_num = OD_data[:,:,0].mean(dim=0)
    selet = np.where(p_num>0.00001,True,False)
    index = torch.linspace(0,len(selet)-1,len(selet)).int()
    OD_node = OD_data.permute(1,0,2)[selet].permute(1,0,2)
    print(OD_node.shape)
    index = index[selet]
    print(OD_node.shape,index.shape, flow.shape)
    A = get_line_graph(OD_node, index, flow)

    OD_node = OD_node.permute(0, 2, 1).reshape(OD_data.shape[0], 2, 10, 10)
    A = A.reshape(A.shape[0],10,10,100).permute(0,3,1,2)
    np.savez("../data/line_graph_data/in_out_image.npz", data=OD_node)
    np.savez("../data/line_graph_data/od_flow_image.npz", data=A)

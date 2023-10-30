import torch
import torch.nn as nn
import torch.optim as optim
import argparse
#from model.model_new import ResidualGraphLSTM
from model.model_ST_plus import ResidualGraphLSTM
from torch.utils.data import DataLoader, Subset
from data_set.OD_dataSet import MyDataset
import os
from tensorboardX import SummaryWriter
import math
from utilities import scale_features, preprocess_adj_tensor,inverse_transform, reset_arg, creat_directory, seed_torch
import numpy as np
import datetime
from config import DATA_ROOT
import warnings
warnings.filterwarnings("ignore")
t = datetime.datetime.now(); s1="-";s2="_"
start_time = str(t.year)+s1+str(t.month)+s1+str(t.day)+s2+str(t.hour)+s1+str(t.minute)+s1+str(t.second)
parser = argparse.ArgumentParser()
#parameter of dataset
# 主要的需要设置的训练结果
parser.add_argument('--use_adj',type=int,default=False,help='grid_num')
parser.add_argument('--use_vec',type=bool,default=False,help='if use vector future')
parser.add_argument('--use_three_branch', type=bool,default=True,help='if use three branch')
parser.add_argument('--use_dynamic_graph', type=bool,default= False,help='if dynamic graph')
parser.add_argument('--self_naming',type=str,default="plus_ST_3",help='self_naming')
parser.add_argument('--grid_num',type=int,default=100,help='grid_num')
parser.add_argument('--root',type=str,default=r"E:\WST_code\zhijiang_data\hangzhou_202005_complete\split_data_0.25_hour",help='if use vector future')
parser.add_argument('--save_folder',type=str,default=r'./result',help='result dir')

parser.add_argument('--len_trend',type=int,default=1,help='length of trend data')
parser.add_argument('--len_period',type=int,default=1,help='length of period data')
parser.add_argument('--len_closeness',type=int,default=2,help='length of closeness data')
parser.add_argument('--train_prop',type=float,default=0.8,help='proportion of training set')
parser.add_argument('--val_prop',type=float,default=0.2,help='proportion of validation set')
parser.add_argument('--batch_size',type=int,default=32,help='batch size')
parser.add_argument('--height',type=int,default=16,help='input flow image height')
parser.add_argument('--width',type=int,default=16,help='input flow image width')
parser.add_argument('--latent_dim',type=int,default=128,help='external factor dimension')
parser.add_argument('--latent_dim_l',type=int,default=32,help='edge channel embedding dimension')
parser.add_argument('--regularizer_rate',type=int,default=0,help='edge channel embedding dimension')
parser.add_argument('--activation',type=str,default='relu',help='edge channel embedding dimension')
parser.add_argument('--is_batch_normalization',type=bool,default=True,help='edge channel embedding dimension')
parser.add_argument('--sym_norm',type=bool,default=True,help='SYM_NORM')
# 新增
parser.add_argument('--train_batchSize',type=int,default=32,help='train batch size')
parser.add_argument('--val_batchSize',type=int,default=32,help='val batch size')
parser.add_argument('--node_num',type=int,default=0.0005,help='node number')
parser.add_argument('--num_filters',type=int,default=10,help='number filters')
# 网络结构，如[h1,h2,h3,h4],则对应的模型结构中的输入输出如下：[h1,h2],[h2,h3],[h3,h4]]
parser.add_argument('--network_structure_e',type=list,default=[4,128,128],help='NETWORK_STRUCTURE_E')
parser.add_argument('--network_structure_d',type=list,default=[128,128,128],help='NETWORK_STRUCTURE_D')
parser.add_argument('--network_structure_l',type=list,default=[100,128,64,30],help='NETWORK_STRUCTURE_L')

#parameter of training
parser.add_argument('--use_GPU',type=bool,default=True,help='use GPU')
parser.add_argument('--epochs',type=int,default=85,help='training epochs')
parser.add_argument('--lr',type=float,default=0.0005,help='learning rate')
parser.add_argument('--seed',type=int,default=99,help='running seed')

parser.add_argument('--save_trained_model',type=str,default='./result/result_without_vec_36/trained_model',help='result dir')
parser.add_argument('--device',type=str,default='cuda:0',help='cuda device')
parser.add_argument('--max_grad_norm',type=int,default=10,help='max gradient norm for gradient clip')
parser.add_argument('--weight_decay',type=float,default=0,help='weight decay rate')
#parameter of model
def get_dataset_args():args = parser.parse_args();return {"len_trend": args.len_trend, "len_period":args.len_period, "len_closeness":args.len_closeness, "train_prop":args.train_prop,"val_prop":args.val_prop}
def get_args():return parser.parse_args()
def run_model():
    args = get_args()
    tensorboard_folder = creat_directory(args)
    writer = SummaryWriter(tensorboard_folder, filename_suffix = "_TrainTime("+start_time +")")
    if args.use_GPU and torch.cuda.is_available(): device = torch.device("cuda:0")
    else:device = torch.device("cpu")
    train_data = MyDataset(args.root, "train", args.grid_num,use_adj= args.use_adj , dynamic_graph = args.use_dynamic_graph, norml_max = None,norml_min = None)
    val_data = MyDataset(args.root, "val", args.grid_num,use_adj= args.use_adj , dynamic_graph = args.use_dynamic_graph, norml_max = train_data.max,norml_min = train_data.min)
    norml_max = train_data.max
    norml_min = train_data.min
    reset_arg(train_data, args)
    train_loader = DataLoader(train_data, batch_size=args.train_batchSize, shuffle=True)

    val_loader = DataLoader(val_data, batch_size=args.val_batchSize, shuffle=False)
    model = ResidualGraphLSTM(args.node_num,args.num_filters ,args.network_structure_e, args.network_structure_l, args.network_structure_d, args.latent_dim,
                              args.latent_dim_l, args.regularizer_rate, args.activation, args.is_batch_normalization,args.use_vec,
                              args.use_three_branch).to(device)
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

    for n in range(args.epochs):
        print("Epoch: ", n)
        train_loss = []
        val_loss = []
        epoch_loss = [math.inf]
        model.train()
        for data in train_loader:
            train_data, targets,norm_targets, vec, adj_matrices = data
            # print(norml_max)
            # print(norml_min)
            dn_range, up_range = 0, 1
            #train_data, batch_range, batch_min = scale_features(train_data, targets, dn_range, up_range)
            adj_matrices = np.nan_to_num(adj_matrices.numpy(), copy=True)
            # print(adj_matrices.max(),adj_matrices.min())
            adj_matrices =torch.from_numpy(preprocess_adj_tensor(adj_matrices, args.sym_norm))
            adj_matrices = torch.from_numpy(np.nan_to_num(adj_matrices, copy=True))
            train_data = train_data.to(torch.float32).to(device)
            adj_matrices = adj_matrices.to(torch.float32).to(device)
            vec = vec.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            norm_targets = norm_targets.to(torch.float32).to(device)
            # 验证数据大小
            optimizer.zero_grad()
            pred = model(train_data, adj_matrices, vec=vec)
            # TODO 为什么反向变化后，误差反而变大了？
            # pred = inverse_transform(pred,norml_max, norml_min).to(torch.float32)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        average_loss = sum(train_loss)/len(train_loss)
        print("Epoch:", n, "  loss:",average_loss)
        writer.add_scalar("train_loss", average_loss, n)
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                # train_data, targets,vec, adj_matrices = data
                train_data, targets, norm_targets, vec, adj_matrices = data
                # train_data, batch_range, batch_min = scale_features(train_data, targets, dn_range, up_range)
                adj_matrices =torch.from_numpy(preprocess_adj_tensor(adj_matrices.numpy(), args.sym_norm))
                adj_matrices = torch.from_numpy(np.nan_to_num(adj_matrices, copy=True))
                train_data = train_data.to(torch.float32).to(device)
                adj_matrices = adj_matrices.to(torch.float32).to(device)
                targets = targets.to(torch.float32).to(device)
                vec = vec.to(torch.float32).to(device)

                pred = model(train_data, adj_matrices, vec=vec)
                # pred = inverse_transform(pred,norml_max, norml_min).to(torch.float32)
                #targets = inverse_transform(targets, norml_max, norml_min).to(torch.float32)
                test_mse = criterion(pred, targets)
                val_loss.append(test_mse.item())
            average_val_loss = sum(val_loss) / len(val_loss)
            print("Epoch:", n, "  average_val_loss:", average_val_loss)
            writer.add_scalar("test_mse", average_val_loss, n)
            if average_loss < min(epoch_loss):
                # 保存所有的数据
                dir_save = os.path.join(args.save_trained_model,"best_model_train_time("+start_time +")")
                torch.save(model, dir_save)
            epoch_loss.append(average_loss)
    writer.close()
if __name__ == "__main__":
    seed_torch(seed=2023)
    run_model()

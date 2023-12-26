import torch
import torch.nn as nn
import torch.optim as optim
import argparse
# from model.model_new import ResidualGraphLSTM
from model.model_ST_plus import ResidualGraphLSTM
from torch.utils.data import DataLoader, Subset
from data_set.OD_dataSet import MyDataset
import os
from tensorboardX import SummaryWriter
import math
from utilities import scale_features, preprocess_adj_tensor, inverse_transform, reset_arg, creat_directory, seed_torch
import numpy as np
import datetime
import warnings
import os
from sklearn.metrics import mean_absolute_error

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
warnings.filterwarnings("ignore")
t = datetime.datetime.now();
s1 = "-";
s2 = "_"
start_time = str(t.year) + s1 + str(t.month) + s1 + str(t.day) + s2 + str(t.hour) + s1 + str(t.minute) + s1 + str(
    t.second)
parser = argparse.ArgumentParser()
# parameter of dataset.
# 主要的需要设置的训练结果
parser.add_argument('--use_adj', type=int, default=False, help='grid_num')
parser.add_argument('--use_vec', type=bool, default=False, help='if use vector future')
parser.add_argument('--use_three_branch', type=bool, default=True, help='if use three branch')
parser.add_argument('--use_dynamic_graph', type=bool, default=False, help='if dynamic graph')
parser.add_argument('--self_naming', type=str, default="plus_ST_4", help='self_naming')
parser.add_argument('--grid_num', type=int, default=100, help='grid_num')
parser.add_argument('--root', type=str, default=r"/app/project/wst/data/hangzhou_202005_1000node/split_data_0.25_hour", help='if use vector future')  ################
parser.add_argument('--save_folder', type=str, default=r'./result', help='result dir')

parser.add_argument('--len_trend', type=int, default=1, help='length of trend data')
parser.add_argument('--len_period', type=int, default=1, help='length of period data')
parser.add_argument('--len_closeness', type=int, default=2, help='length of closeness data')
parser.add_argument('--train_prop', type=float, default=0.8, help='proportion of training set')
parser.add_argument('--val_prop', type=float, default=0.2, help='proportion of validation set')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--height', type=int, default=16, help='input flow image height')
parser.add_argument('--width', type=int, default=16, help='input flow image width')
parser.add_argument('--latent_dim', type=int, default=128, help='external factor dimension')
parser.add_argument('--latent_dim_l', type=int, default=32, help='edge channel embedding dimension')
parser.add_argument('--regularizer_rate', type=int, default=0, help='edge channel embedding dimension')
parser.add_argument('--activation', type=str, default='relu', help='edge channel embedding dimension')
parser.add_argument('--is_batch_normalization', type=bool, default=True, help='edge channel embedding dimension')
parser.add_argument('--sym_norm', type=bool, default=True, help='SYM_NORM')
# 新增
parser.add_argument('--train_batchSize', type=int, default=32, help='train batch size')
parser.add_argument('--val_batchSize', type=int, default=32, help='val batch size')
parser.add_argument('--node_num', type=int, default=1000, help='node number')
parser.add_argument('--num_filters', type=int, default=10, help='number filters')
# 网络结构，如[h1,h2,h3,h4],则对应的模型结构中的输入输出如下：[h1,h2],[h2,h3],[h3,h4]]
parser.add_argument('--network_structure_e', type=list, default=[4, 128, 128], help='NETWORK_STRUCTURE_E')
parser.add_argument('--network_structure_d', type=list, default=[128, 128, 128], help='NETWORK_STRUCTURE_D')
parser.add_argument('--network_structure_l', type=list, default=[100, 128, 64, 128], help='NETWORK_STRUCTURE_L')

# parameter of training
parser.add_argument('--use_GPU', type=bool, default=True, help='use GPU')
parser.add_argument('--epochs', type=int, default=85, help='training epochs')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--seed', type=int, default=99, help='running seed')

parser.add_argument('--save_trained_model', type=str,
                    default='result/result_three_branch_100_plus_ST_4/save_trained_model',
                    help='result dir')
parser.add_argument('--device', type=str, default='cuda:1', help='cuda device')
parser.add_argument('--max_grad_norm', type=int, default=10, help='max gradient norm for gradient clip')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')


# parameter of model
def get_dataset_args(): args = parser.parse_args();return {"len_trend": args.len_trend, "len_period": args.len_period,
                                                           "len_closeness": args.len_closeness,
                                                           "train_prop": args.train_prop, "val_prop": args.val_prop}


def get_args(): return parser.parse_args()


def run_model():
    args = get_args()

    if args.use_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    train_data = MyDataset(args.root, "train", args.grid_num, use_adj=args.use_adj,
                           dynamic_graph=args.use_dynamic_graph, norml_max=None, norml_min=None)
    val_data = MyDataset(args.root, "val", args.grid_num, use_adj=args.use_adj, dynamic_graph=args.use_dynamic_graph,
                         norml_max=train_data.max, norml_min=train_data.min)

    reset_arg(train_data, args)

    val_loader = DataLoader(train_data, batch_size=args.val_batchSize, shuffle=False)
    modelFile = os.path.join(args.save_trained_model, "best_model_train_time(2023-11-30_9-10-2)")  ################
    model = torch.load(modelFile)
    mseLoss = nn.MSELoss(reduction='mean')
    maeLoss = nn.L1Loss(reduction='mean')
    model.eval()
    val_loss = []
    val_mae = []
    total_pred = []
    with torch.no_grad():
        for data in val_loader:
            # train_data, targets,vec, adj_matrices = data
            train_data, targets, norm_targets, vec, adj_matrices = data
            # train_data, batch_range, batch_min = scale_features(train_data, targets, dn_range, up_range)
            adj_matrices = torch.from_numpy(preprocess_adj_tensor(adj_matrices.numpy(), args.sym_norm))
            adj_matrices = torch.from_numpy(np.nan_to_num(adj_matrices, copy=True))
            train_data = train_data.to(torch.float32).to(device)
            adj_matrices = adj_matrices.to(torch.float32).to(device)
            targets = targets.to(torch.float32).to(device)
            vec = vec.to(torch.float32).to(device)

            pred = model(train_data, adj_matrices, vec=vec)
            for p in pred:
                total_pred.append(p.reshape(p.shape[0],).cpu().numpy())
            # pred = inverse_transform(pred,norml_max, norml_min).to(torch.float32)
            # targets = inverse_transform(targets, norml_max, norml_min).to(torch.float32)
            test_mse = mseLoss(pred, targets)
            test_mae = maeLoss(pred, targets)
            val_loss.append(test_mse.item())
            val_mae.append(test_mae.item())
        total_pred = np.array(total_pred)
        print(total_pred)
        average_val_mse = sum(val_loss) / len(val_loss)
        average_val_mae = sum(val_mae) / len(val_mae)
        print("MAE", average_val_mae)
        print("MSE:", average_val_mse)
        return total_pred


if __name__ == "__main__":
    seed_torch(seed=2023)
    pred = run_model()
    np.savez('./result/total_pred_' + str(100) + '.npz', data=pred)
    # data = np.load('./result/total_pred_' + str(100) + '.npz', allow_pickle=True)['data']
    # data=np.load(r"/app/wst/data/hangzhou_202005_1000node/split_data_0.25_hour/train100.npz")['data']
    # print(data)
    # print(data.shape)
    # t = data[0][:-1]
    # print(t)
    # print(t.reshape(100, 100))
    #
    # a=1

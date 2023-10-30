import torch
import torch.nn as nn
from model.MultiGraphCNN import MultiGraphCNN
from model.IdentityBlock import Identity_Block
from model.Convolution_block import Convolution_block
import copy
class ResidualGraphLSTM(nn.Module):
    def __init__(self, node_num,num_filters, network_structure_e, network_structure_l, network_structure_d,
                 latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization, use_vec,
                 use_three_branch):
        super(ResidualGraphLSTM, self).__init__()
        self.num_filters = num_filters
        self.network_structure_e = network_structure_e
        self.network_structure_l = network_structure_l
        self.network_structure_d = network_structure_d
        self.latent_dim = latent_dim
        self.latent_dim_l = latent_dim_l
        self.regularizer_rate = regularizer_rate
        self.node_num = node_num
        self.use_vec =use_vec
        self.node_dim = 4
        self.vec_dim = 256
        self.use_three_branch = use_three_branch
        self.network_structure_e[0] = self.node_dim  # self.node_emb_dim + self.vec_emb_dim
        if self.use_three_branch:
            self.vec_emb_dim = 128
            self.network_structure_vec_e = copy.deepcopy(self.network_structure_e)
            self.emb_vec_1 = nn.Sequential(
                nn.Linear(self.node_dim, 1),
                nn.ReLU())
            self.emb_vec_2 = nn.Sequential(
                nn.Linear(self.vec_dim, self.vec_emb_dim),
                nn.ReLU())
            self.network_structure_vec_e[0] = self.vec_emb_dim
            self.emb_vec_fusion = nn.Sequential(
                nn.Linear(self.network_structure_vec_e[-1] + self.network_structure_e[-1], self.network_structure_e[-1]),
                nn.ReLU())



            # 构建用于处理轨迹特征的图网络
            self.Convolution_vec_block_list = nn.ModuleList([])
            self.Identity_vec_Block_list = nn.ModuleList([])
            for i in range(1, len(self.network_structure_vec_e)):
                units = [int(self.network_structure_vec_e[i - 1]), int(self.network_structure_vec_e[i] / 4),
                         int(self.network_structure_vec_e[i] / 4), int(self.network_structure_vec_e[i])]
                self.Convolution_vec_block_list.append(
                    Convolution_block(num_filters, self.network_structure_vec_e, self.network_structure_l,
                                      self.network_structure_d, latent_dim, latent_dim_l, regularizer_rate, activation,
                                      is_batch_normalization, units))
                units = [int(self.network_structure_vec_e[i]), int(self.network_structure_vec_e[i] / 4),
                         int(self.network_structure_vec_e[i] / 4), int(self.network_structure_vec_e[i])]
                self.Identity_vec_Block_list.append(
                    Identity_Block(num_filters, self.network_structure_vec_e, self.network_structure_l,
                                   self.network_structure_d, latent_dim, latent_dim_l, regularizer_rate, activation,
                                   is_batch_normalization, units))
        self.activation = nn.ReLU()
        self.is_batch_normalization = is_batch_normalization
        # 构建用于处理节点特征的图网络
        self.Convolution_block_list = nn.ModuleList([])
        self.Identity_Block_list = nn.ModuleList([])
        for i in range(1,len(self.network_structure_e)):
            units = [int(self.network_structure_e[i-1]),int(self.network_structure_e[i] / 4), int(self.network_structure_e[i] / 4), int(self.network_structure_e[i])]
            self.Convolution_block_list.append(Convolution_block(num_filters, self.network_structure_e, self.network_structure_l,
                  self.network_structure_d, latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization, units))
            units = [int(self.network_structure_e[i]), int(self.network_structure_e[i] / 4), int(self.network_structure_e[i] / 4), int(self.network_structure_e[i])]
            self.Identity_Block_list.append(Identity_Block(num_filters, self.network_structure_e, self.network_structure_l,
                 self.network_structure_d, latent_dim, latent_dim_l,regularizer_rate,activation, is_batch_normalization, units))

        # 构建LSTM编码器
        self.lstm_encoded = nn.ModuleList([])
        for l in range(1, len(self.network_structure_l)):
            self.lstm_encoded.append(nn.LSTM(self.network_structure_l[l - 1], self.network_structure_l[l], batch_first=True))
        self.Linear_1 = nn.Sequential(nn.Linear(self.network_structure_e[-1]*self.node_num,self.latent_dim), nn.ReLU())
        self.Linear_2 = nn.Sequential(nn.Linear(self.network_structure_l[-1], self.latent_dim_l), nn.ReLU())
        self.Linear_3 = nn.Sequential(nn.Linear(self.latent_dim + self.latent_dim_l, self.node_num*self.network_structure_d[0]), nn.ReLU())

        # 构建解码器
        self.Convolution_block_decoder_list = nn.ModuleList([])
        self.Identity_Block_decoder_list = nn.ModuleList([])
        for i in range(1,len(self.network_structure_d)):
            units = [int(self.network_structure_d[i-1]),int(self.network_structure_d[i] / 4), int(self.network_structure_d[i] / 4), int(self.network_structure_d[i])]
            self.Convolution_block_decoder_list.append(Convolution_block(num_filters, self.network_structure_e, network_structure_l,
                 self.network_structure_d, latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization, units))
            units = [int(self.network_structure_d[i]), int(self.network_structure_d[i] / 4),
                     int(self.network_structure_d[i] / 4), int(self.network_structure_d[i])]
            self.Identity_Block_decoder_list.append(Identity_Block(num_filters, self.network_structure_e, network_structure_l,
                 self.network_structure_d, latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization, units))
        self.MultiGraphCNN = MultiGraphCNN(input_dim=self.network_structure_d[-1], output_dim=1, num_filters=num_filters, activation=torch.relu)
        self.vec_norm = nn.LayerNorm(normalized_shape=256)
        self.decoder_MLP = nn.Sequential(
            nn.Linear(self.network_structure_d[-1], int(self.network_structure_d[-1] / 2)),
            nn.ReLU(),
            nn.Linear(int(self.network_structure_d[-1] / 2), 1)
        )
    def forward(self, X_input, graph_conv_filters_input, vec= None):
        vec = self.vec_norm(vec)
        if self.use_vec:
            input_data = X_input
        else:
            input_data = X_input
        graph_encoded = self.Convolution_block_list[0](input_data, graph_conv_filters_input)
        graph_encoded = self.Identity_Block_list[0](graph_encoded, graph_conv_filters_input)
        for i in range(1,len(self.network_structure_e)-1):
            graph_encoded = self.Convolution_block_list[i](graph_encoded, graph_conv_filters_input)
            graph_encoded = self.Identity_Block_list[i](graph_encoded, graph_conv_filters_input)
        if self.use_three_branch:
            vec_emb_t = self.emb_vec_1(vec.permute(0, 1, 3, 2)).squeeze(-1)
            vec_emb = self.emb_vec_2(vec_emb_t)
            graph_vec_encoded = self.Convolution_vec_block_list[0](vec_emb, graph_conv_filters_input)
            graph_vec_encoded = self.Identity_vec_Block_list[0](graph_vec_encoded, graph_conv_filters_input)
            for i in range(1, len(self.network_structure_vec_e) - 1):
                graph_vec_encoded = self.Convolution_vec_block_list[i](graph_vec_encoded, graph_conv_filters_input)
                graph_vec_encoded = self.Identity_vec_Block_list[i](graph_vec_encoded, graph_conv_filters_input)
            graph_encoded = self.emb_vec_fusion(torch.cat((graph_encoded,graph_vec_encoded),dim=-1))

        X_lstm_input = X_input.permute(0,2,1)
        l_out, (_, _) = self.lstm_encoded[0](X_lstm_input)
        for i in range(1,len(self.network_structure_l)-2):
            l_out,(_,_) = self.lstm_encoded[i](l_out)
        l_out, (l_encoded, _) = self.lstm_encoded[-1](l_out)
        graph_encoded = graph_encoded.reshape(graph_encoded.shape[0],-1)
        l_encoded = l_encoded.squeeze(0)
        graph_encoded = self.Linear_1(graph_encoded)
        l_encoded = self.Linear_2(l_encoded)
        graph_decode = torch.cat((graph_encoded,l_encoded),dim=-1)
        graph_decode = self.Linear_3(graph_decode)
        graph_decode = graph_decode.reshape(graph_encoded.shape[0],self.node_num,-1)
        graph_decode = self.Convolution_block_decoder_list[0](graph_decode, graph_conv_filters_input)
        graph_decode = self.Identity_Block_decoder_list[0](graph_decode, graph_conv_filters_input)
        for i in range(1, len(self.network_structure_d)-1):
            graph_decode = self.Convolution_block_decoder_list[i](graph_decode, graph_conv_filters_input)
            graph_decode = self.Identity_Block_decoder_list[i](graph_decode, graph_conv_filters_input)
        # graph_decode = self.MultiGraphCNN(graph_decode,graph_conv_filters_input)
        graph_decode = self.decoder_MLP(graph_decode)
        return graph_decode

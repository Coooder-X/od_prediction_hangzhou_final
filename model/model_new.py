import torch
import torch.nn as nn
from model.MultiGraphCNN import MultiGraphCNN
from model.IdentityBlock import Identity_Block
from model.Convolution_block import Convolution_block
class ResidualGraphLSTM(nn.Module):
    def __init__(self, node_num,num_filters, network_structure_e, network_structure_l, network_structure_d,
                 latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization, use_vec):
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
        if self.use_vec:
            self.node_emb_dim = 32
            self.vec_emb_dim = 32
            self.network_structure_e[0] = self.node_emb_dim + self.vec_emb_dim
        if activation == "relu":
            self.activation = nn.ReLU()
        if self.use_vec:
            self.emb_node = nn.Sequential(
                nn.Linear(self.node_dim, self.node_emb_dim),
                nn.ReLU())
            self.emb_vec_1 = nn.Sequential(
                nn.Linear(self.node_dim, 1),
                nn.ReLU())
            self.emb_vec_2 = nn.Sequential(
                nn.Linear(self.vec_dim, self.vec_emb_dim),
                nn.ReLU())

        self.is_batch_normalization = is_batch_normalization
        self.Convolution_block_list = nn.ModuleList([])
        self.Identity_Block_list = nn.ModuleList([])
        for i in range(1,len(self.network_structure_e)):
            units = [int(network_structure_e[i-1]),int(network_structure_e[i] / 4), int(self.network_structure_e[i] / 4), int(network_structure_e[i])]
            self.Convolution_block_list.append(Convolution_block(num_filters, network_structure_e, network_structure_l,
                  network_structure_d, latent_dim, latent_dim_l, regularizer_rate,activation, is_batch_normalization, units))
            units = [int(network_structure_e[i]), int(network_structure_e[i] / 4),int(self.network_structure_e[i] / 4), int(network_structure_e[i])]
            self.Identity_Block_list.append(Identity_Block(num_filters, network_structure_e, network_structure_l,
                 network_structure_d, latent_dim, latent_dim_l,regularizer_rate,activation, is_batch_normalization, units))

        # 构建LSTM编码器
        self.lstm_encoded = nn.ModuleList([])
        for l in range(1, len(self.network_structure_l)):
            self.lstm_encoded.append(nn.LSTM(self.network_structure_l[l - 1], self.network_structure_l[l], batch_first=True))
        self.Linear_1 = nn.Sequential(nn.Linear(network_structure_e[-1]*self.node_num,self.latent_dim), nn.ReLU())
        self.Linear_2 = nn.Sequential(nn.Linear(self.latent_dim_l , self.latent_dim_l), nn.ReLU())
        self.Linear_3 = nn.Sequential(nn.Linear(self.latent_dim + self.latent_dim_l, self.node_num*self.network_structure_d[0]), nn.ReLU())
        # 构建解码器
        self.Convolution_block_decoder_list = nn.ModuleList([])
        self.Identity_Block_decoder_list = nn.ModuleList([])
        for i in range(1,len(self.network_structure_d)):
            units = [int(network_structure_d[i-1]),int(network_structure_d[i] / 4), int(self.network_structure_d[i] / 4), int(network_structure_d[i])]
            self.Convolution_block_decoder_list.append(Convolution_block(num_filters, network_structure_e, network_structure_l,
                 network_structure_d, latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization, units))
            units = [int(network_structure_d[i]), int(network_structure_d[i] / 4),
                     int(self.network_structure_d[i] / 4), int(network_structure_d[i])]
            self.Identity_Block_decoder_list.append(Identity_Block(num_filters, network_structure_e, network_structure_l,
                 network_structure_d, latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization, units))
        self.MultiGraphCNN = MultiGraphCNN(input_dim=network_structure_d[-1], output_dim=1, num_filters=num_filters, activation=torch.relu)

    def forward(self, X_input, graph_conv_filters_input, vec= None):
        if self.use_vec:
            vec = self.emb_vec_1(vec.permute(0,1,3,2)).squeeze(-1)
            vec_emb = self.emb_vec_2(vec)
            X_emb = self.emb_node(X_input)
            input_data = torch.cat((X_emb,vec_emb),dim=-1)
        else:
            input_data = X_input
        graph_encoded = self.Convolution_block_list[0](input_data, graph_conv_filters_input)
        graph_encoded = self.Identity_Block_list[0](graph_encoded, graph_conv_filters_input)
        for i in range(1,len(self.network_structure_e)-1):
            graph_encoded = self.Convolution_block_list[i](graph_encoded, graph_conv_filters_input)
            graph_encoded = self.Identity_Block_list[i](graph_encoded, graph_conv_filters_input)

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

        graph_encoded = self.Convolution_block_decoder_list[0](graph_decode, graph_conv_filters_input)
        graph_encoded = self.Identity_Block_decoder_list[0](graph_encoded, graph_conv_filters_input)
        for i in range(1, len(self.network_structure_d)-1):
            graph_encoded = self.Convolution_block_decoder_list[1](graph_encoded, graph_conv_filters_input)
            graph_encoded = self.Identity_Block_decoder_list[1](graph_encoded, graph_conv_filters_input)
        graph_encoded = self.MultiGraphCNN(graph_encoded,graph_conv_filters_input)
        return graph_encoded

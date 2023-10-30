import torch
import torch.nn as nn
from model.MultiGraphCNN import MultiGraphCNN
class Identity_Block(nn.Module):
    def __init__(self, num_filters, network_structure_e, network_structure_l, network_structure_d,
                     latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization, units):
        super(Identity_Block, self).__init__()
        self.num_filters = num_filters
        self.network_structure_e = network_structure_e
        self.network_structure_l = network_structure_l
        self.network_structure_d = network_structure_d
        self.latent_dim = latent_dim
        self.latent_dim_l = latent_dim_l
        self.regularizer_rate = regularizer_rate
        if activation == "relu":
            self.activation = nn.ReLU()
        # self.activation = activation
        self.is_batch_normalization = is_batch_normalization
        u0, u1, u2, u3 = units
        self.batch_norm0 = nn.LayerNorm(u0)
        self.conv1 = MultiGraphCNN(input_dim=u0, output_dim = u1, num_filters = num_filters, activation = torch.relu)
        self.batch_norm1 = nn.LayerNorm(u1)
        self.conv2 = MultiGraphCNN(input_dim=u1, output_dim=u2, num_filters=num_filters, activation=torch.relu)
        self.batch_norm2 = nn.LayerNorm(u2)
        self.conv3 = MultiGraphCNN(input_dim=u2, output_dim=u3, num_filters=num_filters, activation=torch.relu)
        self.batch_norm3 = nn.LayerNorm(u3)
        # self.conv4 = MultiGraphCNN(input_dim=u0, output_dim=u3, num_filters=num_filters, activation=torch.relu)
        # self.batch_norm4 = nn.LayerNorm(u3)

    def forward(self, X, graph_conv_filters_input):
        X = X + self.comput(X, graph_conv_filters_input)
        if self.is_batch_normalization:
            X = self.batch_norm3(X)
        X = self.activation(X)
        return X

    def comput(self,X, graph_conv_filters_input):
        X = self.conv1(X, graph_conv_filters_input)
        if self.is_batch_normalization:
            X = self.batch_norm1(X)
        # Second layer
        X = self.conv2(X, graph_conv_filters_input)
        if self.is_batch_normalization:
            X = self.batch_norm2(X)
        # Third layer
        X = self.conv3(X, graph_conv_filters_input)
        # if self.is_batch_normalization:
        #     X = self.batch_norm3(X)
        return X


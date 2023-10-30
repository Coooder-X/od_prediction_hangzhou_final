import torch
import torch.nn as nn
from model.IdentityBlock import Identity_Block
from model.Convolution_block import Convolution_block
class ResidualGraphLSTM(nn.Module):
    def __init__(self, num_filters, network_structure_e, network_structure_l, network_structure_d,
                 latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization):
        super(ResidualGraphLSTM, self).__init__()
        self.num_filters = num_filters
        self.network_structure_e = network_structure_e
        self.network_structure_l = network_structure_l
        self.network_structure_d = network_structure_d
        self.latent_dim = latent_dim
        self.latent_dim_l = latent_dim_l
        self.regularizer_rate = regularizer_rate
        self.lstm_input_dim = 10
        if activation == "relu":
            self.activation = nn.ReLU()
        # self.activation = activation
        self.is_batch_normalization = is_batch_normalization
        self.Convolution_block_list = nn.ModuleList([Convolution_block(num_filters, network_structure_e, network_structure_l,
                    network_structure_d,latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization,
                    [int(self.network_structure_e[i] / 4), int(self.network_structure_e[i] / 4), self.network_structure_e[i]])
                       for i in range(len(self.network_structure_e))])
        self.Identity_Block_list = nn.ModuleList([Identity_Block(num_filters, network_structure_e, network_structure_l,
                               network_structure_d, latent_dim, latent_dim_l, regularizer_rate, activation,is_batch_normalization,
                               [int(self.network_structure_e[i] / 4), int(self.network_structure_e[i] / 4),
                                self.network_structure_e[i]])for i in range(len(self.network_structure_e))])
        # 构建LSTM解码器
        lstm_layers = []
        lstm_layers.append(nn.LSTM(self.lstm_input_dim, self.network_structure_l[0], batch_first=True,
                                   return_sequences=True, activation=self.activation))
        for l in range(1, len(self.network_structure_l[0])):
            lstm_layers.append(nn.LSTM(self.network_structure_l[l - 1], self.network_structure_l[l], batch_first=True,
                                       return_sequences=True, activation=self.activation))
        lstm_layers.append(nn.LSTM(self.network_structure_l[-1], self.latent_dim, batch_first=True,
                                   return_sequences=False, activation=self.activation))
        self.lstm_encoded = nn.Sequential(*lstm_layers)
        # linear_layer = nn.Linear(graph_encoded.shape[1] + LATENT_DIM, shape[1] * shape[2])
        #

        self.Convolution_block_decoder_list = nn.ModuleList([Convolution_block(num_filters, network_structure_e, network_structure_l,
                    network_structure_d,latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization,
                    [int(self.network_structure_d[i] / 4), int(self.network_structure_d[i] / 4), self.network_structure_d[i]])
                       for i in range(len(self.network_structure_d))])
        self.Identity_Block_decoder_list = nn.ModuleList([Identity_Block(num_filters, network_structure_e, network_structure_l,
                               network_structure_d, latent_dim, latent_dim_l, regularizer_rate, activation,is_batch_normalization,
                               [int(self.network_structure_d[i] / 4), int(self.network_structure_d[i] / 4),
                                self.network_structure_d[i]])for i in range(len(self.network_structure_d))])





    def forward(self, X_input, graph_conv_filters_input):
        graph_encoded = X_input
        for i in range(len(self.network_structure_e)):
            units = [int(self.network_structure_e[i] / 4), int(self.network_structure_e[i] / 4), self.network_structure_e[i]]
            graph_encoded = self.convolution_block(graph_encoded, graph_conv_filters_input, units)

        # Shape info needed to build Decoder Model
        shape = graph_encoded.size()

        # Generate the latent vector
        graph_encoded = graph_encoded.view(graph_encoded.size(0), -1)
        graph_encoded = nn.Linear(graph_encoded.size(1), self.latent_dim)(graph_encoded)
        graph_encoded = self.activation(graph_encoded)

        # Pass through the LSTM encoding layers with residual connections
        lstm_encoded = X_input.permute(0, 2, 1)
        for i in range(len(self.network_structure_l)):
            lstm_encoded, _ = self.lstm_encoded_layers[i](lstm_encoded)

        # Generate the LSTM encoded vector
        lstm_encoded = lstm_encoded[:, -1, :]
        lstm_encoded = nn.Linear(lstm_encoded.size(1), self.latent_dim_l)(lstm_encoded)
        lstm_encoded = self.activation(lstm_encoded)

        # Concatenate the graph and LSTM encoded vectors
        merge_encoded = torch.cat((graph_encoded, lstm_encoded), dim=1)

        # Pass through the graph decoding layers with residual connections
        x = nn.Linear(merge_encoded.size(1), shape[1] * shape[2])(merge_encoded)
        x = self.activation(x)
        x = x.view(-1, shape[1], shape[2])
        for i in range(len(self.network_structure_d)):
            units = [int(self.network_structure_d[i] / 4), int(self.network_structure_d[i] / 4), self.network_structure_d[i]]
            x = self.convolution_block(x, graph_conv_filters_input, units)

        # Pass through the final output layer
        output = self.output_layer(x, graph_conv_filters_input)

        return output


    # def output_layer(self, X, graph_conv_filters_input):
    #     # Pass through the final output layer
    #     output = MultiGraphCNN(train_y.size(2), self.num_filters, activation='linear',
    #                            activity_regularizer=self.regularizer_rate)([X, graph_conv_filters_input])
    #
    #     return output

# class IdentityBlock(nn.Module):
#     def __init__(self, num_filters, network_structure_e, network_structure_l, network_structure_d,
#                      latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization, units):
#         super(IdentityBlock, self).__init__()
#         self.num_filters = num_filters
#         self.network_structure_e = network_structure_e
#         self.network_structure_l = network_structure_l
#         self.network_structure_d = network_structure_d
#         self.latent_dim = latent_dim
#         self.latent_dim_l = latent_dim_l
#         self.regularizer_rate = regularizer_rate
#         if activation == "relu":
#             self.activation = nn.ReLU()
#         # self.activation = activation
#         self.is_batch_normalization = is_batch_normalization
#         u1, u2, u3 = units
#         self.conv1 = MultiGraphCNN(u1, self.num_filters, activation=self.activation,
#                           activity_regularizer=self.regularizer_rate)#([X, graph_conv_filters_input])
#         self.conv2 = MultiGraphCNN(u2, self.num_filters, activation=self.activation,
#                                    activity_regularizer=self.regularizer_rate)#([X, graph_conv_filters_input])
#         self.conv3 = MultiGraphCNN(u3, self.num_filters, activation=self.activation,
#                                    activity_regularizer=self.regularizer_rate)  # ([X, graph_conv_filters_input])
#         self.batch_norm = nn.BatchNorm2d(num_filters)
#
#     def forward(self, X, graph_conv_filters_input):
#         # Save the input value
#         X_shortcut = X
#         X = self.conv1(X, graph_conv_filters_input)
#         X = self.activation(X)
#         if self.is_batch_normalization:
#             X = self.batch_norm(X)
#         # Second layer
#         X = self.conv2(X, graph_conv_filters_input)
#         X = self.activation(X)
#         if self.is_batch_normalization:
#             X = self.batch_norm(X)
#         # Third layer
#         X = self.conv3(X, graph_conv_filters_input)
#         X = self.activation(X)
#         if self.is_batch_normalization:
#             X = self.batch_norm(X)
#         X = X + X_shortcut
#         X = self.activation(X)
#
#         return X
#
#
# class Convolution_block(nn.Module):
#     def __init__(self, num_filters, network_structure_e, network_structure_l, network_structure_d,
#                      latent_dim, latent_dim_l, regularizer_rate, activation, is_batch_normalization, units):
#         super(Convolution_block, self).__init__()
#         self.num_filters = num_filters
#         self.network_structure_e = network_structure_e
#         self.network_structure_l = network_structure_l
#         self.network_structure_d = network_structure_d
#         self.latent_dim = latent_dim
#         self.latent_dim_l = latent_dim_l
#         self.regularizer_rate = regularizer_rate
#         if activation == "relu":
#             self.activation = nn.ReLU()
#         # self.activation = activation
#         self.is_batch_normalization = is_batch_normalization
#         u1, u2, u3 = units
#         self.conv1 = MultiGraphCNN(u1, self.num_filters, activation=self.activation,
#                           activity_regularizer=self.regularizer_rate)#([X, graph_conv_filters_input])
#         self.conv2 = MultiGraphCNN(u2, self.num_filters, activation=self.activation,
#                                    activity_regularizer=self.regularizer_rate)#([X, graph_conv_filters_input])
#         self.conv3 = MultiGraphCNN(u3, self.num_filters, activation=self.activation,
#                                    activity_regularizer=self.regularizer_rate)  # ([X, graph_conv_filters_input])
#         self.conv4 = MultiGraphCNN(u3, self.num_filters, activation=self.activation,
#                                    activity_regularizer=self.regularizer_rate)  # ([X, graph_conv_filters_input])
#         self.batch_norm = nn.BatchNorm2d(num_filters)
#
#     def forward(self, X, graph_conv_filters_input):
#         # Save the input value
#         X_shortcut = X
#         X = self.conv1(X, graph_conv_filters_input)
#         X = self.activation(X)
#         if self.is_batch_normalization:
#             X = self.batch_norm(X)
#         # Second layer
#         X = self.conv2(X, graph_conv_filters_input)
#         X = self.activation(X)
#         if self.is_batch_normalization:
#             X = self.batch_norm(X)
#         # Third layer
#         X = self.conv3(X, graph_conv_filters_input)
#         X = self.activation(X)
#         if self.is_batch_normalization:
#             X = self.batch_norm(X)
#
#         X_shortcut = self.conv3(X_shortcut, graph_conv_filters_input)
#         X_shortcut = self.activation(X_shortcut)
#         if self.is_batch_normalization:
#             X_shortcut = self.batch_norm(X_shortcut)
#
#         X = X + X_shortcut
#         X = self.activation(X)
#
#         return X
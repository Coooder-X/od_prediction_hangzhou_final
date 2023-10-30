import torch
import torch.nn as nn

class MultiGraphCNN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_filters,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None):
        super(MultiGraphCNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        # Initialize weight and bias parameters
        # self.input_dim = None
        self.weight = None
        self.bias = None
        self.build()

    def reset_parameters(self):
        if self.kernel_initializer == 'glorot_uniform':
            nn.init.xavier_uniform_(self.weight)
        elif self.kernel_initializer == 'glorot_normal':
            nn.init.xavier_normal_(self.weight)
        elif self.kernel_initializer == 'he_uniform':
            nn.init.kaiming_uniform_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        elif self.kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        else:
            raise ValueError(f"Invalid kernel_initializer: {self.kernel_initializer}")
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def build(self):
        kernel_shape = (self.num_filters * self.input_dim, self.output_dim)
        self.weight = nn.Parameter(torch.Tensor(*kernel_shape))

        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def forward(self, X, graph_conv_filters_input):
        output = self.graph_conv_op(X, self.num_filters, graph_conv_filters_input)
        if self.use_bias:
            output = output + self.bias.view(1, -1)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1], self.output_dim)
        return output_shape

    def graph_conv_op(self, x, num_filters, graph_conv_filters):
        if len(x.shape) == 2:
            conv_op = torch.mm(graph_conv_filters, x)
            conv_op = torch.split(conv_op, num_filters, dim=0)
            conv_op = torch.cat(conv_op, dim=1)
        elif len(x.shape) == 3:
            conv_op = torch.bmm(graph_conv_filters, x)
            # print(graph_conv_filters.shape, x.shape, conv_op.shape)
            conv_op = torch.split(conv_op, int(conv_op.shape[1]/num_filters), dim=1)
            conv_op = torch.cat(conv_op, dim=2)
            # print(conv_op.shape)
        else:
            raise ValueError('x must be either 2 or 3 dimension tensor'
                             'Got input shape: ' + str(x.shape))

        conv_out = torch.matmul(conv_op, self.weight)
        return conv_out
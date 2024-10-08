import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm
#dropout_rate
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout_rate=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class IMUTCNModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, kernel_size=2, dropout_rate=0.2):
        super(IMUTCNModel, self).__init__()
        layers = []
        num_levels = len(hidden_sizes)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else hidden_sizes[i-1]
            out_channels = hidden_sizes[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout_rate=dropout_rate)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape from (batch, seq_len, features) to (batch, features, seq_len)
        x = self.network(x)
        x = x[:, :, -1]  # Take the last time step
        return self.linear(x)
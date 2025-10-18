from torch import nn as nn

def mlp(input_size, output_size, num_layer, hidden_size):
    layers = []
    for i in range(num_layer):
        in_features = input_size if i == 0 else hidden_size
        out_features = output_size if i == num_layer - 1 else hidden_size

        linear_layer = nn.Linear(in_features, out_features)
        # 权重初始化
        nn.init.xavier_uniform_(linear_layer.weight, gain=1.0)
        nn.init.zeros_(linear_layer.bias)
        layers.append(linear_layer)

        # 非最后一层添加激活函数
        if i != num_layer - 1:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)

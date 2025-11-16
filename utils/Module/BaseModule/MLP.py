from torch import nn as nn

def mlp(input_size, output_size, num_layer, hidden_size, dropout=0.0):
    layers = []
    for i in range(num_layer):
        in_features = input_size if i == 0 else hidden_size
        out_features = output_size if i == num_layer - 1 else hidden_size

        linear_layer = nn.Linear(in_features, out_features)
        # 权重初始化
        nn.init.xavier_uniform_(linear_layer.weight, gain=1.0)
        nn.init.zeros_(linear_layer.bias)
        layers.append(linear_layer)

        # 如果不是最后一层，添加激活函数和可能的 dropout 层
        if i != num_layer - 1:
            layers.append(nn.ReLU())
            # 只有当 dropout 值大于 0 时才添加 dropout 层
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)

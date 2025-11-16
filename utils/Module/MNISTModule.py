import torch
from torch import nn as nn

from utils.Module import BaseModule


class MNISTModule(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.0, num_layers: int = 1):
        super().__init__()
        self.input_size = 28
        self.hidden_size = hidden_size
        self.output_size = 10
        self.dropout_p = dropout
        self.num_layers = num_layers

        # 添加输出层
        self.gru = None
        self.fc = nn.Linear(hidden_size, self.output_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        init_range = 0.1
        nn.init.uniform_(self.fc.weight, -init_range, init_range)
        nn.init.zeros_(self.fc.bias)

    def _init_hidden(self, batch_size, device, dtype):
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, x):
        if self.gru is None:
            # 请先选择gru模型
            raise NotImplementedError("请先选择gru模型")
        x = x.reshape(-1, 28, 28)
        gru_out, hidden = self.gru(x)
        if self.dropout is not None:
            gru_out = self.dropout(gru_out)
        outputs = self.fc(gru_out)
        return outputs[:, -1, :]

    def get_info(self):
        base_info = f"模型={self.name}| hidden_size={self.hidden_size}"
        base_info += f"| num_layers={self.num_layers}"
        if self.dropout_p > 0:
            base_info += f"| dropout={self.dropout_p}"
        return base_info


class LocalGRU(MNISTModule):
    def __init__(self, hidden_size: int, dropout: float = 0.0, num_layers: int = 1):
        super().__init__(hidden_size, dropout, num_layers)
        self.name = "LocalGRU"
        self.gru = BaseModule.LocalGRU(self.input_size, hidden_size)


class TorchGRU(MNISTModule):
    def __init__(self, hidden_size: int, num_layers=1, batch_first=True,
                 dropout: float = 0.0):
        super().__init__(hidden_size, dropout, num_layers)
        self.name = "TorchGRU"
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0  # 只有在多层时才使用dropout
        )


class HGRU(MNISTModule):
    def __init__(self, hidden_size: int, mpl_n=2, mpl_h=144, dropout: float = 0.0, num_layers: int = 1):
        super().__init__(hidden_size, dropout, num_layers)
        self.name = "HGRU"
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        self.gru = BaseModule.HGRU(self.input_size, hidden_size, mpl_n, mpl_h)

    def get_info(self):
        base_info = super().get_info()
        base_info += f"| mlp=[{self.mpl_n},{self.mpl_h}]"
        return base_info


class NormGRU(MNISTModule):
    def __init__(self, hidden_size: int, dropout: float = 0.0, num_layers: int = 1):
        super().__init__(hidden_size, dropout, num_layers)
        self.name = "normGRU"
        self.gru = BaseModule.NormGRU(self.input_size, hidden_size)


class NormHGRU(MNISTModule):
    def __init__(self, hidden_size: int, mpl_n=2, mpl_h=144, dropout: float = 0.0, num_layers: int = 1):
        super().__init__(hidden_size, dropout, num_layers)
        self.name = "normHGRU"
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        self.gru = BaseModule.NormHGRU(self.input_size, hidden_size, mpl_n, mpl_h)

    def get_info(self):
        base_info = super().get_info()
        base_info += f"| mlp=[{self.mpl_n},{self.mpl_h}]"
        return base_info

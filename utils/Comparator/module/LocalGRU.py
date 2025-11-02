import torch
from torch import nn
from .BaseGRU import BaseGRU


class LocalGRU(BaseGRU):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(LocalGRU, self).__init__(input_size, hidden_size, num_layers, dropout)

    def _create_gru_cell(self, input_size, hidden_size):
        """创建单个GRU单元"""
        cell = nn.Module()
        cell.r = nn.Linear(input_size + hidden_size, hidden_size)
        cell.z = nn.Linear(input_size + hidden_size, hidden_size)
        cell.h = nn.Linear(input_size + hidden_size, hidden_size)
        
        # 使用PyTorch GRU默认的初始化策略
        k = 1.0 / hidden_size
        sqrt_k = k ** 0.5
        # 权重初始化
        nn.init.uniform_(cell.r.weight, -sqrt_k, sqrt_k)
        nn.init.uniform_(cell.z.weight, -sqrt_k, sqrt_k)
        nn.init.uniform_(cell.h.weight, -sqrt_k, sqrt_k)
        # 偏置初始化
        nn.init.uniform_(cell.r.bias, -sqrt_k, sqrt_k)
        nn.init.uniform_(cell.z.bias, -sqrt_k, sqrt_k)
        nn.init.uniform_(cell.h.bias, -sqrt_k, sqrt_k)
        
        return cell

    def _init_hidden(self, batch_size, device, dtype):
        """初始化隐藏状态"""
        # 为每一层初始化隐藏状态
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        return hidden

    def _forward_step(self, cell, x, hidden, layer):
        """单步前向传播"""
        # 重置门和更新门计算
        combined = torch.cat([x, hidden], dim=1)
        r = torch.sigmoid(cell.r(combined))
        z = torch.sigmoid(cell.z(combined))
        
        # 候选隐藏状态计算
        combined_reset = torch.cat([x, r * hidden], dim=1)
        h_tilde = torch.tanh(cell.h(combined_reset))
        
        # 更新隐藏状态
        hidden = (1 - z) * hidden + z * h_tilde
        return hidden
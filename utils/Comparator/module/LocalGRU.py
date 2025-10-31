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
        
        # 初始化权重和偏置
        for linear in [cell.r, cell.z, cell.h]:
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
        nn.init.constant_(cell.r.bias, -1.0)  # 使用更温和的初始化值
        
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
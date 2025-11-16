import torch
from torch import nn
from utils.Module.BaseModule import mlp
from .BaseGRU import BaseGRU


class HGRU(BaseGRU):
    def __init__(self, input_size, hidden_size, mpl_n=2, mpl_h=144, num_layers=1, dropout=0.0):
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        super(HGRU, self).__init__(input_size, hidden_size, num_layers, dropout)
        self.name = "HGRU"

    def _create_gru_cell(self, input_size, hidden_size):
        """创建单个HGRU单元"""
        cell = nn.Module()
        cell.r = nn.Linear(input_size + hidden_size, hidden_size)
        cell.z = nn.Linear(input_size + hidden_size, hidden_size)
        cell.h = mlp(input_size + hidden_size, hidden_size, self.mpl_n, self.mpl_h)

        # 初始化权重和偏置
        for linear in [cell.r, cell.z]:
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
        nn.init.constant_(cell.r.bias, -1.0)  # 使用更温和的初始化值
        
        return cell

    def _init_hidden(self, batch_size, device, dtype):
        """初始化隐藏状态"""
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)

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
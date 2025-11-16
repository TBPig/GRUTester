import torch
from torch import nn
from .BaseGRU import BaseGRU


class NormGRU(BaseGRU):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(NormGRU, self).__init__(input_size, hidden_size, num_layers, dropout)
        self.name = "NormGRU"

    def _create_gru_cell(self, input_size, hidden_size):
        """创建单个NormGRU单元"""
        cell = nn.Module()
        cell.h = nn.Linear(input_size + hidden_size, hidden_size)
        
        # 初始化权重和偏置
        nn.init.xavier_uniform_(cell.h.weight, gain=1.0)
        nn.init.zeros_(cell.h.bias)
        
        return cell

    def _init_hidden(self, batch_size, device, dtype):
        """初始化隐藏状态"""
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        # 对于NormGRU，第一维设为1
        hidden[:, :, 0] = 1
        return hidden

    def _forward_step(self, cell, x, hidden, layer):
        """单步前向传播"""
        # 计算候选隐藏状态
        combined = torch.cat([x, hidden], dim=1)
        h_tilde = cell.h(combined)
        
        # 更新隐藏状态并标准化
        h = hidden + h_tilde
        norm = torch.norm(h, dim=1, keepdim=True)
        hidden = h / norm
        return hidden

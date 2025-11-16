import torch
from torch import nn
from utils.Module.BaseModule import mlp
from .BaseGRU import BaseGRU


class NormHGRU(BaseGRU):
    def __init__(self, input_size, hidden_size, mpl_n=2, mpl_h=144, num_layers=1, dropout=0.0):
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h
        super(NormHGRU, self).__init__(input_size, hidden_size, num_layers, dropout)
        self.name = "NormHGRU"

    def _create_gru_cell(self, input_size, hidden_size):
        """创建单个NormHGRU单元"""
        cell = nn.Module()
        cell.h = mlp(input_size + hidden_size, hidden_size, self.mpl_n, self.mpl_h)
        return cell

    def _init_hidden(self, batch_size, device, dtype):
        """初始化隐藏状态"""
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype)
        # 对于NormHGRU，第一维设为1
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

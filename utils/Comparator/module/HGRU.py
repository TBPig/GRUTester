import torch
from torch import nn

from utils.MLP import mlp


class HGRU(nn.Module):
    def __init__(self, input_size, hidden_size, mpl_n=2, mpl_h=144):
        super(HGRU, self).__init__()
        self.hidden_size:int = hidden_size
        self.mpl_n = mpl_n
        self.mpl_h = mpl_h

        self.r = nn.Linear(input_size + hidden_size, hidden_size)
        self.z = nn.Linear(input_size + hidden_size, hidden_size)
        self.h = mlp(input_size + hidden_size, hidden_size, mpl_n, mpl_h)

        # 初始化权重和偏置
        for linear in [self.r, self.z]:
            nn.init.xavier_uniform_(linear.weight, gain=1.0)
            nn.init.zeros_(linear.bias)
        nn.init.constant_(self.r.bias, -1.0)  # 使用更温和的初始化值

    def _init_hidden(self, batch_size, device, dtype):
        """初始化隐藏状态"""
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def forward(self, inputs, hidden=None):
        batch_size, seq_len, input_size = inputs.size()
        if hidden is None:
            hidden = self._init_hidden(batch_size, inputs.device, inputs.dtype)

        # 预分配outputs张量以提高性能
        outputs = torch.empty(batch_size, seq_len, self.hidden_size, device=inputs.device, dtype=inputs.dtype)

        for i in range(seq_len):
            x = inputs[:, i, :]

            # 重置门和更新门计算
            combined = torch.cat([x, hidden], dim=1)
            r = torch.sigmoid(self.r(combined))
            z = torch.sigmoid(self.z(combined))

            # 候选隐藏状态计算
            combined_reset = torch.cat([x, r * hidden], dim=1)
            h_tilde = torch.tanh(self.h(combined_reset))

            # 更新隐藏状态
            hidden = (1 - z) * hidden + z * h_tilde

            outputs[:, i, :] = hidden

        return outputs, hidden